/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/base64"
	"encoding/pem"
	"fmt"
	"math/big"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/security"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	webhookNamespace   = "cluster-system"
	certValidityPeriod = 100 * 365 * 24 * time.Hour
)

// webhookCertBundle holds generated CA + TLS certificate material.
type webhookCertBundle struct {
	caBundleB64 string // base64-encoded CA cert PEM (for caBundle template injection)
	tlsCrt      []byte // PEM-encoded TLS certificate
	tlsKey      []byte // PEM-encoded TLS private key
	caCrt       []byte // PEM-encoded CA certificate (for Secret ca.crt)
}

// generateWebhookCertificates generates a self-signed CA and a TLS certificate
// signed by that CA, using EC P-256 keys.
func generateWebhookCertificates(dnsNames []string) (*webhookCertBundle, error) {
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generate CA key: %w", err)
	}

	caTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "bke-webhook-ca"},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(certValidityPeriod),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}

	caCertDER, err := x509.CreateCertificate(rand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		return nil, fmt.Errorf("create CA cert: %w", err)
	}

	tlsKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generate TLS key: %w", err)
	}

	tlsTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: dnsNames[0]},
		DNSNames:     dnsNames,
		NotBefore:    time.Now(),
		NotAfter:     time.Now().Add(certValidityPeriod),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}

	tlsCertDER, err := x509.CreateCertificate(rand.Reader, tlsTemplate, caTemplate, &tlsKey.PublicKey, caKey)
	if err != nil {
		return nil, fmt.Errorf("create TLS cert: %w", err)
	}

	caCrtPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCertDER})
	tlsCrtPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: tlsCertDER})
	tlsKeyDER, err := x509.MarshalECPrivateKey(tlsKey)
	if err != nil {
		return nil, fmt.Errorf("marshal TLS key: %w", err)
	}
	tlsKeyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: tlsKeyDER})

	return &webhookCertBundle{
		caBundleB64: base64.StdEncoding.EncodeToString(caCrtPEM),
		tlsCrt:      tlsCrtPEM,
		tlsKey:      tlsKeyPEM,
		caCrt:       caCrtPEM,
	}, nil
}

// ensureClusterAPISecrets ensures TLS Secrets and the AES key Secret exist on the
// target cluster. Returns base64-encoded CA bundles for template injection.
// If secrets already exist, their CA certs are reused to keep caBundle consistent.
func (c *Client) ensureClusterAPISecrets() (capiCaBundle, bkeCaBundle string, err error) {
	log.Info("Ensuring cluster-api webhook secrets and encryption key on target cluster")
	ctx := c.Ctx
	if ctx == nil {
		ctx = context.Background()
	}

	// Ensure the cluster-system namespace exists before creating secrets.
	if _, getErr := c.ClientSet.CoreV1().Namespaces().Get(ctx, webhookNamespace, metav1.GetOptions{}); getErr != nil {
		if apierrors.IsNotFound(getErr) {
			_, createErr := c.ClientSet.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{Name: webhookNamespace},
			}, metav1.CreateOptions{})
			if createErr != nil && !apierrors.IsAlreadyExists(createErr) {
				log.Errorf("failed to create namespace %s: %v", webhookNamespace, createErr)
				return "", "", fmt.Errorf("create namespace %s: %w", webhookNamespace, createErr)
			}
			log.Infof("namespace %s created", webhookNamespace)
		} else {
			log.Errorf("failed to get namespace %s: %v", webhookNamespace, getErr)
			return "", "", fmt.Errorf("get namespace %s: %w", webhookNamespace, getErr)
		}
	}

	capiCaBundle, err = c.ensureWebhookTLSSecret(ctx, "capi-webhook-service-cert", []string{
		"capi-webhook-service.cluster-system.svc",
		"capi-webhook-service.cluster-system.svc.cluster.local",
		"capi-webhook-service",
		"capi-kubeadm-bootstrap-webhook-service.cluster-system.svc",
		"capi-kubeadm-bootstrap-webhook-service.cluster-system.svc.cluster.local",
		"capi-kubeadm-bootstrap-webhook-service",
		"capi-kubeadm-control-plane-webhook-service.cluster-system.svc",
		"capi-kubeadm-control-plane-webhook-service.cluster-system.svc.cluster.local",
		"capi-kubeadm-control-plane-webhook-service",
	})
	if err != nil {
		log.Errorf("failed to ensure CAPI webhook secret: %v", err)
		return "", "", fmt.Errorf("ensure CAPI webhook secret: %w", err)
	}

	bkeCaBundle, err = c.ensureWebhookTLSSecret(ctx, "bke-webhook-secret", []string{
		"bke-webhook-service.cluster-system.svc",
		"bke-webhook-service.cluster-system.svc.cluster.local",
	})
	if err != nil {
		log.Errorf("failed to ensure BKE webhook secret: %v", err)
		return "", "", fmt.Errorf("ensure BKE webhook secret: %w", err)
	}

	if err = c.ensurePasswordEncryptionKey(ctx); err != nil {
		log.Errorf("failed to ensure password encryption key: %v", err)
		return "", "", fmt.Errorf("ensure password encryption key: %w", err)
	}

	log.Info("cluster-api webhook secrets and encryption key ensured successfully")
	return capiCaBundle, bkeCaBundle, nil
}

// ensureWebhookTLSSecret creates a TLS Secret if it doesn't exist, or reads the
// existing CA cert if it does. Returns the base64-encoded CA cert for caBundle
// template injection.
//
// When the Secret exists but is missing ca.crt (e.g. leftover from a prior
// incomplete deployment), new certificates are generated and the Secret is
// updated in-place so the returned caBundle always matches the Secret content.
func (c *Client) ensureWebhookTLSSecret(ctx context.Context, secretName string, dnsNames []string) (string, error) {
	existing, err := c.ClientSet.CoreV1().Secrets(webhookNamespace).Get(ctx, secretName, metav1.GetOptions{})
	if err == nil {
		// Secret already exists -- reuse its CA cert so caBundle stays consistent.
		if caCrt, ok := existing.Data["ca.crt"]; ok && len(caCrt) > 0 {
			log.Infof("webhook secret %s already exists, reusing existing CA cert", secretName)
			return base64.StdEncoding.EncodeToString(caCrt), nil
		}
		// Secret exists but ca.crt is missing. Generate new certs and update
		// the Secret so the returned caBundle is consistent with the stored data.
		log.Warnf("webhook secret %s exists but ca.crt is missing, generating new certs and updating", secretName)
		certs, genErr := generateWebhookCertificates(dnsNames)
		if genErr != nil {
			return "", fmt.Errorf("generate webhook certs for %s: %w", secretName, genErr)
		}
		err = phaseutil.RetryOnConflict(func() error {
			latest, getErr := c.ClientSet.CoreV1().Secrets(webhookNamespace).Get(ctx, secretName, metav1.GetOptions{})
			if getErr != nil {
				return getErr
			}
			latest.Data = map[string][]byte{
				"tls.crt": certs.tlsCrt,
				"tls.key": certs.tlsKey,
				"ca.crt":  certs.caCrt,
			}
			_, updateErr := c.ClientSet.CoreV1().Secrets(webhookNamespace).Update(ctx, latest, metav1.UpdateOptions{})
			return updateErr
		})
		if err != nil {
			log.Errorf("retry update webhook secret %s failed: %v", secretName, err)
			return "", fmt.Errorf("retry update secret %s: %w", secretName, err)
		}
		log.Infof("webhook secret %s updated with new certs", secretName)
		return certs.caBundleB64, nil
	} else if !apierrors.IsNotFound(err) {
		log.Errorf("failed to get webhook secret %s: %v", secretName, err)
		return "", fmt.Errorf("get secret %s: %w", secretName, err)
	}

	// Secret does not exist -- generate new certs and create it.
	log.Infof("webhook secret %s does not exist, generating new certs and creating", secretName)
	certs, err := generateWebhookCertificates(dnsNames)
	if err != nil {
		return "", fmt.Errorf("generate webhook certs for %s: %w", secretName, err)
	}

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: webhookNamespace,
		},
		Type: corev1.SecretTypeTLS,
		Data: map[string][]byte{
			"tls.crt": certs.tlsCrt,
			"tls.key": certs.tlsKey,
			"ca.crt":  certs.caCrt,
		},
	}

	_, err = c.ClientSet.CoreV1().Secrets(webhookNamespace).Create(ctx, secret, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		log.Errorf("failed to create webhook secret %s: %v", secretName, err)
		return "", fmt.Errorf("create secret %s: %w", secretName, err)
	}
	log.Infof("webhook secret %s created successfully", secretName)
	return certs.caBundleB64, nil
}

// resolveEncryptionKey returns the base64-encoded AES key of the management
// cluster (this process), so that all clusters managed by the same management
// cluster share the same key. This prevents cross-cluster double-encryption
// of BKENode SSH passwords. It errors out when the management cluster key is
// unavailable instead of generating a random one.
func (c *Client) resolveEncryptionKey(ctx context.Context) (string, error) {
	encoded, err := security.CurrentKeyBase64(ctx)
	if err != nil {
		return "", fmt.Errorf("load management cluster encryption key: %w", err)
	}
	log.Info("reusing management cluster encryption key for target cluster")
	return encoded, nil
}

// ensurePasswordEncryptionKey creates the AES key Secret if it doesn't exist.
func (c *Client) ensurePasswordEncryptionKey(ctx context.Context) error {
	_, err := c.ClientSet.CoreV1().Secrets(webhookNamespace).Get(ctx, "bke-password-encryption-key", metav1.GetOptions{})
	if err == nil {
		log.Info("password encryption key secret already exists")
		return nil
	}
	if !apierrors.IsNotFound(err) {
		log.Errorf("failed to get encryption key secret: %v", err)
		return fmt.Errorf("get encryption key secret: %w", err)
	}

	encoded, err := c.resolveEncryptionKey(ctx)
	if err != nil {
		return err
	}

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "bke-password-encryption-key",
			Namespace: webhookNamespace,
		},
		Type: corev1.SecretTypeOpaque,
		Data: map[string][]byte{
			"encryption-key": []byte(encoded),
		},
	}

	_, err = c.ClientSet.CoreV1().Secrets(webhookNamespace).Create(ctx, secret, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		log.Errorf("failed to create encryption key secret: %v", err)
		return fmt.Errorf("create encryption key secret: %w", err)
	}
	log.Info("password encryption key secret created successfully")
	return nil
}
