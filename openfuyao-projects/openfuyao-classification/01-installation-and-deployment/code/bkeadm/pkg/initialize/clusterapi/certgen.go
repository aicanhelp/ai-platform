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

package clusterapi

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

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
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
	// 1. Generate CA keypair (EC P-256)
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Errorf("failed to generate CA key: %v", err)
		return nil, fmt.Errorf("generate CA key: %w", err)
	}

	// 2. Create self-signed CA certificate
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
		log.Errorf("failed to create CA cert: %v", err)
		return nil, fmt.Errorf("create CA cert: %w", err)
	}

	// 3. Generate TLS keypair (EC P-256)
	tlsKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		log.Errorf("failed to generate TLS key: %v", err)
		return nil, fmt.Errorf("generate TLS key: %w", err)
	}

	// 4. Create TLS certificate signed by CA
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
		log.Errorf("failed to create TLS cert: %v", err)
		return nil, fmt.Errorf("create TLS cert: %w", err)
	}

	// 5. PEM encode
	caCrtPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCertDER})
	tlsCrtPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: tlsCertDER})
	tlsKeyDER, err := x509.MarshalECPrivateKey(tlsKey)
	if err != nil {
		log.Errorf("failed to marshal TLS key: %v", err)
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

// ensureWebhookNamespace ensures the cluster-system namespace exists before
// creating webhook secrets. The namespace was previously defined in the
// deleted webhook-secrets.yaml; now it must be created explicitly.
func ensureWebhookNamespace() error {
	ctx := context.Background()
	client := global.K8s.GetClient()

	_, err := client.CoreV1().Namespaces().Get(ctx, webhookNamespace, metav1.GetOptions{})
	if err == nil {
		return nil
	}
	if !apierrors.IsNotFound(err) {
		log.Errorf("failed to get namespace %s: %v", webhookNamespace, err)
		return fmt.Errorf("get namespace %s: %w", webhookNamespace, err)
	}

	_, err = client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: webhookNamespace},
	}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		log.Errorf("failed to create namespace %s: %v", webhookNamespace, err)
		return fmt.Errorf("create namespace %s: %w", webhookNamespace, err)
	}
	log.Infof("namespace %s ensured", webhookNamespace)
	return nil
}

// ensureWebhookSecrets generates certificates and creates K8s TLS Secrets.
// Returns the base64-encoded CA bundles for template injection.
//
// When a Secret already exists with a valid ca.crt, the existing CA cert is
// reused so the returned caBundle always matches the stored data. If the
// Secret exists but ca.crt is missing (e.g. from a prior incomplete deploy),
// new certificates are generated and the Secret is updated in-place.
func ensureWebhookSecrets() (capiCaBundle string, bkeCaBundle string, err error) {
	log.Info("Generating webhook TLS certificates and ensuring secrets")

	// Ensure the cluster-system namespace exists before creating secrets.
	if err = ensureWebhookNamespace(); err != nil {
		log.Errorf("failed to ensure webhook namespace: %v", err)
		return "", "", fmt.Errorf("ensure webhook namespace: %w", err)
	}

	// 1. Ensure CAPI webhook TLS secret
	capiCaBundle, err = ensureWebhookTLSSecret("capi-webhook-service-cert", []string{
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

	// 2. Ensure BKE webhook TLS secret
	bkeCaBundle, err = ensureWebhookTLSSecret("bke-webhook-secret", []string{
		"bke-webhook-service.cluster-system.svc",
		"bke-webhook-service.cluster-system.svc.cluster.local",
	})
	if err != nil {
		log.Errorf("failed to ensure BKE webhook secret: %v", err)
		return "", "", fmt.Errorf("ensure BKE webhook secret: %w", err)
	}

	log.Info("Webhook TLS secrets ensured successfully")
	return capiCaBundle, bkeCaBundle, nil
}

// ensureWebhookTLSSecret creates a TLS Secret if it doesn't exist, or reads the
// existing CA cert if it does. Returns the base64-encoded CA cert for caBundle
// template injection.
//
// When the Secret exists but is missing ca.crt (e.g. leftover from a prior
// incomplete deployment), new certificates are generated and the Secret is
// updated in-place so the returned caBundle always matches the Secret content.
func ensureWebhookTLSSecret(secretName string, dnsNames []string) (string, error) {
	ctx := context.Background()
	client := global.K8s.GetClient()

	existing, err := client.CoreV1().Secrets(webhookNamespace).Get(ctx, secretName, metav1.GetOptions{})
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
		existing.Data = map[string][]byte{
			"tls.crt": certs.tlsCrt,
			"tls.key": certs.tlsKey,
			"ca.crt":  certs.caCrt,
		}
		if _, updateErr := client.CoreV1().Secrets(webhookNamespace).Update(ctx, existing, metav1.UpdateOptions{}); updateErr != nil {
			log.Errorf("failed to update webhook secret %s: %v", secretName, updateErr)
			return "", fmt.Errorf("update secret %s: %w", secretName, updateErr)
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

	if err := global.K8s.CreateSecret(secret); err != nil {
		log.Errorf("failed to create webhook secret %s: %v", secretName, err)
		return "", fmt.Errorf("create secret %s: %w", secretName, err)
	}
	log.Infof("webhook secret %s created successfully", secretName)
	return certs.caBundleB64, nil
}
