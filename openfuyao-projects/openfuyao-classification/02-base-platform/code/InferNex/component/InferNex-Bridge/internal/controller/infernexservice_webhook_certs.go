/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package controller contains the InferNexService reconciler helpers.
package controller

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
)

const (
	componentWebhookCertSecretSuffix = "webhook-server-cert"
	webhookCertVolumeName            = "webhook-certs"
	webhookCertMountPath             = "/tmp/k8s-webhook-server/serving-certs"
)

func componentNeedsWebhookCert(component string) bool {
	switch component {
	case elasticScalerComponent, rsgComponent:
		return true
	default:
		return false
	}
}

func componentWebhookCertSecretName(ownerName, component string) string {
	return fmt.Sprintf("%s-%s-%s", ownerName, component, componentWebhookCertSecretSuffix)
}

func ensureWebhookCertVolumeAndMount(tpl *corev1.PodTemplateSpec, secretName string) {
	volumeFound := false
	for i := range tpl.Spec.Volumes {
		if tpl.Spec.Volumes[i].Name == webhookCertVolumeName {
			tpl.Spec.Volumes[i].Secret = &corev1.SecretVolumeSource{SecretName: secretName}
			volumeFound = true
			break
		}
	}
	if !volumeFound {
		tpl.Spec.Volumes = append(tpl.Spec.Volumes, corev1.Volume{
			Name: webhookCertVolumeName,
			VolumeSource: corev1.VolumeSource{
				Secret: &corev1.SecretVolumeSource{SecretName: secretName},
			},
		})
	}
	for i := range tpl.Spec.Containers {
		found := false
		for j := range tpl.Spec.Containers[i].VolumeMounts {
			if tpl.Spec.Containers[i].VolumeMounts[j].Name == webhookCertVolumeName {
				tpl.Spec.Containers[i].VolumeMounts[j].MountPath = webhookCertMountPath
				tpl.Spec.Containers[i].VolumeMounts[j].ReadOnly = true
				found = true
				break
			}
		}
		if found {
			continue
		}
		tpl.Spec.Containers[i].VolumeMounts = append(tpl.Spec.Containers[i].VolumeMounts, corev1.VolumeMount{
			Name:      webhookCertVolumeName,
			MountPath: webhookCertMountPath,
			ReadOnly:  true,
		})
	}
}

func (r *InferNexServiceReconciler) ensureComponentWebhookCertSecret(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
) (string, error) {
	secretName := componentWebhookCertSecretName(owner.Name, component)
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: secretName, Namespace: owner.Namespace}}
	_, err := controllerutil.CreateOrUpdate(ctx, r.Client, secret, func() error {
		if secret.Labels == nil {
			secret.Labels = map[string]string{}
		}
		secret.Labels["infernex.io/owner"] = owner.Name
		secret.Labels["infernex.io/component"] = component
		secret.Type = corev1.SecretTypeTLS
		if len(secret.Data[corev1.TLSCertKey]) == 0 || len(secret.Data[corev1.TLSPrivateKeyKey]) == 0 {
			crt, key, certErr := generateSelfSignedCert()
			if certErr != nil {
				return certErr
			}
			if secret.Data == nil {
				secret.Data = map[string][]byte{}
			}
			secret.Data[corev1.TLSCertKey] = crt
			secret.Data[corev1.TLSPrivateKeyKey] = key
		}
		return controllerutil.SetControllerReference(owner, secret, r.Scheme)
	})
	if err != nil {
		return "", err
	}
	return secretName, nil
}

func generateSelfSignedCert() ([]byte, []byte, error) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, err
	}
	serialLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	serial, err := rand.Int(rand.Reader, serialLimit)
	if err != nil {
		return nil, nil, err
	}
	now := time.Now()
	template := x509.Certificate{
		SerialNumber: serial,
		Subject: pkix.Name{
			CommonName: "infernex-component-webhook",
		},
		NotBefore:             now.Add(-1 * time.Hour),
		NotAfter:              now.AddDate(1, 0, 0),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              []string{"localhost"},
	}
	der, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return nil, nil, err
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(priv)})
	return certPEM, keyPEM, nil
}

func (r *InferNexServiceReconciler) deleteComponentWebhookCertSecret(
	ctx context.Context,
	owner *infernexv1alpha1.InferNexService,
	component string,
) error {
	if !componentNeedsWebhookCert(component) {
		return nil
	}
	secretName := componentWebhookCertSecretName(owner.Name, component)
	secret := &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: secretName, Namespace: owner.Namespace}}
	if err := r.Delete(ctx, secret); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}
