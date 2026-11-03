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
	"encoding/base64"
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"

	k8sexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/k8s"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
)

func setupCertgenMockK8s(t *testing.T, client kubernetes.Interface) *k8sexec.MockK8sClient {
	t.Helper()
	mockK8s := &k8sexec.MockK8sClient{}
	patches := gomonkey.NewPatches()
	t.Cleanup(func() {
		patches.Reset()
		global.K8s = nil
	})
	patches.ApplyFunc((*k8sexec.MockK8sClient).GetClient, func(*k8sexec.MockK8sClient) kubernetes.Interface {
		return client
	})
	global.K8s = mockK8s
	return mockK8s
}

func TestGenerateWebhookCertificates(t *testing.T) {
	dnsNames := []string{"webhook.example.svc", "webhook.example.svc.cluster.local"}
	bundle, err := generateWebhookCertificates(dnsNames)
	require.NoError(t, err)
	require.NotNil(t, bundle)
	assert.NotEmpty(t, bundle.caBundleB64)
	assert.NotEmpty(t, bundle.tlsCrt)
	assert.NotEmpty(t, bundle.tlsKey)
	assert.NotEmpty(t, bundle.caCrt)

	decoded, err := base64.StdEncoding.DecodeString(bundle.caBundleB64)
	require.NoError(t, err)
	assert.Equal(t, bundle.caCrt, decoded)
}

func TestEnsureWebhookNamespaceAlreadyExists(t *testing.T) {
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: webhookNamespace}}
	client := fake.NewSimpleClientset(ns)
	setupCertgenMockK8s(t, client)

	err := ensureWebhookNamespace()
	assert.NoError(t, err)
}

func TestEnsureWebhookNamespaceCreatesWhenMissing(t *testing.T) {
	client := fake.NewSimpleClientset()
	setupCertgenMockK8s(t, client)

	err := ensureWebhookNamespace()
	assert.NoError(t, err)

	created, getErr := client.CoreV1().Namespaces().Get(context.Background(), webhookNamespace, metav1.GetOptions{})
	require.NoError(t, getErr)
	assert.Equal(t, webhookNamespace, created.Name)
}

func TestEnsureWebhookNamespaceGetError(t *testing.T) {
	client := fake.NewSimpleClientset()
	client.PrependReactor("get", "namespaces", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("api unavailable")
	})
	setupCertgenMockK8s(t, client)

	err := ensureWebhookNamespace()
	assert.Error(t, err)
}

func TestEnsureWebhookTLSSecretReusesExistingCA(t *testing.T) {
	caPEM := []byte("-----BEGIN CERTIFICATE-----\ntest-ca\n-----END CERTIFICATE-----")
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "test-secret", Namespace: webhookNamespace},
		Data:       map[string][]byte{"ca.crt": caPEM},
	}
	client := fake.NewSimpleClientset(secret)
	setupCertgenMockK8s(t, client)

	caBundle, err := ensureWebhookTLSSecret("test-secret", []string{"svc.example"})
	require.NoError(t, err)
	assert.Equal(t, base64.StdEncoding.EncodeToString(caPEM), caBundle)
}

func TestEnsureWebhookTLSSecretUpdatesMissingCACrt(t *testing.T) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "test-secret", Namespace: webhookNamespace},
		Data:       map[string][]byte{"tls.crt": []byte("stale")},
	}
	client := fake.NewSimpleClientset(secret)
	setupCertgenMockK8s(t, client)

	caBundle, err := ensureWebhookTLSSecret("test-secret", []string{"svc.example"})
	require.NoError(t, err)
	assert.NotEmpty(t, caBundle)

	updated, getErr := client.CoreV1().Secrets(webhookNamespace).Get(context.Background(), "test-secret", metav1.GetOptions{})
	require.NoError(t, getErr)
	assert.NotEmpty(t, updated.Data["ca.crt"])
	assert.NotEmpty(t, updated.Data["tls.crt"])
	assert.NotEmpty(t, updated.Data["tls.key"])
}

func TestEnsureWebhookTLSSecretCreatesNew(t *testing.T) {
	client := fake.NewSimpleClientset()
	setupCertgenMockK8s(t, client)

	var createdSecret *corev1.Secret
	patches := gomonkey.NewPatches()
	t.Cleanup(patches.Reset)
	patches.ApplyFunc((*k8sexec.MockK8sClient).CreateSecret, func(_ *k8sexec.MockK8sClient, secret *corev1.Secret) error {
		createdSecret = secret
		return nil
	})

	caBundle, err := ensureWebhookTLSSecret("new-secret", []string{"new.example.svc"})
	require.NoError(t, err)
	assert.NotEmpty(t, caBundle)
	require.NotNil(t, createdSecret)
	assert.Equal(t, "new-secret", createdSecret.Name)
	assert.Equal(t, webhookNamespace, createdSecret.Namespace)
	assert.NotEmpty(t, createdSecret.Data["ca.crt"])
}

func TestEnsureWebhookTLSSecretGetError(t *testing.T) {
	client := fake.NewSimpleClientset()
	client.PrependReactor("get", "secrets", func(action k8stesting.Action) (bool, runtime.Object, error) {
		return true, nil, errors.New("secret lookup failed")
	})
	setupCertgenMockK8s(t, client)

	_, err := ensureWebhookTLSSecret("bad-secret", []string{"svc.example"})
	assert.Error(t, err)
}

func TestEnsureWebhookSecretsSuccess(t *testing.T) {
	client := fake.NewSimpleClientset()
	setupCertgenMockK8s(t, client)

	patches := gomonkey.NewPatches()
	t.Cleanup(patches.Reset)
	patches.ApplyFunc((*k8sexec.MockK8sClient).CreateSecret, func(*k8sexec.MockK8sClient, *corev1.Secret) error {
		return nil
	})

	capiCA, bkeCA, err := ensureWebhookSecrets()
	require.NoError(t, err)
	assert.NotEmpty(t, capiCA)
	assert.NotEmpty(t, bkeCA)
}
