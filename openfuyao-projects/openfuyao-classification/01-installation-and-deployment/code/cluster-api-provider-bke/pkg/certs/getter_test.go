/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package certs

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
)

const (
	testClusterName = "test-cluster"
	testNamespace   = "default"
	testCertData    = "test-cert-data"
	testKeyData     = "test-key-data"
	testKubeconfig  = "test-kubeconfig-data"
)

func createTestBKEClusterForGetter() *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		TypeMeta: metav1.TypeMeta{
			Kind:       "BKECluster",
			APIVersion: "bke.bocloud.com/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      testClusterName,
			Namespace: testNamespace,
			UID:       "test-uid",
		},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: testHost,
				Port: testPort,
			},
		},
	}
}

func createSchemeForGetter() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	return scheme
}

func createTestGetter(client client.Client) *BKEKubernetesCertGetter {
	bkeCluster := createTestBKEClusterForGetter()
	return NewBKEKubernetesCertGetter(context.TODO(), client, bkeCluster)
}

func createTestSecret(certName, namespace string, certData, keyData []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      NewCertSecretName(testClusterName, certName),
			Namespace: namespace,
		},
		Data: map[string][]byte{
			TLSCrtDataName: certData,
			TLSKeyDataName: keyData,
		},
		Type: utils.BKESecretType,
	}
}

func createKubeconfigSecret(namespace string, kubeconfigData []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      NewCertSecretName(testClusterName, KubeConfigCertName),
			Namespace: namespace,
		},
		Data: map[string][]byte{
			"value": kubeconfigData,
		},
		Type: utils.BKESecretType,
	}
}

func TestNewBKEKubernetesCertGetter(t *testing.T) {
	bkeCluster := createTestBKEClusterForGetter()
	scheme := createSchemeForGetter()
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	getter := NewBKEKubernetesCertGetter(context.TODO(), fakeClient, bkeCluster)

	assert.NotNil(t, getter)
	assert.Equal(t, testNamespace, getter.certNamespace)
	assert.Equal(t, testClusterName, getter.certClusterName)
	assert.Equal(t, bkeCluster, getter.bkeCluster)
}

func TestGetCertContent(t *testing.T) {
	scheme := createSchemeForGetter()

	tests := []struct {
		name          string
		secret        *corev1.Secret
		cert          *pkiutil.BKECert
		expectError   bool
		expectContent *CertContent
	}{
		{
			name:        "secret found with valid data",
			secret:      createTestSecret("ca", testNamespace, []byte(testCertData), []byte(testKeyData)),
			cert:        pkiutil.BKECertRootCA(),
			expectError: false,
			expectContent: &CertContent{
				Key:  testKeyData,
				Cert: testCertData,
			},
		},
		{
			name:          "secret not found",
			secret:        nil,
			cert:          pkiutil.BKECertRootCA(),
			expectError:   true,
			expectContent: nil,
		},
		{
			name:        "secret exists but missing cert data",
			secret:      createTestSecret("ca", testNamespace, nil, []byte(testKeyData)),
			cert:        pkiutil.BKECertRootCA(),
			expectError: false,
			expectContent: &CertContent{
				Key:  testKeyData,
				Cert: "",
			},
		},
		{
			name:        "secret exists but missing key data",
			secret:      createTestSecret("ca", testNamespace, []byte(testCertData), nil),
			cert:        pkiutil.BKECertRootCA(),
			expectError: false,
			expectContent: &CertContent{
				Key:  "",
				Cert: testCertData,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var clientBuilder *fake.ClientBuilder
			if tt.secret != nil {
				clientBuilder = fake.NewClientBuilder().WithScheme(scheme).WithObjects(tt.secret)
			} else {
				clientBuilder = fake.NewClientBuilder().WithScheme(scheme)
			}
			fakeClient := clientBuilder.Build()
			getter := createTestGetter(fakeClient)

			content, err := getter.GetCertContent(tt.cert)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, content)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, content)
				if tt.expectContent != nil {
					assert.Equal(t, tt.expectContent.Cert, content.Cert)
					assert.Equal(t, tt.expectContent.Key, content.Key)
				}
			}
		})
	}
}

func TestGetTargetClusterKubeconfig(t *testing.T) {
	scheme := createSchemeForGetter()

	tests := []struct {
		name             string
		secret           *corev1.Secret
		expectError      bool
		expectKubeconfig string
	}{
		{
			name:             "kubeconfig secret found",
			secret:           createKubeconfigSecret(testNamespace, []byte(testKubeconfig)),
			expectError:      false,
			expectKubeconfig: testKubeconfig,
		},
		{
			name:             "kubeconfig secret not found",
			secret:           nil,
			expectError:      true,
			expectKubeconfig: "",
		},
		{
			name:             "kubeconfig secret exists but empty data",
			secret:           createKubeconfigSecret(testNamespace, []byte{}),
			expectError:      false,
			expectKubeconfig: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var clientBuilder *fake.ClientBuilder
			if tt.secret != nil {
				clientBuilder = fake.NewClientBuilder().WithScheme(scheme).WithObjects(tt.secret)
			} else {
				clientBuilder = fake.NewClientBuilder().WithScheme(scheme)
			}
			fakeClient := clientBuilder.Build()
			getter := createTestGetter(fakeClient)

			kubeconfig, err := getter.GetTargetClusterKubeconfig()

			if tt.expectError {
				assert.Error(t, err)
				assert.Empty(t, kubeconfig)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectKubeconfig, kubeconfig)
			}
		})
	}
}

func TestGetCertContent_DifferentCertTypes(t *testing.T) {
	scheme := createSchemeForGetter()

	certTypes := []struct {
		name string
		cert *pkiutil.BKECert
	}{
		{"root CA", pkiutil.BKECertRootCA()},
		{"etcd CA", pkiutil.BKECertEtcdCA()},
		{"front proxy CA", pkiutil.BKECertFrontProxyCA()},
		{"API server", pkiutil.BKECertAPIServer()},
		{"etcd server", pkiutil.BKECertEtcdServer()},
		{"etcd peer", pkiutil.BKECertEtcdPeer()},
		{"etcd healthcheck client", pkiutil.BKECertEtcdHealthcheck()},
		{"front proxy client", pkiutil.BKECertFrontProxyClient()},
		{"API server etcd client", pkiutil.BKECertEtcdAPIClient()},
		{"service account", pkiutil.BKECertServiceAccount()},
	}

	for _, ct := range certTypes {
		t.Run(ct.name, func(t *testing.T) {
			secret := createTestSecret(ct.cert.Name, testNamespace, []byte(testCertData), []byte(testKeyData))
			fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			getter := createTestGetter(fakeClient)

			content, err := getter.GetCertContent(ct.cert)

			assert.NoError(t, err)
			assert.NotNil(t, content)
			assert.Equal(t, testCertData, content.Cert)
			assert.Equal(t, testKeyData, content.Key)
		})
	}
}

func TestGetCertContent_CrossNamespace(t *testing.T) {
	scheme := createSchemeForGetter()
	differentNamespace := "kube-system"

	bkeCluster := createTestBKEClusterForGetter()
	bkeCluster.Namespace = differentNamespace

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(createTestSecret("ca", differentNamespace, []byte(testCertData), []byte(testKeyData))).
		Build()

	getter := NewBKEKubernetesCertGetter(context.TODO(), fakeClient, bkeCluster)

	content, err := getter.GetCertContent(pkiutil.BKECertRootCA())

	assert.NoError(t, err)
	assert.NotNil(t, content)
	assert.Equal(t, testCertData, content.Cert)
	assert.Equal(t, testKeyData, content.Key)
}
