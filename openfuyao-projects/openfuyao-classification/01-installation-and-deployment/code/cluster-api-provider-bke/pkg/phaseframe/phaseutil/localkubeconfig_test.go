/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package phaseutil

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"testing"
	"time"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/clientcmd/api"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	testNamespace       = "test-namespace"
	testClusterName     = "test-cluster"
	testServerURL       = "https://test.example.com:6443"
	testKubeSystemNS    = "kube-system"
	testBkeSystemNS     = "bke-system"
	testRoleBindingName = "bkeagent"
	testClusterRoleName = "bkeagent-cluster-access"
	testUserCN          = "bkeagent-cert-user"
	testAuthInfoName    = "bkeagent-cert-user"
	testContextName     = "bkeagent-context"
	testClusterNameKC   = "management-cluster"
	testKeySize         = 2048
	testCertValidity    = 100
	testSerialNumber    = 1
)

// testCAData contains test CA certificate and key data
type testCAData struct {
	cert    *x509.Certificate
	key     *rsa.PrivateKey
	certPEM []byte
	keyPEM  []byte
}

// createTestScheme creates a runtime scheme with required API groups
func createTestScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	_ = rbacv1.AddToScheme(scheme)
	_ = bkev1beta1.AddToScheme(scheme)
	return scheme
}

// createFakeClientForKubeConfig creates a fake Kubernetes client with given objects
func createFakeClientForKubeConfig(scheme *runtime.Scheme, objs ...client.Object) client.Client {
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

// testGetRemoteLocalKubeConfig is a test helper that wraps GetRemoteLocalKubeConfig
// to work with k8sfake.Clientset
func testGetRemoteLocalKubeConfig(ctx context.Context, fakeClient *k8sfake.Clientset) ([]byte, error) {
	secret, err := fakeClient.CoreV1().Secrets(testKubeSystemNS).Get(ctx, constant.LocalKubeConfigName, metav1.GetOptions{})
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get localkubeconfig from remote cluster")
	}

	localKubeConfig := secret.Data["config"]
	if len(localKubeConfig) == 0 {
		return nil, errors.New("localkubeconfig from remote cluster is empty")
	}

	return localKubeConfig, nil
}

// runTestWithSecret runs a test case with optional secret object
func runTestWithSecret(t *testing.T, scheme *runtime.Scheme, secret *corev1.Secret, testFn func(*testing.T, client.Client)) {
	var objs []client.Object
	if secret != nil {
		objs = append(objs, secret)
	}
	fakeClient := createFakeClientForKubeConfig(scheme, objs...)
	testFn(t, fakeClient)
}

// verifyKubeConfigResult verifies kubeconfig result
func verifyKubeConfigResult(t *testing.T, result []byte, expected []byte, err error, wantErr bool) {
	if wantErr {
		require.Error(t, err)
		assert.Nil(t, result)
	} else {
		require.NoError(t, err)
		assert.Equal(t, expected, result)
	}
}

// verifyServerURLResult verifies server URL result
func verifyServerURLResult(t *testing.T, result string, expected string, err error, wantErr bool) {
	if wantErr {
		require.Error(t, err)
		assert.Empty(t, result)
	} else {
		require.NoError(t, err)
		assert.Equal(t, expected, result)
	}
}

// verifyCAResult verifies CA certificate and key result
func verifyCAResult(t *testing.T, certBytes, keyBytes []byte, err error, wantErr bool) {
	if wantErr {
		require.Error(t, err)
		assert.Nil(t, certBytes)
		assert.Nil(t, keyBytes)
	} else {
		require.NoError(t, err)
		assert.NotNil(t, certBytes)
		assert.NotNil(t, keyBytes)
	}
}

// expectedCA contains expected CA certificate and key for verification
type expectedCA struct {
	cert *x509.Certificate
	key  *rsa.PrivateKey
}

// verifyManagementCAResult verifies management cluster CA result
func verifyManagementCAResult(t *testing.T, result *managementClusterCA, expected *expectedCA, err error, wantErr bool) {
	if wantErr {
		require.Error(t, err)
		assert.Nil(t, result)
	} else {
		require.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, expected.cert, result.cert)
		assert.Equal(t, expected.key, result.key)
	}
}

// verifyClusterRoleResult verifies ClusterRole result
func verifyClusterRoleResult(t *testing.T, c client.Client, roleName string, expectedRulesCount int) {
	var result rbacv1.ClusterRole
	err := c.Get(context.Background(), client.ObjectKey{Name: roleName}, &result)
	require.NoError(t, err)
	assert.Equal(t, expectedRulesCount, len(result.Rules))
}

// verifyRoleBindingResult verifies RoleBinding result
func verifyRoleBindingResult(t *testing.T, c client.Client, name, namespace string, expectedSubjectsCount int) {
	var result rbacv1.RoleBinding
	err := c.Get(context.Background(), client.ObjectKey{Name: name, Namespace: namespace}, &result)
	require.NoError(t, err)
	assert.Equal(t, expectedSubjectsCount, len(result.Subjects))
}

// verifyClusterRoleBindingResult verifies ClusterRoleBinding result
func verifyClusterRoleBindingResult(t *testing.T, c client.Client, name string, expectedSubjectsCount int) {
	var result rbacv1.ClusterRoleBinding
	err := c.Get(context.Background(), client.ObjectKey{Name: name}, &result)
	require.NoError(t, err)
	assert.Equal(t, expectedSubjectsCount, len(result.Subjects))
}

// baseSecretTestCase represents a base test case for secret tests
type baseSecretTestCase struct {
	name    string
	secret  *corev1.Secret
	wantErr bool
}

// createBaseSecretTestCases creates base test cases for secret tests
func createBaseSecretTestCases(secret *corev1.Secret) []baseSecretTestCase {
	return []baseSecretTestCase{
		{
			name:    "success",
			secret:  secret,
			wantErr: false,
		},
		{
			name:    "secret not found",
			secret:  nil,
			wantErr: true,
		},
	}
}

// createTestSecretCases creates common test cases for secret-based tests
func createTestSecretCases(secret *corev1.Secret) []struct {
	name    string
	secret  *corev1.Secret
	wantErr bool
} {
	baseCases := createBaseSecretTestCases(secret)
	result := make([]struct {
		name    string
		secret  *corev1.Secret
		wantErr bool
	}, len(baseCases))
	for i, bc := range baseCases {
		result[i] = struct {
			name    string
			secret  *corev1.Secret
			wantErr bool
		}{
			name:    bc.name,
			secret:  bc.secret,
			wantErr: bc.wantErr,
		}
	}
	return result
}

// caSecretTestCase represents a test case for CA secret tests
type caSecretTestCase struct {
	name       string
	bkeCluster *bkev1beta1.BKECluster
	secret     *corev1.Secret
	wantErr    bool
}

// createCASecretTestCases creates test cases for CA secret tests
func createCASecretTestCases(bkeCluster *bkev1beta1.BKECluster, caSecret *corev1.Secret, caData *testCAData) []caSecretTestCase {
	baseCases := createBaseCASecretTestCases(bkeCluster, caSecret)
	invalidCases := createInvalidCASecretTestCases(bkeCluster, caData)
	return append(baseCases, invalidCases...)
}

// createBaseCASecretTestCases creates base test cases for CA secret
func createBaseCASecretTestCases(bkeCluster *bkev1beta1.BKECluster, caSecret *corev1.Secret) []caSecretTestCase {
	return []caSecretTestCase{
		{
			name:       "success",
			bkeCluster: bkeCluster,
			secret:     caSecret,
			wantErr:    false,
		},
		{
			name:       "secret not found",
			bkeCluster: bkeCluster,
			secret:     nil,
			wantErr:    true,
		},
	}
}

// createInvalidCASecretTestCases creates test cases for invalid CA secrets
func createInvalidCASecretTestCases(bkeCluster *bkev1beta1.BKECluster, caData *testCAData) []caSecretTestCase {
	return []caSecretTestCase{
		{
			name:       "missing tls.crt",
			bkeCluster: bkeCluster,
			secret:     createSecretWithMissingCert(caData),
			wantErr:    true,
		},
		{
			name:       "missing tls.key",
			bkeCluster: bkeCluster,
			secret:     createSecretWithMissingKey(caData),
			wantErr:    true,
		},
	}
}

// createSecretWithMissingCert creates a secret missing tls.crt
func createSecretWithMissingCert(caData *testCAData) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-ca", testClusterName),
			Namespace: testNamespace,
		},
		Data: map[string][]byte{
			"tls.key": caData.keyPEM,
		},
	}
}

// createSecretWithMissingKey creates a secret missing tls.key
func createSecretWithMissingKey(caData *testCAData) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-ca", testClusterName),
			Namespace: testNamespace,
		},
		Data: map[string][]byte{
			"tls.crt": caData.certPEM,
		},
	}
}

// createTestKubeConfig creates a valid kubeconfig for testing
func createTestKubeConfig(serverURL string) []byte {
	config := api.NewConfig()
	config.Clusters[testClusterNameKC] = &api.Cluster{
		Server:                   serverURL,
		CertificateAuthorityData: []byte("test-ca-data"),
	}
	config.Contexts[testContextName] = &api.Context{
		Cluster:  testClusterNameKC,
		AuthInfo: testAuthInfoName,
	}
	config.AuthInfos[testAuthInfoName] = &api.AuthInfo{
		ClientCertificateData: []byte("test-cert"),
		ClientKeyData:         []byte("test-key"),
	}
	config.CurrentContext = testContextName
	kubeconfigBytes, _ := clientcmd.Write(*config)
	return kubeconfigBytes
}

// createTestSecret creates a test secret with kubeconfig data
func createTestSecret(name, namespace, key string, data []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string][]byte{
			key: data,
		},
	}
}

// createTestBKECluster creates a test BKECluster object
func createTestBKECluster(name, namespace string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
	}
}

// createTestCASecret creates a test CA secret
func createTestCASecret(clusterName, namespace string, certBytes, keyBytes []byte) *corev1.Secret {
	secretName := fmt.Sprintf("%s-ca", clusterName)
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      secretName,
			Namespace: namespace,
		},
		Data: map[string][]byte{
			"tls.crt": certBytes,
			"tls.key": keyBytes,
		},
	}
}

// generateTestCA generates a test CA certificate and key
func generateTestCA() (*testCAData, error) {
	key, err := rsa.GenerateKey(rand.Reader, testKeySize)
	if err != nil {
		return nil, err
	}

	template := &x509.Certificate{
		SerialNumber: big.NewInt(testSerialNumber),
		Subject: pkix.Name{
			CommonName: "test-ca",
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(testCertValidity * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageCRLSign,
		IsCA:                  true,
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return nil, err
	}

	cert, err := x509.ParseCertificate(certDER)
	if err != nil {
		return nil, err
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})

	return &testCAData{
		cert:    cert,
		key:     key,
		certPEM: certPEM,
		keyPEM:  keyPEM,
	}, nil
}

// runKubeConfigTest runs a kubeconfig test with common setup
func runKubeConfigTest(t *testing.T, secretName, namespace string, testFn func(*testing.T, client.Client, []byte, bool)) {
	scheme := createTestScheme()
	kubeconfigData := createTestKubeConfig(testServerURL)
	secret := createTestSecret(secretName, namespace, "config", kubeconfigData)
	tests := createTestSecretCases(secret)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runTestWithSecret(t, scheme, tt.secret, func(t *testing.T, c client.Client) {
				testFn(t, c, kubeconfigData, tt.wantErr)
			})
		})
	}
}

func TestGetLocalKubeConfig(t *testing.T) {
	runKubeConfigTest(t, constant.LocalKubeConfigName, constant.GetLocalKubeConfigObjectKey().Namespace, func(t *testing.T, c client.Client, expected []byte, wantErr bool) {
		result, err := GetLocalKubeConfig(context.Background(), c)
		verifyKubeConfigResult(t, result, expected, err, wantErr)
	})
}

func TestGetLeastPrivilegeKubeConfig(t *testing.T) {
	runKubeConfigTest(t, constant.LeastPrivilegeKubeConfigName, metav1.NamespaceSystem, func(t *testing.T, c client.Client, expected []byte, wantErr bool) {
		result, err := GetLeastPrivilegeKubeConfig(context.Background(), c)
		verifyKubeConfigResult(t, result, expected, err, wantErr)
	})
}

// remoteKubeConfigTestCase represents a test case for remote kubeconfig tests
type remoteKubeConfigTestCase struct {
	name       string
	secret     *corev1.Secret
	wantErr    bool
	errContain string
}

// createRemoteKubeConfigTestCases creates test cases for remote kubeconfig tests
func createRemoteKubeConfigTestCases(kubeconfigData []byte) []remoteKubeConfigTestCase {
	secret := createTestSecret(
		constant.LocalKubeConfigName,
		testKubeSystemNS,
		"config",
		kubeconfigData,
	)
	baseCases := createBaseSecretTestCases(secret)
	result := make([]remoteKubeConfigTestCase, 0, len(baseCases)+1)
	for _, bc := range baseCases {
		tc := remoteKubeConfigTestCase{
			name:    bc.name,
			secret:  bc.secret,
			wantErr: bc.wantErr,
		}
		if bc.wantErr {
			tc.errContain = "failed to get localkubeconfig"
		}
		result = append(result, tc)
	}
	result = append(result, remoteKubeConfigTestCase{
		name: "empty config",
		secret: createTestSecret(
			constant.LocalKubeConfigName,
			testKubeSystemNS,
			"config",
			[]byte{},
		),
		wantErr:    true,
		errContain: "empty",
	})
	return result
}

func TestGetRemoteLocalKubeConfig(t *testing.T) {
	kubeconfigData := createTestKubeConfig(testServerURL)
	tests := createRemoteKubeConfigTestCases(kubeconfigData)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var fakeClientset *k8sfake.Clientset
			if tt.secret != nil {
				fakeClientset = k8sfake.NewSimpleClientset(tt.secret)
			} else {
				fakeClientset = k8sfake.NewSimpleClientset()
			}

			result, err := testGetRemoteLocalKubeConfig(context.Background(), fakeClientset)

			if tt.wantErr {
				require.Error(t, err)
				assert.Nil(t, result)
				if tt.errContain != "" {
					assert.Contains(t, err.Error(), tt.errContain)
				}
			} else {
				require.NoError(t, err)
				assert.Equal(t, kubeconfigData, result)
			}
		})
	}
}

func TestExtractServerURLFromKubeConfig(t *testing.T) {
	tests := []struct {
		name       string
		kubeconfig []byte
		wantErr    bool
	}{
		{
			name:       "success",
			kubeconfig: createTestKubeConfig(testServerURL),
			wantErr:    false,
		},
		{
			name:       "invalid kubeconfig",
			kubeconfig: []byte("invalid"),
			wantErr:    true,
		},
		{
			name:       "empty kubeconfig",
			kubeconfig: []byte{},
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := extractServerURLFromKubeConfig(tt.kubeconfig)
			verifyServerURLResult(t, result, testServerURL, err, tt.wantErr)
		})
	}
}

func TestExtractServerURLFromLocalKubeConfig(t *testing.T) {
	runKubeConfigTest(t, constant.LocalKubeConfigName, constant.GetLocalKubeConfigObjectKey().Namespace, func(t *testing.T, c client.Client, _ []byte, wantErr bool) {
		result, err := extractServerURLFromLocalKubeConfig(context.Background(), c)
		verifyServerURLResult(t, result, testServerURL, err, wantErr)
	})
}

func TestGetManagementClusterCA(t *testing.T) {
	scheme := createTestScheme()
	caData, err := generateTestCA()
	require.NoError(t, err)

	bkeCluster := createTestBKECluster(testClusterName, testNamespace)
	caSecret := createTestCASecret(testClusterName, testNamespace, caData.certPEM, caData.keyPEM)
	tests := createCASecretTestCases(bkeCluster, caSecret, caData)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runTestWithSecret(t, scheme, tt.secret, func(t *testing.T, c client.Client) {
				certBytes, keyBytes, err := getManagementClusterCA(
					context.Background(),
					c,
					tt.bkeCluster,
				)
				verifyCAResult(t, certBytes, keyBytes, err, tt.wantErr)
			})
		})
	}
}

func TestGenerateBKEAgentClientCert(t *testing.T) {
	caData, err := generateTestCA()
	require.NoError(t, err)

	resultCert, resultKey, err := generateBKEAgentClientCert(caData.cert, caData.key)
	require.NoError(t, err)
	assert.NotNil(t, resultCert)
	assert.NotNil(t, resultKey)
	assert.Equal(t, testUserCN, resultCert.Subject.CommonName)
}

func TestCreateKubeConfigWithClientCert(t *testing.T) {
	caData, err := generateTestCA()
	require.NoError(t, err)
	clientCert, clientKey, err := generateBKEAgentClientCert(caData.cert, caData.key)
	require.NoError(t, err)

	ca := &managementClusterCA{
		cert:      caData.cert,
		key:       caData.key,
		certBytes: []byte("test-ca"),
	}

	result, err := createKubeConfigWithClientCert(testServerURL, clientCert, clientKey, ca.certBytes)
	require.NoError(t, err)
	assert.NotEmpty(t, result)

	config, err := clientcmd.Load(result)
	require.NoError(t, err)
	assert.Equal(t, testServerURL, config.Clusters[testClusterNameKC].Server)
	assert.Equal(t, testContextName, config.CurrentContext)
}

func TestGetBKEAgentSubject(t *testing.T) {
	subjects := getBKEAgentSubject()
	require.Len(t, subjects, 1)
	assert.Equal(t, "User", subjects[0].Kind)
	assert.Equal(t, testUserCN, subjects[0].Name)
}

// createNamespaceTestCases creates test cases for namespace tests
func createNamespaceTestCases() []struct {
	name      string
	namespace *corev1.Namespace
	wantErr   bool
} {
	return []struct {
		name      string
		namespace *corev1.Namespace
		wantErr   bool
	}{
		{
			name: "namespace exists",
			namespace: &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name: testNamespace,
				},
			},
			wantErr: false,
		},
		{
			name:      "create namespace",
			namespace: nil,
			wantErr:   false,
		},
	}
}

func TestEnsureNamespaceExists(t *testing.T) {
	scheme := createTestScheme()
	tests := createNamespaceTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []client.Object
			if tt.namespace != nil {
				objs = append(objs, tt.namespace)
			}
			fakeClient := createFakeClientForKubeConfig(scheme, objs...)
			err := ensureNamespaceExists(context.Background(), fakeClient, testNamespace)

			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				var ns corev1.Namespace
				err = fakeClient.Get(context.Background(), client.ObjectKey{Name: testNamespace}, &ns)
				require.NoError(t, err)
			}
		})
	}
}

// createTestClusterRole creates a test ClusterRole
func createTestClusterRole() *rbacv1.ClusterRole {
	return &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterRoleName,
		},
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups: []string{""},
				Resources: []string{"secrets"},
				Verbs:     []string{"get", "list"},
			},
		},
	}
}

// createClusterRoleTestCases creates test cases for ClusterRole tests
func createClusterRoleTestCases(role *rbacv1.ClusterRole) []struct {
	name     string
	role     *rbacv1.ClusterRole
	existing *rbacv1.ClusterRole
	wantErr  bool
} {
	return []struct {
		name     string
		role     *rbacv1.ClusterRole
		existing *rbacv1.ClusterRole
		wantErr  bool
	}{
		{
			name:     "create new role",
			role:     role,
			existing: nil,
			wantErr:  false,
		},
		{
			name: "update existing role",
			role: role,
			existing: &rbacv1.ClusterRole{
				ObjectMeta: metav1.ObjectMeta{
					Name: testClusterRoleName,
				},
				Rules: []rbacv1.PolicyRule{},
			},
			wantErr: false,
		},
	}
}

func TestCreateOrUpdateClusterRole(t *testing.T) {
	scheme := createTestScheme()
	role := createTestClusterRole()
	tests := createClusterRoleTestCases(role)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []client.Object
			if tt.existing != nil {
				objs = append(objs, tt.existing)
			}
			fakeClient := createFakeClientForKubeConfig(scheme, objs...)
			err := createOrUpdateClusterRole(context.Background(), fakeClient, tt.role)

			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				verifyClusterRoleResult(t, fakeClient, tt.role.Name, len(tt.role.Rules))
			}
		})
	}
}

// createTestRoleBinding creates a test RoleBinding
func createTestRoleBinding() *rbacv1.RoleBinding {
	return &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testRoleBindingName,
			Namespace: testNamespace,
		},
		Subjects: getBKEAgentSubject(),
		RoleRef: rbacv1.RoleRef{
			Kind:     "ClusterRole",
			Name:     "test-role",
			APIGroup: "rbac.authorization.k8s.io",
		},
	}
}

// createRoleBindingTestCases creates test cases for RoleBinding tests
func createRoleBindingTestCases(roleBinding *rbacv1.RoleBinding) []struct {
	name        string
	roleBinding *rbacv1.RoleBinding
	existing    *rbacv1.RoleBinding
	wantErr     bool
} {
	return []struct {
		name        string
		roleBinding *rbacv1.RoleBinding
		existing    *rbacv1.RoleBinding
		wantErr     bool
	}{
		{
			name:        "create new rolebinding",
			roleBinding: roleBinding,
			existing:    nil,
			wantErr:     false,
		},
		{
			name:        "update existing rolebinding",
			roleBinding: roleBinding,
			existing: &rbacv1.RoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name:      testRoleBindingName,
					Namespace: testNamespace,
				},
				Subjects: []rbacv1.Subject{},
				RoleRef:  rbacv1.RoleRef{},
			},
			wantErr: false,
		},
	}
}

func TestCreateOrUpdateRoleBinding(t *testing.T) {
	scheme := createTestScheme()
	roleBinding := createTestRoleBinding()
	tests := createRoleBindingTestCases(roleBinding)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []client.Object
			if tt.existing != nil {
				objs = append(objs, tt.existing)
			}
			fakeClient := createFakeClientForKubeConfig(scheme, objs...)
			err := createOrUpdateRoleBinding(context.Background(), fakeClient, tt.roleBinding)

			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				verifyRoleBindingResult(t, fakeClient, tt.roleBinding.Name, tt.roleBinding.Namespace, len(tt.roleBinding.Subjects))
			}
		})
	}
}

// createTestClusterRoleBinding creates a test ClusterRoleBinding
func createTestClusterRoleBinding() *rbacv1.ClusterRoleBinding {
	return &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: testClusterRoleName,
		},
		Subjects: getBKEAgentSubject(),
		RoleRef: rbacv1.RoleRef{
			Kind:     "ClusterRole",
			Name:     testClusterRoleName,
			APIGroup: "rbac.authorization.k8s.io",
		},
	}
}

// createClusterRoleBindingTestCases creates test cases for ClusterRoleBinding tests
func createClusterRoleBindingTestCases(clusterRoleBinding *rbacv1.ClusterRoleBinding) []struct {
	name               string
	clusterRoleBinding *rbacv1.ClusterRoleBinding
	existing           *rbacv1.ClusterRoleBinding
	wantErr            bool
} {
	return []struct {
		name               string
		clusterRoleBinding *rbacv1.ClusterRoleBinding
		existing           *rbacv1.ClusterRoleBinding
		wantErr            bool
	}{
		{
			name:               "create new clusterrolebinding",
			clusterRoleBinding: clusterRoleBinding,
			existing:           nil,
			wantErr:            false,
		},
		{
			name:               "update existing clusterrolebinding",
			clusterRoleBinding: clusterRoleBinding,
			existing: &rbacv1.ClusterRoleBinding{
				ObjectMeta: metav1.ObjectMeta{
					Name: testClusterRoleName,
				},
				Subjects: []rbacv1.Subject{},
				RoleRef:  rbacv1.RoleRef{},
			},
			wantErr: false,
		},
	}
}

func TestCreateOrUpdateClusterRoleBinding(t *testing.T) {
	scheme := createTestScheme()
	clusterRoleBinding := createTestClusterRoleBinding()
	tests := createClusterRoleBindingTestCases(clusterRoleBinding)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []client.Object
			if tt.existing != nil {
				objs = append(objs, tt.existing)
			}
			fakeClient := createFakeClientForKubeConfig(scheme, objs...)
			err := createOrUpdateClusterRoleBinding(context.Background(), fakeClient, tt.clusterRoleBinding)

			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				verifyClusterRoleBindingResult(t, fakeClient, tt.clusterRoleBinding.Name, len(tt.clusterRoleBinding.Subjects))
			}
		})
	}
}

// createRoleBindingForNamespaceTestCases creates test cases for RoleBindingForNamespace tests
func createRoleBindingForNamespaceTestCases() []struct {
	name      string
	namespace *corev1.Namespace
	roleName  string
	wantErr   bool
} {
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNamespace,
		},
	}
	return []struct {
		name      string
		namespace *corev1.Namespace
		roleName  string
		wantErr   bool
	}{
		{
			name:      "success with existing namespace",
			namespace: namespace,
			roleName:  "bkeagent-readonly",
			wantErr:   false,
		},
		{
			name:      "success create namespace",
			namespace: nil,
			roleName:  "bkeagent-readwrite",
			wantErr:   false,
		},
	}
}

func TestCreateRoleBindingForNamespace(t *testing.T) {
	scheme := createTestScheme()
	tests := createRoleBindingForNamespaceTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objs []client.Object
			if tt.namespace != nil {
				objs = append(objs, tt.namespace)
			}
			fakeClient := createFakeClientForKubeConfig(scheme, objs...)
			err := createRoleBindingForNamespace(
				context.Background(),
				fakeClient,
				testNamespace,
				tt.roleName,
			)

			if tt.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				var result rbacv1.RoleBinding
				err = fakeClient.Get(
					context.Background(),
					client.ObjectKey{Name: testRoleBindingName, Namespace: testNamespace},
					&result,
				)
				require.NoError(t, err)
				assert.Equal(t, tt.roleName, result.RoleRef.Name)
			}
		})
	}
}

func TestCreateReadwriteClusterRole(t *testing.T) {
	scheme := createTestScheme()
	fakeClient := createFakeClientForKubeConfig(scheme)

	err := createReadwriteClusterRole(context.Background(), fakeClient)
	require.NoError(t, err)

	var role rbacv1.ClusterRole
	err = fakeClient.Get(
		context.Background(),
		client.ObjectKey{Name: "bkeagent-readwrite"},
		&role,
	)
	require.NoError(t, err)
	assert.NotEmpty(t, role.Rules)
}

func TestCreateClusterAccessClusterRole(t *testing.T) {
	scheme := createTestScheme()
	fakeClient := createFakeClientForKubeConfig(scheme)

	err := createClusterAccessClusterRole(context.Background(), fakeClient)
	require.NoError(t, err)

	var role rbacv1.ClusterRole
	err = fakeClient.Get(
		context.Background(),
		client.ObjectKey{Name: testClusterRoleName},
		&role,
	)
	require.NoError(t, err)
	assert.NotEmpty(t, role.Rules)
}

func TestCreateBKEAgentClusterRoles(t *testing.T) {
	scheme := createTestScheme()
	fakeClient := createFakeClientForKubeConfig(scheme)

	err := createBKEAgentClusterRoles(context.Background(), fakeClient)
	require.NoError(t, err)

	roleNames := []string{"bkeagent-readwrite", "bkeagent-configmap-only", testClusterRoleName}
	for _, roleName := range roleNames {
		var role rbacv1.ClusterRole
		err = fakeClient.Get(
			context.Background(),
			client.ObjectKey{Name: roleName},
			&role,
		)
		require.NoError(t, err)
	}
}

func TestCreateBKEAgentClusterAccessRoleBinding(t *testing.T) {
	scheme := createTestScheme()
	fakeClient := createFakeClientForKubeConfig(scheme)

	err := createBKEAgentClusterAccessRoleBinding(context.Background(), fakeClient)
	require.NoError(t, err)

	var binding rbacv1.ClusterRoleBinding
	err = fakeClient.Get(
		context.Background(),
		client.ObjectKey{Name: testClusterRoleName},
		&binding,
	)
	require.NoError(t, err)
	assert.Equal(t, testClusterRoleName, binding.RoleRef.Name)
}

func TestGenerateKubeConfigWithCert(t *testing.T) {
	caData, err := generateTestCA()
	require.NoError(t, err)

	ca := &managementClusterCA{
		cert:      caData.cert,
		key:       caData.key,
		certBytes: []byte("test-ca"),
	}

	result, err := generateKubeConfigWithCert(testServerURL, ca)
	require.NoError(t, err)
	assert.NotEmpty(t, result)

	config, err := clientcmd.Load(result)
	require.NoError(t, err)
	assert.Equal(t, testServerURL, config.Clusters[testClusterNameKC].Server)
}

func TestGenerateLowPrivilegeKubeConfig(t *testing.T) {
	scheme := createTestScheme()
	caData, err := generateTestCA()
	require.NoError(t, err)

	bkeCluster := createTestBKECluster(testClusterName, testNamespace)
	caSecret := createTestCASecret(testClusterName, testNamespace, caData.certPEM, caData.keyPEM)
	remoteKubeconfig := createTestKubeConfig(testServerURL)

	var objs []client.Object
	objs = append(objs, caSecret)
	fakeClient := createFakeClientForKubeConfig(scheme, objs...)

	result, err := GenerateLowPrivilegeKubeConfig(
		context.Background(),
		fakeClient,
		bkeCluster,
		remoteKubeconfig,
	)
	require.NoError(t, err)
	assert.NotEmpty(t, result)

	config, err := clientcmd.Load(result)
	require.NoError(t, err)
	assert.Equal(t, testServerURL, config.Clusters[testClusterNameKC].Server)
}

func TestCreateBKEAgentRBACWithLocalKubeConfig(t *testing.T) {
	scheme := createTestScheme()
	kubeconfig := createTestKubeConfig(testServerURL)
	bkeCluster := createTestBKECluster(testClusterName, testNamespace)

	restConfig, err := clientcmd.RESTConfigFromKubeConfig(kubeconfig)
	require.NoError(t, err)

	highPrivilegeClient, err := client.New(restConfig, client.Options{Scheme: scheme})
	if err != nil {
		t.Skip("Skipping test that requires valid kubeconfig structure")
		return
	}

	err = CreateBKEAgentRBACWithLocalKubeConfig(
		context.Background(),
		kubeconfig,
		bkeCluster,
	)
	require.NoError(t, err)

	verifyRBACResourcesCreated(t, highPrivilegeClient, bkeCluster)
}

// verifyRBACResourcesCreated verifies that all RBAC resources are created
func verifyRBACResourcesCreated(t *testing.T, c client.Client, bkeCluster *bkev1beta1.BKECluster) {
	clusterRoles := []string{"bkeagent-readonly", "bkeagent-readwrite", testClusterRoleName}
	for _, roleName := range clusterRoles {
		var role rbacv1.ClusterRole
		err := c.Get(context.Background(), client.ObjectKey{Name: roleName}, &role)
		require.NoError(t, err)
	}

	var binding rbacv1.ClusterRoleBinding
	err := c.Get(context.Background(), client.ObjectKey{Name: testClusterRoleName}, &binding)
	require.NoError(t, err)

	namespaces := []string{testKubeSystemNS, testBkeSystemNS}
	if bkeCluster != nil && bkeCluster.Namespace != "" {
		namespaces = append(namespaces, bkeCluster.Namespace)
	}

	for _, ns := range namespaces {
		var roleBinding rbacv1.RoleBinding
		err := c.Get(
			context.Background(),
			client.ObjectKey{Name: testRoleBindingName, Namespace: ns},
			&roleBinding,
		)
		require.NoError(t, err)
	}
}
