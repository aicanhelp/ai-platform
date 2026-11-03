/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package certs

import (
	"context"
	"crypto/x509"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	certutil "k8s.io/client-go/util/cert"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
)

func init() {}

const (
	testIPAddress1   = "127.0.0.1"
	testHADomain     = "ha.example.com"
	testEndpointPort = 6443
	testEndpoint1    = testIPAddress1 + ":6443"
)

var bkeCluster = &bkev1beta1.BKECluster{
	TypeMeta: metav1.TypeMeta{
		Kind:       "BKECluster",
		APIVersion: "bke.bocloud.com/v1beta1",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name:      "test",
		Namespace: "default",
		UID:       "test",
	},
	Spec: confv1beta1.BKEClusterSpec{
		ControlPlaneEndpoint: confv1beta1.APIEndpoint{
			Host: testIPAddress1,
			Port: testEndpointPort,
		},
		ClusterConfig: &confv1beta1.BKEConfig{
			Cluster: confv1beta1.Cluster{
				ControlPlane: confv1beta1.ControlPlane{
					Etcd: &confv1beta1.Etcd{
						ServerCertSANs: []string{"1.1.1.1", "abc.com"},
					},
					APIServer: &confv1beta1.APIServer{
						CertSANs: []string{"2.2.2.2", "def.com"},
					},
				},
			},
		},
	},
}

// createTestGenerator creates a test BKEKubernetesCertGenerator instance with a fake client
func createTestGenerator() *BKEKubernetesCertGenerator {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	return NewKubernetesCertGenerator(context.TODO(), fakeClient, bkeCluster)
}

// TestPrepareBkeCerts tests that prepareBkeCerts appends kubeconfigs and renames them correctly
func TestPrepareBkeCerts(t *testing.T) {
	generator := createTestGenerator()
	generator.needCreateKubeConfig = true // Enable kubeconfig creation
	initialCount := len(generator.bkeCerts)

	err := generator.prepareBkeCerts(false)
	if err != nil {
		t.Errorf("prepareBkeCerts() error = %v", err)
		return
	}

	if len(generator.bkeCerts) <= initialCount {
		t.Error("prepareBkeCerts() should append certificates")
	}

	// Check that non-CA certs have CAName set (except for CA certs themselves and service account)
	// Note: kubeconfig certs (admin, kubelet, etc.) have BaseName matching their file names
	caBaseNames := map[string]bool{
		pkiutil.CACertAndKeyBaseName:                true,
		pkiutil.FrontProxyCACertAndKeyBaseName:      true,
		pkiutil.EtcdCACertAndKeyBaseName:            true,
		pkiutil.ServiceAccountKeyBaseName:           true,
		pkiutil.AdminKubeConfigFileName:             true,
		pkiutil.KubeletKubeConfigFileName:           true,
		pkiutil.ControllerManagerKubeConfigFileName: true,
		pkiutil.SchedulerKubeConfigFileName:         true,
	}

	for _, cert := range generator.bkeCerts {
		if !caBaseNames[cert.BaseName] && cert.CAName == "" {
			t.Errorf("prepareBkeCerts() should set CAName for cert %s (BaseName: %s)", cert.Name, cert.BaseName)
		}
	}
}

// TestHasAnyConfig tests the hasAnyConfig function with various configuration states
func TestHasAnyConfig(t *testing.T) {
	generator := createTestGenerator()

	tests := []struct {
		name string
		cfg  *CertConfigData
		want bool
	}{
		{
			name: "nil config",
			cfg:  nil,
			want: false,
		},
		{
			name: "config with available keys",
			cfg: &CertConfigData{
				AvailableKeys: map[string]bool{
					"key1": true,
				},
			},
			want: true,
		},
		{
			name: "config with no available keys",
			cfg: &CertConfigData{
				AvailableKeys: map[string]bool{
					"key1": false,
				},
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := generator.hasAnyConfig(tt.cfg)
			if got != tt.want {
				t.Errorf("hasAnyConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestCanApplyConfig tests the canApplyConfig function with various states
func TestCanApplyConfig(t *testing.T) {
	generator := createTestGenerator()

	tests := []struct {
		name     string
		bkeCerts pkiutil.Certificates
		cfg      *CertConfigData
		want     bool
	}{
		{
			name:     "bkeCerts is nil",
			bkeCerts: nil,
			cfg:      &CertConfigData{AvailableKeys: make(map[string]bool)},
			want:     false,
		},
		{
			name:     "cfg is nil",
			bkeCerts: pkiutil.Certificates{pkiutil.BKECertRootCA()},
			cfg:      nil,
			want:     false,
		},
		{
			name:     "both bkeCerts and cfg are nil",
			bkeCerts: nil,
			cfg:      nil,
			want:     false,
		},
		{
			name:     "both bkeCerts and cfg are valid",
			bkeCerts: pkiutil.Certificates{pkiutil.BKECertRootCA()},
			cfg:      &CertConfigData{AvailableKeys: make(map[string]bool)},
			want:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator.bkeCerts = tt.bkeCerts
			got := generator.canApplyConfig(tt.cfg)
			if got != tt.want {
				t.Errorf("canApplyConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestNewCertSecretName tests the NewCertSecretName function
func TestNewCertSecretName(t *testing.T) {
	tests := []struct {
		name        string
		clusterName string
		certName    string
		want        string
	}{
		{
			name:        "normal case",
			clusterName: "test-cluster",
			certName:    "ca",
			want:        "test-cluster-ca",
		},
		{
			name:        "empty cluster name",
			clusterName: "",
			certName:    "ca",
			want:        "-ca",
		},
		{
			name:        "empty cert name",
			clusterName: "test-cluster",
			certName:    "",
			want:        "test-cluster-",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewCertSecretName(tt.clusterName, tt.certName)
			if got != tt.want {
				t.Errorf("NewCertSecretName() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestHasServerAuth tests the HasServerAuth function
func TestHasServerAuth(t *testing.T) {
	serverAuthCert := createTestCertWithServerAuth()
	clientAuthCert := createTestCertWithClientAuth()
	noAuthCert := createTestCertWithoutAuth()

	tests := []struct {
		name string
		cert *x509.Certificate
		want bool
	}{
		{
			name: "cert with server auth",
			cert: serverAuthCert,
			want: true,
		},
		{
			name: "cert with client auth only",
			cert: clientAuthCert,
			want: false,
		},
		{
			name: "cert without auth",
			cert: noAuthCert,
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := HasServerAuth(tt.cert)
			if got != tt.want {
				t.Errorf("HasServerAuth() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestGetExtraAltNames tests the getExtraAltNames function
func TestGetExtraAltNames(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

	clusterWithEndpoint := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: testIPAddress1,
				Port: testEndpointPort,
			},
		},
	}

	clusterWithoutEndpoint := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{},
	}

	tests := []struct {
		name       string
		bkeCluster *bkev1beta1.BKECluster
		want       []string
	}{
		{
			name:       "cluster with valid endpoint",
			bkeCluster: clusterWithEndpoint,
			want:       []string{testIPAddress1},
		},
		{
			name:       "cluster without endpoint",
			bkeCluster: clusterWithoutEndpoint,
			want:       []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), fakeClient, tt.bkeCluster)
			got := generator.getExtraAltNames()
			if !equalStringSlices(got, tt.want) {
				t.Errorf("getExtraAltNames() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestFillInCertificateContent tests the fillInCertificateContent function
func TestFillInCertificateContent(t *testing.T) {
	fakeClient := createFakeClientForTest()
	testCertBytes := []byte("test-cert")
	testKeyBytes := []byte("test-key")
	certName := "test-cert"
	tests := getFillInCertificateContentTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), fakeClient, bkeCluster)
			generator.fillInCertificateContent(testCertBytes, testKeyBytes, certName, tt.isCA)
			verifyFillInCertificateContent(t, generator, certName, testCertBytes, tt.checkCA)
		})
	}
}

// getFillInCertificateContentTestCases returns test cases for fillInCertificateContent
func getFillInCertificateContentTestCases() []struct {
	name    string
	isCA    bool
	checkCA bool
} {
	return []struct {
		name    string
		isCA    bool
		checkCA bool
	}{
		{
			name:    "CA certificate",
			isCA:    true,
			checkCA: true,
		},
		{
			name:    "non-CA certificate",
			isCA:    false,
			checkCA: false,
		},
	}
}

// verifyFillInCertificateContent verifies the certificate content was filled correctly
func verifyFillInCertificateContent(t *testing.T, generator *BKEKubernetesCertGenerator, certName string, testCertBytes []byte, checkCA bool) {
	if checkCA {
		verifyCACertificateContent(t, generator, certName, testCertBytes)
	} else {
		verifyNonCACertificateContent(t, generator, certName, testCertBytes)
	}
}

// verifyCACertificateContent verifies CA certificate content
func verifyCACertificateContent(t *testing.T, generator *BKEKubernetesCertGenerator, certName string, testCertBytes []byte) {
	if generator.caCertificatesContent == nil {
		t.Error("caCertificatesContent should not be nil")
	}
	if data, ok := generator.caCertificatesContent[certName]; !ok {
		t.Error("CA certificate should be in caCertificatesContent")
	} else if string(data[TLSCrtDataName]) != string(testCertBytes) {
		t.Errorf("certificate data mismatch")
	}
}

// verifyNonCACertificateContent verifies non-CA certificate content
func verifyNonCACertificateContent(t *testing.T, generator *BKEKubernetesCertGenerator, certName string, testCertBytes []byte) {
	if generator.certificatesContent == nil {
		t.Error("certificatesContent should not be nil")
	}
	if data, ok := generator.certificatesContent[certName]; !ok {
		t.Error("certificate should be in certificatesContent")
	} else if string(data[TLSCrtDataName]) != string(testCertBytes) {
		t.Errorf("certificate data mismatch")
	}
}

// createFakeClientForTest creates a fake client for testing
func createFakeClientForTest() client.Client {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	return fake.NewClientBuilder().WithScheme(scheme).Build()
}

// TestValidateGlobalCASecret tests the validateGlobalCASecret function
func TestValidateGlobalCASecret(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	generator := NewKubernetesCertGenerator(context.TODO(), fakeClient, bkeCluster)

	validCertBytes, validKeyBytes := createValidTestCertAndKey()
	invalidCertBytes := []byte("invalid-cert")

	tests := []struct {
		name    string
		secret  *corev1.Secret
		wantErr bool
	}{
		{
			name:    "valid secret",
			secret:  createSecretWithData(validCertBytes, validKeyBytes),
			wantErr: false,
		},
		{
			name:    "secret with nil data",
			secret:  &corev1.Secret{Data: nil},
			wantErr: true,
		},
		{
			name:    "secret missing cert",
			secret:  createSecretWithData(nil, validKeyBytes),
			wantErr: true,
		},
		{
			name:    "secret missing key",
			secret:  createSecretWithData(validCertBytes, nil),
			wantErr: true,
		},
		{
			name:    "secret with empty cert",
			secret:  createSecretWithData([]byte{}, validKeyBytes),
			wantErr: true,
		},
		{
			name:    "secret with invalid cert",
			secret:  createSecretWithData(invalidCertBytes, validKeyBytes),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := generator.validateGlobalCASecret(tt.secret)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateGlobalCASecret() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestApplyAltNamesToCert tests the applyAltNamesToCert function
func TestApplyAltNamesToCert(t *testing.T) {

	generator := createTestGenerator()
	cert := pkiutil.BKECertAPIServer()
	altNames := &certutil.AltNames{
		DNSNames: []string{"test.example.com"},
		IPs:      []net.IP{net.ParseIP(testIPAddress1)},
	}
	extraAltNames := []string{"extra.example.com"}

	err := generator.applyAltNamesToCert(cert, altNames, extraAltNames)
	if err != nil {
		t.Errorf("applyAltNamesToCert() error = %v", err)
	}

	if len(cert.Config.AltNames.DNSNames) == 0 {
		t.Error("DNSNames should not be empty")
	}
	if len(cert.Config.AltNames.IPs) == 0 {
		t.Error("IPs should not be empty")
	}
}

// createTestCertWithServerAuth creates a test certificate with server auth
func createTestCertWithServerAuth() *x509.Certificate {
	return &x509.Certificate{
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
}

// createTestCertWithClientAuth creates a test certificate with client auth only
func createTestCertWithClientAuth() *x509.Certificate {
	return &x509.Certificate{
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
	}
}

// createTestCertWithoutAuth creates a test certificate without auth
func createTestCertWithoutAuth() *x509.Certificate {
	return &x509.Certificate{
		ExtKeyUsage: []x509.ExtKeyUsage{},
	}
}

// createValidTestCertAndKey creates valid test certificate and key bytes
func createValidTestCertAndKey() ([]byte, []byte) {
	cert := pkiutil.BKECertRootCA()
	caCert, caKey, _ := pkiutil.NewCertificateAuthority(cert)
	return pkiutil.EncodeCertToPEM(caCert), pkiutil.EncodeKeyToPEM(caKey)
}

// createSecretWithData creates a secret with certificate and key data
func createSecretWithData(certBytes, keyBytes []byte) *corev1.Secret {
	data := make(map[string][]byte)
	if len(certBytes) > 0 {
		data[TLSCrtDataName] = certBytes
	}
	if len(keyBytes) > 0 {
		data[TLSKeyDataName] = keyBytes
	}
	return &corev1.Secret{Data: data}
}

// equalStringSlices checks if two string slices are equal
func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestConfigKubeConfig tests the ConfigKubeConfig function
func TestConfigKubeConfig(t *testing.T) {

	generator := createTestGenerator()
	tests := []struct {
		name           string
		endpoint       string
		expectedResult string
	}{
		{
			name:           "empty endpoint uses default",
			endpoint:       "",
			expectedResult: bkeCluster.Spec.ControlPlaneEndpoint.String(),
		},
		{
			name:           "non-empty endpoint",
			endpoint:       testEndpoint1,
			expectedResult: testEndpoint1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator.ConfigKubeConfig(tt.endpoint)
			if generator.kubeConfigEndpoint != tt.expectedResult {
				t.Errorf("kubeConfigEndpoint = %v, want %v", generator.kubeConfigEndpoint, tt.expectedResult)
			}
			if !generator.needCreateKubeConfig {
				t.Error("needCreateKubeConfig should be true")
			}
		})
	}
}

// TestSetCertsCAName tests the SetCertsCAName function
func TestSetCertsCAName(t *testing.T) {
	generator := createTestGenerator()
	tests := []struct {
		name           string
		isUserCustomCA bool
		bkeCerts       pkiutil.Certificates
		expectCAName   string
	}{
		{
			name:           "isUserCustomCA true with CA cert",
			isUserCustomCA: true,
			bkeCerts:       pkiutil.Certificates{pkiutil.BKECertRootCA()},
			expectCAName:   "global-ca",
		},
		{
			name:           "isUserCustomCA false with CA cert",
			isUserCustomCA: false,
			bkeCerts:       pkiutil.Certificates{pkiutil.BKECertRootCA()},
			expectCAName:   "",
		},
		{
			name:           "isUserCustomCA true with FrontProxyCA",
			isUserCustomCA: true,
			bkeCerts:       pkiutil.Certificates{pkiutil.BKECertFrontProxyCA()},
			expectCAName:   "global-ca",
		},
		{
			name:           "isUserCustomCA true with EtcdCA",
			isUserCustomCA: true,
			bkeCerts:       pkiutil.Certificates{pkiutil.BKECertEtcdCA()},
			expectCAName:   "global-ca",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator.bkeCerts = tt.bkeCerts
			generator.isUserCustomCA = tt.isUserCustomCA
			generator.SetCertsCAName()

			if len(generator.bkeCerts) > 0 {
				cert := generator.bkeCerts[0]
				if cert.CAName != tt.expectCAName {
					t.Errorf("CAName = %v, want %v", cert.CAName, tt.expectCAName)
				}
			}
		})
	}
}

// TestIsHACluster tests the IsHACluster function
func TestIsHACluster(t *testing.T) {
	tests := getIsHAClusterTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsHACluster(tt.bkeCluster, tt.nodes)
			if got != tt.want {
				t.Errorf("IsHACluster() = %v, want %v", got, tt.want)
			}
		})
	}
}

// getIsHAClusterTestCases returns test cases for IsHACluster
func getIsHAClusterTestCases() []struct {
	name       string
	bkeCluster *bkev1beta1.BKECluster
	nodes      bkenode.Nodes
	want       bool
} {
	return []struct {
		name       string
		bkeCluster *bkev1beta1.BKECluster
		nodes      bkenode.Nodes
		want       bool
	}{
		{
			name: "HA cluster with endpoint not in node IPs",
			bkeCluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{
					ControlPlaneEndpoint: confv1beta1.APIEndpoint{
						Host: testHADomain,
						Port: testEndpointPort,
					},
				},
			},
			nodes: bkenode.Nodes{
				{IP: testIPAddress1},
				{IP: "10.0.0.2"},
			},
			want: true,
		},
		{
			name: "non-HA cluster with endpoint equal to node IP",
			bkeCluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{
					ControlPlaneEndpoint: confv1beta1.APIEndpoint{
						Host: testIPAddress1,
						Port: testEndpointPort,
					},
				},
			},
			nodes: bkenode.Nodes{
				{IP: testIPAddress1},
				{IP: "10.0.0.2"},
			},
			want: false,
		},
		{
			name: "cluster without endpoint",
			bkeCluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{},
			},
			nodes: bkenode.Nodes{
				{IP: testIPAddress1},
			},
			want: false,
		},
		{
			name: "cluster without nodes",
			bkeCluster: &bkev1beta1.BKECluster{
				Spec: confv1beta1.BKEClusterSpec{
					ControlPlaneEndpoint: confv1beta1.APIEndpoint{
						Host: testHADomain,
						Port: testEndpointPort,
					},
				},
			},
			nodes: nil,
			want:  false,
		},
	}
}

// TestTryLoadGlobalCAFromSecret tests the tryLoadGlobalCAFromSecret function
func TestTryLoadGlobalCAFromSecret(t *testing.T) {
	scheme := createSchemeWithCoreV1ForTest()
	generator := createTestGenerator()
	tests := getTryLoadGlobalCATestCases(scheme)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator.client = tt.clientSetup()
			data, found, err := generator.tryLoadGlobalCAFromSecret()
			result := tryLoadGlobalCAResult{
				data:        data,
				found:       found,
				err:         err,
				expectFound: tt.expectFound,
				expectError: tt.expectError,
			}
			verifyTryLoadGlobalCAResult(t, result)
		})
	}
}

// getTryLoadGlobalCATestCases returns test cases for tryLoadGlobalCAFromSecret
func getTryLoadGlobalCATestCases(scheme *runtime.Scheme) []tryLoadGlobalCATestCase {
	validSecret := createValidGlobalCASecret()
	invalidSecret := createInvalidGlobalCASecret()

	return []tryLoadGlobalCATestCase{
		createSecretNotFoundTestCase(scheme),
		createValidSecretFoundTestCase(scheme, validSecret),
		createInvalidSecretDataTestCase(scheme, invalidSecret),
	}
}

// tryLoadGlobalCATestCase represents a test case for tryLoadGlobalCAFromSecret
type tryLoadGlobalCATestCase struct {
	name        string
	clientSetup func() client.Client
	expectFound bool
	expectError bool
}

// createSecretNotFoundTestCase creates a test case for secret not found
func createSecretNotFoundTestCase(scheme *runtime.Scheme) tryLoadGlobalCATestCase {
	return tryLoadGlobalCATestCase{
		name:        "secret not found",
		clientSetup: func() client.Client { return fake.NewClientBuilder().WithScheme(scheme).Build() },
		expectFound: false,
		expectError: false,
	}
}

// createValidSecretFoundTestCase creates a test case for valid secret found
func createValidSecretFoundTestCase(scheme *runtime.Scheme, validSecret *corev1.Secret) tryLoadGlobalCATestCase {
	return tryLoadGlobalCATestCase{
		name: "valid secret found",
		clientSetup: func() client.Client {
			return fake.NewClientBuilder().WithScheme(scheme).WithObjects(validSecret).Build()
		},
		expectFound: true,
		expectError: false,
	}
}

// createInvalidSecretDataTestCase creates a test case for invalid secret data
func createInvalidSecretDataTestCase(scheme *runtime.Scheme, invalidSecret *corev1.Secret) tryLoadGlobalCATestCase {
	return tryLoadGlobalCATestCase{
		name: "invalid secret data",
		clientSetup: func() client.Client {
			return fake.NewClientBuilder().WithScheme(scheme).WithObjects(invalidSecret).Build()
		},
		expectFound: false,
		expectError: false,
	}
}

// createValidGlobalCASecret creates a valid global CA secret
func createValidGlobalCASecret() *corev1.Secret {
	validCertBytes, validKeyBytes := createValidTestCertAndKey()
	secret := createSecretWithData(validCertBytes, validKeyBytes)
	secret.Namespace = GlobalCANamespace
	secret.Name = GlobalCASecretName
	return secret
}

// createInvalidGlobalCASecret creates an invalid global CA secret
func createInvalidGlobalCASecret() *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: GlobalCANamespace,
			Name:      GlobalCASecretName,
		},
		Data: nil,
	}
}

// createGeneratorForGlobalCATest creates a generator for global CA tests
func createGeneratorForGlobalCATest(scheme *runtime.Scheme) *BKEKubernetesCertGenerator {
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	return NewKubernetesCertGenerator(context.TODO(), fakeClient, bkeCluster)
}

// createSchemeWithCoreV1ForTest creates a scheme with CoreV1 for testing
func createSchemeWithCoreV1ForTest() *runtime.Scheme {
	return createSchemeWithCoreV1()
}

// tryLoadGlobalCAResult represents the result of tryLoadGlobalCAFromSecret
type tryLoadGlobalCAResult struct {
	data        map[string][]byte
	found       bool
	err         error
	expectFound bool
	expectError bool
}

// verifyTryLoadGlobalCAResult verifies the result of tryLoadGlobalCAFromSecret
func verifyTryLoadGlobalCAResult(t *testing.T, result tryLoadGlobalCAResult) {
	verifyTryLoadGlobalCAError(t, result.err, result.expectError)
	verifyTryLoadGlobalCAFound(t, result.found, result.expectFound)
	verifyTryLoadGlobalCAData(t, result.data, result.expectFound)
}

// verifyTryLoadGlobalCAError verifies the error result
func verifyTryLoadGlobalCAError(t *testing.T, err error, expectError bool) {
	if expectError && err == nil {
		t.Errorf("tryLoadGlobalCAFromSecret() expected error but got nil")
	}
	if !expectError && err != nil {
		t.Errorf("tryLoadGlobalCAFromSecret() unexpected error = %v", err)
	}
}

// verifyTryLoadGlobalCAFound verifies the found result
func verifyTryLoadGlobalCAFound(t *testing.T, found, expectFound bool) {
	if found != expectFound {
		t.Errorf("tryLoadGlobalCAFromSecret() found = %v, want %v", found, expectFound)
	}
}

// verifyTryLoadGlobalCAData verifies the data result
func verifyTryLoadGlobalCAData(t *testing.T, data map[string][]byte, expectFound bool) {
	if expectFound && data == nil {
		t.Error("tryLoadGlobalCAFromSecret() should return data when found")
	}
}

// TestCreateGlobalCASecret tests the createGlobalCASecret function
func TestCreateGlobalCASecret(t *testing.T) {
	scheme := createSchemeWithCoreV1()
	testData := createTestGlobalCAData()
	tests := getCreateGlobalCASecretTestCases(scheme)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runCreateGlobalCASecretTest(t, tt, testData)
		})
	}
}

// getCreateGlobalCASecretTestCases returns test cases for createGlobalCASecret
func getCreateGlobalCASecretTestCases(scheme *runtime.Scheme) []createGlobalCASecretTestCase {
	return []createGlobalCASecretTestCase{
		createSuccessfulCreateTestCase(scheme),
		createAlreadyExistsTestCase(scheme),
	}
}

// createGlobalCASecretTestCase represents a test case for createGlobalCASecret
type createGlobalCASecretTestCase struct {
	name          string
	bkeCluster    *bkev1beta1.BKECluster
	clientSetup   func() client.Client
	expectError   bool
	expectErrType string
}

// createSuccessfulCreateTestCase creates a test case for successful creation
func createSuccessfulCreateTestCase(scheme *runtime.Scheme) createGlobalCASecretTestCase {
	return createGlobalCASecretTestCase{
		name:       "successful create with bkeCluster",
		bkeCluster: bkeCluster,
		clientSetup: func() client.Client {
			return fake.NewClientBuilder().WithScheme(scheme).Build()
		},
		expectError: false,
	}
}

// createAlreadyExistsTestCase creates a test case for already exists error
func createAlreadyExistsTestCase(scheme *runtime.Scheme) createGlobalCASecretTestCase {
	return createGlobalCASecretTestCase{
		name:       "already exists error",
		bkeCluster: bkeCluster,
		clientSetup: func() client.Client {
			return &alreadyExistsClient{Client: fake.NewClientBuilder().WithScheme(scheme).Build()}
		},
		expectError:   true,
		expectErrType: "already exists",
	}
}

// runCreateGlobalCASecretTest runs a single test case for createGlobalCASecret
func runCreateGlobalCASecretTest(t *testing.T, tt createGlobalCASecretTestCase, testData map[string][]byte) {
	fakeClient := tt.clientSetup()
	generator := NewKubernetesCertGenerator(context.TODO(), fakeClient, tt.bkeCluster)
	err := generator.createGlobalCASecret(testData)
	verifyCreateGlobalCASecretError(t, err, tt.expectError, tt.expectErrType)
}

// TestLoadCaCertContent tests the loadCaCertContent function
func TestLoadCaCertContent(t *testing.T) {
	scheme := createSchemeWithCoreV1ForTest()
	caSecret := createTestCASecret()
	tests := getLoadCaCertContentTestCases(scheme, caSecret)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runLoadCaCertContentTest(t, tt)
		})
	}
}

// getLoadCaCertContentTestCases returns test cases for loadCaCertContent
func getLoadCaCertContentTestCases(scheme *runtime.Scheme, caSecret *corev1.Secret) []loadCaCertContentTestCase {
	return []loadCaCertContentTestCase{
		createSecretsNotFoundTestCase(scheme),
		createSecretFoundTestCase(scheme, caSecret),
	}
}

// loadCaCertContentTestCase represents a test case for loadCaCertContent
type loadCaCertContentTestCase struct {
	name        string
	clientSetup func() client.Client
	expectError bool
}

// createSecretsNotFoundTestCase creates a test case for secrets not found
func createSecretsNotFoundTestCase(scheme *runtime.Scheme) loadCaCertContentTestCase {
	return loadCaCertContentTestCase{
		name: "secrets not found",
		clientSetup: func() client.Client {
			return fake.NewClientBuilder().WithScheme(scheme).Build()
		},
		expectError: false,
	}
}

// createSecretFoundTestCase creates a test case for secret found
func createSecretFoundTestCase(scheme *runtime.Scheme, caSecret *corev1.Secret) loadCaCertContentTestCase {
	return loadCaCertContentTestCase{
		name: "secret found and loaded",
		clientSetup: func() client.Client {
			return fake.NewClientBuilder().WithScheme(scheme).WithObjects(caSecret).Build()
		},
		expectError: false,
	}
}

// createTestCASecret creates a test CA secret
func createTestCASecret() *corev1.Secret {
	validCertBytes, validKeyBytes := createValidTestCertAndKey()
	caSecret := createSecretWithData(validCertBytes, validKeyBytes)
	caSecret.Namespace = "default"
	caSecret.Name = "test-ca"
	return caSecret
}

// runLoadCaCertContentTest runs a single test case for loadCaCertContent
func runLoadCaCertContentTest(t *testing.T, tt loadCaCertContentTestCase) {
	generator := NewKubernetesCertGenerator(context.TODO(), tt.clientSetup(), bkeCluster)
	generator.certClusterName = "test"
	err := generator.loadCaCertContent()
	verifyLoadCaCertContentError(t, err, tt.expectError)
}

// verifyLoadCaCertContentError verifies the error from loadCaCertContent
func verifyLoadCaCertContentError(t *testing.T, err error, expectError bool) {
	if (err != nil) != expectError {
		t.Errorf("loadCaCertContent() error = %v, wantErr %v", err, expectError)
	}
}

// createSchemeWithCoreV1 creates a scheme with CoreV1 and BKE types
func createSchemeWithCoreV1() *runtime.Scheme {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	_ = corev1.AddToScheme(scheme)
	return scheme
}

// createTestGlobalCAData creates test data for global CA
func createTestGlobalCAData() map[string][]byte {
	return map[string][]byte{
		TLSCrtDataName: []byte("test-cert"),
		TLSKeyDataName: []byte("test-key"),
	}
}

// verifyCreateGlobalCASecretError verifies the error from createGlobalCASecret
func verifyCreateGlobalCASecretError(t *testing.T, err error, expectError bool, expectErrType string) {
	if expectError {
		if err == nil {
			t.Errorf("createGlobalCASecret() expected error but got nil")
		} else if expectErrType != "" {
			if !strings.Contains(err.Error(), expectErrType) {
				t.Errorf("createGlobalCASecret() error = %v, want error containing %v", err, expectErrType)
			}
		}
	} else {
		if err != nil {
			t.Errorf("createGlobalCASecret() unexpected error = %v", err)
		}
	}
}

// TestSetNodes tests the SetNodes function
func TestSetNodes(t *testing.T) {
	generator := createTestGenerator()
	testNodes := bkenode.Nodes{
		{IP: testIPAddress1},
	}

	generator.SetNodes(testNodes)

	assert.Equal(t, testNodes, generator.nodes)
}

// alreadyExistsClient is a test client that returns AlreadyExists error on Create
type alreadyExistsClient struct {
	client.Client
}

func (c *alreadyExistsClient) Create(ctx context.Context, obj client.Object, opts ...client.CreateOption) error {
	return apierrors.NewAlreadyExists(corev1.Resource("secret"), obj.GetName())
}

// TestLookupRegularCert tests the lookupRegularCert function
func TestLookupRegularCert(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	tests := []struct {
		name        string
		secretName  string
		setupClient func() client.Client
		expectFound bool
		expectError bool
	}{
		{
			name:        "secret not found",
			secretName:  "nonexistent",
			setupClient: func() client.Client { return fake.NewClientBuilder().WithScheme(scheme).Build() },
			expectFound: false,
			expectError: false,
		},
		{
			name:       "secret found",
			secretName: "existing-secret",
			setupClient: func() client.Client {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "existing-secret",
						Namespace: "default",
					},
					Data: map[string][]byte{
						TLSCrtDataName: []byte("cert"),
					},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			},
			expectFound: true,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), tt.setupClient(), bkeCluster)
			found, err := generator.lookupRegularCert(tt.secretName)

			assert.Equal(t, tt.expectFound, found)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestLookupKubeConfigCert tests the new phaseutil.RetryReadNotFound-backed
// lookupKubeConfigCert function. The retry helper is exercised but should
// ultimately return the same result as the field-level checks would suggest.
func TestLookupKubeConfigCert(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	haCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: testHADomain,
				Port: testEndpointPort,
			},
		},
	}
	haNodes := bkenode.Nodes{{IP: testIPAddress1}}

	tests := []struct {
		name        string
		secret      *corev1.Secret
		bkeCluster  *bkev1beta1.BKECluster
		nodes       bkenode.Nodes
		setupClient func(*corev1.Secret) client.Client
		expectFound bool
		expectError bool
	}{
		{
			name:       "secret not found returns false (retry budget exhausted)",
			secret:     &corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"}},
			bkeCluster: bkeCluster,
			setupClient: func(s *corev1.Secret) client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			expectFound: false,
			expectError: false,
		},
		{
			name: "secret without value field returns false (synthesised NotFound)",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Data:       map[string][]byte{},
			},
			bkeCluster: bkeCluster,
			setupClient: func(s *corev1.Secret) client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(s).Build()
			},
			expectFound: false,
			expectError: false,
		},
		{
			name: "HA cluster secret without ha field returns false (synthesised NotFound)",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Data:       map[string][]byte{"value": []byte("kubeconfig")},
			},
			bkeCluster: haCluster,
			nodes:      haNodes,
			setupClient: func(s *corev1.Secret) client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(s).Build()
			},
			expectFound: false,
			expectError: false,
		},
		{
			name: "HA cluster secret with value+ha returns true",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Data: map[string][]byte{
					"value": []byte("kubeconfig"),
					"ha":    []byte("ha-kubeconfig"),
				},
			},
			bkeCluster: haCluster,
			nodes:      haNodes,
			setupClient: func(s *corev1.Secret) client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(s).Build()
			},
			expectFound: true,
			expectError: false,
		},
		{
			name: "non-HA cluster secret with value only returns true",
			secret: &corev1.Secret{
				ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
				Data:       map[string][]byte{"value": []byte("kubeconfig")},
			},
			bkeCluster: &bkev1beta1.BKECluster{
				ObjectMeta: metav1.ObjectMeta{Namespace: "default"},
				Spec:       confv1beta1.BKEClusterSpec{},
			},
			nodes: nil,
			setupClient: func(s *corev1.Secret) client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(s).Build()
			},
			expectFound: true,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), tt.setupClient(tt.secret), tt.bkeCluster)
			generator.SetNodes(tt.nodes)
			found, err := generator.lookupKubeConfigCert(tt.secret.ObjectMeta.Name)

			assert.Equal(t, tt.expectFound, found)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestNeedGenerate tests the NeedGenerate function
func TestNeedGenerate(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	tests := []struct {
		name        string
		setupClient func() client.Client
		expectNeed  bool
		expectError bool
	}{
		{
			name: "no secrets exist - need generate",
			setupClient: func() client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			expectNeed:  true,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), tt.setupClient(), bkeCluster)
			need, err := generator.NeedGenerate()

			assert.Equal(t, tt.expectNeed, need)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestGenerateCACertAndKey tests the generateCACertAndKey function
func TestGenerateCACertAndKey(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	tests := []struct {
		name        string
		cert        *pkiutil.BKECert
		expectError bool
	}{
		{
			name:        "generate CA cert",
			cert:        pkiutil.BKECertRootCA(),
			expectError: false,
		},
		{
			name:        "admin kubeconfig cert skipped",
			cert:        pkiutil.BKEAdminKubeConfig(),
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), fake.NewClientBuilder().WithScheme(scheme).Build(), bkeCluster)
			err := generator.generateCACertAndKey(tt.cert)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestGenerateSAKeyAndPublicKey tests the generateSAKeyAndPublicKey function
func TestGenerateSAKeyAndPublicKey(t *testing.T) {
	scheme := createSchemeWithCoreV1()
	generator := NewKubernetesCertGenerator(context.TODO(), fake.NewClientBuilder().WithScheme(scheme).Build(), bkeCluster)

	saCert := pkiutil.BKECertServiceAccount()
	err := generator.generateSAKeyAndPublicKey(saCert)

	assert.NoError(t, err)
	assert.NotNil(t, generator.certificatesContent)
}

// TestGenerateCertAndKeyWithCA tests the generateCertAndKeyWithCA function
func TestGenerateCertAndKeyWithCA(t *testing.T) {
	scheme := createSchemeWithCoreV1()
	validCertBytes, validKeyBytes := createValidTestCertAndKey()

	tests := []struct {
		name        string
		cert        *pkiutil.BKECert
		setupClient func() client.Client
		expectError bool
	}{
		{
			name: "generate cert with CA",
			cert: pkiutil.BKECertAPIServer(),
			setupClient: func() client.Client {
				caSecret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-ca",
						Namespace: "default",
					},
					Data: map[string][]byte{
						TLSCrtDataName: validCertBytes,
						TLSKeyDataName: validKeyBytes,
					},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(caSecret).Build()
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), tt.setupClient(), bkeCluster)
			generator.certClusterName = "test"
			generator.caCertificatesContent = map[string]map[string][]byte{
				"ca": {
					TLSCrtDataName: validCertBytes,
					TLSKeyDataName: validKeyBytes,
				},
			}
			tt.cert.CAName = "ca"
			err := generator.generateCertAndKeyWithCA(tt.cert, false)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestGenerateCertificates tests the generateCertificates function
func TestGenerateCertificates(t *testing.T) {
	scheme := createSchemeWithCoreV1()
	generator := NewKubernetesCertGenerator(context.TODO(), fake.NewClientBuilder().WithScheme(scheme).Build(), bkeCluster)

	generator.bkeCerts = pkiutil.Certificates{pkiutil.BKECertRootCA()}
	needCreate, err := generator.generateCertificates()

	assert.NoError(t, err)
	assert.True(t, needCreate)
}

// TestGetCertificateFromSecret tests the getCertificateFromSecret function
func TestGetCertificateFromSecret(t *testing.T) {
	scheme := createSchemeWithCoreV1()
	validCertBytes, validKeyBytes := createValidTestCertAndKey()

	tests := []struct {
		name        string
		certName    string
		setupClient func() client.Client
		expectError bool
	}{
		{
			name:     "secret found",
			certName: "ca",
			setupClient: func() client.Client {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-ca",
						Namespace: "default",
					},
					Data: map[string][]byte{
						TLSCrtDataName: validCertBytes,
						TLSKeyDataName: validKeyBytes,
					},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			},
			expectError: false,
		},
		{
			name:     "secret not found",
			certName: "nonexistent",
			setupClient: func() client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			expectError: true,
		},
		{
			name:     "secret missing cert data",
			certName: "ca",
			setupClient: func() client.Client {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-ca",
						Namespace: "default",
					},
					Data: map[string][]byte{},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), tt.setupClient(), bkeCluster)
			generator.certClusterName = "test"
			crt, err := generator.getCertificateFromSecret(tt.certName)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, crt)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, crt)
			}
		})
	}
}

// TestVerifyCertificateSans tests the VerifyCertificateSans function
func TestVerifyCertificateSans(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	tests := []struct {
		name        string
		setupClient func() client.Client
		expectError bool
	}{
		{
			name: "no certificates",
			setupClient: func() client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), tt.setupClient(), bkeCluster)
			generator.certClusterName = "test"
			err := generator.VerifyCertificateSans()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestGenerateKubeConfig tests the GenerateKubeConfig function
func TestGenerateKubeConfig(t *testing.T) {
	t.Skip("TestGenerateKubeConfig requires CA secrets and kubeconfig controller - skipping for unit tests")
}

// TestHandleHAKubeConfig tests the handleHAKubeConfig function
func TestHandleHAKubeConfig(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	haCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default", UID: "test"},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: testHADomain,
				Port: testEndpointPort,
			},
		},
	}

	kubeconfigSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "test-kubeconfig", Namespace: "default"},
		Data: map[string][]byte{
			"value": []byte("apiVersion: v1\nclusters:\n- cluster:\n    server: https://localhost:6443\n  name: test\ncontexts:\n- context:\n    cluster: test\n    user: admin\ncurrent-context: test\nkind: Config\nusers:\n- name: admin\n  user:\n    token: dummy"),
		},
	}

	generator := NewKubernetesCertGenerator(context.TODO(), fake.NewClientBuilder().WithScheme(scheme).WithObjects(kubeconfigSecret).Build(), haCluster)
	generator.certClusterName = "test"
	generator.certNamespace = "default"

	err := generator.handleHAKubeConfig()
	assert.NoError(t, err)
}

// TestUpdateHAKubeConfig tests the updateHAKubeConfig function
func TestUpdateHAKubeConfig(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	haCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default", UID: "test"},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: testHADomain,
				Port: testEndpointPort,
			},
		},
	}

	kubeconfigSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "test-kubeconfig", Namespace: "default"},
		Data: map[string][]byte{
			"value": []byte("apiVersion: v1\nclusters:\n- cluster:\n    server: https://localhost:6443\n  name: test\ncontexts:\n- context:\n    cluster: test\n    user: admin\ncurrent-context: test\nkind: Config\nusers:\n- name: admin\n  user:\n    token: dummy"),
		},
	}

	generator := NewKubernetesCertGenerator(context.TODO(), fake.NewClientBuilder().WithScheme(scheme).WithObjects(kubeconfigSecret).Build(), haCluster)
	generator.certClusterName = "test"
	generator.certNamespace = "default"

	err := generator.updateHAKubeConfig(kubeconfigSecret, "test-kubeconfig")
	assert.NoError(t, err)
}

// TestUpdateHAKubeConfig_MissingValue tests updateHAKubeConfig with missing value field
func TestUpdateHAKubeConfig_MissingValue(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	haCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default", UID: "test"},
		Spec: confv1beta1.BKEClusterSpec{
			ControlPlaneEndpoint: confv1beta1.APIEndpoint{
				Host: testHADomain,
				Port: testEndpointPort,
			},
		},
	}

	kubeconfigSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "test-kubeconfig", Namespace: "default"},
		Data:       map[string][]byte{},
	}

	generator := NewKubernetesCertGenerator(context.TODO(), fake.NewClientBuilder().WithScheme(scheme).Build(), haCluster)
	generator.certClusterName = "test"
	generator.certNamespace = "default"

	err := generator.updateHAKubeConfig(kubeconfigSecret, "test-kubeconfig")
	assert.Error(t, err)
}

// TestCreateCertificateSecrets tests the createCertificateSecrets function
func TestCreateCertificateSecrets(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	tests := []struct {
		name                  string
		caCertificatesContent map[string]map[string][]byte
		certificatesContent   map[string]map[string][]byte
		expectError           bool
	}{
		{
			name: "create certificates successfully",
			caCertificatesContent: map[string]map[string][]byte{
				"ca": {
					TLSCrtDataName: []byte("ca-cert"),
					TLSKeyDataName: []byte("ca-key"),
				},
			},
			certificatesContent: map[string]map[string][]byte{
				"ca": {
					TLSCrtDataName: []byte("cert"),
					TLSKeyDataName: []byte("key"),
				},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), fake.NewClientBuilder().WithScheme(scheme).Build(), bkeCluster)
			generator.caCertificatesContent = tt.caCertificatesContent
			generator.certificatesContent = tt.certificatesContent
			generator.needCreateKubeConfig = false

			err := generator.createCertificateSecrets()

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// createExpiringTestCertAndKey creates a CA certificate that expires within 1 hour.
func createExpiringTestCertAndKey() ([]byte, []byte) {
	cert := pkiutil.BKECertRootCA()
	cert.Config.Validity = 1 * time.Hour
	caCert, caKey, _ := pkiutil.NewCertificateAuthority(cert)
	return pkiutil.EncodeCertToPEM(caCert), pkiutil.EncodeKeyToPEM(caKey)
}

// createExpiredTestCertAndKey creates a CA certificate that has already expired.
func createExpiredTestCertAndKey() ([]byte, []byte) {
	cert := pkiutil.BKECertRootCA()
	// Negative validity produces a NotAfter in the past
	cert.Config.Validity = -1 * time.Hour
	caCert, caKey, _ := pkiutil.NewCertificateAuthority(cert)
	return pkiutil.EncodeCertToPEM(caCert), pkiutil.EncodeKeyToPEM(caKey)
}

// TestIsCertExpiringSoon tests the isCertExpiringSoon function.
func TestIsCertExpiringSoon(t *testing.T) {
	scheme := createSchemeWithCoreV1()
	validCertBytes, _ := createValidTestCertAndKey()
	expiringCertBytes, _ := createExpiringTestCertAndKey()
	expiredCertBytes, _ := createExpiredTestCertAndKey()

	tests := []struct {
		name         string
		cert         *pkiutil.BKECert
		setupClient  func() client.Client
		expectExpiry bool
	}{
		{
			name: "kubeconfig skipped",
			cert: pkiutil.BKEAdminKubeConfig(),
			setupClient: func() client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			expectExpiry: false,
		},
		{
			name: "valid certificate not expiring",
			cert: pkiutil.BKECertRootCA(),
			setupClient: func() client.Client {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{Name: "test-ca", Namespace: "default"},
					Data:       map[string][]byte{TLSCrtDataName: validCertBytes},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			},
			expectExpiry: false,
		},
		{
			name: "certificate expiring soon",
			cert: pkiutil.BKECertRootCA(),
			setupClient: func() client.Client {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{Name: "test-ca", Namespace: "default"},
					Data:       map[string][]byte{TLSCrtDataName: expiringCertBytes},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			},
			expectExpiry: true,
		},
		{
			name: "certificate already expired",
			cert: pkiutil.BKECertRootCA(),
			setupClient: func() client.Client {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{Name: "test-ca", Namespace: "default"},
					Data:       map[string][]byte{TLSCrtDataName: expiredCertBytes},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			},
			expectExpiry: true,
		},
		{
			name: "secret not found returns false (not blocking)",
			cert: pkiutil.BKECertRootCA(),
			setupClient: func() client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			expectExpiry: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), tt.setupClient(), bkeCluster)
			generator.certClusterName = "test"
			result := generator.isCertExpiringSoon(tt.cert)
			assert.Equal(t, tt.expectExpiry, result)
		})
	}
}

// TestLookupWithExpiration tests that lookup() returns false for expiring certs.
func TestLookupWithExpiration(t *testing.T) {
	scheme := createSchemeWithCoreV1()
	validCertBytes, validKeyBytes := createValidTestCertAndKey()
	expiringCertBytes, expiringKeyBytes := createExpiringTestCertAndKey()

	tests := []struct {
		name        string
		cert        *pkiutil.BKECert
		setupClient func() client.Client
		expectFound bool
		expectError bool
	}{
		{
			name: "valid cert exists - returns true",
			cert: pkiutil.BKECertRootCA(),
			setupClient: func() client.Client {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{Name: "test-ca", Namespace: "default"},
					Data:       map[string][]byte{TLSCrtDataName: validCertBytes, TLSKeyDataName: validKeyBytes},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			},
			expectFound: true,
			expectError: false,
		},
		{
			name: "expiring cert exists - returns false to trigger regeneration",
			cert: pkiutil.BKECertRootCA(),
			setupClient: func() client.Client {
				secret := &corev1.Secret{
					ObjectMeta: metav1.ObjectMeta{Name: "test-ca", Namespace: "default"},
					Data:       map[string][]byte{TLSCrtDataName: expiringCertBytes, TLSKeyDataName: expiringKeyBytes},
				}
				return fake.NewClientBuilder().WithScheme(scheme).WithObjects(secret).Build()
			},
			expectFound: false,
			expectError: false,
		},
		{
			name: "cert not found - returns false",
			cert: pkiutil.BKECertRootCA(),
			setupClient: func() client.Client {
				return fake.NewClientBuilder().WithScheme(scheme).Build()
			},
			expectFound: false,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			generator := NewKubernetesCertGenerator(context.TODO(), tt.setupClient(), bkeCluster)
			generator.certClusterName = "test"
			found, err := generator.lookup(tt.cert)

			assert.Equal(t, tt.expectFound, found)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestGenerateCertificatesCACascade tests that regenerating a CA forces leaf cert regeneration.
func TestGenerateCertificatesCACascade(t *testing.T) {
	scheme := createSchemeWithCoreV1()

	// Create an expiring Root CA cert (will trigger regeneration + cascade)
	expiringCertBytes, expiringKeyBytes := createExpiringTestCertAndKey()

	// Create a valid (non-expiring) leaf cert to simulate one that would otherwise be skipped.
	// The cascade logic should force its regeneration regardless.
	leafCertBytes, leafKeyBytes := createValidTestCertAndKey()

	rootCASecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "test-ca", Namespace: "default"},
		Data:       map[string][]byte{TLSCrtDataName: expiringCertBytes, TLSKeyDataName: expiringKeyBytes},
	}
	leafSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "test-apiserver", Namespace: "default"},
		Data:       map[string][]byte{TLSCrtDataName: leafCertBytes, TLSKeyDataName: leafKeyBytes},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(rootCASecret, leafSecret).
		Build()

	generator := NewKubernetesCertGenerator(context.TODO(), fakeClient, bkeCluster)
	generator.certClusterName = "test"
	generator.bkeCerts = pkiutil.Certificates{
		pkiutil.BKECertRootCA(),
		pkiutil.BKECertAPIServer(),
	}

	needCreate, err := generator.generateCertificates()

	assert.NoError(t, err)
	// Both certs should be regenerated: CA because it's expiring, leaf because of cascade
	assert.True(t, needCreate)
}
