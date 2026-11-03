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

package pkiutil

import (
	"crypto/rsa"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	certutil "k8s.io/client-go/util/cert"
)

const (
	// test constants for cluster configuration
	testClusterName  = "test-cluster"
	testServerPort   = "6443"
	testCertValidity = 8760 * time.Hour // 1 year
)

// testIPGenerator generates test IP addresses dynamically
type testIPGenerator struct {
	octet1, octet2, octet3, octet4 byte
}

// newTestIPGenerator creates a new IP generator starting from a base IP
func newTestIPGenerator(baseIP string) *testIPGenerator {
	ip := net.ParseIP(baseIP)
	if ip == nil || ip.To4() == nil {
		ip = net.ParseIP("127.0.0.1")
	}
	ipv4 := ip.To4()
	return &testIPGenerator{
		octet1: ipv4[0],
		octet2: ipv4[1],
		octet3: ipv4[2],
		octet4: ipv4[3],
	}
}

// nextIP generates the next IP address
func (g *testIPGenerator) nextIP() string {
	ip := fmt.Sprintf("%d.%d.%d.%d", g.octet1, g.octet2, g.octet3, g.octet4)
	g.octet4++
	if g.octet4 == 0 {
		g.octet3++
		if g.octet3 == 0 {
			g.octet2++
			if g.octet2 == 0 {
				g.octet1++
			}
		}
	}
	return ip
}

// createTestTempDir creates a temporary directory for testing
func createTestTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "kubeconfig-cert-test-*")
	require.NoError(t, err)
	return dir
}

// createTestCACertAndKey creates a CA certificate and key for testing
func createTestCACertAndKey(t *testing.T, pkiPath string) (*x509.Certificate, *rsa.PrivateKey) {
	t.Helper()
	caCertSpec := BKECertRootCA()
	caCertSpec.PkiPath = pkiPath

	caKey, err := newPrivateKey()
	require.NoError(t, err)

	caCert, err := newSelfSignedCACert(caCertSpec, caKey)
	require.NoError(t, err)

	err = WriteCertAndKey(caCertSpec, caCert, caKey)
	require.NoError(t, err)

	return caCert, caKey
}

// createTestKubeConfigGenerater creates a KubeConfigGenerater for testing
func createTestKubeConfigGenerater(t *testing.T, pkiPath, fileName, hostIP string) *KubeConfigGenerater {
	t.Helper()
	return NewKubeConfigGenerater(KubeConfigOptions{
		PkiPath:     pkiPath,
		ClusterName: testClusterName,
		FileName:    fileName,
		ServerPort:  testServerPort,
		HostIP:      hostIP,
	})
}

// createTestBKEConfig creates a BKEConfig with master nodes for testing
func createTestBKEConfig(t *testing.T, ipGen *testIPGenerator, masterCount int) *bkev1beta1.BKEConfig {
	t.Helper()
	nodes := make([]bkev1beta1.Node, 0, masterCount)
	for i := 0; i < masterCount; i++ {
		ip := ipGen.nextIP()
		nodes = append(nodes, bkev1beta1.Node{
			IP:       ip,
			Hostname: fmt.Sprintf("master-%d", i),
			Role:     []string{"master"},
		})
	}
	return &bkev1beta1.BKEConfig{
		Nodes: nodes,
	}
}

// createTestBKEConfigWithF5IP creates a BKEConfig with F5 load balancer IP
func createTestBKEConfigWithF5IP(t *testing.T, ipGen *testIPGenerator, f5IP string) *bkev1beta1.BKEConfig {
	t.Helper()
	config := createTestBKEConfig(t, ipGen, 1)
	config.CustomExtra = map[string]string{
		"extraLoadBalanceIP": f5IP,
	}
	return config
}

// createTestCSRFile creates a CSR configuration file for testing
func createTestCSRFile(t *testing.T, configDir, filename string, csr *CertCSR) {
	t.Helper()
	err := os.MkdirAll(configDir, DirDefaultPermission)
	require.NoError(t, err)

	filePath := filepath.Join(configDir, filename)
	data, err := json.Marshal(csr)
	require.NoError(t, err)

	err = os.WriteFile(filePath, data, FileDefaultPermission)
	require.NoError(t, err)
}

// createTestPolicyFile creates a sign policy file for testing
func createTestPolicyFile(t *testing.T, configDir string, policy *CertPolicy) {
	t.Helper()
	err := os.MkdirAll(configDir, DirDefaultPermission)
	require.NoError(t, err)

	filePath := filepath.Join(configDir, signPolicyConfigFile)
	data, err := json.Marshal(policy)
	require.NoError(t, err)

	err = os.WriteFile(filePath, data, FileDefaultPermission)
	require.NoError(t, err)
}

// createTestCSR creates a test CSR configuration
func createTestCSR() *CertCSR {
	return &CertCSR{
		CN: "test-admin",
		O:  "system:masters",
		C:  "CN",
		ST: "Beijing",
		L:  "Beijing",
		OU: "Kubernetes",
		Key: struct {
			Algo string `json:"algo"`
			Size int    `json:"size"`
		}{
			Algo: "rsa",
			Size: 2048,
		},
		Hosts: []string{},
	}
}

// createTestPolicy creates a test sign policy
func createTestPolicy() *CertPolicy {
	policy := &CertPolicy{}
	policy.Signing.Default.Usages = []string{"digital signature", "key encipherment", "client auth"}
	policy.Signing.Default.Expiry = "8760h"
	policy.Signing.Profiles = make(map[string]struct {
		Usages       []string `json:"usages,omitempty"`
		Expiry       string   `json:"expiry,omitempty"`
		CAConstraint *struct {
			IsCA           bool `json:"is_ca,omitempty"`
			MaxPathLen     int  `json:"max_path_len,omitempty"`
			MaxPathLenZero bool `json:"max_path_len_zero,omitempty"`
		} `json:"ca_constraint,omitempty"`
	})
	profileUsages := []string{"digital signature", "key encipherment", "client auth"}
	profileExpiry := "8760h"
	profile := struct {
		Usages       []string `json:"usages,omitempty"`
		Expiry       string   `json:"expiry,omitempty"`
		CAConstraint *struct {
			IsCA           bool `json:"is_ca,omitempty"`
			MaxPathLen     int  `json:"max_path_len,omitempty"`
			MaxPathLenZero bool `json:"max_path_len_zero,omitempty"`
		} `json:"ca_constraint,omitempty"`
	}{
		Usages: profileUsages,
		Expiry: profileExpiry,
	}
	policy.Signing.Profiles[AdminKubeConfigFileName] = profile
	return policy
}

// validateClientCert validates the generated client certificate
func validateClientCert(t *testing.T, cert *x509.Certificate, expectedCN, expectedOrg string) {
	t.Helper()
	assert.Equal(t, expectedCN, cert.Subject.CommonName)
	assert.Contains(t, cert.Subject.Organization, expectedOrg)
	assert.Contains(t, cert.ExtKeyUsage, x509.ExtKeyUsageClientAuth)
	assert.False(t, cert.IsCA)
}

// TestGenerateClientCertAndKey_WithDefaultConfig tests generating client cert with default config
func TestGenerateClientCertAndKey_WithDefaultConfig(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()
	generater.BKECertSpec.PkiPath = pkiPath

	caCert, caKey := createTestCACertAndKey(t, pkiPath)

	result, err := generater.GenerateClientCertAndKey(caCert, caKey)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.Cert)
	require.NotNil(t, result.Key)

	validateClientCert(t, result.Cert, "kubernetes-admin", "system:masters")
}

// TestApplyCSRToCert tests applying CSR configuration to certificate
func TestApplyCSRToCert(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()

	csr := createTestCSR()
	csr.CN = "custom-admin"
	csr.O = "custom-org"
	csr.C = "US"
	csr.ST = "California"
	csr.L = "San Francisco"
	csr.OU = "Engineering"
	csr.Key.Size = 4096
	csr.Key.Algo = "rsa"

	generater.applyCSRToCert(csr)

	assert.Equal(t, csr.CN, generater.BKECertSpec.Config.CommonName)
	assert.Equal(t, []string{csr.O}, generater.BKECertSpec.Config.Organization)
	assert.Equal(t, []string{csr.C}, generater.BKECertSpec.Config.Country)
	assert.Equal(t, []string{csr.ST}, generater.BKECertSpec.Config.Province)
	assert.Equal(t, []string{csr.L}, generater.BKECertSpec.Config.Locality)
	assert.Equal(t, []string{csr.OU}, generater.BKECertSpec.Config.OrganizationalUnit)
	assert.Equal(t, csr.Key.Size, generater.BKECertSpec.Config.KeySize)
	assert.Equal(t, x509.RSA, generater.BKECertSpec.Config.PublicKeyAlgorithm)
}

// TestApplyCSRToCert_KubeletPreservesCN tests that kubelet CN is preserved
func TestApplyCSRToCert_KubeletPreservesCN(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubelet", hostIP)
	generater.BKECertSpec = BKEKubeletKubeConfig()
	originalCN := generater.BKECertSpec.Config.CommonName

	csr := createTestCSR()
	csr.CN = "should-not-be-used"

	generater.applyCSRToCert(csr)

	assert.Equal(t, originalCN, generater.BKECertSpec.Config.CommonName)
}

// TestApplyPolicyToCert_WithProfile tests applying policy with profile
func TestApplyPolicyToCert_WithProfile(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()

	policy := createTestPolicy()
	profile := policy.Signing.Profiles[AdminKubeConfigFileName]
	profile.Usages = []string{"digital signature", "key encipherment", "client auth"}
	profile.Expiry = "4380h"
	policy.Signing.Profiles[AdminKubeConfigFileName] = profile

	generater.applyPolicyToCert(policy)

	assert.Contains(t, generater.BKECertSpec.Config.Usages, x509.ExtKeyUsageClientAuth)
	assert.Equal(t, testCertValidity/two, generater.BKECertSpec.Config.Validity)
}

// TestApplyPolicyToCert_WithDefault tests applying policy with default profile
func TestApplyPolicyToCert_WithDefault(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()
	generater.BKECertSpec.BaseName = "unknown-base-name"

	policy := createTestPolicy()

	generater.applyPolicyToCert(policy)

	assert.Contains(t, generater.BKECertSpec.Config.Usages, x509.ExtKeyUsageClientAuth)
}

// TestApplyHostsToAltNames_WithMasterNodes tests applying hosts with master nodes
func TestApplyHostsToAltNames_WithMasterNodes(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()
	generater.bkeConfig = createTestBKEConfig(t, ipGen, two)

	// Get expected IPs from the created BKEConfig
	expectedIPs := make([]string, 0, len(generater.bkeConfig.Nodes))
	for _, node := range generater.bkeConfig.Nodes {
		expectedIPs = append(expectedIPs, node.IP)
	}

	err := generater.applyHostsToAltNames([]string{})
	require.NoError(t, err)

	ips := generater.BKECertSpec.Config.AltNames.IPs
	ipStrings := make([]string, 0, len(ips))
	for _, ip := range ips {
		ipStrings = append(ipStrings, ip.String())
	}
	for _, expectedIP := range expectedIPs {
		assert.Contains(t, ipStrings, expectedIP)
	}
}

// TestApplyHostsToAltNames_WithCSRHosts tests applying hosts from CSR
func TestApplyHostsToAltNames_WithCSRHosts(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()

	csrHostIP := ipGen.nextIP()
	csrHostDNS := "test.example.com"
	hosts := []string{csrHostIP, csrHostDNS}

	err := generater.applyHostsToAltNames(hosts)
	require.NoError(t, err)

	ips := generater.BKECertSpec.Config.AltNames.IPs
	ipStrings := make([]string, 0, len(ips))
	for _, ip := range ips {
		ipStrings = append(ipStrings, ip.String())
	}
	assert.Contains(t, ipStrings, csrHostIP)
	assert.Contains(t, generater.BKECertSpec.Config.AltNames.DNSNames, csrHostDNS)
}

// TestApplyHostsToAltNames_WithF5IP tests applying hosts with F5 load balancer IP
func TestApplyHostsToAltNames_WithF5IP(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()
	f5IP := ipGen.nextIP()
	generater.bkeConfig = createTestBKEConfigWithF5IP(t, ipGen, f5IP)

	err := generater.applyHostsToAltNames([]string{})
	require.NoError(t, err)

	ips := generater.BKECertSpec.Config.AltNames.IPs
	ipStrings := make([]string, 0, len(ips))
	for _, ip := range ips {
		ipStrings = append(ipStrings, ip.String())
	}
	assert.Contains(t, ipStrings, f5IP)
}

// TestAddF5IPToAltNames_WithValidIP tests adding F5 IP when valid
func TestAddF5IPToAltNames_WithValidIP(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()
	f5IP := ipGen.nextIP()
	generater.bkeConfig = createTestBKEConfigWithF5IP(t, ipGen, f5IP)

	altNames := &certutil.AltNames{
		IPs:      []net.IP{},
		DNSNames: []string{},
	}
	generater.addF5IPToAltNames(altNames)

	assert.Len(t, altNames.IPs, 1)
	assert.Equal(t, f5IP, altNames.IPs[0].String())
}

// TestAddF5IPToAltNames_WithNilAltNames tests adding F5 IP with nil altNames
func TestAddF5IPToAltNames_WithNilAltNames(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.bkeConfig = createTestBKEConfigWithF5IP(t, ipGen, ipGen.nextIP())

	generater.addF5IPToAltNames(nil)
	// Should not panic
}

// TestAddF5IPToAltNames_WithDuplicateIP tests adding F5 IP when already exists
func TestAddF5IPToAltNames_WithDuplicateIP(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	f5IP := ipGen.nextIP()
	generater.bkeConfig = createTestBKEConfigWithF5IP(t, ipGen, f5IP)

	existingIP := net.ParseIP(f5IP)
	altNames := &certutil.AltNames{
		IPs:      []net.IP{existingIP},
		DNSNames: []string{},
	}
	generater.addF5IPToAltNames(altNames)

	assert.Len(t, altNames.IPs, 1)
}

// TestLoadCSRFromFile_WithMissingFile tests loading CSR from missing file
func TestLoadCSRFromFile_WithMissingFile(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)

	result, err := generater.loadCSRFromFile("non-existent.json")
	require.NoError(t, err)
	assert.Nil(t, result)
}

// TestApplyCertConfig_WithNilBKECertSpec tests applying config with nil BKECertSpec
func TestApplyCertConfig_WithNilBKECertSpec(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = nil

	err := generater.applyCertConfig()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "BKECertSpec is nil")
}

// TestApplyCertConfig_WithUnknownBaseName tests applying config with unknown base name
func TestApplyCertConfig_WithUnknownBaseName(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()
	generater.BKECertSpec.BaseName = "unknown-base-name"

	err := generater.applyCertConfig()
	require.NoError(t, err)
}

// TestParsePublicKeyAlgorithm tests parsing public key algorithm
func TestParsePublicKeyAlgorithm(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected x509.PublicKeyAlgorithm
	}{
		{"RSA", "rsa", x509.RSA},
		{"RSA uppercase", "RSA", x509.RSA},
		{"ECDSA", "ecdsa", x509.ECDSA},
		{"EC", "ec", x509.ECDSA},
		{"Ed25519", "ed25519", x509.Ed25519},
		{"Unknown", "unknown", x509.RSA},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parsePublicKeyAlgorithm(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestParseKeyUsages tests parsing key usages
func TestParseKeyUsages(t *testing.T) {
	usages := []string{"digital signature", "key encipherment", "cert sign"}
	result := parseKeyUsages(usages)

	assert.Contains(t, result, x509.KeyUsageDigitalSignature)
	assert.Contains(t, result, x509.KeyUsageKeyEncipherment)
	assert.Contains(t, result, x509.KeyUsageCertSign)
}

// TestParseExtKeyUsages tests parsing extended key usages
func TestParseExtKeyUsages(t *testing.T) {
	usages := []string{"client auth", "server auth", "any"}
	result := parseExtKeyUsages(usages)

	assert.Contains(t, result, x509.ExtKeyUsageClientAuth)
	assert.Contains(t, result, x509.ExtKeyUsageServerAuth)
	assert.Contains(t, result, x509.ExtKeyUsageAny)
}

// TestParseDuration tests parsing duration string
func TestParseDuration(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected time.Duration
	}{
		{"1 hour", "1h", time.Hour},
		{"1 day", "24h", 24 * time.Hour},
		{"1 year", "8760h", testCertValidity},
		{"Invalid", "invalid", defaultCertValidity},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := parseDuration(tt.input)
			if tt.input == "invalid" {
				assert.Equal(t, defaultCertValidity, result)
			} else {
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

// TestGenerateClientCertAndKey_AllKubeConfigTypes tests all kubeconfig types
func TestGenerateClientCertAndKey_AllKubeConfigTypes(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	caCert, caKey := createTestCACertAndKey(t, pkiPath)

	kubeConfigs := GetKubeConfigs()
	for _, kubeConfigSpec := range kubeConfigs {
		t.Run(kubeConfigSpec.BaseName, func(t *testing.T) {
			generater := createTestKubeConfigGenerater(t, pkiPath, kubeConfigSpec.Name, hostIP)
			generater.BKECertSpec = kubeConfigSpec
			generater.BKECertSpec.PkiPath = pkiPath

			result, err := generater.GenerateClientCertAndKey(caCert, caKey)
			require.NoError(t, err)
			require.NotNil(t, result)
			require.NotNil(t, result.Cert)
			require.NotNil(t, result.Key)

			assert.Contains(t, result.Cert.ExtKeyUsage, x509.ExtKeyUsageClientAuth)
		})
	}
}

// TestApplyHostsToAltNames_WithInvalidBKEConfig tests with invalid BKEConfig
func TestApplyHostsToAltNames_WithInvalidBKEConfig(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()
	generater.bkeConfig = &bkev1beta1.BKEConfig{
		Nodes: []bkev1beta1.Node{
			{IP: "invalid-ip", Role: []string{"master"}},
		},
	}

	err := generater.applyHostsToAltNames([]string{})
	// Should handle gracefully or return error
	if err != nil {
		assert.Contains(t, err.Error(), "failed to get master node alt names")
	}
}

// TestApplyProfileConfig tests applying profile configuration
func TestApplyProfileConfig(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()

	usages := []string{"digital signature", "key encipherment", "client auth"}
	expiry := "8760h"
	isCA := false

	generater.applyProfileConfig(usages, expiry, isCA)

	assert.Contains(t, generater.BKECertSpec.Config.Usages, x509.ExtKeyUsageClientAuth)
	assert.Equal(t, testCertValidity, generater.BKECertSpec.Config.Validity)
	assert.False(t, generater.BKECertSpec.IsCA)
}

// TestApplyProfileConfig_WithCA tests applying profile config with CA constraint
func TestApplyProfileConfig_WithCA(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()

	usages := []string{"cert sign", "crl sign"}
	expiry := "87600h"
	isCA := true

	generater.applyProfileConfig(usages, expiry, isCA)

	assert.True(t, generater.BKECertSpec.IsCA)
}

// TestApplyHostsToAltNames_WithEmptyHosts tests with empty hosts list
func TestApplyHostsToAltNames_WithEmptyHosts(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()
	generater.bkeConfig = createTestBKEConfig(t, ipGen, 1)

	err := generater.applyHostsToAltNames([]string{})
	require.NoError(t, err)
	// Should still add master node IPs
	assert.NotEmpty(t, generater.BKECertSpec.Config.AltNames.IPs)
}

// TestApplyHostsToAltNames_WithMixedHosts tests with mixed IP and DNS hosts
func TestApplyHostsToAltNames_WithMixedHosts(t *testing.T) {
	pkiPath := createTestTempDir(t)
	defer os.RemoveAll(pkiPath)

	ipGen := newTestIPGenerator("127.0.0.1")
	hostIP := ipGen.nextIP()
	generater := createTestKubeConfigGenerater(t, pkiPath, "kubeconfig", hostIP)
	generater.BKECertSpec = BKEAdminKubeConfig()

	hostIP1 := ipGen.nextIP()
	hostIP2 := ipGen.nextIP()
	hostDNS1 := "example.com"
	hostDNS2 := "test.example.com"
	hosts := []string{hostIP1, hostDNS1, hostIP2, hostDNS2}

	err := generater.applyHostsToAltNames(hosts)
	require.NoError(t, err)

	ips := generater.BKECertSpec.Config.AltNames.IPs
	ipStrings := make([]string, 0, len(ips))
	for _, ip := range ips {
		ipStrings = append(ipStrings, ip.String())
	}
	assert.Contains(t, ipStrings, hostIP1)
	assert.Contains(t, ipStrings, hostIP2)
	assert.Contains(t, generater.BKECertSpec.Config.AltNames.DNSNames, hostDNS1)
	assert.Contains(t, generater.BKECertSpec.Config.AltNames.DNSNames, hostDNS2)
}
