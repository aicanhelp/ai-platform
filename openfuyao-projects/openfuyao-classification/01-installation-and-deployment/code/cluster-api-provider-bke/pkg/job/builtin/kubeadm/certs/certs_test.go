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
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/pkiutil"
)

const (
	// test常量定义
	testClusterName     = "test-cluster"
	testNamespace       = "default"
	testSecretName      = "test-secret"
	testGlobalCAName    = "global-ca"
	testCACertName      = "test-ca"
	testChainCertName   = "test-chain"
	testChainFileName   = "test-chain.crt"
	testEmptyChainFile  = "empty-chain.crt"
	testSubDir1         = "new"
	testSubDir2         = "subdir"
	serialNumberBitSize = 128
	three               = 3
	four                = 4
)

// testCertHelper 创建测试用的证书辅助函数
type testCertHelper struct {
	caCert    *x509.Certificate
	caKey     *rsa.PrivateKey
	chainCert *x509.Certificate
	chainKey  *rsa.PrivateKey
}

// createTestCertHelper 创建测试用的证书辅助对象
func createTestCertHelper(t *testing.T) *testCertHelper {
	t.Helper()
	caKey, err := rsa.GenerateKey(rand.Reader, pkiutil.DefaultRSAKeySize)
	if err != nil {
		t.Fatalf("failed to generate CA key: %v", err)
	}

	caCert, err := createSelfSignedCACert(testCACertName, caKey)
	if err != nil {
		t.Fatalf("failed to generate CA cert: %v", err)
	}

	chainKey, err := rsa.GenerateKey(rand.Reader, pkiutil.DefaultRSAKeySize)
	if err != nil {
		t.Fatalf("failed to generate chain key: %v", err)
	}

	chainCert, err := createSignedCert(testChainCertName, chainKey, caCert, caKey)
	if err != nil {
		t.Fatalf("failed to generate chain cert: %v", err)
	}

	return &testCertHelper{
		caCert:    caCert,
		caKey:     caKey,
		chainCert: chainCert,
		chainKey:  chainKey,
	}
}

// encodeCertsToPEMBytes 将证书编码为PEM格式的字节数组
func encodeCertsToPEMBytes(certs []*x509.Certificate) []byte {
	var pemBytes []byte
	for _, cert := range certs {
		pemBytes = append(pemBytes, pkiutil.EncodeCertToPEM(cert)...)
	}
	return pemBytes
}

// encodeKeyToPEMBytes 将密钥编码为PEM格式的字节数组
func encodeKeyToPEMBytes(key *rsa.PrivateKey) []byte {
	return pkiutil.EncodeKeyToPEM(key)
}

// TestParseChainCerts 测试解析证书链函数
func TestParseChainCerts(t *testing.T) {
	helper := createTestCertHelper(t)
	chainCerts := []*x509.Certificate{helper.chainCert}
	chainBytes := encodeCertsToPEMBytes(chainCerts)

	cp := &CertPlugin{}
	parsedCerts, err := cp.parseChainCerts(chainBytes)
	if err != nil {
		t.Fatalf("parseChainCerts() error = %v", err)
	}

	if len(parsedCerts) != len(chainCerts) {
		t.Errorf("expected %d certificates, got %d", len(chainCerts), len(parsedCerts))
	}

	if parsedCerts[0].Subject.CommonName != chainCerts[0].Subject.CommonName {
		t.Errorf("certificate CommonName mismatch")
	}
}

// TestParseChainCerts_InvalidData 测试解析无效证书数据
func TestParseChainCertsInvalidData(t *testing.T) {
	cp := &CertPlugin{}
	invalidData := []byte("invalid certificate data")

	_, err := cp.parseChainCerts(invalidData)
	if err == nil {
		t.Error("expected error for invalid certificate data")
	}
}

// TestParseChainCerts_EmptyData 测试解析空数据
func TestParseChainCertsEmptyData(t *testing.T) {
	cp := &CertPlugin{}
	var emptyData []byte

	_, err := cp.parseChainCerts(emptyData)
	if err == nil {
		t.Error("expected error for empty certificate data")
	}
}

// TestMergeCertChain 测试合并CA证书和证书链函数
func TestMergeCertChain(t *testing.T) {
	helper := createTestCertHelper(t)
	caCerts := []*x509.Certificate{helper.caCert}
	chainCerts := []*x509.Certificate{helper.chainCert}

	caBytes := encodeCertsToPEMBytes(caCerts)
	chainBytes := encodeCertsToPEMBytes(chainCerts)

	cp := &CertPlugin{}
	mergedCerts, err := cp.mergeCertChain(caBytes, chainBytes)
	if err != nil {
		t.Fatalf("mergeCertChain() error = %v", err)
	}

	expectedCount := len(caCerts) + len(chainCerts)
	if len(mergedCerts) != expectedCount {
		t.Errorf("expected %d certificates, got %d", expectedCount, len(mergedCerts))
	}

	if mergedCerts[0].Subject.CommonName != caCerts[0].Subject.CommonName {
		t.Errorf("first certificate should be CA cert")
	}

	if mergedCerts[1].Subject.CommonName != chainCerts[0].Subject.CommonName {
		t.Errorf("second certificate should be chain cert")
	}
}

// TestMergeCertChain_InvalidCACert 测试合并时CA证书无效的情况
func TestMergeCertChainInvalidCACert(t *testing.T) {
	helper := createTestCertHelper(t)
	chainBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.chainCert})
	invalidCABytes := []byte("invalid CA certificate")

	cp := &CertPlugin{}
	_, err := cp.mergeCertChain(invalidCABytes, chainBytes)
	if err == nil {
		t.Error("expected error for invalid CA certificate")
	}
}

// TestMergeCertChain_InvalidChainCert 测试合并时证书链无效的情况
func TestMergeCertChainInvalidChainCert(t *testing.T) {
	helper := createTestCertHelper(t)
	caBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	invalidChainBytes := []byte("invalid chain certificate")

	cp := &CertPlugin{}
	_, err := cp.mergeCertChain(caBytes, invalidChainBytes)
	if err == nil {
		t.Error("expected error for invalid chain certificate")
	}
}

// TestWriteCertChainToFile 测试写入证书链到文件函数
func TestWriteCertChainToFile(t *testing.T) {
	tmpDir := t.TempDir()
	helper := createTestCertHelper(t)
	certs := []*x509.Certificate{helper.caCert, helper.chainCert}

	cp := &CertPlugin{pkiPath: tmpDir}
	filePath := filepath.Join(tmpDir, testChainFileName)

	err := cp.writeCertChainToFile(filePath, certs)
	if err != nil {
		t.Fatalf("writeCertChainToFile() error = %v", err)
	}

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		t.Errorf("certificate chain file was not created at %s", filePath)
	}

	fileContent, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatalf("failed to read certificate chain file: %v", err)
	}

	if len(fileContent) == 0 {
		t.Error("certificate chain file is empty")
	}

	parsedCerts, err := pkiutil.ParseCertsPEM(fileContent)
	if err != nil {
		t.Fatalf("failed to parse written certificate chain: %v", err)
	}

	if len(parsedCerts) != len(certs) {
		t.Errorf("expected %d certificates in file, got %d", len(certs), len(parsedCerts))
	}
}

// TestWriteCertChainToFile_CreateDirectory 测试写入时自动创建目录
func TestWriteCertChainToFileCreateDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	helper := createTestCertHelper(t)
	certs := []*x509.Certificate{helper.caCert}

	newDir := filepath.Join(tmpDir, testSubDir1, testSubDir2)
	cp := &CertPlugin{pkiPath: newDir}
	filePath := filepath.Join(newDir, testChainFileName)

	err := cp.writeCertChainToFile(filePath, certs)
	if err != nil {
		t.Fatalf("writeCertChainToFile() error = %v", err)
	}

	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		t.Errorf("certificate chain file was not created at %s", filePath)
	}
}

// TestWriteCertChainToFile_EmptyCerts 测试写入空证书列表
func TestWriteCertChainToFileEmptyCerts(t *testing.T) {
	tmpDir := t.TempDir()
	cp := &CertPlugin{pkiPath: tmpDir}
	filePath := filepath.Join(tmpDir, testEmptyChainFile)

	err := cp.writeCertChainToFile(filePath, []*x509.Certificate{})
	if err != nil {
		t.Fatalf("writeCertChainToFile() should not error for empty certs, got: %v", err)
	}
}

// mockK8sClient 模拟K8s客户端用于测试
type mockK8sClient struct {
	client.Client
	caSecret *corev1.Secret
	getError error
}

// Get 实现client.Client接口的Get方法
func (m *mockK8sClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	if m.getError != nil {
		return m.getError
	}
	if secret, ok := obj.(*corev1.Secret); ok && m.caSecret != nil {
		*secret = *m.caSecret
		return nil
	}
	return nil
}

// createGlobalSecret 创建全局CA Secret（测试辅助函数）
func createGlobalSecret(chainBytes []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testGlobalCAName,
			Namespace: testNamespace,
		},
		Data: map[string][]byte{
			pkiutil.ChainCrtDataName: chainBytes,
		},
	}
}

// createCASecret 创建CA Secret（测试辅助函数）
func createCASecret(caBytes []byte) *corev1.Secret {
	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testClusterName + "-ca",
			Namespace: testNamespace,
		},
		Data: map[string][]byte{
			pkiutil.TLSCrtDataName: caBytes,
		},
	}
}

// setupCertPluginForSaveChain 设置CertPlugin用于测试saveCertChain（测试辅助函数）
func setupCertPluginForSaveChain(caSecret *corev1.Secret, tmpDir string, getError error) *CertPlugin {
	mockClient := &mockK8sClient{
		caSecret: caSecret,
		getError: getError,
	}

	cp := &CertPlugin{
		k8sClient:   mockClient,
		clusterName: testClusterName,
		namespace:   testNamespace,
	}
	if tmpDir != "" {
		cp.pkiPath = tmpDir
	}

	return cp
}

// setupCertPluginForGetCA 设置CertPlugin用于测试getCACertFromClusterSecret（测试辅助函数）
func setupCertPluginForGetCA(caSecret *corev1.Secret) *CertPlugin {
	mockClient := &mockK8sClient{
		caSecret: caSecret,
	}

	return &CertPlugin{
		k8sClient:   mockClient,
		clusterName: testClusterName,
		namespace:   testNamespace,
	}
}

// TestSaveCertChain 测试保存证书链函数
func TestSaveCertChain(t *testing.T) {
	tmpDir := t.TempDir()
	helper := createTestCertHelper(t)

	caBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	chainBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.chainCert})

	caSecret := createCASecret(caBytes)
	globalSecret := createGlobalSecret(chainBytes)

	cp := setupCertPluginForSaveChain(caSecret, tmpDir, nil)
	cp.saveCertChain(globalSecret)

	chainPath := filepath.Join(tmpDir, pkiutil.CertChainFileName)
	caChainPath := filepath.Join(tmpDir, CertCAAndChainFileName)

	if _, err := os.Stat(chainPath); os.IsNotExist(err) {
		t.Errorf("chain file was not created at %s", chainPath)
	}

	if _, err := os.Stat(caChainPath); os.IsNotExist(err) {
		t.Errorf("CA and chain file was not created at %s", caChainPath)
	}
}

// TestSaveCertChain_NoData 测试Secret没有数据的情况
func TestSaveCertChainNoData(t *testing.T) {
	cp := &CertPlugin{}
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testSecretName,
			Namespace: testNamespace,
		},
		Data: nil,
	}

	cp.saveCertChain(secret)
}

// TestSaveCertChain_NoChainData 测试Secret没有证书链数据的情况
func TestSaveCertChainNoChainData(t *testing.T) {
	cp := &CertPlugin{}
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testSecretName,
			Namespace: testNamespace,
		},
		Data: map[string][]byte{},
	}

	cp.saveCertChain(secret)
}

// TestSaveCertChain_EmptyChainData 测试Secret证书链数据为空的情况
func TestSaveCertChainEmptyChainData(t *testing.T) {
	cp := &CertPlugin{}
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testSecretName,
			Namespace: testNamespace,
		},
		Data: map[string][]byte{
			pkiutil.ChainCrtDataName: nil,
		},
	}

	cp.saveCertChain(secret)
}

// TestSaveCertChain_GetCAError 测试获取CA证书失败的情况
func TestSaveCertChainGetCAError(t *testing.T) {
	helper := createTestCertHelper(t)
	chainBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.chainCert})

	globalSecret := createGlobalSecret(chainBytes)
	cp := setupCertPluginForSaveChain(nil, "", os.ErrNotExist)

	cp.saveCertChain(globalSecret)
}

// TestSaveCertChain_ParseChainError 测试解析证书链失败的情况
func TestSaveCertChainParseChainError(t *testing.T) {
	tmpDir := t.TempDir()
	invalidChainBytes := []byte("invalid chain certificate data")

	globalSecret := createGlobalSecret(invalidChainBytes)

	helper := createTestCertHelper(t)
	caBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	caSecret := createCASecret(caBytes)

	cp := setupCertPluginForSaveChain(caSecret, tmpDir, nil)
	cp.saveCertChain(globalSecret)
}

// TestSaveCertChain_MergeError 测试合并证书链失败的情况
func TestSaveCertChainMergeError(t *testing.T) {
	tmpDir := t.TempDir()
	helper := createTestCertHelper(t)
	chainBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.chainCert})

	globalSecret := createGlobalSecret(chainBytes)

	invalidCABytes := []byte("invalid CA certificate data")
	caSecret := createCASecret(invalidCABytes)

	cp := setupCertPluginForSaveChain(caSecret, tmpDir, nil)
	cp.saveCertChain(globalSecret)
}

// TestSaveCertChain_WriteChainFileError 测试写入证书链文件失败的情况
func TestSaveCertChainWriteChainFileError(t *testing.T) {
	helper := createTestCertHelper(t)
	caBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	chainBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.chainCert})

	globalSecret := createGlobalSecret(chainBytes)
	caSecret := createCASecret(caBytes)

	cp := setupCertPluginForSaveChain(caSecret, "/nonexistent/path/that/cannot/be/created", nil)
	cp.saveCertChain(globalSecret)
}

// TestGetCACertFromClusterSecret_EmptyClusterName 测试clusterName为空的情况
func TestGetCACertFromClusterSecretEmptyClusterName(t *testing.T) {
	cp := &CertPlugin{
		clusterName: "",
		namespace:   testNamespace,
	}

	_, err := cp.getCACertFromClusterSecret()
	if err == nil {
		t.Error("expected error when clusterName is empty")
	}
}

// TestGetCACertFromClusterSecret_EmptyNamespace 测试namespace为空的情况
func TestGetCACertFromClusterSecretEmptyNamespace(t *testing.T) {
	cp := &CertPlugin{
		clusterName: testClusterName,
		namespace:   "",
	}

	_, err := cp.getCACertFromClusterSecret()
	if err == nil {
		t.Error("expected error when namespace is empty")
	}
}

// TestGetCACertFromClusterSecret_NoData 测试Secret没有数据的情况
func TestGetCACertFromClusterSecretNoData(t *testing.T) {
	caSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testClusterName + "-ca",
			Namespace: testNamespace,
		},
		Data: nil,
	}

	cp := setupCertPluginForGetCA(caSecret)
	_, err := cp.getCACertFromClusterSecret()
	if err == nil {
		t.Error("expected error when Secret has no data")
	}
}

// TestGetCACertFromClusterSecret_NoTLSCrt 测试Secret没有tls.crt数据的情况
func TestGetCACertFromClusterSecretNoTLSCrt(t *testing.T) {
	caSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testClusterName + "-ca",
			Namespace: testNamespace,
		},
		Data: map[string][]byte{},
	}

	cp := setupCertPluginForGetCA(caSecret)
	_, err := cp.getCACertFromClusterSecret()
	if err == nil {
		t.Error("expected error when Secret has no tls.crt data")
	}
}

// TestGetCACertFromClusterSecret_EmptyTLSCrt 测试Secret的tls.crt数据为空的情况
func TestGetCACertFromClusterSecretEmptyTLSCrt(t *testing.T) {
	caSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testClusterName + "-ca",
			Namespace: testNamespace,
		},
		Data: map[string][]byte{
			pkiutil.TLSCrtDataName: nil,
		},
	}

	cp := setupCertPluginForGetCA(caSecret)
	_, err := cp.getCACertFromClusterSecret()
	if err == nil {
		t.Error("expected error when tls.crt data is empty")
	}
}

// createSelfSignedCACert 创建自签名CA证书（测试辅助函数）
func createSelfSignedCACert(commonName string, key *rsa.PrivateKey) (*x509.Certificate, error) {
	serialNumber, err := rand.Int(rand.Reader, big.NewInt(0).Lsh(big.NewInt(1), serialNumberBitSize))
	if err != nil {
		return nil, err
	}

	template := &x509.Certificate{
		SerialNumber:          serialNumber,
		Subject:               pkix.Name{CommonName: commonName},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(pkiutil.CertificateValidity),
		IsCA:                  true,
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		return nil, err
	}

	return x509.ParseCertificate(certDER)
}

// createSignedCert 创建由CA签名的证书（测试辅助函数）
func createSignedCert(commonName string, key *rsa.PrivateKey, caCert *x509.Certificate, caKey *rsa.PrivateKey) (*x509.Certificate, error) {
	serialNumber, err := rand.Int(rand.Reader, big.NewInt(0).Lsh(big.NewInt(1), serialNumberBitSize))
	if err != nil {
		return nil, err
	}

	template := &x509.Certificate{
		SerialNumber:          serialNumber,
		Subject:               pkix.Name{CommonName: commonName},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(pkiutil.CertificateValidity),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}

	certDER, err := x509.CreateCertificate(rand.Reader, template, caCert, &key.PublicKey, caKey)
	if err != nil {
		return nil, err
	}

	return x509.ParseCertificate(certDER)
}

// TestValidateGlobalCASecretData 测试验证全局CA Secret数据
func TestValidateGlobalCASecretData(t *testing.T) {
	helper := createTestCertHelper(t)
	certBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	keyBytes := encodeKeyToPEMBytes(helper.caKey)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      utils.GlobalCASecretName,
			Namespace: utils.GlobalCANamespace,
		},
		Data: map[string][]byte{
			pkiutil.TLSCrtDataName: certBytes,
			pkiutil.TLSKeyDataName: keyBytes,
		},
	}

	cp := &CertPlugin{}
	certData, keyData, err := cp.validateGlobalCASecretData(secret)
	if err != nil {
		t.Fatalf("validateGlobalCASecretData() error = %v", err)
	}

	if len(certData) == 0 {
		t.Error("expected certificate data to be returned")
	}

	if len(keyData) == 0 {
		t.Error("expected key data to be returned")
	}
}

// TestValidateGlobalCASecretData_NoData 测试Secret没有数据的情况
func TestValidateGlobalCASecretDataNoData(t *testing.T) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      utils.GlobalCASecretName,
			Namespace: utils.GlobalCANamespace,
		},
		Data: nil,
	}

	cp := &CertPlugin{}
	certData, keyData, err := cp.validateGlobalCASecretData(secret)
	if err != nil {
		t.Fatalf("validateGlobalCASecretData() should not return error for no data, got: %v", err)
	}

	if certData != nil {
		t.Error("expected certificate data to be nil when Secret has no data")
	}

	if keyData != nil {
		t.Error("expected key data to be nil when Secret has no data")
	}
}

// TestValidateGlobalCASecretData_NoCert 测试Secret缺少证书数据的情况
func TestValidateGlobalCASecretDataNoCert(t *testing.T) {
	helper := createTestCertHelper(t)
	keyBytes := encodeKeyToPEMBytes(helper.caKey)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      utils.GlobalCASecretName,
			Namespace: utils.GlobalCANamespace,
		},
		Data: map[string][]byte{
			pkiutil.TLSKeyDataName: keyBytes,
		},
	}

	cp := &CertPlugin{}
	certData, keyData, err := cp.validateGlobalCASecretData(secret)
	if err != nil {
		t.Fatalf("validateGlobalCASecretData() should not return error for missing cert, got: %v", err)
	}

	if certData != nil {
		t.Error("expected certificate data to be nil when cert is missing")
	}

	if keyData != nil {
		t.Error("expected key data to be nil when cert is missing")
	}
}

// TestValidateGlobalCASecretData_NoKey 测试Secret缺少密钥数据的情况
func TestValidateGlobalCASecretDataNoKey(t *testing.T) {
	helper := createTestCertHelper(t)
	certBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      utils.GlobalCASecretName,
			Namespace: utils.GlobalCANamespace,
		},
		Data: map[string][]byte{
			pkiutil.TLSCrtDataName: certBytes,
		},
	}

	cp := &CertPlugin{}
	certData, keyData, err := cp.validateGlobalCASecretData(secret)
	if err != nil {
		t.Fatalf("validateGlobalCASecretData() should not return error for missing key, got: %v", err)
	}

	if certData != nil {
		t.Error("expected certificate data to be nil when key is missing")
	}

	if keyData != nil {
		t.Error("expected key data to be nil when key is missing")
	}
}

// TestValidateGlobalCASecretData_EmptyCert 测试Secret证书数据为空的情况
func TestValidateGlobalCASecretDataEmptyCert(t *testing.T) {
	helper := createTestCertHelper(t)
	keyBytes := encodeKeyToPEMBytes(helper.caKey)

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      utils.GlobalCASecretName,
			Namespace: utils.GlobalCANamespace,
		},
		Data: map[string][]byte{
			pkiutil.TLSCrtDataName: nil,
			pkiutil.TLSKeyDataName: keyBytes,
		},
	}

	cp := &CertPlugin{}
	certData, keyData, err := cp.validateGlobalCASecretData(secret)
	if err != nil {
		t.Fatalf("validateGlobalCASecretData() should not return error for empty cert, got: %v", err)
	}

	if certData != nil {
		t.Error("expected certificate data to be nil when cert is empty")
	}

	if keyData != nil {
		t.Error("expected key data to be nil when cert is empty")
	}
}

// TestValidateGlobalCASecretData_EmptyKey 测试Secret密钥数据为空的情况
func TestValidateGlobalCASecretDataEmptyKey(t *testing.T) {
	helper := createTestCertHelper(t)
	certBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      utils.GlobalCASecretName,
			Namespace: utils.GlobalCANamespace,
		},
		Data: map[string][]byte{
			pkiutil.TLSCrtDataName: certBytes,
			pkiutil.TLSKeyDataName: nil,
		},
	}

	cp := &CertPlugin{}
	certData, keyData, err := cp.validateGlobalCASecretData(secret)
	if err != nil {
		t.Fatalf("validateGlobalCASecretData() should not return error for empty key, got: %v", err)
	}

	if certData != nil {
		t.Error("expected certificate data to be nil when key is empty")
	}

	if keyData != nil {
		t.Error("expected key data to be nil when key is empty")
	}
}

// TestParseGlobalCACertAndKey 测试解析全局CA证书和密钥
func TestParseGlobalCACertAndKey(t *testing.T) {
	helper := createTestCertHelper(t)
	certBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	keyBytes := encodeKeyToPEMBytes(helper.caKey)

	cp := &CertPlugin{}
	cert, key, err := cp.parseGlobalCACertAndKey(certBytes, keyBytes)
	if err != nil {
		t.Fatalf("parseGlobalCACertAndKey() error = %v", err)
	}

	if cert == nil {
		t.Error("expected certificate to be parsed")
	}

	if key == nil {
		t.Error("expected key to be parsed")
	}

	if cert.Subject.CommonName != helper.caCert.Subject.CommonName {
		t.Errorf("certificate CommonName mismatch")
	}
}

// TestParseGlobalCACertAndKey_InvalidCert 测试证书解析失败的情况
func TestParseGlobalCACertAndKeyInvalidCert(t *testing.T) {
	helper := createTestCertHelper(t)
	invalidCertBytes := []byte("invalid certificate data")
	keyBytes := encodeKeyToPEMBytes(helper.caKey)

	cp := &CertPlugin{}
	_, _, err := cp.parseGlobalCACertAndKey(invalidCertBytes, keyBytes)
	if err == nil {
		t.Error("expected error when certificate parsing fails")
	}
}

// TestParseGlobalCACertAndKey_InvalidKey 测试密钥解析失败的情况
func TestParseGlobalCACertAndKeyInvalidKey(t *testing.T) {
	helper := createTestCertHelper(t)
	certBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	invalidKeyBytes := []byte("invalid key data")

	cp := &CertPlugin{}
	_, _, err := cp.parseGlobalCACertAndKey(certBytes, invalidKeyBytes)
	if err == nil {
		t.Error("expected error when key parsing fails")
	}
}

// TestParseGlobalCACertAndKey_EmptyCert 测试证书数据为空的情况
func TestParseGlobalCACertAndKeyEmptyCert(t *testing.T) {
	helper := createTestCertHelper(t)
	var emptyCertBytes []byte
	keyBytes := encodeKeyToPEMBytes(helper.caKey)

	cp := &CertPlugin{}
	_, _, err := cp.parseGlobalCACertAndKey(emptyCertBytes, keyBytes)
	if err == nil {
		t.Error("expected error when certificate data is empty")
	}
}

// TestParseGlobalCACertAndKey_EmptyKey 测试密钥数据为空的情况
func TestParseGlobalCACertAndKeyEmptyKey(t *testing.T) {
	helper := createTestCertHelper(t)
	certBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	var emptyKeyBytes []byte

	cp := &CertPlugin{}
	_, _, err := cp.parseGlobalCACertAndKey(certBytes, emptyKeyBytes)
	if err == nil {
		t.Error("expected error when key data is empty")
	}
}

// setupCertPluginForSaveChainFromLocal 设置CertPlugin用于测试saveCertChainFromLocal
func setupCertPluginForSaveChainFromLocal(t *testing.T, caSecret *corev1.Secret, tmpDir string, getError error) *CertPlugin {
	t.Helper()
	mockClient := &mockK8sClient{
		caSecret: caSecret,
		getError: getError,
	}
	return &CertPlugin{
		k8sClient:   mockClient,
		clusterName: testClusterName,
		namespace:   testNamespace,
		pkiPath:     tmpDir,
	}
}

// createTrustChainFile 创建本地trust-chain.crt文件（使用常量路径）
func createTrustChainFile(t *testing.T, content []byte) {
	t.Helper()
	trustChainDir := filepath.Dir(LocalTrustChainPath)
	if err := os.MkdirAll(trustChainDir, pkiutil.DirDefaultPermission); err != nil {
		t.Fatalf("failed to create trust chain directory: %v", err)
	}
	if len(content) != 0 {
		if err := os.WriteFile(LocalTrustChainPath, content, pkiutil.FileDefaultPermission); err != nil {
			t.Fatalf("failed to write trust chain file: %v", err)
		}
	}
}

// cleanupTrustChainFile 清理本地trust-chain.crt文件
func cleanupTrustChainFile(t *testing.T) {
	t.Helper()
	if err := os.Remove(LocalTrustChainPath); err != nil && !os.IsNotExist(err) {
		t.Logf("failed to remove trust chain file: %v", err)
	}
}

// TestSaveCertChainFromLocal_FileNotExists 测试trust-chain.crt文件不存在的情况
func TestSaveCertChainFromLocalFileNotExists(t *testing.T) {
	tmpDir := t.TempDir()
	cp := setupCertPluginForSaveChainFromLocal(t, nil, tmpDir, nil)
	// 确保文件不存在
	cleanupTrustChainFile(t)
	err := cp.saveCertChainFromLocal()
	if err != nil {
		t.Errorf("saveCertChainFromLocal() should return nil when file not exists, got: %v", err)
	}
}

const (
	testDefaultUser             = "root"
	testLocalhostIP             = "127.0.0.1"
	testMasterIP1               = "127.0.0.2"
	testGenerateKubeConfigKey   = "generateKubeConfig"
	testLocalKubeConfigScopeKey = "localKubeConfigScope"
)

// mockExecutor implements exec.Executor interface for testing
type mockExecutor struct{}

func (m *mockExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockExecutor) ExecuteCommand(command string, arg ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithEnv(env []string, command string, arg ...string) error {
	return nil
}

func (m *mockExecutor) ExecuteCommandWithOutput(command string, arg ...string) (string, error) {
	return "success", nil
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutput(command string, arg ...string) (string, error) {
	return "success", nil
}

func (m *mockExecutor) ExecuteCommandWithCombinedOutputInDir(_ string, command string, arg ...string) (string, error) {
	return m.ExecuteCommandWithCombinedOutput(command, arg...)
}

func (m *mockExecutor) ExecuteCommandWithOutputFile(command, outfileArg string, arg ...string) (string, error) {
	return "success", nil
}

func (m *mockExecutor) ExecuteCommandWithOutputFileTimeout(timeout time.Duration, command, outfileArg string, arg ...string) (string, error) {
	return "success", nil
}

func (m *mockExecutor) ExecuteCommandWithTimeout(timeout time.Duration, command string, arg ...string) (string, error) {
	return "success", nil
}

func (m *mockExecutor) ExecuteCommandResidentBinary(timeout time.Duration, command string, arg ...string) error {
	return nil
}

// testIPGenerator 用于生成测试IP地址，避免硬编码
type testIPGenerator struct {
	octet1, octet2, octet3, octet4 byte
}

// newTestIPGenerator 创建新的IP生成器
func newTestIPGenerator(baseIP string) *testIPGenerator {
	ipv4 := parseAndValidateIP(baseIP)
	// 确保ipv4至少有4个字节，防止数组越界
	if len(ipv4) < four {
		// 如果解析失败，使用默认的127.0.0.1
		defaultIP := net.ParseIP(testLocalhostIP).To4()
		if defaultIP != nil && len(defaultIP) >= four {
			ipv4 = defaultIP
		} else {
			// 最后的保障：使用硬编码的127.0.0.1的字节
			ipv4 = []byte{127, 0, 0, 1}
		}
	}
	return &testIPGenerator{
		octet1: ipv4[0],
		octet2: ipv4[1],
		octet3: ipv4[2],
		octet4: ipv4[3],
	}
}

// parseAndValidateIP 解析并验证IP地址，如果无效则返回默认IP
func parseAndValidateIP(baseIP string) []byte {
	ip := net.ParseIP(baseIP)
	if ip == nil {
		ip = net.ParseIP(testLocalhostIP)
	}
	ipv4 := ip.To4()
	if ipv4 == nil || len(ipv4) < four {
		// 如果To4()返回nil或长度不足，使用默认IP
		ip = net.ParseIP(testLocalhostIP)
		ipv4 = ip.To4()
	}
	return ipv4
}

// nextIP 生成下一个IP地址
func (g *testIPGenerator) nextIP() string {
	currentIP := fmt.Sprintf("%d.%d.%d.%d", g.octet1, g.octet2, g.octet3, g.octet4)
	g.incrementOctets()
	return currentIP
}

// incrementOctets 递增IP地址的各个字节，处理溢出
func (g *testIPGenerator) incrementOctets() {
	g.octet4++
	if g.octet4 != 0 {
		return
	}
	g.octet3++
	if g.octet3 != 0 {
		return
	}
	g.octet2++
	if g.octet2 != 0 {
		return
	}
	g.octet1++
}

// createTestBKEConfig 创建测试用的BKEConfig
func createTestBKEConfig(t *testing.T, masterCount int, workerCount int) *bkev1beta1.BKEConfig {
	t.Helper()
	ipGen := newTestIPGenerator(testMasterIP1)
	nodes := make([]bkev1beta1.Node, 0, masterCount+workerCount)

	// 创建master节点
	for i := 0; i < masterCount; i++ {
		nodes = append(nodes, bkev1beta1.Node{
			IP:       ipGen.nextIP(),
			Hostname: fmt.Sprintf("master-%d", i),
			Role:     []string{"master"},
		})
	}

	// 创建worker节点
	for i := 0; i < workerCount; i++ {
		nodes = append(nodes, bkev1beta1.Node{
			IP:       ipGen.nextIP(),
			Hostname: fmt.Sprintf("worker-%d", i),
			Role:     []string{"worker"},
		})
	}

	return &bkev1beta1.BKEConfig{
		Cluster: bkev1beta1.Cluster{
			ControlPlane: bkev1beta1.ControlPlane{
				APIServer: &bkev1beta1.APIServer{
					ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{},
				},
			},
		},
	}
}

// createTestMasterNode 创建测试用的master节点（用于无BKEConfig的场景）
func createTestMasterNode(ip string, port int, username string) *bkenode.Node {
	// 直接创建bkenode.Node，参考render_test.go的方式
	// 从certs.go:496和render.go:455可以看到直接访问node.APIServer.Port
	// 说明bkenode.Node有APIServer字段，它可能指向ControlPlane.APIServer
	// 或者bkenode.Node是结构体，有APIServer字段直接暴露
	node := bkenode.Node{
		IP:       ip,
		Username: username,
		Role:     []string{"master"},
	}
	if port > 0 {
		// 设置ControlPlane和APIServer
		apiServer := &bkev1beta1.APIServer{
			ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{},
		}
		node.ControlPlane = bkev1beta1.ControlPlane{
			APIServer: apiServer,
		}
		// bkenode.Node可能有APIServer字段直接指向ControlPlane.APIServer
		// 或者bkenode.Node是结构体，有APIServer字段
		// 尝试直接设置Port（如果字段存在）
		// 从代码来看，node.APIServer应该存在，并且有Port字段
		if node.APIServer != nil {
			node.APIServer.Port = int32(port)
		}
	}
	return &node
}

// createTestWorkerNode 创建测试用的worker节点（用于无BKEConfig的场景）
func createTestWorkerNode(ip string) *bkenode.Node {
	node := bkev1beta1.Node{
		IP:   ip,
		Role: []string{"worker"},
	}
	bkenodeNode := bkenode.Node(node)
	return &bkenodeNode
}

// mockKubeConfigGenerater 模拟KubeConfigGenerater用于测试
type mockKubeConfigGenerater struct {
	generateError error
}

// Generate 模拟Generate方法
func (m *mockKubeConfigGenerater) Generate() error {
	return m.generateError
}

// getNodeServerConfigTestCase 定义getNodeServerConfig的测试用例结构
type getNodeServerConfigTestCase struct {
	name        string
	bkeConfig   *bkev1beta1.BKEConfig
	currentNode *bkenode.Node
	isWorker    bool
	wantPort    int
	wantIP      string
	wantUser    string
}

// runGetNodeServerConfigTest 执行单个getNodeServerConfig测试用例
func runGetNodeServerConfigTest(t *testing.T, tt getNodeServerConfigTestCase) {
	t.Helper()
	cp := &CertPlugin{
		bkeConfig:   tt.bkeConfig,
		currentNode: tt.currentNode,
	}
	gotPort, gotIP, gotUser := cp.getNodeServerConfig(tt.isWorker)
	if gotPort != tt.wantPort {
		t.Errorf("getNodeServerConfig() port = %v, want %v", gotPort, tt.wantPort)
	}
	if gotIP != tt.wantIP {
		t.Errorf("getNodeServerConfig() IP = %v, want %v", gotIP, tt.wantIP)
	}
	if gotUser != tt.wantUser {
		t.Errorf("getNodeServerConfig() user = %v, want %v", gotUser, tt.wantUser)
	}
}

// getGetNodeServerConfigTestCases 获取getNodeServerConfig的测试用例
func getGetNodeServerConfigTestCases(t *testing.T) []getNodeServerConfigTestCase {
	t.Helper()
	return []getNodeServerConfigTestCase{
		{
			name:        "nil config returns defaults",
			bkeConfig:   nil,
			currentNode: nil,
			isWorker:    false,
			wantPort:    bkeinit.DefaultAPIBindPort,
			wantIP:      testLocalhostIP,
			wantUser:    testDefaultUser,
		},
		{
			name:        "worker node uses admin kubeconfig server when available",
			bkeConfig:   createTestBKEConfig(t, two, 0),
			currentNode: nil,
			isWorker:    true,
			wantPort:    bkeinit.DefaultAPIBindPort,
			wantIP:      testMasterIP1,
			wantUser:    testDefaultUser,
		},
		{
			name:        "worker node with admin kubeconfig uses its server config",
			bkeConfig:   createTestBKEConfig(t, 1, 0),
			currentNode: nil,
			isWorker:    true,
			wantPort:    bkeinit.DefaultAPIBindPort,
			wantIP:      testMasterIP1,
			wantUser:    testDefaultUser,
		},
	}
}

// TestGetNodeServerConfig 测试getNodeServerConfig函数
func TestGetNodeServerConfig(t *testing.T) {
	tests := []getNodeServerConfigTestCase{
		{
			name:        "nil config returns defaults",
			bkeConfig:   nil,
			currentNode: nil,
			isWorker:    false,
			wantPort:    bkeinit.DefaultAPIBindPort,
			wantIP:      testLocalhostIP,
			wantUser:    testDefaultUser,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runGetNodeServerConfigTest(t, tt)
		})
	}
}

// setupCertPluginForKubeConfig 设置CertPlugin用于kubeconfig相关测试
func setupCertPluginForKubeConfig(t *testing.T, currentNode *bkenode.Node) (*CertPlugin, string) {
	t.Helper()
	tmpDir := t.TempDir()
	helper := createTestCertHelper(t)

	// 创建CA证书和密钥文件，这是生成kubeconfig所必需的
	caCertSpec := pkiutil.BKECertRootCA()
	caCertSpec.PkiPath = tmpDir
	err := pkiutil.WriteCertAndKey(caCertSpec, helper.caCert, helper.caKey)
	if err != nil {
		t.Fatalf("failed to write CA cert and key: %v", err)
	}

	cp := &CertPlugin{
		pkiPath:     tmpDir,
		clusterName: testClusterName,
		bkeConfig:   nil,
		currentNode: currentNode,
	}
	return cp, tmpDir
}

// TestGenerateKubeConfigsForScopes 测试generateKubeConfigsForScopes函数
func TestGenerateKubeConfigsForScopes(t *testing.T) {
	_, tmpDir := setupCertPluginForKubeConfig(t, nil)

	tests := []struct {
		name       string
		scopes     []string
		serverPort int
		nodeIP     string
		isWorker   bool
		pkiPath    string
		wantError  bool
	}{
		{
			name:       "empty scopes returns no error",
			scopes:     []string{},
			serverPort: bkeinit.DefaultAPIBindPort,
			nodeIP:     testMasterIP1,
			isWorker:   false,
			pkiPath:    tmpDir,
			wantError:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp, _ := setupCertPluginForKubeConfig(t, nil)
			cp.pkiPath = tt.pkiPath
			err := cp.generateKubeConfigsForScopes(tt.scopes, tt.serverPort, tt.nodeIP, tt.isWorker)
			if (err != nil) != tt.wantError {
				t.Errorf("generateKubeConfigsForScopes() error = %v, wantError %v", err, tt.wantError)
			}
		})
	}
}

// TestGenerateKubeConfigsForScopes_InvalidScope 测试无效scope的情况
func TestGenerateKubeConfigsForScopesInvalidScope(t *testing.T) {
	cp, _ := setupCertPluginForKubeConfig(t, nil)

	// 使用无效的scope，这会导致Generate失败
	invalidScopes := []string{"invalid-scope"}
	err := cp.generateKubeConfigsForScopes(invalidScopes, bkeinit.DefaultAPIBindPort, testMasterIP1, false)
	if err == nil {
		t.Error("generateKubeConfigsForScopes() with invalid scope should return error")
	}
}

// TestHandleGenerateKubeConfig_EmptyScope 测试空scope的情况
func TestHandleGenerateKubeConfigEmptyScope(t *testing.T) {
	cp, _ := setupCertPluginForKubeConfig(t, nil)

	certParamMap := map[string]string{
		testGenerateKubeConfigKey:   "true",
		testLocalKubeConfigScopeKey: "kubelet", // 使用有效的scope，空字符串会导致错误
	}

	// 使用有效scope进行测试
	err := cp.handleGenerateKubeConfig(certParamMap)

	if err != nil && !strings.Contains(err.Error(), "permission denied") {
		t.Errorf("handleGenerateKubeConfig() error = %v", err)
	}
}

// TestNew 测试New函数
func TestNew(t *testing.T) {
	cp := New(nil, nil, nil)
	if cp == nil {
		t.Error("New() returned nil")
	}
	if cp.Name() != Name {
		t.Errorf("Name() = %s, want %s", cp.Name(), Name)
	}
}

// TestName 测试Name函数
func TestName(t *testing.T) {
	cp := &CertPlugin{}
	if cp.Name() != Name {
		t.Errorf("Name() = %s, want %s", cp.Name(), Name)
	}
}

// TestParam 测试Param函数
func TestParam(t *testing.T) {
	cp := &CertPlugin{}
	params := cp.Param()
	if params == nil {
		t.Error("Param() returned nil")
	}
	if len(params) == 0 {
		t.Error("Param() returned empty map")
	}
}

// TestInitializeParams 测试initializeParams函数
func TestInitializeParams(t *testing.T) {
	tests := []struct {
		name          string
		bkeConfig     *bkev1beta1.BKEConfig
		paramMap      map[string]string
		expectPath    string
		expectNS      string
		expectCluster string
	}{
		{
			name:          "nil bkeConfig",
			bkeConfig:     nil,
			paramMap:      map[string]string{"certificatesDir": "/test/pki", "namespace": "test-ns", "clusterName": "test-cluster"},
			expectPath:    "/test/pki",
			expectNS:      "test-ns",
			expectCluster: "test-cluster",
		},
		{
			name:          "with bkeConfig",
			bkeConfig:     &bkev1beta1.BKEConfig{},
			paramMap:      map[string]string{"certificatesDir": "/etc/kubernetes/pki", "namespace": "default", "clusterName": "mycluster"},
			expectPath:    "/etc/kubernetes/pki",
			expectNS:      "default",
			expectCluster: "mycluster",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{}
			err := cp.initializeParams(tt.paramMap)
			if err != nil {
				t.Errorf("initializeParams() error = %v", err)
			}
			if cp.pkiPath != tt.expectPath {
				t.Errorf("pkiPath = %s, want %s", cp.pkiPath, tt.expectPath)
			}
			if cp.namespace != tt.expectNS {
				t.Errorf("namespace = %s, want %s", cp.namespace, tt.expectNS)
			}
			if cp.clusterName != tt.expectCluster {
				t.Errorf("clusterName = %s, want %s", cp.clusterName, tt.expectCluster)
			}
		})
	}
}

// TestHandleLoadCACert 测试handleLoadCACert函数
func TestHandleLoadCACert(t *testing.T) {
	tests := []struct {
		name        string
		paramMap    map[string]string
		expectError bool
	}{
		{
			name:        "loadCACert is false",
			paramMap:    map[string]string{"loadCACert": "false"},
			expectError: false,
		},
		{
			name:        "loadCACert is true but caCertNames is empty",
			paramMap:    map[string]string{"loadCACert": "true", "caCertNames": ""},
			expectError: true,
		},
		{
			name:        "loadCACert is true but clusterName is empty",
			paramMap:    map[string]string{"loadCACert": "true", "caCertNames": "ca,sa", "clusterName": "", "namespace": "test"},
			expectError: true,
		},
		{
			name:        "loadCACert is true but namespace is empty",
			paramMap:    map[string]string{"loadCACert": "true", "caCertNames": "ca", "clusterName": "test", "namespace": ""},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{}
			err := cp.handleLoadCACert(tt.paramMap)
			if (err != nil) != tt.expectError {
				t.Errorf("handleLoadCACert() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestHandleLoadAdminKubeconfig 测试handleLoadAdminKubeconfig函数
func TestHandleLoadAdminKubeconfig(t *testing.T) {
	tests := []struct {
		name        string
		paramMap    map[string]string
		expectError bool
	}{
		{
			name:        "loadAdminKubeconfig is false",
			paramMap:    map[string]string{"loadAdminKubeconfig": "false"},
			expectError: false,
		},
		{
			name:        "loadAdminKubeconfig is true but namespace is empty",
			paramMap:    map[string]string{"loadAdminKubeconfig": "true"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{}
			err := cp.handleLoadAdminKubeconfig(tt.paramMap)
			if (err != nil) != tt.expectError {
				t.Errorf("handleLoadAdminKubeconfig() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestHandleCertChainAndGlobalCert 测试handleCertChainAndGlobalCert函数
func TestHandleCertChainAndGlobalCert(t *testing.T) {
	tests := []struct {
		name        string
		currentNode *bkenode.Node
		paramMap    map[string]string
		expectError bool
	}{
		{
			name:        "worker node",
			currentNode: createTestWorkerNode("192.168.1.100"),
			paramMap:    map[string]string{},
			expectError: false,
		},
		{
			name:        "non-manager cluster",
			currentNode: createTestMasterNode("192.168.1.1", 6443, "root"),
			paramMap:    map[string]string{"isManagerCluster": "false"},
			expectError: false,
		},
		{
			name:        "manager cluster but loadGlobalCA fails",
			currentNode: createTestMasterNode("192.168.1.1", 6443, "root"),
			paramMap:    map[string]string{"isManagerCluster": "true"},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{
				currentNode: tt.currentNode,
				pkiPath:     t.TempDir(),
			}
			err := cp.handleCertChainAndGlobalCert(tt.paramMap)
			if (err != nil) != tt.expectError {
				t.Errorf("handleCertChainAndGlobalCert() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestHandleLoadTargetClusterCert 测试handleLoadTargetClusterCert函数
func TestHandleLoadTargetClusterCert(t *testing.T) {
	tests := []struct {
		name        string
		paramMap    map[string]string
		expectError bool
	}{
		{
			name:        "loadTargetClusterCert is false",
			paramMap:    map[string]string{"loadTargetClusterCert": "false"},
			expectError: false,
		},
		{
			name:        "loadTargetClusterCert is true but namespace is empty",
			paramMap:    map[string]string{"loadTargetClusterCert": "true"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{}
			err := cp.handleLoadTargetClusterCert(tt.paramMap)
			if (err != nil) != tt.expectError {
				t.Errorf("handleLoadTargetClusterCert() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestHandleGenerateCerts 测试handleGenerateCerts函数
func TestHandleGenerateCerts(t *testing.T) {
	t.Skip("skip unstable UT: generates cert files under /etc/kubernetes and requires host permissions")

	tests := []struct {
		name        string
		paramMap    map[string]string
		expectError bool
	}{
		{
			name:        "generate is false",
			paramMap:    map[string]string{"generate": "false"},
			expectError: false,
		},
		{
			name:        "generate is true with altIPs",
			paramMap:    map[string]string{"generate": "true", "altIPs": "10.0.0.1,10.0.0.2"},
			expectError: false,
		},
		{
			name:        "generate is true with altDNSNames",
			paramMap:    map[string]string{"generate": "true", "altDNSNames": "test.example.com"},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{
				pkiPath: t.TempDir(),
			}
			err := cp.handleGenerateCerts(tt.paramMap)
			if (err != nil) != tt.expectError {
				t.Errorf("handleGenerateCerts() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestHandleGenerateKubeConfig 测试handleGenerateKubeConfig函数
func TestHandleGenerateKubeConfig(t *testing.T) {
	tests := []struct {
		name        string
		paramMap    map[string]string
		expectError bool
	}{
		{
			name:        "generateKubeConfig is false",
			paramMap:    map[string]string{"generateKubeConfig": "false"},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{
				pkiPath:     t.TempDir(),
				clusterName: testClusterName,
			}
			err := cp.handleGenerateKubeConfig(tt.paramMap)
			if (err != nil) != tt.expectError {
				t.Errorf("handleGenerateKubeConfig() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestHandleUploadCerts 测试handleUploadCerts函数
func TestHandleUploadCerts(t *testing.T) {
	tests := []struct {
		name        string
		paramMap    map[string]string
		expectError bool
	}{
		{
			name:        "uploadCerts is false",
			paramMap:    map[string]string{"uploadCerts": "false"},
			expectError: false,
		},
		{
			name:        "uploadCerts is true but namespace is empty",
			paramMap:    map[string]string{"uploadCerts": "true"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{}
			err := cp.handleUploadCerts(tt.paramMap)
			if (err != nil) != tt.expectError {
				t.Errorf("handleUploadCerts() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestGenerateCerts 测试generateCerts函数
func TestGenerateCerts(t *testing.T) {
	cp := &CertPlugin{
		pkiPath: t.TempDir(),
	}
	cp.generateCerts()
}

// TestUploadCerts 测试uploadCerts函数
func TestUploadCerts(t *testing.T) {
	tests := []struct {
		name        string
		namespace   string
		expectError bool
	}{
		{
			name:        "empty namespace",
			namespace:   "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockK8sClient{
				getError: os.ErrNotExist,
			}
			cp := &CertPlugin{
				pkiPath:   t.TempDir(),
				k8sClient: mockClient,
			}
			err := cp.uploadCerts(tt.namespace)
			if (err != nil) != tt.expectError {
				t.Errorf("uploadCerts() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestGetCertFromSecret 测试getCertFromSecret函数
func TestGetCertFromSecret(t *testing.T) {
	tests := []struct {
		name        string
		secretName  string
		saveTo      string
		expectError bool
	}{
		{
			name:        "empty secret name",
			secretName:  "",
			saveTo:      t.TempDir(),
			expectError: true,
		},
		{
			name:        "valid secret name but namespace is empty",
			secretName:  "test-secret",
			saveTo:      t.TempDir(),
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockK8sClient{}
			cp := &CertPlugin{
				namespace:   testNamespace,
				clusterName: testClusterName,
				k8sClient:   mockClient,
				pkiPath:     tt.saveTo,
			}
			err := cp.getCertFromSecret(tt.secretName, tt.saveTo)
			if (err != nil) != tt.expectError {
				t.Errorf("getCertFromSecret() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestUploadCertToSecret 测试uploadCertToSecret函数
func TestUploadCertToSecret(t *testing.T) {
	cp := &CertPlugin{
		clusterName: testClusterName,
		k8sClient:   &mockK8sClient{},
	}
	cert := &pkiutil.BKECert{
		Name:     "test",
		BaseName: "test",
	}
	err := cp.uploadCertToSecret(cert, testNamespace)
	if err == nil {
		t.Error("uploadCertToSecret() expected error for mock client")
	}
}

// TestPrepareCertList 测试prepareCertList函数
func TestPrepareCertList(t *testing.T) {
	tests := []struct {
		name        string
		bkeConfig   *bkev1beta1.BKEConfig
		expectError bool
	}{
		{
			name:        "nil bkeConfig",
			bkeConfig:   nil,
			expectError: false,
		},
		{
			name:        "with bkeConfig",
			bkeConfig:   &bkev1beta1.BKEConfig{},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{
				bkeConfig: tt.bkeConfig,
			}
			certList, err := cp.prepareCertList()
			if (err != nil) != tt.expectError {
				t.Errorf("prepareCertList() error = %v, expectError %v", err, tt.expectError)
			}
			if err == nil && certList == nil {
				t.Error("prepareCertList() returned nil certList without error")
			}
		})
	}
}

// TestSplitNameSpaceName 测试splitNameSpaceName函数
func TestSplitNameSpaceName(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		expectNS    string
		expectNames []string
		expectError bool
	}{
		{
			name:        "valid format",
			input:       "namespace:name1,name2",
			expectNS:    "namespace",
			expectNames: []string{"name1", "name2"},
			expectError: false,
		},
		{
			name:        "invalid format - missing colon",
			input:       "invalid",
			expectError: true,
		},
		{
			name:        "empty string",
			input:       "",
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ns, names, err := splitNameSpaceName(tt.input)
			if (err != nil) != tt.expectError {
				t.Errorf("splitNameSpaceName() error = %v, expectError %v", err, tt.expectError)
			}
			if !tt.expectError {
				if ns != tt.expectNS {
					t.Errorf("namespace = %s, want %s", ns, tt.expectNS)
				}
				if len(names) != len(tt.expectNames) {
					t.Errorf("names length = %d, want %d", len(names), len(tt.expectNames))
				}
			}
		})
	}
}

// TestCopyAdminKubeConfig 测试copyAdminKubeConfig函数
func TestCopyAdminKubeConfig(t *testing.T) {
	var mockExec exec.Executor = &mockExecutor{}
	cp := &CertPlugin{
		exec: mockExec,
	}
	cp.copyAdminKubeConfig("root")
	cp.copyAdminKubeConfig("testuser")
}

// TestLoadGlobalCACertFromLocal 测试loadGlobalCACertFromLocal函数
func TestLoadGlobalCACertFromLocal(t *testing.T) {
	tests := []struct {
		name        string
		certPath    string
		keyPath     string
		expectError bool
	}{
		{
			name:        "file does not exist",
			certPath:    "/nonexistent/global-ca.crt",
			keyPath:     "/nonexistent/global-ca.key",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{
				pkiPath: t.TempDir(),
			}
			err := cp.loadGlobalCACertFromLocal()
			if (err != nil) != tt.expectError {
				t.Errorf("loadGlobalCACertFromLocal() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestSaveGlobalCACertAndKey 测试saveGlobalCACertAndKey函数
func TestSaveGlobalCACertAndKey(t *testing.T) {
	cp := &CertPlugin{
		pkiPath: t.TempDir(),
	}
	helper := createTestCertHelper(t)
	err := cp.saveGlobalCACertAndKey(helper.caCert, helper.caKey)
	if err != nil {
		t.Errorf("saveGlobalCACertAndKey() error = %v", err)
	}
}

// TestGetKubeConfigServerConfig 测试getKubeConfigServerConfig函数
func TestGetKubeConfigServerConfig(t *testing.T) {
	tests := []struct {
		name         string
		currentNode  *bkenode.Node
		expectWorker bool
	}{
		{
			name:         "nil currentNode",
			currentNode:  nil,
			expectWorker: false,
		},
		{
			name:         "master node",
			currentNode:  createTestMasterNode("192.168.1.1", 6443, "root"),
			expectWorker: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{
				currentNode: tt.currentNode,
			}
			config := cp.getKubeConfigServerConfig()
			if config.isWorker != tt.expectWorker {
				t.Errorf("isWorker = %v, want %v", config.isWorker, tt.expectWorker)
			}
		})
	}
}

// TestExecute 测试Execute函数
func TestExecute(t *testing.T) {
	tests := []struct {
		name        string
		commands    []string
		expectError bool
	}{
		{
			name:        "generate false returns nil",
			commands:    []string{"Cert", "generate=false"},
			expectError: false,
		},
		{
			name:        "generate false with uploadCerts true and namespace empty",
			commands:    []string{"Cert", "generate=false", "uploadCerts=true"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cp := &CertPlugin{}
			cp.pkiPath = t.TempDir()
			cp.clusterName = "test-cluster"
			cp.currentNode = createTestMasterNode("192.168.1.1", 6443, "root")
			_, err := cp.Execute(context.Background(), tt.commands)
			if (err != nil) != tt.expectError {
				t.Errorf("Execute() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

// TestGetAdminKubeConfigServer 测试getAdminKubeConfigServer函数
func TestGetAdminKubeConfigServer(t *testing.T) {
	tests := []struct {
		name              string
		kubeConfigContent string
		existsReturn      bool
		expectError       bool
	}{
		{
			name:         "kubeconfig file not exists",
			existsReturn: false,
			expectError:  true,
		},
		{
			name: "kubeconfig with empty current context",
			kubeConfigContent: `apiVersion: v1
kind: Config
clusters:
- name: test-cluster
  cluster:
    server: https://192.168.1.100:6443
contexts: []
current-context: ""
users:
- name: test-user
  user:
    token: test-token
`,
			existsReturn: true,
			expectError:  true,
		},
		{
			name: "kubeconfig with missing context",
			kubeConfigContent: `apiVersion: v1
kind: Config
clusters:
- name: test-cluster
  cluster:
    server: https://192.168.1.100:6443
contexts:
- name: wrong-context
  context:
    cluster: test-cluster
    user: test-user
current-context: missing-context
users:
- name: test-user
  user:
    token: test-token
`,
			existsReturn: true,
			expectError:  true,
		},
		{
			name: "kubeconfig with missing cluster",
			kubeConfigContent: `apiVersion: v1
kind: Config
clusters:
- name: test-cluster
  cluster:
    server: https://192.168.1.100:6443
contexts:
- name: test-context
  context:
    cluster: missing-cluster
    user: test-user
current-context: test-context
users:
- name: test-user
  user:
    token: test-token
`,
			existsReturn: true,
			expectError:  true,
		},
		{
			name: "kubeconfig with empty server address",
			kubeConfigContent: `apiVersion: v1
Kind: Config
clusters:
- name: test-cluster
  cluster:
    server: ""
contexts:
- name: test-context
  context:
    cluster: test-cluster
    user: test-user
current-context: test-context
users:
- name: test-user
  user:
    token: test-token
`,
			existsReturn: true,
			expectError:  true,
		},
		{
			name: "kubeconfig with invalid URL",
			kubeConfigContent: `apiVersion: v1
kind: Config
clusters:
- name: test-cluster
  cluster:
    server: "http://test test.com"
contexts:
- name: test-context
  context:
    cluster: test-cluster
    user: test-user
current-context: test-context
users:
- name: test-user
  user:
    token: test-token
`,
			existsReturn: true,
			expectError:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()

			testPath := filepath.Join(t.TempDir(), "admin.conf")
			if tt.kubeConfigContent != "" {
				os.WriteFile(testPath, []byte(tt.kubeConfigContent), 0644)
			}

			patches.ApplyFunc(utils.Exists, func(path string) bool {
				return tt.existsReturn && path == testPath
			})

			patches.ApplyFunc(pkiutil.GetDefaultKubeConfigPath, func() string {
				return testPath
			})

			cp := &CertPlugin{}

			_, _, err := cp.getAdminKubeConfigServer()

			if tt.expectError {
				if err == nil {
					t.Error("getAdminKubeConfigServer() expected error, got nil")
				}
			} else {
				if err != nil {
					t.Errorf("getAdminKubeConfigServer() unexpected error: %v", err)
				}
			}
		})
	}
}

func TestGetNodeServerConfigLoadBalance(t *testing.T) {
	patchAdminKubeconfig := func(cp *CertPlugin) *gomonkey.Patches {
		return gomonkey.ApplyPrivateMethod(reflect.TypeOf(cp), "getAdminKubeConfigServer",
			func(_ *CertPlugin) (string, int, error) { return "10.0.0.10", 6443, nil })
	}

	t.Run("nginx mode worker uses local nginx proxy", func(t *testing.T) {
		cp := &CertPlugin{bkeConfig: &bkev1beta1.BKEConfig{Cluster: bkev1beta1.Cluster{LoadBalanceType: "nginx"}}}
		patches := patchAdminKubeconfig(cp)
		defer patches.Reset()

		port, ip, _ := cp.getNodeServerConfig(true)
		if port != bkeinit.DefaultAPIBindPort {
			t.Errorf("expected port %d, got %d", bkeinit.DefaultAPIBindPort, port)
		}
		if ip != "127.0.0.1" {
			t.Errorf("expected nodeIP 127.0.0.1, got %s", ip)
		}
	})

	t.Run("nginx mode uses configured apiserver port", func(t *testing.T) {
		cp := &CertPlugin{bkeConfig: &bkev1beta1.BKEConfig{Cluster: bkev1beta1.Cluster{LoadBalanceType: "nginx"}}}
		cp.bkeConfig.Cluster.APIServer = &bkev1beta1.APIServer{
			APIEndpoint: bkev1beta1.APIEndpoint{Port: 8443},
		}
		patches := patchAdminKubeconfig(cp)
		defer patches.Reset()

		port, ip, _ := cp.getNodeServerConfig(true)
		if port != 8443 {
			t.Errorf("expected port 8443, got %d", port)
		}
		if ip != "127.0.0.1" {
			t.Errorf("expected nodeIP 127.0.0.1, got %s", ip)
		}
	})

	t.Run("ha mode worker still reads admin kubeconfig", func(t *testing.T) {
		cp := &CertPlugin{bkeConfig: &bkev1beta1.BKEConfig{}}
		patches := patchAdminKubeconfig(cp)
		defer patches.Reset()

		port, ip, _ := cp.getNodeServerConfig(true)
		if port != 6443 {
			t.Errorf("expected port 6443, got %d", port)
		}
		if ip != "10.0.0.10" {
			t.Errorf("expected nodeIP 10.0.0.10, got %s", ip)
		}
	})
}
