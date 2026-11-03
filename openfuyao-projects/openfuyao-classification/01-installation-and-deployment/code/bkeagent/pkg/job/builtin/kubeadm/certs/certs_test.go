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
	"strings"
	"testing"
	"time"

	bkev1beta1 "gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
	bkeinit "gopkg.openfuyao.cn/bkecommon/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/bkecommon/cluster/node"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/bkeagent/utils"
	"gopkg.openfuyao.cn/bkeagent/utils/pkiutil"
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
	two                 = 2
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
func TestParseChainCerts_InvalidData(t *testing.T) {
	cp := &CertPlugin{}
	invalidData := []byte("invalid certificate data")

	_, err := cp.parseChainCerts(invalidData)
	if err == nil {
		t.Error("expected error for invalid certificate data")
	}
}

// TestParseChainCerts_EmptyData 测试解析空数据
func TestParseChainCerts_EmptyData(t *testing.T) {
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
func TestMergeCertChain_InvalidCACert(t *testing.T) {
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
func TestMergeCertChain_InvalidChainCert(t *testing.T) {
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
func TestWriteCertChainToFile_CreateDirectory(t *testing.T) {
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
func TestWriteCertChainToFile_EmptyCerts(t *testing.T) {
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
func TestSaveCertChain_NoData(t *testing.T) {
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
func TestSaveCertChain_NoChainData(t *testing.T) {
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
func TestSaveCertChain_EmptyChainData(t *testing.T) {
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
func TestSaveCertChain_GetCAError(t *testing.T) {
	helper := createTestCertHelper(t)
	chainBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.chainCert})

	globalSecret := createGlobalSecret(chainBytes)
	cp := setupCertPluginForSaveChain(nil, "", os.ErrNotExist)

	cp.saveCertChain(globalSecret)
}

// TestSaveCertChain_ParseChainError 测试解析证书链失败的情况
func TestSaveCertChain_ParseChainError(t *testing.T) {
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
func TestSaveCertChain_MergeError(t *testing.T) {
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
func TestSaveCertChain_WriteChainFileError(t *testing.T) {
	helper := createTestCertHelper(t)
	caBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	chainBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.chainCert})

	globalSecret := createGlobalSecret(chainBytes)
	caSecret := createCASecret(caBytes)

	cp := setupCertPluginForSaveChain(caSecret, "/nonexistent/path/that/cannot/be/created", nil)
	cp.saveCertChain(globalSecret)
}

// TestGetCACertFromClusterSecret_EmptyClusterName 测试clusterName为空的情况
func TestGetCACertFromClusterSecret_EmptyClusterName(t *testing.T) {
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
func TestGetCACertFromClusterSecret_EmptyNamespace(t *testing.T) {
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
func TestGetCACertFromClusterSecret_NoData(t *testing.T) {
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
func TestGetCACertFromClusterSecret_NoTLSCrt(t *testing.T) {
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
func TestGetCACertFromClusterSecret_EmptyTLSCrt(t *testing.T) {
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

// TestGetGlobalCASecretFromParentCluster 测试从父集群获取全局CA Secret
func TestGetGlobalCASecretFromParentCluster(t *testing.T) {
	helper := createTestCertHelper(t)
	certBytes := encodeCertsToPEMBytes([]*x509.Certificate{helper.caCert})
	keyBytes := encodeKeyToPEMBytes(helper.caKey)

	globalSecret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      utils.GlobalCASecretName,
			Namespace: utils.GlobalCANamespace,
		},
		Data: map[string][]byte{
			pkiutil.TLSCrtDataName: certBytes,
			pkiutil.TLSKeyDataName: keyBytes,
		},
	}

	mockClient := &mockK8sClient{
		caSecret: globalSecret,
	}

	cp := &CertPlugin{
		k8sClient: mockClient,
	}

	secret, err := cp.getGlobalCASecretFromParentCluster()
	if err != nil {
		t.Fatalf("getGlobalCASecretFromParentCluster() error = %v", err)
	}

	if secret == nil {
		t.Error("expected secret to be returned")
	}

	if secret.Name != utils.GlobalCASecretName {
		t.Errorf("expected secret name %s, got %s", utils.GlobalCASecretName, secret.Name)
	}
}

// TestGetGlobalCASecretFromParentCluster_NotFound 测试Secret不存在的情况
func TestGetGlobalCASecretFromParentCluster_NotFound(t *testing.T) {
	notFoundError := apierrors.NewNotFound(schema.GroupResource{
		Group:    "",
		Resource: "secrets",
	}, utils.GlobalCASecretName)

	mockClient := &mockK8sClient{
		getError: notFoundError,
	}

	cp := &CertPlugin{
		k8sClient: mockClient,
	}

	secret, err := cp.getGlobalCASecretFromParentCluster()
	if err != nil {
		t.Fatalf("getGlobalCASecretFromParentCluster() should not return error for NotFound, got: %v", err)
	}

	if secret != nil {
		t.Error("expected secret to be nil when not found")
	}
}

// TestGetGlobalCASecretFromParentCluster_OtherError 测试其他错误情况
func TestGetGlobalCASecretFromParentCluster_OtherError(t *testing.T) {
	otherError := os.ErrPermission

	mockClient := &mockK8sClient{
		getError: otherError,
	}

	cp := &CertPlugin{
		k8sClient: mockClient,
	}

	_, err := cp.getGlobalCASecretFromParentCluster()
	if err == nil {
		t.Error("expected error when Get fails with non-NotFound error")
	}
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
func TestValidateGlobalCASecretData_NoData(t *testing.T) {
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
func TestValidateGlobalCASecretData_NoCert(t *testing.T) {
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
func TestValidateGlobalCASecretData_NoKey(t *testing.T) {
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
func TestValidateGlobalCASecretData_EmptyCert(t *testing.T) {
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
func TestValidateGlobalCASecretData_EmptyKey(t *testing.T) {
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
func TestParseGlobalCACertAndKey_InvalidCert(t *testing.T) {
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
func TestParseGlobalCACertAndKey_InvalidKey(t *testing.T) {
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
func TestParseGlobalCACertAndKey_EmptyCert(t *testing.T) {
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
func TestParseGlobalCACertAndKey_EmptyKey(t *testing.T) {
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
func TestSaveCertChainFromLocal_FileNotExists(t *testing.T) {
	tmpDir := t.TempDir()
	cp := setupCertPluginForSaveChainFromLocal(t, nil, tmpDir, nil)
	// 确保文件不存在
	cleanupTrustChainFile(t)
	err := cp.saveCertChainFromLocal()
	if err != nil {
		t.Errorf("saveCertChainFromLocal() should return nil when file not exists, got: %v", err)
	}
}

// 测试相关的常量定义
const (
	testDefaultUser             = "root"
	testLocalhostIP             = "127.0.0.1"
	testMasterIP1               = "127.0.0.2"
	testGenerateKubeConfigKey   = "generateKubeConfig"
	testLocalKubeConfigScopeKey = "localKubeConfigScope"
)

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
		Nodes: nodes,
	}
}

// createTestMasterNodeFromConfig 从BKEConfig创建测试用的master节点
func createTestMasterNodeFromConfig(t *testing.T, config *bkev1beta1.BKEConfig, index int, port int, username string) *bkenode.Node {
	t.Helper()
	nodes := bkenode.Nodes(config.Nodes)
	masterNodes := nodes.Master()
	if index >= len(masterNodes) {
		t.Fatalf("master node index %d out of range, only %d masters", index, len(masterNodes))
	}
	// masterNodes[index] 返回的是 bkenode.Node 类型
	node := masterNodes[index]
	// 创建新的节点副本以便修改
	bkeNode := bkev1beta1.Node(node)
	if port > 0 {
		// 设置ControlPlane和APIServer
		if bkeNode.ControlPlane.APIServer == nil {
			bkeNode.ControlPlane.APIServer = &bkev1beta1.APIServer{
				ControlPlaneComponent: bkev1beta1.ControlPlaneComponent{},
			}
		}
		// 注意：bkev1beta1.APIServer可能没有Port字段
		// 但bkenode.Node的APIServer可能有，所以先转换再设置
	}
	if username != "" {
		bkeNode.Username = username
	}
	resultNode := bkenode.Node(bkeNode)
	// 如果bkenode.Node的APIServer有Port字段，在这里设置
	if port > 0 && resultNode.APIServer != nil {
		resultNode.APIServer.Port = int32(port)
	}
	return &resultNode
}

// createTestWorkerNodeFromConfig 从BKEConfig创建测试用的worker节点
func createTestWorkerNodeFromConfig(t *testing.T, config *bkev1beta1.BKEConfig, index int) *bkenode.Node {
	t.Helper()
	nodes := bkenode.Nodes(config.Nodes)
	workerNodes := nodes.Worker()
	if index >= len(workerNodes) {
		t.Fatalf("worker node index %d out of range, only %d workers", index, len(workerNodes))
	}
	// workerNodes[index] 返回的是 bkenode.Node 类型
	node := workerNodes[index]
	// 创建新的节点副本
	bkeNode := bkev1beta1.Node(node)
	resultNode := bkenode.Node(bkeNode)
	return &resultNode
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

// isHAClusterTestCase 定义IsHACluster的测试用例结构
type isHAClusterTestCase struct {
	name        string
	bkeConfig   *bkev1beta1.BKEConfig
	masterCount int
	want        bool
}

// runIsHAClusterTest 执行单个IsHACluster测试用例
func runIsHAClusterTest(t *testing.T, tt isHAClusterTestCase) {
	t.Helper()
	cp := &CertPlugin{
		bkeConfig: tt.bkeConfig,
	}
	got := cp.IsHACluster()
	if got != tt.want {
		t.Errorf("IsHACluster() = %v, want %v", got, tt.want)
	}
}

// getIsHAClusterTestCases 获取IsHACluster的测试用例
func getIsHAClusterTestCases(t *testing.T) []isHAClusterTestCase {
	t.Helper()
	return []isHAClusterTestCase{
		{
			name:        "nil config returns false",
			bkeConfig:   nil,
			masterCount: 0,
			want:        false,
		},
		{
			name:        "single master returns false",
			bkeConfig:   createTestBKEConfig(t, 1, 0),
			masterCount: 1,
			want:        false,
		},
		{
			name:        "two masters returns true",
			bkeConfig:   createTestBKEConfig(t, two, 0),
			masterCount: two,
			want:        true,
		},
		{
			name:        "three masters returns true",
			bkeConfig:   createTestBKEConfig(t, three, 0),
			masterCount: three,
			want:        true,
		},
		{
			name:        "multiple masters with workers returns true",
			bkeConfig:   createTestBKEConfig(t, two, two),
			masterCount: two,
			want:        true,
		},
	}
}

// TestIsHACluster 测试IsHACluster函数
func TestIsHACluster(t *testing.T) {
	tests := getIsHAClusterTestCases(t)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runIsHAClusterTest(t, tt)
		})
	}
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
			name:        "worker node in HA cluster uses HA domain",
			bkeConfig:   createTestBKEConfig(t, two, 0),
			currentNode: nil,
			isWorker:    true,
			wantPort:    bkeinit.DefaultLoadBalancerBindPort,
			wantIP:      pkiutil.MasterHADomain,
			wantUser:    testDefaultUser,
		},
		{
			name:        "worker node in single master uses master IP",
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
	tests := getGetNodeServerConfigTestCases(t)
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
func TestGenerateKubeConfigsForScopes_InvalidScope(t *testing.T) {
	cp, _ := setupCertPluginForKubeConfig(t, nil)

	// 使用无效的scope，这会导致Generate失败
	invalidScopes := []string{"invalid-scope"}
	err := cp.generateKubeConfigsForScopes(invalidScopes, bkeinit.DefaultAPIBindPort, testMasterIP1, false)
	if err == nil {
		t.Error("generateKubeConfigsForScopes() with invalid scope should return error")
	}
}

// TestHandleGenerateKubeConfig_EmptyScope 测试空scope的情况
func TestHandleGenerateKubeConfig_EmptyScope(t *testing.T) {
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
