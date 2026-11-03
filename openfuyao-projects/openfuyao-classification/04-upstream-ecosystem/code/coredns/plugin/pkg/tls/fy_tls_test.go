// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package tls

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/coredns/coredns/fypkg/crypto"
)

func TestFYConvertToTLSConfig(t *testing.T) {
	certPath := "tmpCertFromMetricServer.pem"
	keyPath := "tmpKeyFromMetricServer.pem"
	caCertPath := "tmpCaCertFromMetricServer.pem"
	caKeyPath := "tmpCaKeyFromMetricServer.pem"
	crlPath := "tmpCRL.crl"

	generateCertificate(certPath, keyPath)
	generateCertificate(caCertPath, caKeyPath)
	generateCRL(caCertPath, caKeyPath, crlPath)

	defer os.Remove(certPath)
	defer os.Remove(keyPath)
	defer os.Remove(caCertPath)
	defer os.Remove(caKeyPath)
	defer os.Remove(crlPath)

	hookOld := crypto.GetDecryptKeyBytesHookFunc()
	crypto.SetDecryptKeyBytesHookFunc(func(b []byte) ([]byte, error) {
		return b, nil
	})
	defer crypto.SetDecryptKeyBytesHookFunc(hookOld)

	tests := []struct {
		certPath   string
		keyPath    string
		caPath     string
		crlPath    string
		clientAuth tls.ClientAuthType
		shouldErr  bool
	}{
		// oks
		{certPath, keyPath, "", "", tls.NoClientCert, false},
		{certPath, keyPath, caCertPath, "", tls.RequireAndVerifyClientCert, false},
		{certPath, keyPath, caCertPath, crlPath, tls.RequireAndVerifyClientCert, false},
		// fails
		{"fake.pem", keyPath, "", "", tls.NoClientCert, true},
		{certPath, "fake.pem", "", "", tls.NoClientCert, true},
		{certPath, keyPath, "fake.pem", "", tls.RequireAndVerifyClientCert, true},
	}
	for i, test := range tests {
		tlsArgs := NewFyTLSArgs()
		tlsArgs.ServerCertPath = test.certPath
		tlsArgs.ServerKeyPath = test.keyPath
		tlsArgs.ClientCACertPath = test.caPath
		tlsArgs.CRLPath = test.crlPath
		tlsArgs.ClientAuth = test.clientAuth
		_, err := tlsArgs.fyConvertToTLSConfig()
		if test.shouldErr {
			if err == nil {
				t.Errorf("Test %v: Expected error but found nil", i)
			}
		} else {
			if err != nil {
				t.Errorf("Test %v: Expected no error but found error: %v", i, err)
			}
		}
	}
}

func generateCertificate(certPath, keyPath string) {
	// 生成密钥对
	privateKey, _ := rsa.GenerateKey(rand.Reader, 2048)

	// 创建证书模板
	template := x509.Certificate{
		SubjectKeyId: []byte{0},
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName:   "example.com",
			Organization: []string{"Example Organization"},
		},
		NotBefore: time.Now(),
		NotAfter:  time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:  x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage: []x509.ExtKeyUsage{
			x509.ExtKeyUsageServerAuth,
			x509.ExtKeyUsageClientAuth,
		},
		IPAddresses: []net.IP{net.ParseIP("127.0.0.1")},
	}

	// 使用私钥生成证书
	derBytes, _ := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)

	// 将证书保存到文件
	certOut, _ := os.OpenFile(certPath, os.O_CREATE|os.O_WRONLY, 0600)
	defer certOut.Close()
	pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})

	// 将私钥保存到文件
	keyOut, _ := os.OpenFile(keyPath, os.O_CREATE|os.O_WRONLY, 0600)
	defer keyOut.Close()
	pem.Encode(keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})
}

func generateCRL(caCertPath, caKeyPath, crlPath string) {
	certPEMBlock, _ := os.ReadFile(filepath.Clean(caCertPath))
	keyPEMBlock, _ := os.ReadFile(filepath.Clean(caKeyPath))

	certBlock, _ := pem.Decode(certPEMBlock)
	keyBlock, _ := pem.Decode(keyPEMBlock)

	cert, _ := x509.ParseCertificate(certBlock.Bytes)
	privateKey, _ := x509.ParsePKCS1PrivateKey(keyBlock.Bytes)

	// 创建一个空的吊销列表模板
	crlTemplate := &x509.RevocationList{
		SignatureAlgorithm:  x509.SHA256WithRSA,
		Issuer:              pkix.Name{CommonName: "example.com"},
		ThisUpdate:          time.Now(),
		NextUpdate:          time.Now().AddDate(1, 0, 0),
		RevokedCertificates: []pkix.RevokedCertificate{},
		Number:              big.NewInt(1),
	}

	// 使用私钥创建CRL
	crlBytes, _ := x509.CreateRevocationList(rand.Reader, crlTemplate, cert, privateKey)
	// 将CRL保存到文件中
	crlFile, _ := os.OpenFile(crlPath, os.O_CREATE|os.O_WRONLY, 0600)
	defer crlFile.Close()

	pem.Encode(crlFile, &pem.Block{
		Type:  "X509 CRL",
		Bytes: crlBytes,
	})
}
