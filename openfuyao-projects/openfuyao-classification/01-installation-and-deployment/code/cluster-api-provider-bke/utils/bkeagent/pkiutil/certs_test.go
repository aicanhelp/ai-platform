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
	"net"
	"testing"
	"time"

	certutil "k8s.io/client-go/util/cert"
)

const (
	// 测试常量定义
	testCAValidityYears      = 10
	testCertValidityYears    = 1
	testTimeToleranceMinutes = 1440
	testIPOctet1             = 172
	testIPOctet2             = 16
	testIPOctet3             = 0
	testIPOctet4             = 1
	oneYearDay               = 365
	oneDayHour               = 24
	oneMonthHour             = 730
	two                      = 2
	one                      = 1
)

// testCACertificate 创建用于测试的 CA 证书和密钥
func testCACertificate(t *testing.T) (*x509.Certificate, *rsa.PrivateKey) {
	t.Helper()
	caKey, err := newPrivateKey()
	if err != nil {
		t.Fatalf("Failed to generate CA key: %v", err)
	}

	caCertSpec := &BKECert{
		Name: "ca",
		IsCA: true,
		Config: CertConfig{
			Validity: time.Duration(testCAValidityYears) * oneYearDay * oneDayHour * time.Hour,
			Config: certutil.Config{
				CommonName: "test-ca",
			},
		},
	}

	caCert, err := newSelfSignedCACert(caCertSpec, caKey)
	if err != nil {
		t.Fatalf("Failed to generate CA cert: %v", err)
	}
	return caCert, caKey
}

// testPrivateKey 生成测试用的私钥
func testPrivateKey(t *testing.T) *rsa.PrivateKey {
	t.Helper()
	key, err := newPrivateKey()
	if err != nil {
		t.Fatalf("Failed to generate key: %v", err)
	}
	return key
}

// testIPAddress 生成测试用的 IP 地址
func testIPAddress() net.IP {
	return net.IPv4(testIPOctet1, testIPOctet2, testIPOctet3, testIPOctet4)
}

// testIPv6Address 生成测试用的 IPv6 地址
func testIPv6Address() net.IP {
	return net.IPv6loopback
}

// validateBasicCertProperties 验证证书的基本属性
func validateBasicCertProperties(t *testing.T, cert *x509.Certificate, caCert *x509.Certificate) {
	t.Helper()
	if cert == nil {
		t.Fatal("Certificate is nil")
	}
	if cert.SerialNumber == nil {
		t.Error("Expected SerialNumber to be set")
	}
	if !cert.BasicConstraintsValid {
		t.Error("Expected BasicConstraintsValid to be true")
	}
	if err := cert.CheckSignatureFrom(caCert); err != nil {
		t.Errorf("Certificate signature verification failed: %v", err)
	}
}

// validateCACertProperties 验证CA证书的属性
func validateCACertProperties(t *testing.T, cert *x509.Certificate) {
	t.Helper()
	if !cert.IsCA {
		t.Error("Expected IsCA to be true")
	}
	if cert.KeyUsage&x509.KeyUsageCertSign == 0 {
		t.Error("Expected KeyUsageCertSign to be set")
	}
}

// createAndValidateCert 创建证书并验证基本属性
func createAndValidateCert(t *testing.T, certSpec *BKECert, key *rsa.PrivateKey, caCert *x509.Certificate, caKey *rsa.PrivateKey) *x509.Certificate {
	t.Helper()
	cert, err := newSignedCert(certSpec, key, caCert, caKey)
	if err != nil {
		t.Fatalf("newSignedCert() error = %v", err)
	}
	validateBasicCertProperties(t, cert, caCert)
	return cert
}

// validateAltNames 验证AltNames（DNSNames和IPAddresses）
func validateAltNames(t *testing.T, cert *x509.Certificate, expectedDNSCount, expectedIPCount int) {
	t.Helper()
	if len(cert.DNSNames) != expectedDNSCount {
		t.Errorf("Expected %d DNSNames, got %d", expectedDNSCount, len(cert.DNSNames))
	}
	if len(cert.IPAddresses) != expectedIPCount {
		t.Errorf("Expected %d IPAddresses, got %d", expectedIPCount, len(cert.IPAddresses))
	}
}

// validateExtKeyUsages 验证扩展密钥用途
func validateExtKeyUsages(t *testing.T, cert *x509.Certificate, expectedUsages []x509.ExtKeyUsage) {
	t.Helper()
	if len(cert.ExtKeyUsage) != len(expectedUsages) {
		t.Errorf("Expected %d ExtKeyUsage, got %d", len(expectedUsages), len(cert.ExtKeyUsage))
	}
	usageMap := make(map[x509.ExtKeyUsage]bool)
	for _, usage := range cert.ExtKeyUsage {
		usageMap[usage] = true
	}
	for _, expectedUsage := range expectedUsages {
		if !usageMap[expectedUsage] {
			t.Errorf("Expected ExtKeyUsage %v to be set", expectedUsage)
		}
	}
}

// TestNewSignedCert_Normal 测试正常证书创建
func TestNewSignedCertNormal(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "test-cert",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName:   "test.example.com",
				Organization: []string{"Test Org"},
				Usages:       []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
				AltNames: certutil.AltNames{
					DNSNames: []string{"test.example.com", "www.test.example.com"},
					IPs:      []net.IP{testIPAddress()},
				},
			},
		},
	}

	cert := createAndValidateCert(t, certSpec, key, caCert, caKey)

	if cert.Subject.CommonName != "test.example.com" {
		t.Errorf("Expected CommonName 'test.example.com', got '%s'", cert.Subject.CommonName)
	}
	if len(cert.Subject.Organization) != 1 || cert.Subject.Organization[0] != "Test Org" {
		t.Errorf("Expected Organization 'Test Org', got %v", cert.Subject.Organization)
	}
	validateAltNames(t, cert, two, one)
	if cert.IsCA {
		t.Error("Expected IsCA to be false")
	}
	if cert.KeyUsage&x509.KeyUsageCertSign != 0 {
		t.Error("Expected KeyUsageCertSign to not be set")
	}
	if !cert.NotBefore.Equal(caCert.NotBefore) {
		t.Errorf("Expected NotBefore to match CA cert, got %v, expected %v", cert.NotBefore, caCert.NotBefore)
	}
	if !cert.NotAfter.Equal(caCert.NotAfter) {
		t.Errorf("Expected NotAfter to match CA cert, got %v, expected %v", cert.NotAfter, caCert.NotAfter)
	}
}

// TestNewSignedCert_CustomValidity 测试自定义有效期的证书
func TestNewSignedCertCustomValidity(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "test-cert-validity",
		Config: CertConfig{
			Validity: time.Duration(testCertValidityYears) * oneYearDay * oneDayHour * time.Hour,
			Config: certutil.Config{
				CommonName: "test-validity.example.com",
				Usages:     []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth},
			},
		},
	}

	cert, err := newSignedCert(certSpec, key, caCert, caKey)
	if err != nil {
		t.Fatalf("newSignedCert() error = %v", err)
	}

	validateBasicCertProperties(t, cert, caCert)

	now := time.Now().UTC()
	expectedNotBefore := now.Add(-time.Hour * oneMonthHour)
	expectedNotAfter := now.Add(time.Duration(testCertValidityYears) * oneYearDay * oneDayHour * time.Hour)
	tolerance := time.Duration(testTimeToleranceMinutes) * time.Minute

	if cert.NotBefore.Before(expectedNotBefore.Add(-tolerance)) || cert.NotBefore.After(expectedNotBefore.Add(tolerance)) {
		t.Errorf("Expected NotBefore around %v, got %v", expectedNotBefore, cert.NotBefore)
	}
	if cert.NotAfter.Before(expectedNotAfter.Add(-tolerance)) || cert.NotAfter.After(expectedNotAfter.Add(tolerance)) {
		t.Errorf("Expected NotAfter around %v, got %v", expectedNotAfter, cert.NotAfter)
	}
}

// TestNewSignedCert_IsCA 测试 CA 证书创建
func TestNewSignedCertIsCA(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "test-ca-cert",
		IsCA: true,
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "test-ca.example.com",
			},
		},
	}

	cert := createAndValidateCert(t, certSpec, key, caCert, caKey)
	validateCACertProperties(t, cert)
}

// TestNewSignedCert_NameCA 测试名称为 'ca' 的证书
func TestNewSignedCertNameCA(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "ca",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "ca.example.com",
			},
		},
	}

	cert, err := newSignedCert(certSpec, key, caCert, caKey)
	if err != nil {
		t.Fatalf("newSignedCert() error = %v", err)
	}

	validateBasicCertProperties(t, cert, caCert)

	if cert.KeyUsage&x509.KeyUsageCRLSign == 0 {
		t.Error("Expected KeyUsageCRLSign to be set when Name is 'ca'")
	}
}

// TestNewSignedCert_FullSubject 测试包含完整主题属性的证书
func TestNewSignedCertFullSubject(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "test-full-cert",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "full.example.com",
				AltNames: certutil.AltNames{
					DNSNames: []string{"dns1.example.com", "dns2.example.com"},
					IPs:      []net.IP{testIPAddress(), testIPv6Address()},
				},
			},
			Country:            []string{"CN"},
			Province:           []string{"Beijing"},
			Locality:           []string{"Beijing"},
			OrganizationalUnit: []string{"IT"},
		},
	}

	cert := createAndValidateCert(t, certSpec, key, caCert, caKey)

	if len(cert.Subject.Country) != 1 || cert.Subject.Country[0] != "CN" {
		t.Errorf("Expected Country 'CN', got %v", cert.Subject.Country)
	}
	if len(cert.Subject.Province) != 1 || cert.Subject.Province[0] != "Beijing" {
		t.Errorf("Expected Province 'Beijing', got %v", cert.Subject.Province)
	}
	if len(cert.Subject.Locality) != 1 || cert.Subject.Locality[0] != "Beijing" {
		t.Errorf("Expected Locality 'Beijing', got %v", cert.Subject.Locality)
	}
	if len(cert.Subject.OrganizationalUnit) != 1 || cert.Subject.OrganizationalUnit[0] != "IT" {
		t.Errorf("Expected OrganizationalUnit 'IT', got %v", cert.Subject.OrganizationalUnit)
	}
	validateAltNames(t, cert, two, two)
}

// TestNewSignedCert_MultipleExtKeyUsages 测试多个扩展密钥用途的证书
func TestNewSignedCertMultipleExtKeyUsages(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "test-multi-usage",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "multi.example.com",
				Usages: []x509.ExtKeyUsage{
					x509.ExtKeyUsageServerAuth,
					x509.ExtKeyUsageClientAuth,
				},
			},
		},
	}

	cert := createAndValidateCert(t, certSpec, key, caCert, caKey)

	expectedUsages := []x509.ExtKeyUsage{
		x509.ExtKeyUsageServerAuth,
		x509.ExtKeyUsageClientAuth,
	}
	validateExtKeyUsages(t, cert, expectedUsages)
}

// TestNewSignedCert_ZeroValidity 测试零有效期的证书（应使用 CA 证书有效期）
func TestNewSignedCertZeroValidity(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "test-zero-validity",
		Config: CertConfig{
			Validity: 0,
			Config: certutil.Config{
				CommonName: "test.example.com",
			},
		},
	}

	cert := createAndValidateCert(t, certSpec, key, caCert, caKey)

	if !cert.NotBefore.Equal(caCert.NotBefore) {
		t.Errorf("Expected NotBefore to match CA cert, got %v, expected %v", cert.NotBefore, caCert.NotBefore)
	}
	if !cert.NotAfter.Equal(caCert.NotAfter) {
		t.Errorf("Expected NotAfter to match CA cert, got %v, expected %v", cert.NotAfter, caCert.NotAfter)
	}
}

// TestNewSignedCert_EmptyAltNames 测试空 AltNames 的证书
func TestNewSignedCertEmptyAltNames(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "test-empty-altnames",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "test.example.com",
				AltNames:   certutil.AltNames{},
			},
		},
	}

	cert := createAndValidateCert(t, certSpec, key, caCert, caKey)

	validateAltNames(t, cert, 0, 0)
}

// TestNewSignedCert_IsCAAndNameCA 测试同时设置 IsCA 和 Name 为 'ca' 的证书
func TestNewSignedCertIsCAAndNameCA(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "ca",
		IsCA: true,
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "test-ca.example.com",
			},
		},
	}

	cert := createAndValidateCert(t, certSpec, key, caCert, caKey)
	validateCACertProperties(t, cert)

	if cert.KeyUsage&x509.KeyUsageCRLSign == 0 {
		t.Error("Expected KeyUsageCRLSign to be set when Name is 'ca'")
	}
}

// TestNewSignedCert_KeyUsage 测试密钥用途设置
func TestNewSignedCertKeyUsage(t *testing.T) {
	caCert, caKey := testCACertificate(t)
	key := testPrivateKey(t)

	certSpec := &BKECert{
		Name: "test-keyusage",
		Config: CertConfig{
			Config: certutil.Config{
				CommonName: "test.example.com",
			},
		},
	}

	cert := createAndValidateCert(t, certSpec, key, caCert, caKey)

	expectedKeyUsage := x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature
	if cert.KeyUsage&expectedKeyUsage != expectedKeyUsage {
		t.Errorf("Expected KeyUsage to include KeyEncipherment and DigitalSignature, got %v", cert.KeyUsage)
	}
}
