package httputil

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"helm.sh/helm/v3/pkg/repo"
)

func generateCertAndKeyPEM(t *testing.T) (string, string) {
	t.Helper()

	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	assert.NoError(t, err)

	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName: "localhost",
		},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	der, err := x509.CreateCertificate(rand.Reader, template, template, &privateKey.PublicKey, privateKey)
	assert.NoError(t, err)

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(privateKey)})
	return string(certPEM), string(keyPEM)
}

func TestIsHttpsEnabled(t *testing.T) {
	dir := t.TempDir()
	certPath := filepath.Join(dir, "tls.crt")
	keyPath := filepath.Join(dir, "tls.key")
	caPath := filepath.Join(dir, "ca.crt")

	ok, err := IsHttpsEnabled(certPath, keyPath, caPath)
	assert.NoError(t, err)
	assert.False(t, ok)

	assert.NoError(t, os.WriteFile(certPath, []byte("a"), 0600))
	assert.NoError(t, os.WriteFile(keyPath, []byte("b"), 0600))
	assert.NoError(t, os.WriteFile(caPath, []byte("c"), 0600))

	ok, err = IsHttpsEnabled(certPath, keyPath, caPath)
	assert.NoError(t, err)
	assert.True(t, ok)
}

func TestGetHttpConfig(t *testing.T) {
	cfg, err := GetHttpConfig("", "", "", false)
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.True(t, cfg.InsecureSkipVerify)

	certPEM, keyPEM := generateCertAndKeyPEM(t)
	dir := t.TempDir()
	certPath := filepath.Join(dir, "tls.crt")
	keyPath := filepath.Join(dir, "tls.key")
	caPath := filepath.Join(dir, "ca.crt")

	assert.NoError(t, os.WriteFile(certPath, []byte(certPEM), 0600))
	assert.NoError(t, os.WriteFile(keyPath, []byte(keyPEM), 0600))
	assert.NoError(t, os.WriteFile(caPath, []byte(certPEM), 0600))

	cfg, err = GetHttpConfig(certPath, keyPath, caPath, true)
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.NotEmpty(t, cfg.Certificates)
	assert.NotNil(t, cfg.ClientCAs)
}

func TestIsHttpsEnabledByPEM(t *testing.T) {
	ok, err := IsHttpsEnabledByPEM("", "k", "c")
	assert.NoError(t, err)
	assert.False(t, ok)

	ok, err = IsHttpsEnabledByPEM("c", "", "c")
	assert.NoError(t, err)
	assert.False(t, ok)

	ok, err = IsHttpsEnabledByPEM("c", "k", "")
	assert.NoError(t, err)
	assert.False(t, ok)

	ok, err = IsHttpsEnabledByPEM("c", "k", "c")
	assert.NoError(t, err)
	assert.True(t, ok)
}

func TestGetHttpConfigByPEM(t *testing.T) {
	cfg, err := GetHttpConfigByPEM("", "", "", false)
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.True(t, cfg.InsecureSkipVerify)

	certPEM, keyPEM := generateCertAndKeyPEM(t)
	cfg, err = GetHttpConfigByPEM(certPEM, keyPEM, certPEM, true)
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.NotEmpty(t, cfg.Certificates)
	assert.NotNil(t, cfg.RootCAs)
}

func TestGetHarborHTTPConfig(t *testing.T) {
	cfg, err := GetHarborHTTPConfig(&repo.Entry{Name: "openfuyao"})
	assert.NoError(t, err)
	assert.Nil(t, cfg)

	certPEM, keyPEM := generateCertAndKeyPEM(t)
	cfg, err = GetHarborHTTPConfig(&repo.Entry{
		Name:                  "local",
		CertFile:              certPEM,
		KeyFile:               keyPEM,
		CAFile:                certPEM,
		InsecureSkipTLSverify: true,
	})
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.Equal(t, uint16(0x0303), cfg.MinVersion)
	assert.Equal(t, uint16(0x0303), cfg.MaxVersion)

	cfg, err = GetHarborHTTPConfig(&repo.Entry{
		Name:                  "repo",
		CertFile:              "bad-cert",
		KeyFile:               "bad-key",
		CAFile:                "bad-ca",
		InsecureSkipTLSverify: true,
	})
	assert.Error(t, err)
	assert.Nil(t, cfg)
}

func TestGetHarborHttpClientByRepo(t *testing.T) {
	client, err := GetHarborHttpClientByRepo(&repo.Entry{Name: "openfuyao"})
	assert.NoError(t, err)
	assert.NotNil(t, client)
}

