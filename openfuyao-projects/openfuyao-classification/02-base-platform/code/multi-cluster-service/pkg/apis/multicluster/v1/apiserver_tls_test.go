/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1

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

	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/config"
	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/runtime"
)

// newTestRunConfig builds a secure-serving RunConfig for initServer tests.
func newTestRunConfig(port int, cert, key, ca string) *config.RunConfig {
	return &config.RunConfig{Server: &runtime.ServerConfig{
		SecurePort:     port,
		CertFile:       cert,
		PrivateKeyFile: key,
		CAFile:         ca,
	}}
}

// writeSelfSignedCert generates a throwaway self-signed certificate/key pair.
func writeSelfSignedCert(t *testing.T, dir string) (certPath, keyPath, caPath string) {
	t.Helper()

	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	template := x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "mcs-test"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	der, err := x509.CreateCertificate(rand.Reader, &template, &template, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("create certificate: %v", err)
	}

	certPath = filepath.Join(dir, "tls.crt")
	keyPath = filepath.Join(dir, "tls.key")
	caPath = filepath.Join(dir, "ca.crt")

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
	for path, data := range map[string][]byte{certPath: certPEM, keyPath: keyPEM, caPath: certPEM} {
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}
	return certPath, keyPath, caPath
}

func TestInitServerTLS(t *testing.T) {
	t.Run("secure serving with valid certificates", func(t *testing.T) {
		dir := t.TempDir()
		certPath, keyPath, caPath := writeSelfSignedCert(t, dir)

		cfg := newTestRunConfig(18443, certPath, keyPath, caPath)
		server, err := initServer(cfg)
		if err != nil {
			t.Fatalf("initServer() error = %v", err)
		}
		if server.TLSConfig == nil || len(server.TLSConfig.Certificates) == 0 {
			t.Error("TLSConfig not configured for secure serving")
		}
		if server.Addr != ":18443" {
			t.Errorf("Addr = %q, want :18443", server.Addr)
		}
	})

	t.Run("missing certificate file", func(t *testing.T) {
		cfg := newTestRunConfig(18444, "/nonexistent/tls.crt", "/nonexistent/tls.key", "/nonexistent/ca.crt")
		if _, err := initServer(cfg); err == nil {
			t.Error("initServer() expected error for missing certificate")
		}
	})

	t.Run("missing ca file", func(t *testing.T) {
		dir := t.TempDir()
		certPath, keyPath, _ := writeSelfSignedCert(t, dir)

		cfg := newTestRunConfig(18445, certPath, keyPath, "/nonexistent/ca.crt")
		if _, err := initServer(cfg); err == nil {
			t.Error("initServer() expected error for missing ca file")
		}
	})
}
