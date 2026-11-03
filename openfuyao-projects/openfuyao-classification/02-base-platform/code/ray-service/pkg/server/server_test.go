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

// Package server create apiserver
package server

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"gorm.io/driver/sqlite"
	"gorm.io/gorm"
	"k8s.io/client-go/rest"

	"openfuyao-rayservice/pkg/client/k8s"
	"openfuyao-rayservice/pkg/cmd/config"
	"openfuyao-rayservice/pkg/constant"
	dbruntime "openfuyao-rayservice/pkg/database/runtime"
	"openfuyao-rayservice/pkg/server/runtime"
)

// generateTestCertFiles writes a self-signed certificate and key pair usable
// by tls.LoadX509KeyPair.
func generateTestCertFiles(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("failed to generate private key: %v", err)
	}
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{
			CommonName:   "openfuyao-rayservice-test",
			Organization: []string{"openFuyao"},
		},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IPAddresses:           []net.IP{net.ParseIP("127.0.0.1")},
	}
	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		t.Fatalf("failed to create certificate: %v", err)
	}

	certFile := filepath.Join(dir, "tls.crt")
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	if err := os.WriteFile(certFile, certPEM, 0600); err != nil {
		t.Fatalf("failed to write certificate file: %v", err)
	}

	keyFile := filepath.Join(dir, "tls.key")
	keyDER, err := x509.MarshalECPrivateKey(priv)
	if err != nil {
		t.Fatalf("failed to marshal private key: %v", err)
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})
	if err := os.WriteFile(keyFile, keyPEM, 0600); err != nil {
		t.Fatalf("failed to write key file: %v", err)
	}
	return certFile, keyFile
}

func TestInitServerInsecure(t *testing.T) {
	cfg := &config.RunConfig{Server: runtime.NewServerConfig()}
	httpServer, err := initServer(cfg)
	if err != nil {
		t.Fatalf("initServer() error = %v, want nil", err)
	}
	if httpServer.Addr != ":12345" {
		t.Errorf("initServer() Addr = %q, want :12345", httpServer.Addr)
	}
	if httpServer.TLSConfig != nil {
		t.Errorf("initServer() TLSConfig should be nil for insecure serving")
	}
}

func TestInitServerSecureWithBadCert(t *testing.T) {
	cfg := &config.RunConfig{Server: &runtime.ServerConfig{
		BindAddress:  "0.0.0.0",
		InsecurePort: 0,
		SecurePort:   6443,
		CertFile:     "/not/exist/tls.crt",
		PrivateKey:   "/not/exist/tls.key",
	}}
	if _, err := initServer(cfg); err == nil {
		t.Errorf("initServer() error = nil, want error for not exist cert files")
	}
}

func TestInitServerSecure(t *testing.T) {
	certFile, keyFile := generateTestCertFiles(t)
	cfg := &config.RunConfig{Server: &runtime.ServerConfig{
		BindAddress:  "0.0.0.0",
		InsecurePort: 0,
		SecurePort:   6443,
		CertFile:     certFile,
		PrivateKey:   keyFile,
	}}
	httpServer, err := initServer(cfg)
	if err != nil {
		t.Fatalf("initServer() error = %v, want nil", err)
	}
	if httpServer.Addr != ":6443" {
		t.Errorf("initServer() Addr = %q, want :6443", httpServer.Addr)
	}
	if httpServer.TLSConfig == nil {
		t.Fatalf("initServer() TLSConfig should be set for secure serving")
	}
	if len(httpServer.TLSConfig.Certificates) != 1 {
		t.Errorf("initServer() should load exactly 1 certificate, got %d",
			len(httpServer.TLSConfig.Certificates))
	}
}

func newTestRunConfig(t *testing.T, securePort, insecurePort int,
	certFile, keyFile, fileLocation string) *config.RunConfig {
	t.Helper()
	return &config.RunConfig{
		Server: &runtime.ServerConfig{
			BindAddress:  "127.0.0.1",
			InsecurePort: insecurePort,
			SecurePort:   securePort,
			CertFile:     certFile,
			PrivateKey:   keyFile,
			FileLocation: fileLocation,
		},
		DbConfig: dbruntime.NewDbConfig(),
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{Host: "http://127.0.0.1:18080"},
			QPS:        100,
			Burst:      200,
		},
	}
}

func TestNewServer(t *testing.T) {
	t.Run("nil kubeconfig returns error", func(t *testing.T) {
		cfg := newTestRunConfig(t, 0, 12345, "", "", t.TempDir())
		cfg.KubernetesCfg = &k8s.KubernetesCfg{}
		if _, err := NewServer(cfg, context.Background()); err == nil {
			t.Errorf("NewServer() error = nil, want error for nil kubeconfig")
		}
	})

	t.Run("valid config returns server", func(t *testing.T) {
		cfg := newTestRunConfig(t, 0, 12345, "", "", t.TempDir())
		cServer, err := NewServer(cfg, context.Background())
		if err != nil {
			t.Fatalf("NewServer() error = %v, want nil", err)
		}
		if cServer.Server == nil || cServer.container == nil || cServer.KubernetesClient == nil {
			t.Errorf("NewServer() should initialize Server, container and KubernetesClient")
		}
		if cServer.RunConfig != cfg {
			t.Errorf("NewServer() should keep the given RunConfig")
		}
	})
}

func TestBuildHandlerChain(t *testing.T) {
	cfg := newTestRunConfig(t, 0, 12345, "", "", t.TempDir())
	cServer, err := NewServer(cfg, context.Background())
	if err != nil {
		t.Fatalf("NewServer() error = %v, want nil", err)
	}

	cServer.buildHandlerChain()
	if cServer.Server.Handler == nil {
		t.Fatalf("buildHandlerChain() should set the server handler")
	}
}

// freeTCPPort asks the kernel for a free tcp port on 127.0.0.1.
func freeTCPPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to find a free port: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatalf("failed to close the probe listener: %v", err)
	}
	return port
}

// waitServerReady polls until the given tcp address accepts connections.
func waitServerReady(t *testing.T, address string) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", address, 500*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("server at %s did not become ready in time", address)
}

// newTestDB opens an isolated in-memory sqlite database for the server run tests.
func newTestDB(t *testing.T, name string) *gorm.DB {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(fmt.Sprintf("file:%s?mode=memory&cache=shared", name)), &gorm.Config{})
	if err != nil {
		t.Fatalf("failed to open in-memory db: %v", err)
	}
	return db
}

func TestCServerRunInsecure(t *testing.T) {
	staticDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(staticDir, "hello.txt"), []byte("static-content"), 0600); err != nil {
		t.Fatalf("failed to write static file: %v", err)
	}

	port := freeTCPPort(t)
	cfg := newTestRunConfig(t, 0, port, "", "", staticDir)
	ctx := context.WithValue(context.Background(), constant.ContextDBKey, newTestDB(t, "server_run_insecure"))

	cServer, err := NewServer(cfg, ctx)
	if err != nil {
		t.Fatalf("NewServer() error = %v, want nil", err)
	}

	runCtx, cancel := context.WithCancel(ctx)
	errCh := make(chan error, 1)
	go func() {
		errCh <- cServer.Run(runCtx)
	}()

	address := fmt.Sprintf("127.0.0.1:%d", port)
	waitServerReady(t, address)

	resp, err := http.Get("http://" + address + "/static/hello.txt")
	if err != nil {
		t.Fatalf("failed to request static file: %v", err)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("failed to read response body: %v", err)
	}
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("static file status = %d, want %d", resp.StatusCode, http.StatusOK)
	}
	if string(body) != "static-content" {
		t.Errorf("static file body = %q, want static-content", string(body))
	}

	cancel()
	select {
	case err := <-errCh:
		if err != http.ErrServerClosed {
			t.Errorf("Run() error = %v, want http.ErrServerClosed", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("Run() did not return after context cancel")
	}
}

func TestCServerRunSecure(t *testing.T) {
	certFile, keyFile := generateTestCertFiles(t)
	port := freeTCPPort(t)
	cfg := newTestRunConfig(t, port, 0, certFile, keyFile, t.TempDir())
	ctx := context.WithValue(context.Background(), constant.ContextDBKey, newTestDB(t, "server_run_secure"))

	cServer, err := NewServer(cfg, ctx)
	if err != nil {
		t.Fatalf("NewServer() error = %v, want nil", err)
	}
	if cServer.Server.TLSConfig == nil {
		t.Fatalf("expected TLS config for secure serving")
	}

	runCtx, cancel := context.WithCancel(ctx)
	errCh := make(chan error, 1)
	go func() {
		errCh <- cServer.Run(runCtx)
	}()

	waitServerReady(t, fmt.Sprintf("127.0.0.1:%d", port))

	cancel()
	select {
	case err := <-errCh:
		if err != http.ErrServerClosed {
			t.Errorf("Run() error = %v, want http.ErrServerClosed", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatalf("Run() did not return after context cancel")
	}
}

func TestResourceHelper(t *testing.T) {
	gr := resource("clusters")
	if gr.Group != groupName || gr.Resource != "clusters" {
		t.Errorf("resource() = %v, want group %q resource clusters", gr, groupName)
	}
	if version != "v1alpha1" {
		t.Errorf("version = %q, want v1alpha1", version)
	}
}
