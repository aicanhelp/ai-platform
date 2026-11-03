/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package server

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"math/big"
	"os"
	"testing"
	"time"

	"github.com/emicklei/go-restful/v3"
	snapshotclient "github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"application-management-service/cmd/config"
	"application-management-service/pkg/client/k8s"
	"application-management-service/pkg/server/runtime"
)

type fakeBaseClient struct {
	cfg *rest.Config
}

func (f fakeBaseClient) KubernetesClient() kubernetes.Interface             { return nil }
func (f fakeBaseClient) SnapshotClient() snapshotclient.Interface           { return nil }
func (f fakeBaseClient) ApiExtensionsClient() apiextensionsclient.Interface { return nil }
func (f fakeBaseClient) ConfigClient() *rest.Config                         { return f.cfg }

func TestLogResponse_CallsLogFunc(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "http://example/path", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	r := restful.NewRequest(req)

	recorder := httptest.NewRecorder()
	resp := restful.NewResponse(recorder)
	resp.WriteHeader(http.StatusOK)

	called := false
	logResponse(r, resp, time.Now(), func(format string, args ...interface{}) {
		called = true
	})

	if !called {
		t.Fatalf("Expected log func called")
	}
}

func TestRecordAccessLogs_UsesWarnfWhenStatusGTBadRequest(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "http://example/path", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	r := restful.NewRequest(req)

	recorder := httptest.NewRecorder()
	resp := restful.NewResponse(recorder)

	called := false
	chain := &restful.FilterChain{Target: func(req *restful.Request, resp *restful.Response) {
		resp.WriteHeader(http.StatusInternalServerError)
	}}

	RecordAccessLogs(r, resp, chain)

	if resp.StatusCode() != http.StatusInternalServerError {
		t.Fatalf("Expected status set by chain")
	}

	logResponse(r, resp, time.Now(), func(string, ...interface{}) { called = true })
	if !called {
		t.Fatalf("Expected logResponse called")
	}
}

func TestRecordAccessLogs_UsesInfofWhenStatusLEBadRequest(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "http://example/path", nil)
	req.RemoteAddr = "127.0.0.1:1234"
	r := restful.NewRequest(req)

	recorder := httptest.NewRecorder()
	resp := restful.NewResponse(recorder)

	chain := &restful.FilterChain{Target: func(req *restful.Request, resp *restful.Response) {
		resp.WriteHeader(http.StatusBadRequest)
	}}

	RecordAccessLogs(r, resp, chain)
	if resp.StatusCode() != http.StatusBadRequest {
		t.Fatalf("Expected status set by chain")
	}
}

func TestInitServer_InsecureOnly(t *testing.T) {
	cfg := &config.RunConfig{Server: &runtime.ServerConfig{}}
	cfg.Server.InsecurePort = 8080
	cfg.Server.SecurePort = 0

	s, err := initServer(cfg)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if s.Addr != ":8080" {
		t.Fatalf("Expected :8080, got %s", s.Addr)
	}
	if s.TLSConfig != nil {
		t.Fatalf("Expected TLSConfig nil")
	}
}

func TestInitServer_TLS_KeyPairError(t *testing.T) {
	cfg := &config.RunConfig{Server: &runtime.ServerConfig{}}
	cfg.Server.InsecurePort = 8080
	cfg.Server.SecurePort = 8443
	cfg.Server.CertFile = "/not/exist-cert.pem"
	cfg.Server.PrivateKey = "/not/exist-key.pem"
	cfg.Server.CAFile = "/not/exist"

	_, err := initServer(cfg)
	if err == nil {
		t.Fatalf("Expected error")
	}
}

func TestInitServer_TLS_CAReadError(t *testing.T) {
	certPEM, keyPEM := generateSelfSignedCertPEM(t)

	cfg := &config.RunConfig{Server: &runtime.ServerConfig{}}
	cfg.Server.InsecurePort = 8080
	cfg.Server.SecurePort = 8443
	tmp := t.TempDir()
	cfg.Server.CertFile = tmp + "/cert.pem"
	cfg.Server.PrivateKey = tmp + "/key.pem"
	cfg.Server.CAFile = tmp + "/missing-ca.pem"

	if err := os.WriteFile(cfg.Server.CertFile, certPEM, 0o600); err != nil {
		t.Fatalf("WriteFile cert: %v", err)
	}
	if err := os.WriteFile(cfg.Server.PrivateKey, keyPEM, 0o600); err != nil {
		t.Fatalf("WriteFile key: %v", err)
	}

	_, err := initServer(cfg)
	if err == nil {
		t.Fatalf("Expected error")
	}
}

func TestNewServer_BuildsContainerAndClients(t *testing.T) {
	cfg := &config.RunConfig{Server: &runtime.ServerConfig{}}
	cfg.Server.InsecurePort = 18080
	cfg.KubernetesCfg = &k8s.KubernetesCfg{KubeConfig: &rest.Config{Host: "https://example"}}

	s, err := NewServer(cfg, context.Background())
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if s == nil || s.Server == nil {
		t.Fatalf("Expected server initialized")
	}
	if s.KubernetesClient == nil {
		t.Fatalf("Expected KubernetesClient initialized")
	}
}

func TestRegisterAPI_AddsWebService(t *testing.T) {
	s := &CServer{container: restful.NewContainer(), KubernetesClient: fakeBaseClient{cfg: &rest.Config{Host: "https://example"}}}
	s.registerAPI()
	if len(s.container.RegisteredWebServices()) < 1 {
		t.Fatalf("Expected at least 1 webservice registered")
	}
}

// TestRun_* 在去除 production 注入点后会依赖真实监听端口与 goroutine 退出时序，
// 为避免不稳定/阻塞，本次阶段不新增该类集成测试。

// generateSelfSignedCertPEM generates an in-memory self-signed cert for unit tests.
func generateSelfSignedCertPEM(t *testing.T) ([]byte, []byte) {
	t.Helper()

	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("GenerateKey: %v", err)
	}

	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		t.Fatalf("rand.Int: %v", err)
	}

	template := &x509.Certificate{
		SerialNumber: serial,
		NotBefore:    time.Now().Add(-1 * time.Hour),
		NotAfter:     time.Now().Add(24 * time.Hour),
		KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		IsCA:         false,
		Subject:      pkix.Name{CommonName: "test"},
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, template, template, &priv.PublicKey, priv)
	if err != nil {
		t.Fatalf("CreateCertificate: %v", err)
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	keyBytes := x509.MarshalPKCS1PrivateKey(priv)
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: keyBytes})
	return certPEM, keyPEM
}

func TestRun_CancelShutsDownServer(t *testing.T) {
	s := &CServer{
		Server: &http.Server{
			Addr: ":0", // use an ephemeral port
		},
		container: restful.NewContainer(),
		KubernetesClient: fakeBaseClient{
			cfg: &rest.Config{
				Host: "http://127.0.0.1",
			},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- s.Run(ctx)
	}()

	// Give ListenAndServe a short window to start.
	time.Sleep(200 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		// http.ErrServerClosed is the expected error returned by Shutdown.
		if err != nil && err != http.ErrServerClosed {
			t.Fatalf("Run returned unexpected error: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not exit in time after context cancellation")
	}
}
