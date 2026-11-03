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

package server

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/emicklei/go-restful/v3"
	snapshotclient "github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"plugin-management-service/pkg/client/k8s"
	"plugin-management-service/pkg/server/config"
	"plugin-management-service/pkg/server/runtime"
)

type mockK8sClient struct {
	k8s.BaseClient
}

func (m *mockK8sClient) ConfigClient() *rest.Config {
	return &rest.Config{Host: "http://localhost"}
}
func (m *mockK8sClient) KubernetesClient() kubernetes.Interface             { return nil }
func (m *mockK8sClient) SnapshotClient() snapshotclient.Interface           { return nil }
func (m *mockK8sClient) ApiExtensionsClient() apiextensionsclient.Interface { return nil }

// writeFakeKubeconfig writes a syntactically valid kubeconfig into a temp
// HOME so that k8s.GetKubeConfig() succeeds on machines without in-cluster
// credentials or ~/.kube/config (e.g. CI runners).
func writeFakeKubeconfig(t *testing.T) {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	if err := os.MkdirAll(filepath.Join(home, ".kube"), 0o755); err != nil {
		t.Fatalf("create .kube dir: %v", err)
	}
	content := `apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://127.0.0.1:6443
    insecure-skip-tls-verify: true
  name: dummy
contexts:
- context:
    cluster: dummy
    user: dummy
  name: dummy
current-context: dummy
users:
- name: dummy
  user:
    token: dummy
`
	path := filepath.Join(home, ".kube", "config")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write kubeconfig: %v", err)
	}
}

func TestNewServerAndRegisterAPI(t *testing.T) {
	// Backup and restore
	oldNewK8s := newKubernetesClient
	defer func() { newKubernetesClient = oldNewK8s }()

	// Inject mock
	newKubernetesClient = func(cfg *k8s.KubernetesCfg) (k8s.BaseClient, error) {
		return &mockK8sClient{}, nil
	}

	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			InsecurePort: 8080,
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{},
		},
	}

	ctx := context.Background()
	srv, err := NewServer(cfg, ctx)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	if srv.container == nil {
		t.Fatal("srv.container is nil")
	}

	// Test registerAPI
	srv.registerAPI()

	// Verify routes are added (at least one)
	if len(srv.container.RegisteredWebServices()) == 0 {
		t.Error("No web services registered")
	}
}

func TestRun(t *testing.T) {
	oldNewK8s := newKubernetesClient
	defer func() { newKubernetesClient = oldNewK8s }()

	newKubernetesClient = func(cfg *k8s.KubernetesCfg) (k8s.BaseClient, error) {
		return &mockK8sClient{}, nil
	}

	cfg := &config.RunConfig{
		Server: &runtime.ServerConfig{
			InsecurePort: 0,
		},
		KubernetesCfg: &k8s.KubernetesCfg{
			KubeConfig: &rest.Config{},
		},
	}

	srv, _ := NewServer(cfg, context.Background())

	ctx, cancel := context.WithCancel(context.Background())

	go func() {
		_ = srv.Run(ctx)
	}()

	cancel()
}

func TestRecordAccessLogs(t *testing.T) {
	type args struct {
		req   *restful.Request
		resp  *restful.Response
		chain *restful.FilterChain
	}
	okFoundFilter := func(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
		resp.WriteHeader(http.StatusOK)
		chain.ProcessFilter(req, resp)
	}
	notFoundFilter := func(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
		resp.WriteHeader(http.StatusNotFound)
		chain.ProcessFilter(req, resp)
	}
	dummyTarget := func(req *restful.Request, resp *restful.Response) {
		// dummyTarget function does nothing
	}
	tests := []struct {
		name string
		args args
	}{
		{
			"TestWarnLogs",
			args{
				restful.NewRequest(httptest.NewRequest("GET", "http://example.com", nil)),
				restful.NewResponse(httptest.NewRecorder()),
				&restful.FilterChain{
					Filters: []restful.FilterFunction{notFoundFilter},
					Target:  dummyTarget,
				},
			},
		},
		{
			"TestInfoLogs",
			args{
				restful.NewRequest(httptest.NewRequest("GET", "http://example.com", nil)),
				restful.NewResponse(httptest.NewRecorder()),
				&restful.FilterChain{
					Filters: []restful.FilterFunction{okFoundFilter},
					Target:  dummyTarget,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			RecordAccessLogs(tt.args.req, tt.args.resp, tt.args.chain)
		})
	}
}

func TestInitServer(t *testing.T) {
	writeFakeKubeconfig(t)

	createMockCertFiles := func(t *testing.T) (string, string, string) {
		dir := t.TempDir()
		caFile := filepath.Join(dir, "ca.pem")
		certFile := filepath.Join(dir, "server.crt")
		keyFile := filepath.Join(dir, "server.key")

		key, _ := rsa.GenerateKey(rand.Reader, 2048)
		template := x509.Certificate{
			SerialNumber: big.NewInt(1),
			Subject:      pkix.Name{Organization: []string{"Test"}},
			NotBefore:    time.Now(),
			NotAfter:     time.Now().Add(time.Hour),
			KeyUsage:     x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
			ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		}

		derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &key.PublicKey, key)
		if err != nil {
			t.Fatalf("Failed to create certificate: %v", err)
		}

		certOut, _ := os.Create(certFile)
		pem.Encode(certOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
		certOut.Close()

		keyOut, _ := os.OpenFile(keyFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0600)
		pem.Encode(keyOut, &pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
		keyOut.Close()

		caOut, _ := os.Create(caFile)
		pem.Encode(caOut, &pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
		caOut.Close()

		return caFile, certFile, keyFile
	}

	ca, cert, key := createMockCertFiles(t)

	tests := []struct {
		name    string
		cfg     *config.RunConfig
		wantErr bool
	}{
		{
			"TestInsecureConfig",
			&config.RunConfig{
				Server: &runtime.ServerConfig{
					BindAddress:    "0.0.0.0",
					SecurePort:     0,
					InsecurePort:   8080,
					PrivateKeyFile: "",
					CertFile:       "",
					CAFile:         "",
				},
				KubernetesCfg: k8s.NewKubernetesCfg(),
			},
			false,
		},
		{
			"TestSecureConfigSuccess",
			&config.RunConfig{
				Server: &runtime.ServerConfig{
					BindAddress:    "0.0.0.0",
					SecurePort:     8443,
					InsecurePort:   0,
					PrivateKeyFile: key,
					CertFile:       cert,
					CAFile:         ca,
				},
				KubernetesCfg: k8s.NewKubernetesCfg(),
			},
			false,
		},
		{
			"TestSecureConfigNoFile",
			&config.RunConfig{
				Server: &runtime.ServerConfig{
					BindAddress:    "0.0.0.0",
					SecurePort:     8443,
					InsecurePort:   0,
					PrivateKeyFile: "non-existent",
					CertFile:       "non-existent",
					CAFile:         "non-existent",
				},
				KubernetesCfg: k8s.NewKubernetesCfg(),
			},
			true,
		},
		{
			"TestSecureConfigInvalidCAFile",
			&config.RunConfig{
				Server: &runtime.ServerConfig{
					BindAddress:    "0.0.0.0",
					SecurePort:     8443,
					InsecurePort:   0,
					PrivateKeyFile: key,
					CertFile:       cert,
					CAFile:         "non-existent",
				},
				KubernetesCfg: k8s.NewKubernetesCfg(),
			},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := initServer(tt.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("initServer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && tt.cfg.Server.SecurePort != 0 {
				if s.TLSConfig == nil {
					t.Errorf("initServer() expected TLSConfig for secure port")
				}
				if s.Addr != fmt.Sprintf(":%d", tt.cfg.Server.SecurePort) {
					t.Errorf("initServer() addr = %v, want :%d", s.Addr, tt.cfg.Server.SecurePort)
				}
			}
		})
	}
}
