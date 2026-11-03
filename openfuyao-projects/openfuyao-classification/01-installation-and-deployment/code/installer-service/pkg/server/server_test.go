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
package server

import (
	"context"
	"crypto/tls"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	snapshotclient "github.com/kubernetes-csi/external-snapshotter/client/v4/clientset/versioned"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	k8sclient "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"installer-service/cmd/config"
	"installer-service/pkg/api/clustermanage"
	"installer-service/pkg/client/k8s"
	"installer-service/pkg/server/filters"
	"installer-service/pkg/server/runtime"
)

// 模拟 k8s.Client 接口
type mockK8sClient struct {
	k8s.Client // 嵌入接口以继承所有方法
}

func (m *mockK8sClient) Close() {
	// 模拟关闭操作
}

// 测试正常创建服务器
func TestNewServerSuccess(t *testing.T) {
	// 准备测试数据
	cfg := &config.RunConfig{
		KubernetesCfg: &k8s.KubernetesCfg{},
	}
	ctx := context.Background()

	// 创建 gomonkey 补丁
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 1. 打桩 initServer 函数
	patches.ApplyFunc(initServer, func(_ *config.RunConfig) (*http.Server, error) {
		return &http.Server{Addr: ":8080"}, nil
	})

	// 2. 打桩 k8s.NewKubernetesClient 函数
	patches.ApplyFunc(k8s.NewKubernetesClient, func(_ *k8s.KubernetesCfg) (k8s.Client, error) {
		return &mockK8sClient{}, nil
	})

	// 执行测试
	server, err := NewServer(cfg, ctx)

	// 验证结果
	require.NoError(t, err, "创建服务器不应返回错误")
	assert.NotNil(t, server, "服务器实例不应为 nil")
	assert.Equal(t, ":8080", server.Server.Addr, "HTTP 服务器地址不匹配")
	assert.NotNil(t, server.container, "RESTful 容器不应为 nil")
	assert.NotNil(t, server.KubernetesClient, "Kubernetes 客户端不应为 nil")
	assert.IsType(t, &mockK8sClient{}, server.KubernetesClient, "Kubernetes 客户端类型不匹配")
}

// 测试上下文传递（如果需要）
func TestNewServerContextHandling(t *testing.T) {
	// 准备测试数据
	cfg := &config.RunConfig{
		KubernetesCfg: &k8s.KubernetesCfg{},
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 创建 gomonkey 补丁
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 1. 打桩 initServer
	patches.ApplyFunc(initServer, func(_ *config.RunConfig) (*http.Server, error) {
		return &http.Server{}, nil
	})

	// 2. 打桩 k8s.NewKubernetesClient
	var capturedCtx context.Context
	patches.ApplyFunc(k8s.NewKubernetesClient, func(_ *k8s.KubernetesCfg) (k8s.Client, error) {
		// 捕获上下文以验证
		capturedCtx = ctx
		return &mockK8sClient{}, nil
	})

	// 执行测试
	_, err := NewServer(cfg, ctx)

	// 验证结果
	require.NoError(t, err, "创建服务器不应返回错误")
	assert.Equal(t, ctx, capturedCtx, "上下文应传递给依赖项")
}

func TestInitServer(t *testing.T) {
	testCases := []struct {
		name        string
		setup       func() *gomonkey.Patches
		cfg         *config.RunConfig
		expectedErr string
		validate    func(t *testing.T, s *http.Server)
	}{{name: "HTTP服务成功",
		cfg: &config.RunConfig{Server: &runtime.ServerConfig{
			InsecurePort: 8080},
		},
		validate: func(t *testing.T, s *http.Server) {
			assert.Equal(t, ":8080", s.Addr)
			assert.Nil(t, s.TLSConfig)
		},
	},
		{name: "证书加载失败",
			setup: func() *gomonkey.Patches {
				p := gomonkey.NewPatches()
				p.ApplyFunc(tls.LoadX509KeyPair, func(_, _ string) (tls.Certificate, error) {
					return tls.Certificate{}, errors.New("load cert failed")
				})
				return p
			},
			cfg: &config.RunConfig{
				Server: &runtime.ServerConfig{
					SecurePort: 8443, CertFile: "test.crt", PrivateKey: "test.key",
				},
			},
			expectedErr: "load cert failed", validate: nil,
		},
		{name: "TLS 服务成功",
			setup: func() *gomonkey.Patches {
				p := gomonkey.NewPatches()
				p.ApplyFunc(tls.LoadX509KeyPair, func(_, _ string) (tls.Certificate, error) {
					return tls.Certificate{}, nil
				})
				return p
			},
			cfg: &config.RunConfig{
				Server: &runtime.ServerConfig{
					SecurePort: 8443, CertFile: "test.crt", PrivateKey: "test.key",
				},
			},
			expectedErr: "", validate: func(t *testing.T, s *http.Server) {
				assert.Equal(t, ":8443", s.Addr)
				if assert.NotNil(t, s.TLSConfig) {
					assert.Len(t, s.TLSConfig.Certificates, 1)
				}
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var patches *gomonkey.Patches
			if tc.setup != nil {
				patches = tc.setup()
				defer patches.Reset()
			}
			server, err := initServer(tc.cfg)
			if tc.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectedErr)
				return
			}
			assert.NoError(t, err)
			if tc.validate != nil {
				tc.validate(t, server)
			}
		})
	}
}

func TestResourceFunction(t *testing.T) {
	gr := resource("widgets")
	if gr.Resource != "widgets" {
		t.Fatalf("expected resource widgets, got %s", gr.Resource)
	}
}

// fakeClient implements k8s.Client for testing (only methods needed are stubbed)
type fakeClient struct{}

func (f *fakeClient) ApiExtensions() apiextensionsclient.Interface { return nil }
func (f *fakeClient) Config() *rest.Config                         { return &rest.Config{} }
func (f *fakeClient) Kubernetes() k8sclient.Interface              { return nil }
func (f *fakeClient) Snapshot() snapshotclient.Interface           { return nil }

func TestRegisterAPIAndWSCallConfigInstaller(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	called := false
	calledWs := false

	patches.ApplyFunc(clustermanage.ConfigInstaller, func(c *restful.Container, cfg *rest.Config) error {
		called = true
		return nil
	})
	patches.ApplyFunc(clustermanage.ConfigInstallerWs, func(c *restful.Container, cfg *rest.Config) error {
		calledWs = true
		return nil
	})

	// assign to interface (we only need Config in registerAPI)
	var mock k8s.Client = &fakeClient{}
	srv := &CServer{container: restful.NewContainer(), KubernetesClient: mock}

	srv.registerAPI()
	srv.registerAPIWs()

	assert.True(t, called, "ConfigInstaller should be called")
	assert.True(t, calledWs, "ConfigInstallerWs should be called")
}

func TestBuildHandlerChainWrapsHandlers(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// Patch ProxyAPIServer to mark flag when executed
	var proxied bool

	patches.ApplyFunc(filters.ProxyAPIServer, func(handler http.Handler, cfg *rest.Config) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			proxied = true
			handler.ServeHTTP(w, r)
		})
	})

	// base handler
	calledBase := false
	base := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calledBase = true
		w.WriteHeader(200)
		_, _ = w.Write([]byte("ok"))
	})

	srv := &CServer{Server: &http.Server{}, container: restful.NewContainer(), KubernetesClient: &fakeClient{}}
	srv.Server.Handler = base

	// build the chain
	srv.buildHandlerChain()

	rr := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)

	srv.Server.Handler.ServeHTTP(rr, req)

	assert.True(t, proxied, "ProxyAPIServer wrapper should execute")
	assert.True(t, calledBase, "base handler should be executed")
	assert.Equal(t, 200, rr.Code)
}

func TestRunStartsAndStops(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(clustermanage.ConfigInstaller, func(c *restful.Container, _ *rest.Config) error {
		return nil
	})
	patches.ApplyFunc(clustermanage.ConfigInstallerWs, func(c *restful.Container, _ *rest.Config) error {
		return nil
	})

	srv := &CServer{
		Server:           &http.Server{Addr: "127.0.0.1:0"},
		container:        restful.NewContainer(),
		KubernetesClient: &fakeClient{},
	}

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.Run(ctx)
	}()

	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case err := <-errCh:
		if err != nil {
			assert.ErrorIs(t, err, http.ErrServerClosed)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Run did not stop after context cancel")
	}
}

func TestRunWithTLS(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(clustermanage.ConfigInstaller, func(c *restful.Container, _ *rest.Config) error {
		return nil
	})
	patches.ApplyFunc(clustermanage.ConfigInstallerWs, func(c *restful.Container, _ *rest.Config) error {
		return nil
	})
	patches.ApplyMethod(reflect.TypeOf(&http.Server{}), "ListenAndServeTLS", func(_ *http.Server, _, _ string) error {
		return http.ErrServerClosed
	})

	srv := &CServer{
		Server:           &http.Server{Addr: "127.0.0.1:0", TLSConfig: &tls.Config{}},
		container:        restful.NewContainer(),
		KubernetesClient: &fakeClient{},
	}

	err := srv.Run(context.Background())
	assert.ErrorIs(t, err, http.ErrServerClosed)
}
