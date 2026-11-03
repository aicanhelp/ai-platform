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
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/client/k8s"
	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/config"
	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/runtime"
)

func TestInitServer(t *testing.T) {
	tests := createTestCases()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch1 := gomonkey.ApplyFunc(os.ReadFile, func(path string) ([]byte, error) {
				if tt.mockErr != nil {
					return nil, tt.mockErr
				}
				return []byte("OK"), nil
			})
			patch2 := gomonkey.ApplyFunc(tls.LoadX509KeyPair, func(cf string, kf string) (tls.Certificate, error) {
				if tt.mockErr != nil {
					return tls.Certificate{}, tt.mockErr
				}
				return tls.Certificate{}, nil
			})
			patch3 := gomonkey.ApplyFunc(x509.NewCertPool, func() *x509.CertPool {
				return nil
			})
			patch4 := gomonkey.ApplyMethod(reflect.TypeOf(&x509.CertPool{}), "AppendCertsFromPEM",
				func(_ *x509.CertPool, p []byte) (ok bool) {
					return true
				})
			defer patch1.Reset()
			defer patch2.Reset()
			defer patch3.Reset()
			defer patch4.Reset()
			got, err := initServer(tt.args.cfg)
			if (err != nil) != tt.wantErr {
				t.Errorf("initServer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got == nil && tt.want == nil {
				return
			}
			if !reflect.DeepEqual(got.Addr, tt.want.Addr) {
				t.Errorf("initServer() got = %v, want %v", got.TLSConfig, tt.want.TLSConfig)
			}
		})
	}
}

type InitServerTestCase struct {
	name    string
	args    struct{ cfg *config.RunConfig }
	want    *http.Server
	wantErr bool
	mockErr error
}

func createTestCases() []InitServerTestCase {
	certData, err := tls.X509KeyPair([]byte("cert"), []byte("key"))
	if err != nil {
		fmt.Println(err)
	}
	return []InitServerTestCase{
		{
			name: "success",
			args: struct{ cfg *config.RunConfig }{
				cfg: &config.RunConfig{
					Server: &runtime.ServerConfig{SecurePort: 0},
				},
			},
			want: &http.Server{
				Addr: ":0",
			},
			wantErr: false,
			mockErr: nil,
		},
		{
			name: "port not equal 0",
			args: struct{ cfg *config.RunConfig }{
				cfg: &config.RunConfig{
					Server: &runtime.ServerConfig{SecurePort: 2},
				},
			},
			want: &http.Server{
				Addr: ":2",
				TLSConfig: &tls.Config{
					Certificates: []tls.Certificate{certData},
					ClientAuth:   tls.RequireAndVerifyClientCert,
					MinVersion:   tls.VersionTLS12,
					ClientCAs:    x509.NewCertPool(),
				},
			},
			wantErr: false,
			mockErr: nil,
		},
		{
			name: "mock error",
			args: struct{ cfg *config.RunConfig }{
				cfg: &config.RunConfig{
					Server: &runtime.ServerConfig{SecurePort: 1, CertFile: "s", PrivateKey: "S"},
				},
			},
			want:    nil,
			wantErr: true,
			mockErr: errors.New("err"),
		},
	}
}

func TestAddSecurityHeader(t *testing.T) {
	// 创建一个模拟的 HTTP 请求
	req := httptest.NewRequest("GET", "/test", nil)
	// 创建一个 ResponseRecorder，用于捕获响应
	recorder := httptest.NewRecorder()

	// 模拟下游处理器
	mockHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, err := w.Write([]byte("OK"))
		if err != nil {
			t.Fatalf("error is %v", err)
		}
	})

	// 包装下游处理器
	handler := addSecurityHeader(mockHandler)

	// 调用处理器
	handler.ServeHTTP(recorder, req)

	// 检查响应状态码
	assert.Equal(t, http.StatusOK, recorder.Code)

	// 验证响应头是否设置正确
	expectedHeaders := map[string]string{
		"Content-Security-Policy":   "connect-src 'self' https:;frame-ancestors 'none';object-src 'none'",
		"Cache-Control":             "no-cache, no-store, must-revalidate",
		"X-Frame-Options":           "DENY",
		"X-Content-Type-Options":    "nosniff",
		"X-XSS-Protection":          "1",
		"Strict-Transport-Security": "max-age=31536000",
	}

	for key, expectedValue := range expectedHeaders {
		t.Run(key, func(t *testing.T) {
			assert.Equal(t, expectedValue, recorder.Header().Get(key), "Header %s should match", key)
		})
	}

	// 验证响应体
	assert.Equal(t, "OK", recorder.Body.String())
}
func TestAPIServerRun(t *testing.T) {
	apiServer := &APIServer{
		Server: &http.Server{
			Addr: ":0", // 使用随机端口
		},
		container: restful.NewContainer(),
	}

	// Monkey Patch AddToContainer
	patch := gomonkey.ApplyFunc(AddToContainer, func(container *restful.Container, client client.Client) error {
		webService := new(restful.WebService)
		container.Add(webService)
		return nil
	})
	defer patch.Reset()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		time.Sleep(500 * time.Millisecond) // 确保服务器有足够时间启动
		cancel()                           // 模拟关闭服务器
	}()

	// 启动服务器
	err := apiServer.Run(ctx)
	if err != nil && err.Error() != "http: Server closed" {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestAPIServerRegisterAPI_PanicOnError(t *testing.T) {
	apiServer := &APIServer{
		container: restful.NewContainer(),
	}

	patch := gomonkey.ApplyFunc(AddToContainer, func(container *restful.Container, client client.Client) error {
		return errors.New("boom")
	})
	defer patch.Reset()

	assert.Panics(t, func() {
		apiServer.registerAPI()
	})
}

func TestAPIServerRegisterAPI_Success(t *testing.T) {
	apiServer := &APIServer{
		container: restful.NewContainer(),
	}

	patch := gomonkey.ApplyFunc(AddToContainer, func(container *restful.Container, client client.Client) error {
		return nil
	})
	defer patch.Reset()

	assert.NotPanics(t, func() {
		apiServer.registerAPI()
	})
}

type fakeK8sClient struct{}

func (f *fakeK8sClient) K8s() kubernetes.Interface {
	return nil
}

func (f *fakeK8sClient) Cfg() *rest.Config {
	return nil
}

func TestNewServer_InitServerError(t *testing.T) {
	patch := gomonkey.ApplyFunc(initServer, func(cfg *config.RunConfig) (*http.Server, error) {
		return nil, errors.New("init error")
	})
	defer patch.Reset()

	_, err := NewServer(&config.RunConfig{}, context.Background(), nil)
	assert.Error(t, err)
}

func TestNewServer_K8sClientError(t *testing.T) {
	patch1 := gomonkey.ApplyFunc(initServer, func(cfg *config.RunConfig) (*http.Server, error) {
		return &http.Server{}, nil
	})
	patch2 := gomonkey.ApplyFunc(k8s.NewKubernetesClient, func(cfg *k8s.KubernetesCfg) (k8s.Client, error) {
		return nil, errors.New("k8s error")
	})
	defer patch1.Reset()
	defer patch2.Reset()

	_, err := NewServer(&config.RunConfig{KubernetesCfg: &k8s.KubernetesCfg{}}, context.Background(), nil)
	assert.Error(t, err)
}

func TestNewServer_Success(t *testing.T) {
	patch1 := gomonkey.ApplyFunc(initServer, func(cfg *config.RunConfig) (*http.Server, error) {
		return &http.Server{}, nil
	})
	patch2 := gomonkey.ApplyFunc(k8s.NewKubernetesClient, func(cfg *k8s.KubernetesCfg) (k8s.Client, error) {
		return &fakeK8sClient{}, nil
	})
	defer patch1.Reset()
	defer patch2.Reset()

	server, err := NewServer(&config.RunConfig{KubernetesCfg: &k8s.KubernetesCfg{}}, context.Background(), nil)
	assert.NoError(t, err)
	assert.NotNil(t, server)
	assert.NotNil(t, server.Server)
	assert.NotNil(t, server.container)
	assert.NotNil(t, server.KubernetesClient)
}

func TestAPIServerRun_Insecure(t *testing.T) {
	apiServer := &APIServer{
		Server:    &http.Server{},
		container: restful.NewContainer(),
	}
	patch1 := gomonkey.ApplyFunc(AddToContainer, func(container *restful.Container, client client.Client) error {
		return nil
	})
	var called bool
	patch2 := gomonkey.ApplyMethod(reflect.TypeOf(&http.Server{}), "ListenAndServe",
		func(_ *http.Server) error {
			called = true
			return http.ErrServerClosed
		})
	defer patch1.Reset()
	defer patch2.Reset()

	err := apiServer.Run(context.Background())
	assert.True(t, called)
	assert.Equal(t, http.ErrServerClosed, err)
}

func TestAPIServerRun_TLS(t *testing.T) {
	apiServer := &APIServer{
		Server: &http.Server{
			TLSConfig: &tls.Config{},
		},
		container: restful.NewContainer(),
	}
	patch1 := gomonkey.ApplyFunc(AddToContainer, func(container *restful.Container, client client.Client) error {
		return nil
	})
	var called bool
	patch2 := gomonkey.ApplyMethod(reflect.TypeOf(&http.Server{}), "ListenAndServeTLS",
		func(_ *http.Server, certFile, keyFile string) error {
			called = true
			return http.ErrServerClosed
		})
	defer patch1.Reset()
	defer patch2.Reset()

	err := apiServer.Run(context.Background())
	assert.True(t, called)
	assert.Equal(t, http.ErrServerClosed, err)
}

func TestInitServer_CAReadError(t *testing.T) {
	patch1 := gomonkey.ApplyFunc(tls.LoadX509KeyPair, func(cf string, kf string) (tls.Certificate, error) {
		return tls.Certificate{}, nil
	})
	patch2 := gomonkey.ApplyFunc(os.ReadFile, func(path string) ([]byte, error) {
		return nil, errors.New("read error")
	})
	defer patch1.Reset()
	defer patch2.Reset()

	_, err := initServer(&config.RunConfig{
		Server: &runtime.ServerConfig{
			SecurePort: 1,
			CertFile:   "cert",
			PrivateKey: "key",
			CAFile:     "ca",
		},
	})
	assert.Error(t, err)
}

func TestInitServer_TLSConfigFields(t *testing.T) {
	patch1 := gomonkey.ApplyFunc(tls.LoadX509KeyPair, func(cf string, kf string) (tls.Certificate, error) {
		return tls.Certificate{}, nil
	})
	patch2 := gomonkey.ApplyFunc(os.ReadFile, func(path string) ([]byte, error) {
		return []byte("ca"), nil
	})
	patch3 := gomonkey.ApplyFunc(x509.NewCertPool, func() *x509.CertPool {
		return &x509.CertPool{}
	})
	patch4 := gomonkey.ApplyMethod(reflect.TypeOf(&x509.CertPool{}), "AppendCertsFromPEM",
		func(_ *x509.CertPool, p []byte) (ok bool) {
			return true
		})
	defer patch1.Reset()
	defer patch2.Reset()
	defer patch3.Reset()
	defer patch4.Reset()

	server, err := initServer(&config.RunConfig{
		Server: &runtime.ServerConfig{
			SecurePort: 9443,
			CertFile:   "cert",
			PrivateKey: "key",
			CAFile:     "ca",
		},
	})
	assert.NoError(t, err)
	if assert.NotNil(t, server.TLSConfig) {
		assert.Equal(t, tls.RequireAndVerifyClientCert, server.TLSConfig.ClientAuth)
		assert.EqualValues(t, tls.VersionTLS13, server.TLSConfig.MinVersion)
		assert.NotNil(t, server.TLSConfig.ClientCAs)
	}
}
