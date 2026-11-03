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

package filters

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"k8s.io/client-go/rest"

	"installer-service/pkg/server/request"
)

// 测试无效 Kubernetes 主机 URL 的情况
func TestProxyAPIServerInvalidKubeHost(t *testing.T) {
	// 创建模拟的下一个处理器
	nextHandlerCalled := false
	nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		nextHandlerCalled = true
		w.WriteHeader(http.StatusOK)
	})

	// 创建包含无效主机 URL 的配置
	config := &rest.Config{
		Host: "::invalid::url::", // 无效的 URL
	}

	// 创建代理处理器
	proxyHandler := ProxyAPIServer(nextHandler, config)

	// 创建测试请求和响应记录器
	req := httptest.NewRequest("GET", "/api/v1/pods", nil)
	rec := httptest.NewRecorder()

	// 发送请求
	proxyHandler.ServeHTTP(rec, req)

	// 验证：应该调用下一个处理器
	if !nextHandlerCalled {
		t.Error("Next handler should have been called for invalid host URL")
	}
}

// 测试创建传输层失败的情况
func TestProxyAPIServerTransportCreationFailure(t *testing.T) {
	// 创建模拟的下一个处理器
	nextHandlerCalled := false
	nextHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		nextHandlerCalled = true
		w.WriteHeader(http.StatusOK)
	})

	// 创建会导致 TransportFor 失败的配置
	config := &rest.Config{
		Host: "http://valid-url-but-invalid-transport",
		// 添加无效的 TLS 配置会导致 TransportFor 失败
		TLSClientConfig: rest.TLSClientConfig{
			CertFile: "/path/to/non-existent/cert",
			KeyFile:  "/path/to/non-existent/key",
		},
	}

	// 创建代理处理器
	proxyHandler := ProxyAPIServer(nextHandler, config)

	// 创建测试请求和响应记录器
	req := httptest.NewRequest("GET", "/api/v1/pods", nil)
	rec := httptest.NewRecorder()

	// 发送请求
	proxyHandler.ServeHTTP(rec, req)

	// 验证：应该调用下一个处理器
	if !nextHandlerCalled {
		t.Error("Next handler should have been called when transport creation fails")
	}
}

func TestApiServerProxyServeHTTP(t *testing.T) {
	mockServer := createMockAPIServer(t)
	defer mockServer.Close()

	kubeUrl, ok := url.Parse(mockServer.URL)
	if ok != nil {
		t.Errorf("donot parse correctly")
	}

	tests := []struct {
		name           string
		requestPath    string
		isK8sRequest   bool
		hasRequestInfo bool
		wantStatus     int
	}{
		{"Missing RequestInfo", "/", false, false, http.StatusInternalServerError},
		{"Kubernetes API request", "/api/kubernetes/pods", true, true, http.StatusOK},
		{"Non-Kubernetes request", "/healthz", false, true, http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := createTestRequest(tt.requestPath, tt.hasRequestInfo, tt.isK8sRequest)
			rec := httptest.NewRecorder()
			proxy := createProxy(kubeUrl, mockServer.Client().Transport)

			proxy.ServeHTTP(rec, req)

			if rec.Code != tt.wantStatus {
				t.Errorf("Want status %d, got %d", tt.wantStatus, rec.Code)
			}
		})
	}
}

// 创建模拟 API 服务器
func createMockAPIServer(t *testing.T) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/pods" {
			t.Errorf("Want path '/pods', got '%s'", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "" {
			t.Error("Authorization header should be deleted")
		}
		w.WriteHeader(http.StatusOK)
	}))
}

// 创建测试请求
func createTestRequest(path string, hasInfo, isK8s bool) *http.Request {
	req := httptest.NewRequest("GET", "http://example.com"+path, nil)
	req.Header.Set("Authorization", "Bearer token")

	if hasInfo {
		ctx := request.WithRequestInfo(req.Context(), &request.RequestInfo{
			IsK8sRequest: isK8s,
		})
		req = req.WithContext(ctx)
	}
	return req
}

// 创建代理实例
func createProxy(kubeUrl *url.URL, transport http.RoundTripper) *apiServerProxy {
	return &apiServerProxy{
		nextHandler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}),
		kubeUrl:      kubeUrl,
		roundTripper: transport,
	}
}
