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

package filters

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"k8s.io/client-go/rest"

	"console-service/pkg/plugin"
)

const (
	customHttpsPort = 8443
)

func createTestCPProxy(t *testing.T) *consolePluginProxy {
	proxy, ok := ProxyConsolePlugin(testHandler, &rest.Config{}).(*consolePluginProxy)
	if !ok {
		t.Fatalf("Failed to create test consoleplugin proxy")
		return nil
	}

	backendUrl := createTestBackend(t)
	proxy.multiClusterProxy = httputil.NewSingleHostReverseProxy(backendUrl)

	return proxy
}

func TestCPProxyServeNonPluginReq(t *testing.T) {
	proxy := createTestCPProxy(t)

	tests := []struct {
		name           string
		context        context.Context
		expectedStatus int
	}{
		{
			"TestWithoutReqInfo",
			context.Background(),
			http.StatusInternalServerError,
		},
		{
			"TestNonPluginReq",
			singleClusterCtx,
			http.StatusTeapot,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			req := httptest.NewRequest("GET", "/", nil).WithContext(tt.context)
			proxy.ServeHTTP(recorder, req)
			resp := recorder.Result()
			defer resp.Body.Close()
			if resp.StatusCode != tt.expectedStatus {
				t.Errorf("Expected status code %d, but got %d", tt.expectedStatus, resp.StatusCode)
			}
		})
	}
}

func TestCPProxyServePluginReq(t *testing.T) {
	proxy := createTestCPProxy(t)
	tests := []struct {
		name           string
		req            *http.Request
		expectedStatus int
		patch          bool
	}{
		{
			name:           "TestMjsReqInvalid",
			req:            httptest.NewRequest("GET", "/proxy", nil).WithContext(singleClusterCtx),
			expectedStatus: http.StatusBadRequest,
			patch:          false,
		},
		{
			name:           "TestPluginReqFail",
			req:            httptest.NewRequest("GET", "/rest/multicluster/v1beta1/resources/clusters", nil).WithContext(multiClusterCtx),
			expectedStatus: http.StatusOK,
			patch:          false,
		},
		{
			name:           "TestMjsReqMultiCluster",
			req:            httptest.NewRequest("GET", "/proxy/multicluster", nil).WithContext(multiClusterCtx),
			expectedStatus: http.StatusOK,
			patch:          true,
		},
		{
			name:           "TestMjsReqSingleCluster",
			req:            httptest.NewRequest("GET", "/proxy/test", nil).WithContext(singleClusterCtx),
			expectedStatus: http.StatusOK,
			patch:          true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			if tt.patch {
				patchRequestConsolePlugin(t)
				patchProxySingleClusterConsolePluginRequest(t)
			}
			proxy.ServeHTTP(recorder, tt.req)
			resp := recorder.Result()
			if resp.StatusCode != tt.expectedStatus {
				t.Errorf("Expected status code %d, but got %d", tt.expectedStatus, resp.StatusCode)
			}
			defer resp.Body.Close()
		})
	}
}

func patchProxySingleClusterConsolePluginRequest(t *testing.T) {
	patches := gomonkey.ApplyFunc(proxySingleClusterConsolePluginRequest, func(_ string, cp *plugin.ConsolePlugin,
		w http.ResponseWriter, r *http.Request) {
		backendUrl := createTestBackend(t)
		reverseProxy := httputil.NewSingleHostReverseProxy(backendUrl)
		reverseProxy.ServeHTTP(w, r)
	})
	t.Cleanup(func() {
		patches.Reset()
	})
}

// Helper function to create test ConsolePlugin
func createTestConsolePlugin(namespace, name, scheme string, port int32,
	basePath string) *plugin.ConsolePlugin {
	return &plugin.ConsolePlugin{
		Spec: plugin.ConsolePluginSpec{
			Backend: &plugin.ConsolePluginBackend{
				Type: plugin.ServiceBackendType,
				Service: &plugin.ConsolePluginService{
					Name:      name,
					Namespace: namespace,
					BasePath:  basePath,
					Scheme:    scheme,
					Port:      port,
				},
			},
		},
	}
}

type testCaseArgs struct {
	cp *plugin.ConsolePlugin
}

func TestGetPluginApiServerProxyPathPrefix(t *testing.T) {
	tests := getPluginApiServerProxyPathPrefixTests()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getPluginApiServerProxyPathPrefix(tt.args.cp); got != tt.want {
				t.Errorf("getPluginApiServerProxyPathPrefix() = %v, want %v", got, tt.want)
			}
		})
	}
}

func getPluginApiServerProxyPathPrefixTests() []struct {
	args testCaseArgs
	name string
	want string
} {
	return []struct {
		args testCaseArgs
		name string
		want string
	}{
		{
			name: "HTTP service with default port", args: testCaseArgs{
				cp: createTestConsolePlugin("test-ns", "test-service", "http", 0, "/api"),
			},
			want: "/api/v1/namespaces/test-ns/services/test-service:80/proxy/api",
		},
		{
			name: "HTTPS service with custom port", args: testCaseArgs{
				cp: createTestConsolePlugin("secure-ns", "secure-service", "https", customHttpsPort, "/secure"),
			},
			want: "/api/v1/namespaces/secure-ns/services/https:secure-service:8443/proxy/secure",
		},
		{
			name: "HTTP service with custom port",
			args: testCaseArgs{
				cp: createTestConsolePlugin("custom-ns", "custom-service", "http", 8080, ""),
			},
			want: "/api/v1/namespaces/custom-ns/services/custom-service:8080/proxy",
		},
		{
			name: "HTTPS service with default port", args: testCaseArgs{
				cp: createTestConsolePlugin("https-ns", "default-https", "https", 0, "/"),
			},
			want: "/api/v1/namespaces/https-ns/services/https:default-https:443/proxy",
		},
		{
			name: "Non-service backend type",
			args: testCaseArgs{
				cp: &plugin.ConsolePlugin{
					Spec: plugin.ConsolePluginSpec{
						Backend: &plugin.ConsolePluginBackend{
							Type: "OtherType",
						},
					},
				},
			},
			want: "",
		},
	}
}

func TestGetSingleClusterConsolePluginProxyUrl(t *testing.T) {
	tests := getSingleClusterConsolePluginProxyUrlTests()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getSingleClusterConsolePluginProxyUrl(tt.args.cp); got != tt.want {
				t.Errorf("getSingleClusterConsolePluginProxyUrl() = %v, want %v", got, tt.want)
			}
		})
	}
}

func getSingleClusterConsolePluginProxyUrlTests() []struct {
	name string
	args testCaseArgs
	want string
} {
	return []struct {
		name string
		args testCaseArgs
		want string
	}{
		{
			name: "HTTP service with default port", args: testCaseArgs{
				cp: createTestConsolePlugin("test-ns", "test-service", "http", 0, "/api"),
			},
			want: "http://test-service.test-ns.svc:80/api",
		},
		{
			name: "HTTPS service with custom port", args: testCaseArgs{
				cp: createTestConsolePlugin("secure-ns", "secure-service", "https", customHttpsPort, "/secure"),
			},
			want: "https://secure-service.secure-ns.svc:8443/secure",
		},
		{
			name: "HTTP service with custom port",
			args: testCaseArgs{
				cp: createTestConsolePlugin("custom-ns", "custom-service", "http", 8080, ""),
			},
			want: "http://custom-service.custom-ns.svc:8080",
		},
		{
			name: "HTTPS service with default port", args: testCaseArgs{
				cp: createTestConsolePlugin("https-ns", "default-https", "https", 0, "/"),
			},
			want: "https://default-https.https-ns.svc:443/",
		},
		{
			name: "Non-service backend type", want: "",
			args: testCaseArgs{
				cp: &plugin.ConsolePlugin{
					Spec: plugin.ConsolePluginSpec{
						Backend: &plugin.ConsolePluginBackend{
							Type: "OtherType",
						},
					},
				},
			},
		},
	}
}

func TestProxySingleClusterConsolePluginRequest(t *testing.T) {
	backendServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter,
		r *http.Request) {
		fmt.Fprintf(w, "Hello from backend: %s", r.URL.Path)
	}))
	defer backendServer.Close()

	tests := getProxySingleClusterConsolePluginRequestTests(backendServer.URL)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proxySingleClusterConsolePluginRequest(tt.args.url, tt.args.cp, tt.args.w, tt.args.r)
		})
	}
}

func getProxySingleClusterConsolePluginRequestTests(backendServerURL string) []struct {
	name string
	args struct {
		url string
		cp  *plugin.ConsolePlugin
		w   http.ResponseWriter
		r   *http.Request
	}
} {
	return []struct {
		name string
		args struct {
			url string
			cp  *plugin.ConsolePlugin
			w   http.ResponseWriter
			r   *http.Request
		}
	}{
		{
			name: "Valid service backend proxy request",
			args: struct {
				url string
				cp  *plugin.ConsolePlugin
				w   http.ResponseWriter
				r   *http.Request
			}{
				cp:  createTestConsolePlugin("test-ns", "test-service", "http", 0, "/api"),
				url: backendServerURL, w: httptest.NewRecorder(), r: httptest.NewRequest("GET", "/test", nil),
			},
		},
		{
			name: "Invalid backend type",
			args: struct {
				url string
				cp  *plugin.ConsolePlugin
				w   http.ResponseWriter
				r   *http.Request
			}{
				cp: &plugin.ConsolePlugin{
					Spec: plugin.ConsolePluginSpec{
						Backend: &plugin.ConsolePluginBackend{
							Type: "InvalidType",
						},
					},
				},
				url: "http://example.com", w: httptest.NewRecorder(), r: httptest.NewRequest("GET", "/test", nil),
			},
		},
	}
}
