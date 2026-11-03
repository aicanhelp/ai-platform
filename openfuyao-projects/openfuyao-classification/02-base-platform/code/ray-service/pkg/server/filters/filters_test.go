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
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	k8srequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/rest"

	"openfuyao-rayservice/pkg/server/request"
)

// recordingHandler records whether it was called and what request info was
// found in the context.
type recordingHandler struct {
	called bool
	info   *request.RequestInfo
}

func (h *recordingHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	h.called = true
	h.info, _ = request.RequestInfoFrom(req.Context())
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("next-handler"))
}

// failingResolver always fails to build request info.
type failingResolver struct{}

func (failingResolver) NewRequestInfo(req *http.Request) (*request.RequestInfo, error) {
	return nil, errors.New("resolve failed")
}

func newTestRequestInfoFactory() request.RequestInfoResolver {
	return &request.RequestInfoFactory{
		RequestInfoFactory: &k8srequest.RequestInfoFactory{
			APIPrefixes:          sets.NewString("api", "apis", "fuyao"),
			GrouplessAPIPrefixes: sets.NewString("api"),
		},
	}
}

func TestBuildRequestInfoResolverError(t *testing.T) {
	next := &recordingHandler{}
	handler := BuildRequestInfo(next, failingResolver{})

	req := httptest.NewRequest(http.MethodGet, "/api/v1/pods", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("ServeHTTP() status = %d, want %d", w.Code, http.StatusInternalServerError)
	}
	if !strings.Contains(w.Body.String(), "RequestInfo create failed") {
		t.Errorf("ServeHTTP() body = %q, want it to contain the resolve error", w.Body.String())
	}
	if next.called {
		t.Errorf("next handler should not be called when the resolver fails")
	}
}

func TestBuildRequestInfoSuccess(t *testing.T) {
	next := &recordingHandler{}
	handler := BuildRequestInfo(next, newTestRequestInfoFactory())

	req := httptest.NewRequest(http.MethodGet, "/api/v1/namespaces/default/pods", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("ServeHTTP() status = %d, want %d", w.Code, http.StatusOK)
	}
	if !next.called {
		t.Fatalf("next handler should be called when the resolver succeeds")
	}
	if next.info == nil {
		t.Fatalf("next handler should find RequestInfo in the request context")
	}
	if next.info.Resource != "pods" || next.info.Namespace != "default" {
		t.Errorf("RequestInfo resource/namespace = %q/%q, want pods/default",
			next.info.Resource, next.info.Namespace)
	}
}

func TestDispatchCluster(t *testing.T) {
	t.Run("no request info in context", func(t *testing.T) {
		next := &recordingHandler{}
		handler := DispatchCluster(next)

		req := httptest.NewRequest(http.MethodGet, "/api/v1/pods", nil)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusInternalServerError {
			t.Errorf("ServeHTTP() status = %d, want %d", w.Code, http.StatusInternalServerError)
		}
		if next.called {
			t.Errorf("next handler should not be called without request info")
		}
	})

	t.Run("request info with empty cluster dispatches to next", func(t *testing.T) {
		next := &recordingHandler{}
		handler := DispatchCluster(next)

		req := httptest.NewRequest(http.MethodGet, "/api/v1/pods", nil)
		ctx := request.WithRequestInfo(req.Context(), &request.RequestInfo{
			RequestInfo: &k8srequest.RequestInfo{Path: "/api/v1/pods", Verb: "GET"},
		})
		req = req.WithContext(ctx)

		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("ServeHTTP() status = %d, want %d", w.Code, http.StatusOK)
		}
		if !next.called {
			t.Errorf("next handler should be called for empty cluster info")
		}
	})

	t.Run("request info with cluster name dispatches to next", func(t *testing.T) {
		next := &recordingHandler{}
		handler := DispatchCluster(next)

		req := httptest.NewRequest(http.MethodGet, "/api/v1/pods", nil)
		ctx := request.WithRequestInfo(req.Context(), &request.RequestInfo{
			RequestInfo: &k8srequest.RequestInfo{Path: "/api/v1/pods", Verb: "GET"},
			Cluster:     "guest",
		})
		req = req.WithContext(ctx)

		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("ServeHTTP() status = %d, want %d", w.Code, http.StatusOK)
		}
		if !next.called {
			t.Errorf("next handler should be called for cluster request")
		}
	})
}

func TestProxyAPIServerInvalidHost(t *testing.T) {
	next := &recordingHandler{}
	handler := ProxyAPIServer(next, &rest.Config{Host: "://invalid-host"})

	if _, ok := handler.(*recordingHandler); !ok {
		t.Errorf("ProxyAPIServer() should return the original handler when the host url is invalid")
	}
}

func TestProxyAPIServer(t *testing.T) {
	var proxiedPath string
	var proxiedAuthHeader string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxiedPath = r.URL.Path
		proxiedAuthHeader = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("backend-ok"))
	}))
	defer backend.Close()

	next := &recordingHandler{}
	config := &rest.Config{Host: backend.URL}
	handler := ProxyAPIServer(next, config)

	proxy, ok := handler.(*apiServerProxy)
	if !ok {
		t.Fatalf("ProxyAPIServer() should return an apiServerProxy for a valid host")
	}
	if proxy.kubeUrl.Host == "" {
		t.Errorf("apiServerProxy kubeUrl host should be set")
	}

	t.Run("no request info in context", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/kubernetes/api/v1/pods", nil)
		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusInternalServerError {
			t.Errorf("ServeHTTP() status = %d, want %d", w.Code, http.StatusInternalServerError)
		}
		if !next.called {
			t.Errorf("next handler should still be called without request info")
		}
	})

	t.Run("non k8s request goes to next handler", func(t *testing.T) {
		next.called = false
		req := httptest.NewRequest(http.MethodGet, "/rest/v1/overview/all", nil)
		ctx := request.WithRequestInfo(req.Context(), &request.RequestInfo{
			RequestInfo: &k8srequest.RequestInfo{Path: "/rest/v1/overview/all", Verb: "GET"},
		})
		req = req.WithContext(ctx)

		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if !next.called {
			t.Errorf("next handler should be called for non k8s request")
		}
		if proxiedPath != "" {
			t.Errorf("request should not be proxied to the k8s apiserver")
		}
	})

	t.Run("k8s request is proxied to apiserver", func(t *testing.T) {
		next.called = false
		req := httptest.NewRequest(http.MethodGet, "/api/kubernetes/api/v1/namespaces/default/pods", nil)
		req.Header.Set("Authorization", "Bearer original-token")
		ctx := request.WithRequestInfo(req.Context(), &request.RequestInfo{
			RequestInfo:  &k8srequest.RequestInfo{Path: "/api/v1/namespaces/default/pods", Verb: "GET"},
			IsK8sRequest: true,
		})
		req = req.WithContext(ctx)

		w := httptest.NewRecorder()
		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("ServeHTTP() status = %d, want %d, body: %s", w.Code, http.StatusOK, w.Body.String())
		}
		if !strings.Contains(w.Body.String(), "backend-ok") {
			t.Errorf("ServeHTTP() body = %q, want backend response", w.Body.String())
		}
		if proxiedPath != "/api/v1/namespaces/default/pods" {
			t.Errorf("proxied path = %q, want %q", proxiedPath, "/api/v1/namespaces/default/pods")
		}
		if proxiedAuthHeader != "" {
			t.Errorf("Authorization header should be removed before proxying, got %q", proxiedAuthHeader)
		}
		if next.called {
			t.Errorf("next handler should not be called for k8s request")
		}
	})
}

func TestResponderError(t *testing.T) {
	r := &responder{}
	req := httptest.NewRequest(http.MethodGet, "/api/v1/pods", nil)
	w := httptest.NewRecorder()

	r.Error(w, req, errors.New("proxy boom"))

	if w.Code != http.StatusInternalServerError {
		t.Errorf("Error() status = %d, want %d", w.Code, http.StatusInternalServerError)
	}
	if !strings.Contains(w.Body.String(), "proxy boom") {
		t.Errorf("Error() body = %q, want it to contain the error message", w.Body.String())
	}
}
