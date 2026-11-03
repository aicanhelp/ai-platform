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
	"net/http"
	"net/http/httptest"
	"testing"

	reqpkg "installer-service/pkg/server/request"
)

func TestDispatchCluster(t *testing.T) {
	// 创建一个简单的 handler
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	// 调用 DispatchCluster 函数
	clusterHandler := DispatchCluster(handler)

	// 检查返回的 clusterHandler 是否不为 nil
	if clusterHandler == nil {
		t.Error("Expected clusterHandler to be non-nil")
	}
}

func TestServeHTTPNoRequestInfo(t *testing.T) {
	called := false
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	})

	cd := DispatchCluster(handler)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/foo", nil)

	cd.ServeHTTP(rr, req)

	if called {
		t.Fatalf("next handler should not be called when RequestInfo missing")
	}
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d", rr.Code)
	}
}

func TestServeHTTPWithAndWithoutCluster(t *testing.T) {
	tests := []struct {
		name       string
		cluster    string
		wantCalled bool
	}{
		{name: "empty-cluster", cluster: "", wantCalled: true},
		{name: "with-cluster", cluster: "c1", wantCalled: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			called := false
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				called = true
				w.WriteHeader(http.StatusOK)
			})

			cd := DispatchCluster(handler)
			rr := httptest.NewRecorder()
			req := httptest.NewRequest("GET", "/api/clusters/"+tc.cluster+"/x", nil)
			// attach RequestInfo in context
			info := &reqpkg.RequestInfo{Cluster: tc.cluster}
			ctx := req.Context()
			ctx = reqpkg.WithRequestInfo(ctx, info)
			req = req.WithContext(ctx)

			cd.ServeHTTP(rr, req)
			if called != tc.wantCalled {
				t.Fatalf("handler called = %v, want %v", called, tc.wantCalled)
			}
			if rr.Code != http.StatusOK {
				t.Fatalf("expected 200, got %d", rr.Code)
			}
		})
	}
}
