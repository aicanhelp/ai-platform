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

	"console-service/pkg/auth"
	"console-service/pkg/constant"
	"console-service/pkg/utils/authutil"
)

func TestRequestAuthCheckerGetTokenFromOpenFuyaoAuthHeader(t *testing.T) {
	tests := []struct {
		name    string
		header  string
		want    string
		wantErr bool
	}{
		{
			name:    "missing_bearer_prefix",
			header:  "invalid_token",
			want:    "",
			wantErr: true,
		},
		{
			name:    "empty_header",
			header:  "",
			want:    "",
			wantErr: true,
		},
		{
			name:    "bearer_without_token",
			header:  "Bearer ",
			want:    "",
			wantErr: true,
		},
		{
			name:    "invalid_jwt_token",
			header:  "Bearer invalid.jwt.token",
			want:    "",
			wantErr: true,
		},
		{
			name:    "valid_bearer_prefix_with_valid_token",
			header:  "Bearer " + authutil.GenerateToken("test"),
			want:    authutil.GenerateToken("test"),
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rc := &requestAuthChecker{}
			req := &http.Request{
				Header: map[string][]string{},
			}
			if tt.header != "" {
				req.Header.Set(constant.OpenFuyaoAuthHeader, tt.header)
			}

			got, err := rc.getTokenFromOpenFuyaoAuthHeader(req)
			if (err != nil) != tt.wantErr {
				t.Errorf("requestAuthChecker.getTokenFromOpenFuyaoAuthHeader() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("requestAuthChecker.getTokenFromOpenFuyaoAuthHeader() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAuthFailureSetsReferrerCookieForFuyaoApi(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	rc := &requestAuthChecker{
		authHttpHandler:    next,
		nonAuthHttpHandler: next,
		authHandler:        &auth.Handler{},
	}

	refererURL := "http://example.com/container_platform/workload/pods"
	req := httptest.NewRequest("GET", "http://example.com/rest/alert/v1/test", nil)
	req.Header.Set("Referer", refererURL)

	recorder := httptest.NewRecorder()
	rc.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusUnauthorized {
		t.Errorf("ServeHTTP() status = %v, want %v", recorder.Code, http.StatusUnauthorized)
	}

	var referrerCookie *http.Cookie
	for _, cookie := range recorder.Result().Cookies() {
		if cookie.Name == constant.CookieNameReferrer {
			referrerCookie = cookie
			break
		}
	}
	if referrerCookie == nil {
		t.Fatal("ServeHTTP() did not set referrer cookie for Fuyao API request")
	}
	if referrerCookie.Value != refererURL {
		t.Errorf("ServeHTTP() referrer cookie = %v, want %v", referrerCookie.Value, refererURL)
	}
}

func TestAuthFailurePreservesExistingReferrerOnOAuthBackNavigation(t *testing.T) {
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	rc := &requestAuthChecker{
		authHttpHandler:    next,
		nonAuthHttpHandler: next,
		authHandler:        &auth.Handler{},
	}

	originalPage := "http://example.com/container_platform/workload/pods"
	oauthLoginPage := "http://example.com/oauth2/auth/login/fuyaoPasswordProvider?state=abc"

	req := httptest.NewRequest("GET", "http://example.com/rest/alert/v1/test", nil)
	req.Header.Set("Referer", oauthLoginPage)
	req.AddCookie(&http.Cookie{Name: constant.CookieNameReferrer, Value: originalPage})

	recorder := httptest.NewRecorder()
	rc.ServeHTTP(recorder, req)

	var referrerCookie *http.Cookie
	for _, cookie := range recorder.Result().Cookies() {
		if cookie.Name == constant.CookieNameReferrer {
			referrerCookie = cookie
			break
		}
	}
	if referrerCookie != nil {
		t.Fatalf("ServeHTTP() should not overwrite referrer cookie, got %v", referrerCookie.Value)
	}
}

func TestRequestPathMatchers(t *testing.T) {
	tests := []struct {
		name     string
		fn       func(string) bool
		path     string
		expected bool
	}{
		{
			name:     "ws_auth_request",
			fn:       isWSAuthRequest,
			path:     "/ws/auth/login",
			expected: true,
		},
		{
			name:     "kube_api_request",
			fn:       isKubeApiRequest,
			path:     "/api/kubernetes/api/v1/pods",
			expected: true,
		},
		{
			name:     "non_kube_api_request",
			fn:       isKubeApiRequest,
			path:     "/rest/console/v1beta1/pods",
			expected: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.fn(tt.path); got != tt.expected {
				t.Errorf("%s(%q) = %v, want %v", tt.name, tt.path, got, tt.expected)
			}
		})
	}
}
