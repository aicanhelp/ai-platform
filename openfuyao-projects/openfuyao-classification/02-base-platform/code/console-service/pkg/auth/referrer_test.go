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

package auth

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"console-service/pkg/constant"
)

func TestIsValidReferrerURL(t *testing.T) {
	tests := []struct {
		name  string
		url   string
		valid bool
	}{
		{
			name:  "console_page",
			url:   "http://example.com/container_platform/workload/pods",
			valid: true,
		},
		{
			name:  "console_login",
			url:   "http://example.com/rest/auth/login",
			valid: false,
		},
		{
			name:  "oauth_authorize",
			url:   "http://example.com/oauth2/oauth/authorize?client_id=abc",
			valid: false,
		},
		{
			name:  "oauth_password_login",
			url:   "http://example.com/oauth2/auth/login/fuyaoPasswordProvider?state=abc",
			valid: false,
		},
		{
			name:  "password_reset",
			url:   "http://example.com/password/reset",
			valid: false,
		},
		{
			name:  "empty",
			url:   "",
			valid: false,
		},
		{
			name:  "invalid_url",
			url:   "://invalid",
			valid: false,
		},
		{
			name:  "password_path",
			url:   "http://example.com/password",
			valid: false,
		},
		{
			name:  "ws_auth",
			url:   "http://example.com/ws/auth/heartbeat",
			valid: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsValidReferrerURL(tt.url); got != tt.valid {
				t.Errorf("IsValidReferrerURL(%q) = %v, want %v", tt.url, got, tt.valid)
			}
		})
	}
}

func TestShouldSkipReferrerUpdate(t *testing.T) {
	originalPage := "http://example.com/container_platform/workload/pods"
	oauthLoginPage := "http://example.com/oauth2/auth/login/fuyaoPasswordProvider?state=abc"

	req := httptest.NewRequest("GET", oauthLoginPage, nil)
	req.AddCookie(&http.Cookie{Name: constant.CookieNameReferrer, Value: originalPage})

	if !shouldSkipReferrerUpdate(req, oauthLoginPage) {
		t.Error("shouldSkipReferrerUpdate() should skip when new URL is login page and valid cookie exists")
	}

	reqWithoutCookie := httptest.NewRequest("GET", oauthLoginPage, nil)
	if !shouldSkipReferrerUpdate(reqWithoutCookie, oauthLoginPage) {
		t.Error("shouldSkipReferrerUpdate() should skip invalid login URL even without existing cookie")
	}

	reqWithoutCookie = httptest.NewRequest("GET", "http://example.com/container_platform/workload/deployments", nil)
	if shouldSkipReferrerUpdate(reqWithoutCookie, "http://example.com/container_platform/workload/deployments") {
		t.Error("shouldSkipReferrerUpdate() should not skip when setting a new valid referrer")
	}
}

func TestSetReferrerCookie(t *testing.T) {
	tests := []struct {
		name        string
		originalURL string
		wantCookie  bool
		wantValue   string
	}{
		{
			name:        "set_cookie_with_valid_url",
			originalURL: "/container_platform/workload/pods",
			wantCookie:  true,
			wantValue:   "/container_platform/workload/pods",
		},
		{
			name:        "set_cookie_with_full_url",
			originalURL: "http://example.com/container_platform/workload/pods?param=value",
			wantCookie:  true,
			wantValue:   "http://example.com/container_platform/workload/pods?param=value",
		},
		{
			name:        "set_cookie_with_empty_url",
			originalURL: "",
			wantCookie:  true,
			wantValue:   "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			SetReferrerCookie(recorder, tt.originalURL)

			if !tt.wantCookie {
				return
			}
			cookies := recorder.Result().Cookies()
			if len(cookies) == 0 {
				t.Errorf("SetReferrerCookie() did not set any cookie")
				return
			}
			cookie := findReferrerCookie(cookies)
			if cookie == nil {
				t.Errorf("SetReferrerCookie() did not set cookie with name %s", constant.CookieNameReferrer)
				return
			}
			assertReferrerCookieAttrs(t, cookie, tt.wantValue)
		})
	}
}

// findReferrerCookie returns the referrer cookie from the given cookie list,
// or nil when no referrer cookie is present.
func findReferrerCookie(cookies []*http.Cookie) *http.Cookie {
	for _, cookie := range cookies {
		if cookie.Name == constant.CookieNameReferrer {
			return cookie
		}
	}
	return nil
}

// assertReferrerCookieAttrs asserts all attribute invariants of the referrer cookie.
func assertReferrerCookieAttrs(t *testing.T, cookie *http.Cookie, wantValue string) {
	t.Helper()
	if cookie.Value != wantValue {
		t.Errorf("SetReferrerCookie() cookie value = %v, want %v", cookie.Value, wantValue)
	}
	if cookie.Path != "/" {
		t.Errorf("SetReferrerCookie() cookie path = %v, want /", cookie.Path)
	}
	if !cookie.HttpOnly {
		t.Errorf("SetReferrerCookie() cookie should be HttpOnly")
	}
	if !cookie.Secure {
		t.Errorf("SetReferrerCookie() cookie should be Secure")
	}
	if cookie.SameSite != http.SameSiteLaxMode {
		t.Errorf("SetReferrerCookie() cookie SameSite = %v, want Lax", cookie.SameSite)
	}
	expectedExpiry := time.Now().Add(referrerCookieExpireSeconds * time.Second)
	if cookie.Expires.Sub(expectedExpiry) > time.Minute {
		t.Errorf("SetReferrerCookie() cookie expiry = %v, want approximately %v", cookie.Expires, expectedExpiry)
	}
}

func TestGetReferrerURL(t *testing.T) {
	baseURL := "http://example.com"
	tests := []struct {
		name          string
		requestURL    string
		refererHeader string
		want          string
	}{
		{
			name:          "use_request_url_when_no_referer",
			requestURL:    "/container_platform/workload/pods",
			refererHeader: "",
			want:          baseURL + "/container_platform/workload/pods",
		},
		{
			name:          "use_referer_header_when_present",
			requestURL:    "/rest/console/v1beta1/pods",
			refererHeader: "http://example.com/container_platform/workload/pods",
			want:          "http://example.com/container_platform/workload/pods",
		},
		{
			name:          "skip_login_referer_and_use_valid_request_url",
			requestURL:    "/container_platform/workload/pods",
			refererHeader: "http://example.com/oauth2/auth/login/fuyaoPasswordProvider?state=abc",
			want:          baseURL + "/container_platform/workload/pods",
		},
		{
			name:          "empty_when_both_login_related",
			requestURL:    "/oauth2/oauth/authorize",
			refererHeader: "http://example.com/oauth2/auth/login/fuyaoPasswordProvider",
			want:          "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", baseURL+tt.requestURL, nil)
			if tt.refererHeader != "" {
				req.Header.Set("Referer", tt.refererHeader)
			}

			got := GetReferrerURL(req)
			if got != tt.want {
				t.Errorf("GetReferrerURL() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResolvePostLoginRedirectURL(t *testing.T) {
	defaultURL := "/"
	validReferrer := "/container_platform/workload/pods"
	loginReferrer := "http://example.com/oauth2/auth/login/fuyaoPasswordProvider?state=abc"

	t.Run("use_valid_referrer_and_clear_cookie", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/", nil)
		req.AddCookie(&http.Cookie{Name: constant.CookieNameReferrer, Value: validReferrer})
		recorder := httptest.NewRecorder()

		got := ResolvePostLoginRedirectURL(req, recorder, defaultURL)
		if got != validReferrer {
			t.Errorf("ResolvePostLoginRedirectURL() = %v, want %v", got, validReferrer)
		}
		for _, cookie := range recorder.Result().Cookies() {
			if cookie.Name == constant.CookieNameReferrer && cookie.Value != "" {
				t.Errorf("ResolvePostLoginRedirectURL() should clear referrer cookie, got %v", cookie.Value)
			}
		}
	})

	t.Run("fallback_without_cookie", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/", nil)
		recorder := httptest.NewRecorder()

		got := ResolvePostLoginRedirectURL(req, recorder, defaultURL)
		if got != defaultURL {
			t.Errorf("ResolvePostLoginRedirectURL() = %v, want %v", got, defaultURL)
		}
	})

	t.Run("fallback_with_login_referrer", func(t *testing.T) {
		req := httptest.NewRequest("GET", "/", nil)
		req.AddCookie(&http.Cookie{Name: constant.CookieNameReferrer, Value: loginReferrer})
		recorder := httptest.NewRecorder()

		got := ResolvePostLoginRedirectURL(req, recorder, defaultURL)
		if got != defaultURL {
			t.Errorf("ResolvePostLoginRedirectURL() = %v, want %v", got, defaultURL)
		}
	})
}

func TestSaveReferrerCookieIfNeeded(t *testing.T) {
	t.Run("set_cookie_for_valid_page", func(t *testing.T) {
		req := httptest.NewRequest("GET", "http://example.com/container_platform/workload/pods", nil)
		recorder := httptest.NewRecorder()

		SaveReferrerCookieIfNeeded(recorder, req)

		cookies := recorder.Result().Cookies()
		if len(cookies) == 0 {
			t.Fatal("SaveReferrerCookieIfNeeded() should set referrer cookie")
		}
		if cookies[0].Name != constant.CookieNameReferrer {
			t.Errorf("cookie name = %q, want %q", cookies[0].Name, constant.CookieNameReferrer)
		}
	})

	t.Run("skip_when_login_related", func(t *testing.T) {
		req := httptest.NewRequest("GET", "http://example.com/rest/auth/login", nil)
		recorder := httptest.NewRecorder()

		SaveReferrerCookieIfNeeded(recorder, req)

		if len(recorder.Result().Cookies()) != 0 {
			t.Error("SaveReferrerCookieIfNeeded() should not set cookie for login page")
		}
	})

	t.Run("skip_when_valid_cookie_exists", func(t *testing.T) {
		originalPage := "http://example.com/container_platform/workload/pods"
		oauthLoginPage := "http://example.com/oauth2/auth/login/fuyaoPasswordProvider?state=abc"
		req := httptest.NewRequest("GET", oauthLoginPage, nil)
		req.AddCookie(&http.Cookie{Name: constant.CookieNameReferrer, Value: originalPage})
		recorder := httptest.NewRecorder()

		SaveReferrerCookieIfNeeded(recorder, req)

		if len(recorder.Result().Cookies()) != 0 {
			t.Error("SaveReferrerCookieIfNeeded() should skip when valid referrer cookie exists")
		}
	})
}

func TestClearReferrerCookie(t *testing.T) {
	recorder := httptest.NewRecorder()
	ClearReferrerCookie(recorder)

	cookies := recorder.Result().Cookies()
	if len(cookies) == 0 {
		t.Fatal("ClearReferrerCookie() should set clearing cookie")
	}
	for _, cookie := range cookies {
		if cookie.Name == constant.CookieNameReferrer && cookie.Value != "" {
			t.Errorf("ClearReferrerCookie() value = %q, want empty", cookie.Value)
		}
	}
}
