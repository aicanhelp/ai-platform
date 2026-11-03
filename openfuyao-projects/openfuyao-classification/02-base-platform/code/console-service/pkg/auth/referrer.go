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
	"net/url"
	"strings"
	"time"

	"console-service/pkg/constant"
)

const referrerCookieExpireSeconds = 7200

var loginRelatedPathPrefixes = []string{
	"/rest/auth/",
	"/ws/auth/",
	"/oauth2/oauth/authorize",
	"/oauth2/auth/login",
	"/oauth2/oauth/token",
	"/password/",
}

var loginRelatedPathContains = []string{
	"fuyaoPasswordProvider",
}

// IsValidReferrerURL reports whether urlStr is a safe post-login redirect target
// (not a login or OAuth intermediate page).
func IsValidReferrerURL(urlStr string) bool {
	if urlStr == "" {
		return false
	}
	u, err := url.Parse(urlStr)
	if err != nil {
		return false
	}
	return !isLoginRelatedPath(u.Path)
}

func isLoginRelatedPath(pathname string) bool {
	if pathname == "/password" {
		return true
	}
	for _, prefix := range loginRelatedPathPrefixes {
		if strings.HasPrefix(pathname, prefix) {
			return true
		}
	}
	for _, part := range loginRelatedPathContains {
		if strings.Contains(pathname, part) {
			return true
		}
	}
	return false
}

// GetReferrerURL returns the page URL to save before OAuth redirect.
// For API requests (XHR/Fetch), the Referer header contains the original page URL.
func GetReferrerURL(req *http.Request) string {
	if referer := req.Header.Get("Referer"); referer != "" {
		if IsValidReferrerURL(referer) {
			return referer
		}
	}
	if IsValidReferrerURL(req.URL.String()) {
		return req.URL.String()
	}
	return ""
}

// SaveReferrerCookieIfNeeded saves the original page URL unless the URL is login-related
// or a valid referrer cookie is already present (e.g. browser back during OAuth flow).
func SaveReferrerCookieIfNeeded(w http.ResponseWriter, req *http.Request) {
	referrerURL := GetReferrerURL(req)
	if referrerURL == "" || shouldSkipReferrerUpdate(req, referrerURL) {
		return
	}
	SetReferrerCookie(w, referrerURL)
}

func shouldSkipReferrerUpdate(req *http.Request, newURL string) bool {
	if !IsValidReferrerURL(newURL) {
		return true
	}
	existing, err := req.Cookie(constant.CookieNameReferrer)
	if err != nil || existing.Value == "" {
		return false
	}
	return IsValidReferrerURL(existing.Value)
}

// SetReferrerCookie sets a cookie to save the original request URL before OAuth redirect.
func SetReferrerCookie(w http.ResponseWriter, originalURL string) {
	cookie := http.Cookie{}
	applyReferrerCookieAttributes(&cookie, originalURL, time.Now().Add(referrerCookieExpireSeconds*time.Second))
	http.SetCookie(w, &cookie)
}

// ClearReferrerCookie removes the referrer cookie from the response.
func ClearReferrerCookie(w http.ResponseWriter) {
	cookie := http.Cookie{}
	applyReferrerCookieAttributes(&cookie, "", time.Unix(1, 0))
	http.SetCookie(w, &cookie)
}

// ResolvePostLoginRedirectURL returns the redirect target after login.
// When a valid referrer cookie is present, it is cleared via w.
func ResolvePostLoginRedirectURL(req *http.Request, w http.ResponseWriter, defaultURL string) string {
	cookie, err := req.Cookie(constant.CookieNameReferrer)
	if err != nil || cookie.Value == "" || !IsValidReferrerURL(cookie.Value) {
		return defaultURL
	}
	ClearReferrerCookie(w)
	return cookie.Value
}

func applyReferrerCookieAttributes(cookie *http.Cookie, value string, expires time.Time) {
	cookie.Name = constant.CookieNameReferrer
	cookie.Value = value
	cookie.Path = "/"
	cookie.HttpOnly = true
	cookie.Secure = true
	cookie.SameSite = http.SameSiteLaxMode
	cookie.Expires = expires
}
