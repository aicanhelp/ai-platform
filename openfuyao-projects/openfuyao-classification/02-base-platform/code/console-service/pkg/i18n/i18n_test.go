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

package i18n

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGetLocaleFromRequest(t *testing.T) {
	tests := []struct {
		name       string
		header     string
		wantLocale string
	}{
		{
			name:       "empty_header_returns_default",
			header:     "",
			wantLocale: DefaultLocale,
		},
		{
			name:       "exact_zh_cn",
			header:     "zh-CN",
			wantLocale: "zh-CN",
		},
		{
			name:       "exact_en_us",
			header:     "en-US",
			wantLocale: "en-US",
		},
		{
			name:       "accept_language_with_quality",
			header:     "en-US,en;q=0.9,zh-CN;q=0.8",
			wantLocale: "en-US",
		},
		{
			name:       "language_prefix_en",
			header:     "en",
			wantLocale: "en-US",
		},
		{
			name:       "unsupported_language_returns_default",
			header:     "fr-FR",
			wantLocale: DefaultLocale,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			if tt.header != "" {
				req.Header.Set("Accept-Language", tt.header)
			}
			if got := GetLocaleFromRequest(req); got != tt.wantLocale {
				t.Errorf("GetLocaleFromRequest() = %q, want %q", got, tt.wantLocale)
			}
		})
	}
}

func TestT(t *testing.T) {
	tests := []struct {
		name   string
		locale string
		key    string
		want   string
	}{
		{
			name:   "zh_cn_known_key",
			locale: "zh-CN",
			key:    "auth.login_required",
			want:   "请先登录",
		},
		{
			name:   "en_us_known_key",
			locale: "en-US",
			key:    "auth.login_required",
			want:   "Please login first",
		},
		{
			name:   "fallback_to_default_locale",
			locale: "fr-FR",
			key:    "auth.login_required",
			want:   "请先登录",
		},
		{
			name:   "unknown_key_returns_key",
			locale: "zh-CN",
			key:    "unknown.key",
			want:   "unknown.key",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := T(tt.locale, tt.key); got != tt.want {
				t.Errorf("T(%q, %q) = %q, want %q", tt.locale, tt.key, got, tt.want)
			}
		})
	}
}

func TestWithLocaleAndGetLocaleFromContext(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	reqWithLocale := WithLocale(req, "en-US")

	if got := GetLocaleFromContext(reqWithLocale.Context()); got != "en-US" {
		t.Errorf("GetLocaleFromContext() = %q, want en-US", got)
	}

	if got := GetLocaleFromContext(context.Background()); got != DefaultLocale {
		t.Errorf("GetLocaleFromContext() without locale = %q, want %q", got, DefaultLocale)
	}
}
