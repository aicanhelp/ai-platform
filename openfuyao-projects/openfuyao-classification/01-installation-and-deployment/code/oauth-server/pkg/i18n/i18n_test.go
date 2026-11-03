/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN 'AS IS' BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package i18n

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGetLocaleFromRequest(t *testing.T) {
	tests := []struct {
		name   string
		cookie string
		header string
		want   string
	}{
		{
			name:   "cookie_takes_precedence",
			cookie: "en-US",
			header: "zh-CN",
			want:   "en-US",
		},
		{
			name:   "accept_language_header",
			header: "en-US,en;q=0.9",
			want:   "en-US",
		},
		{
			name:   "language_prefix_match",
			header: "en",
			want:   "en-US",
		},
		{
			name: "default_when_missing",
			want: DefaultLocale,
		},
		{
			name:   "zh_CN_header",
			header: "zh-CN,zh;q=0.9",
			want:   "zh-CN",
		},
		{
			name:   "unsupported_locale_returns_default",
			header: "fr-FR",
			want:   DefaultLocale,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			if tt.cookie != "" {
				req.AddCookie(&http.Cookie{Name: "locale", Value: tt.cookie})
			}
			if tt.header != "" {
				req.Header.Set("Accept-Language", tt.header)
			}
			if got := GetLocaleFromRequest(req); got != tt.want {
				t.Errorf("GetLocaleFromRequest() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestT(t *testing.T) {
	// Test existing key
	if got := T("zh-CN", "login.username"); got != "用户名" {
		t.Errorf("T(\"zh-CN\", \"login.username\") = %q, want %q", got, "用户名")
	}
	if got := T("en-US", "login.username"); got != "Username" {
		t.Errorf("T(\"en-US\", \"login.username\") = %q, want %q", got, "Username")
	}

	// Test fallback to default locale
	if got := T("fr-FR", "login.username"); got != "用户名" {
		t.Errorf("T(\"fr-FR\", \"login.username\") = %q, want %q (fallback)", got, "用户名")
	}

	// Test unknown key returns the key itself
	if got := T("zh-CN", "nonexistent.key"); got != "nonexistent.key" {
		t.Errorf("T(\"zh-CN\", \"nonexistent.key\") = %q, want %q", got, "nonexistent.key")
	}
}

func TestTFormattable(t *testing.T) {
	// Test with format args
	got := TFormattable("zh-CN", "error.username_or_password_count", "3")
	want := "用户名或密码错误，再输错 3 次用户将锁定"
	if got != want {
		t.Errorf("TFormattable() = %q, want %q", got, want)
	}

	got = TFormattable("en-US", "error.username_or_password_count", "3")
	want = "Incorrect username or password, account will be locked after 3 more failed attempt(s)"
	if got != want {
		t.Errorf("TFormattable() = %q, want %q", got, want)
	}

	// Test without format args
	got = TFormattable("zh-CN", "login.username")
	want = "用户名"
	if got != want {
		t.Errorf("TFormattable() = %q, want %q", got, want)
	}
}
