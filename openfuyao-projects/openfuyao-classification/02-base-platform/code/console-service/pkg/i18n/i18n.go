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
	"net/http"
	"strings"
)

// DefaultLocale is the fallback locale
const DefaultLocale = "zh-CN"

// messages maps locale -> message key -> translated string
func messages() map[string]map[string]string {
	return map[string]map[string]string{
		"zh-CN": {
			"auth.login_required":   "请先登录",
			"auth.session_expired":  "会话已过期，请重新登录",
			"auth.forbidden":        "权限不足",
			"auth.login_success":    "登录成功",
			"auth.login_failed":     "登录失败",
			"auth.logout_success":   "退出成功",
			"auth.callback_success": "认证成功",
			"auth.callback_failed":  "认证失败",
			"error.not_found":       "资源未找到",
			"error.internal":        "内部服务器错误",
			"error.bad_request":     "请求参数错误",
		},
		"en-US": {
			"auth.login_required":   "Please login first",
			"auth.session_expired":  "Session expired, please login again",
			"auth.forbidden":        "Permission denied",
			"auth.login_success":    "Login successful",
			"auth.login_failed":     "Login failed",
			"auth.logout_success":   "Logged out successfully",
			"auth.callback_success": "Authentication successful",
			"auth.callback_failed":  "Authentication failed",
			"error.not_found":       "Resource not found",
			"error.internal":        "Internal server error",
			"error.bad_request":     "Bad request",
		},
	}
}

// GetLocaleFromRequest extracts the locale from Accept-Language header
func GetLocaleFromRequest(r *http.Request) string {
	supportedLocales := []string{"zh-CN", "en-US"}
	al := r.Header.Get("Accept-Language")
	if al == "" {
		return DefaultLocale
	}
	// Parse Accept-Language: e.g. "en-US,en;q=0.9,zh-CN;q=0.8"
	for _, part := range strings.Split(al, ",") {
		lang := strings.TrimSpace(strings.SplitN(part, ";", 2)[0])
		for _, supported := range supportedLocales {
			if strings.EqualFold(lang, supported) {
				return supported
			}
		}
		// Check language prefix (e.g. "en" matches "en-US")
		for _, supported := range supportedLocales {
			if strings.HasPrefix(strings.ToLower(supported), strings.ToLower(lang)) {
				return supported
			}
		}
	}
	return DefaultLocale
}

// T translates a message key for the given locale
func T(locale, key string) string {
	msgs := messages()
	// nil map lookups are safe, so both levels can be indexed directly
	if msg, ok := msgs[locale][key]; ok {
		return msg
	}
	// Fallback to default locale
	if msg, ok := msgs[DefaultLocale][key]; ok {
		return msg
	}
	return key
}
