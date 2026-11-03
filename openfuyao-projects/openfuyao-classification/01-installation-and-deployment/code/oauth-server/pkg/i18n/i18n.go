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
	"fmt"
	"net/http"
	"strings"
)

// SupportedLocales lists all supported locale codes
var SupportedLocales = []string{"zh-CN", "en-US"}

// DefaultLocale is the fallback locale
const DefaultLocale = "zh-CN"

// Messages maps locale -> message key -> translated string
var Messages = map[string]map[string]string{
	"zh-CN": {
		// page titles
		"page.title":              "openFuyao管理平台",
		"login.welcome":           "欢迎登录openFuyao",
		"login.username":          "用户名",
		"login.password":          "密码",
		"login.button":            "登录",
		"login.username_required": "请输入用户名!",
		"login.password_required": "请输入密码!",
		// password confirm page
		"password_confirm.title":           "初次修改密码",
		"password_confirm.prompt":          "为确保您的账户安全，初次登录后请修改密码",
		"password_confirm.new_password":    "新密码",
		"password_confirm.confirm":         "确认密码",
		"password_confirm.length_rule":     "密码长度8~32位",
		"password_confirm.complexity":      "包含英文字母、数字、特殊字符`~!@#$%^&*()-_=+\\|[{}];:'\",<.>/?",
		"password_confirm.same_as_account": "不能和账号及账号逆序相同",
		"password_confirm.mismatch":        "两次输入密码需要一致",
		"password_confirm.cancel":          "取消",
		"password_confirm.confirm_btn":     "确认",
		// error messages
		"error.username_or_password":        "用户名或密码错误",
		"error.current_password":            "当前密码错误",
		"error.username_or_password_count":  "用户名或密码错误，再输错 %s 次用户将锁定",
		"error.username_or_password_locked": "用户名或密码错误，用户已经被锁定，请 %s 分钟后重试",
		"error.password_same":               "新输入的密码和旧密码相同",
		"error.login_blocked":               "账户锁定中，请等待 %s 分钟后重试",
		"error.invalid_then":                "错误的重定向地址，请从首页重新登录",
	},
	"en-US": {
		// page titles
		"page.title":              "openFuyao Management Platform",
		"login.welcome":           "Welcome to openFuyao",
		"login.username":          "Username",
		"login.password":          "Password",
		"login.button":            "Login",
		"login.username_required": "Please enter username!",
		"login.password_required": "Please enter password!",
		// password confirm page
		"password_confirm.title":           "Initial Password Change",
		"password_confirm.prompt":          "To ensure your account security, please change your password after first login",
		"password_confirm.new_password":    "New Password",
		"password_confirm.confirm":         "Confirm Password",
		"password_confirm.length_rule":     "Password length 8~32 characters",
		"password_confirm.complexity":      "Contains letters, numbers, special characters `~!@#$%^&*()-_=+\\|[{}];:'\",<.>/?",
		"password_confirm.same_as_account": "Cannot be the same as account or reverse of account",
		"password_confirm.mismatch":        "The two passwords must match",
		"password_confirm.cancel":          "Cancel",
		"password_confirm.confirm_btn":     "Confirm",
		// error messages
		"error.username_or_password":        "Incorrect username or password",
		"error.current_password":            "Current password is incorrect",
		"error.username_or_password_count":  "Incorrect username or password, account will be locked after %s more failed attempt(s)",
		"error.username_or_password_locked": "Incorrect username or password, account has been locked, please try again in %s minutes",
		"error.password_same":               "The new password is the same as the old password",
		"error.login_blocked":               "Account is locked, please try again in %s minutes",
		"error.invalid_then":                "Invalid redirect address, please login again from the home page",
	},
}

// GetLocaleFromRequest extracts the locale from Accept-Language header or cookie
func GetLocaleFromRequest(r *http.Request) string {
	// Try cookie first
	if cookie, err := r.Cookie("locale"); err == nil && cookie.Value != "" {
		for _, supported := range SupportedLocales {
			if strings.EqualFold(cookie.Value, supported) {
				return supported
			}
		}
	}
	// Try Accept-Language header
	al := r.Header.Get("Accept-Language")
	if al == "" {
		return DefaultLocale
	}
	// Parse Accept-Language: e.g. "en-US,en;q=0.9,zh-CN;q=0.8"
	for _, part := range strings.Split(al, ",") {
		lang := strings.TrimSpace(strings.SplitN(part, ";", 2)[0])
		for _, supported := range SupportedLocales {
			if strings.EqualFold(lang, supported) {
				return supported
			}
		}
		// Check language prefix (e.g. "en" matches "en-US")
		for _, supported := range SupportedLocales {
			if strings.HasPrefix(strings.ToLower(supported), strings.ToLower(lang)) {
				return supported
			}
		}
	}
	return DefaultLocale
}

// T translates a message key for the given locale
func T(locale, key string) string {
	if msgs, ok := Messages[locale]; ok {
		if msg, ok := msgs[key]; ok {
			return msg
		}
	}
	// Fallback to default locale
	if msgs, ok := Messages[DefaultLocale]; ok {
		if msg, ok := msgs[key]; ok {
			return msg
		}
	}
	return key
}

// TFormattable translates a key for a given locale with fmt.Sprintf-style formatting.
func TFormattable(locale, key string, args ...interface{}) string {
	msg := T(locale, key)
	if len(args) > 0 {
		return fmt.Sprintf(msg, args...)
	}
	return msg
}
