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

package httpserver

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"

	"openfuyao/oauth-server/pkg/utils"
	"openfuyao/oauth-server/pkg/zlog"
)

// TestResponseLogger 测试响应记录器
func TestResponseLogger(t *testing.T) {
	rec := httptest.NewRecorder()
	rl := &responseLogger{ResponseWriter: rec}

	// 测试状态码设置
	rl.WriteHeader(http.StatusNotFound)
	if rl.status != http.StatusNotFound {
		t.Errorf("WriteHeader failed, got %d, want %d", rl.status, http.StatusNotFound)
	}

	// 测试写入大小计算
	data := []byte("test data")
	size, _ := rl.Write(data)
	if rl.size != len(data) || size != len(data) {
		t.Errorf("Write size mismatch, got %d, want %d", rl.size, len(data))
	}
}

func TestAccessLoggingMiddleware(t *testing.T) {
	// 使用 gomonkey 捕获日志调用
	var (
		logLevel string
		logMsg   string
	)

	patches := gomonkey.ApplyFunc(zlog.LogInfof, func(format string, args ...interface{}) {
		logLevel = "info"
		logMsg = fmt.Sprintf(format, args...)
	})

	patches.ApplyFunc(zlog.LogWarnf, func(format string, args ...interface{}) {
		logLevel = "warn"
		logMsg = fmt.Sprintf(format, args...)
	})

	defer patches.Reset()

	// 创建测试环境和执行请求
	req, rec, start := setupTestEnvironment(t)

	// 验证安全头
	verifySecurityHeaders(t, rec)

	// 验证日志记录
	verifyAccessLog(t, logLevel, logMsg, req, start)
}

// 辅助函数：设置测试环境并执行请求
func setupTestEnvironment(t *testing.T) (*http.Request, *httptest.ResponseRecorder, time.Time) {
	t.Helper()
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, err := w.Write([]byte("OK"))
		if err != nil {
			t.Errorf("Failed to write response: %v", err)
		}
	})
	req := httptest.NewRequest("GET", "/test?param=value", nil)
	req.RemoteAddr = "127.0.0.1:12345"
	rec := httptest.NewRecorder()
	start := time.Now()
	AccessLoggingMiddleware(handler).ServeHTTP(rec, req)
	return req, rec, start
}

// 辅助函数：验证安全头
func verifySecurityHeaders(t *testing.T, rec *httptest.ResponseRecorder) {
	t.Helper()
	headers := map[string]string{
		"Content-Security-Policy":   "connect-src 'self' https:;frame-ancestors 'none';object-src 'none'",
		"Cache-Control":             "no-cache, no-store, must-revalidate",
		"X-Frame-Options":           "DENY",
		"X-Content-Type-Options":    "nosniff",
		"X-XSS-Protection":          "1",
		"Strict-Transport-Security": "max-age=31536000",
		"Referrer-Policy":           "same-origin",
	}

	for key, want := range headers {
		if got := rec.Header().Get(key); got != want {
			t.Errorf("Header %s: got %q, want %q", key, got, want)
		}
	}
}

// 辅助函数：验证访问日志
func verifyAccessLog(t *testing.T, level, msg string, req *http.Request, start time.Time) {
	t.Helper()
	if level != "info" {
		t.Errorf("Expected info log, got %s", level)
	}

	expected := fmt.Sprintf(`%s - - [%s] "%s %s %s", status=%d, length=%d, duration=%dms`,
		req.RemoteAddr,
		start.Format("02/Jan/2006:15:04:05 -0700"),
		req.Method,
		utils.EscapeSpecialChars(req.RequestURI),
		req.Proto,
		http.StatusOK,
		len("OK"),
		time.Since(start).Milliseconds(),
	)

	const leng = 50
	if !strings.Contains(msg, expected[:leng]) {
		t.Errorf("Log mismatch\nGot: %s\nWant: %s", msg, expected)
	}
}

// TestAccessLoggingMiddlewareErrorStatus 测试错误状态日志
func TestAccessLoggingMiddlewareErrorStatus(t *testing.T) {
	// 使用 gomonkey 只捕获警告日志
	var logCalled bool

	patches := gomonkey.ApplyFunc(zlog.LogWarnf, func(format string, args ...interface{}) {
		logCalled = true
	})

	defer patches.Reset()

	// 创建返回400的处理器
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	})
	mw := AccessLoggingMiddleware(handler)

	// 创建测试请求
	req := httptest.NewRequest("GET", "/error", nil)
	rec := httptest.NewRecorder()
	mw.ServeHTTP(rec, req)

	// 验证警告日志
	if !logCalled {
		t.Error("Expected LogWarnf to be called for 400 status")
	}
}
