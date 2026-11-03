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

// Package httpserver defines the httpserver options, middlewares and responses
package httpserver

import (
	"net/http"
	"time"

	"openfuyao/oauth-server/pkg/utils"
	"openfuyao/oauth-server/pkg/zlog"
)

// responseLogger is a custom response logger for recording response status codes and sizes
type responseLogger struct {
	http.ResponseWriter
	status int
	size   int
}

// WriteHeader rewrites the WriteHeader method of http.ResponseWriter
func (rl *responseLogger) WriteHeader(code int) {
	rl.status = code
	rl.ResponseWriter.WriteHeader(code)
}

// Write rewrites the Write method of http.ResponseWriter
func (rl *responseLogger) Write(b []byte) (int, error) {
	size, err := rl.ResponseWriter.Write(b)
	rl.size += size
	return size, err
}

// AccessLoggingMiddleware logs the access entries
func AccessLoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// add http security header
		csp := "connect-src 'self' https:;frame-ancestors 'none';object-src 'none'"
		w.Header().Set("Content-Security-Policy", csp)
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-XSS-Protection", "1")
		w.Header().Set("Strict-Transport-Security", "max-age=31536000")
		w.Header().Set("Referrer-Policy", "same-origin")

		// Create a new responseLogger
		rl := &responseLogger{
			ResponseWriter: w,
			status:         http.StatusOK, // Default status code is 200
		}
		next.ServeHTTP(rl, r)

		// Log access
		logFunc := zlog.LogInfof
		if rl.status >= http.StatusBadRequest {
			logFunc = zlog.LogWarnf
		}
		logFunc(
			`%s - - [%s] "%s %s %s", status=%d, length=%d, duration=%dms`,
			r.RemoteAddr,
			start.Format("02/Jan/2006:15:04:05 -0700"),
			r.Method,
			utils.EscapeSpecialChars(r.RequestURI),
			r.Proto,
			rl.status,
			rl.size,
			time.Since(start).Milliseconds(),
		)
	})
}
