/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package wsutil

import (
	"net/http/httptest"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCheckOrigin(t *testing.T) {
	t.Cleanup(func() {
		_ = os.Unsetenv(allowedOriginsEnv)
		ResetAllowedOriginsForTest()
	})

	tests := []struct {
		name       string
		envOrigins string
		origin     string
		host       string
		xfwdHost   string
		xfwdProto  string
		allowed    bool
	}{
		{name: "empty origin allowed for non-browser", origin: "", host: "installer.local:9210", allowed: true},
		{name: "cross-site origin rejected", origin: "https://attacker.com", host: "installer.local:9210", allowed: false},
		{name: "allowlist match", envOrigins: "https://console.openfuyao.com", origin: "https://console.openfuyao.com", host: "installer.local:9210", allowed: true},
		{name: "allowlist trailing slash normalized", envOrigins: "https://console.openfuyao.com/", origin: "https://console.openfuyao.com", host: "installer.local:9210", allowed: true},
		{name: "same host via ingress proxy", origin: "https://console.openfuyao.com", host: "console.openfuyao.com:443", allowed: true},
		{name: "installer-website nginx proxy with X-Forwarded-Host", origin: "https://console.openfuyao.com", host: "installer-service.openfuyao-system.svc:9210", allowed: true, xfwdHost: "console.openfuyao.com", xfwdProto: "https"},
		{name: "cross-site still rejected", origin: "https://attacker.com", host: "installer-service.openfuyao-system.svc:9210", xfwdHost: "console.openfuyao.com", allowed: false},
		{name: "same host different scheme still matches hostname", origin: "http://127.0.0.1:8080", host: "127.0.0.1:8080", allowed: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ResetAllowedOriginsForTest()
			if tt.envOrigins != "" {
				t.Setenv(allowedOriginsEnv, tt.envOrigins)
			} else {
				_ = os.Unsetenv(allowedOriginsEnv)
			}

			req := httptest.NewRequest("GET", "http://example/ws", nil)
			req.Host = tt.host
			if tt.origin != "" {
				req.Header.Set("Origin", tt.origin)
			}
			if tt.xfwdHost != "" {
				req.Header.Set("X-Forwarded-Host", tt.xfwdHost)
			}
			if tt.xfwdProto != "" {
				req.Header.Set("X-Forwarded-Proto", tt.xfwdProto)
			}

			assert.Equal(t, tt.allowed, CheckOrigin(req))
		})
	}
}
