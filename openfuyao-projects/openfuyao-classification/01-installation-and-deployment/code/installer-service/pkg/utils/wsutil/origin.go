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

// Package wsutil provides WebSocket security helpers.
package wsutil

import (
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
)

const allowedOriginsEnv = "INSTALLER_WS_ALLOWED_ORIGINS"

var (
	allowedOnce sync.Once
	allowedSet  map[string]struct{}
)

func loadAllowedOrigins() {
	allowedSet = make(map[string]struct{})
	for _, item := range strings.Split(os.Getenv(allowedOriginsEnv), ",") {
		if normalized := normalizeOrigin(item); normalized != "" {
			allowedSet[normalized] = struct{}{}
		}
	}
}

func allowedOrigins() map[string]struct{} {
	allowedOnce.Do(loadAllowedOrigins)
	return allowedSet
}

func normalizeOrigin(origin string) string {
	origin = strings.TrimSpace(origin)
	if origin == "" {
		return ""
	}
	return strings.TrimRight(origin, "/")
}

func hostname(host string) string {
	if host == "" {
		return ""
	}
	h, _, err := net.SplitHostPort(host)
	if err != nil {
		return host
	}
	return h
}

func originMatchesRequestHost(origin, requestHost string) bool {
	originURL, err := url.Parse(origin)
	if err != nil || originURL.Host == "" {
		return false
	}
	return strings.EqualFold(hostname(originURL.Host), hostname(requestHost))
}

// requestHostCandidates returns Host values that represent the client-facing endpoint.
// [MOD-4] 反代场景下 r.Host 常为 K8s 内部 svc 名，需结合 X-Forwarded-* 与 Origin 对齐。
func requestHostCandidates(r *http.Request) []string {
	var hosts []string
	add := func(host string) {
		host = strings.TrimSpace(host)
		if host != "" {
			hosts = append(hosts, host)
		}
	}

	add(r.Host)
	if xfwd := r.Header.Get("X-Forwarded-Host"); xfwd != "" {
		for _, h := range strings.Split(xfwd, ",") {
			add(h)
		}
	}

	if proto := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto")); proto != "" {
		base := append([]string(nil), hosts...)
		for _, h := range base {
			if !strings.Contains(h, "://") {
				add(proto + "://" + h)
			}
		}
	}

	return hosts
}

func originAllowedForRequest(origin string, r *http.Request) bool {
	if _, ok := allowedOrigins()[origin]; ok {
		return true
	}
	for _, candidate := range requestHostCandidates(r) {
		if normalizeOrigin(candidate) == origin {
			return true
		}
		if originMatchesRequestHost(origin, candidate) {
			return true
		}
	}
	return false
}

// CheckOrigin validates the WebSocket handshake Origin header.
// [MOD-1] CSWSH 防护：拒绝未在白名单且非同 Host 的跨站 Origin。
//
// 放行策略（按序）：
//  1. Origin 为空：非浏览器客户端 / 单测（浏览器跨站 CSWSH 必带 Origin）
//  2. INSTALLER_WS_ALLOWED_ORIGINS 精确匹配（如 https://console.example.com）
//  3. Origin 与请求 Host / X-Forwarded-Host 同主机（installer-website nginx 反代场景）
func CheckOrigin(r *http.Request) bool {
	origin := normalizeOrigin(r.Header.Get("Origin"))
	if origin == "" {
		return true
	}

	return originAllowedForRequest(origin, r)
}

// ResetAllowedOriginsForTest clears cached allowlist (tests only).
func ResetAllowedOriginsForTest() {
	allowedOnce = sync.Once{}
	allowedSet = nil
}
