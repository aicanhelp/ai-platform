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

// Package constants defines constants used for oauth-proxy
package constants

const (
	// SessionIDParam param for session_id
	SessionIDParam = "session_id"
	// LogoutEndpointParam param for logoutendpoint
	LogoutEndpointParam = "logout_endpoint"
	// OpenFuyaoAuthHeaderKey authn header for openfuyao pods
	OpenFuyaoAuthHeaderKey = "X-OpenFuyao-Authorization"
)

const (
	// DefaultUpstreamFlush default upstream flush milliseconds
	DefaultUpstreamFlush = 5
	// DefaultCookieExpireMin default cookie expire min
	DefaultCookieExpireMin = 10
)

const (
	// AuthnCacheTTL default authenticator cache ttl
	AuthnCacheTTL = 2
	// AuthzAllowCacheTTL default authenticator allow cache ttl
	AuthzAllowCacheTTL = 2
	// AuthzDenyCacheTTL default authenticator deny cache ttl
	AuthzDenyCacheTTL = 5
)

const (
	// SessionIDLength the default length for oauth-proxy sessionID
	SessionIDLength = 32
)
