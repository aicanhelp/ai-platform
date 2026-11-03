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

// Package filters provides HTTP middleware functions to enhance the functionality
// of web services by applying additional processing to the HTTP requests and responses.
// This package is primarily designed to integrate seamlessly into the HTTP request handling
// pipeline, enabling pre- and post-processing of HTTP traffic.
package filters

import (
	"context"
	"strings"

	"github.com/emicklei/go-restful/v3"
	"github.com/golang-jwt/jwt/v4"

	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

// JWTAccessClaims structure
type JWTAccessClaims struct {
	jwt.StandardClaims
}

const (
	defaultAuthHeader   = "Authorization"
	openFuyaoAuthHeader = "X-OpenFuyao-Authorization"
)

// ExactSubjectAccess checks the authorization header for a beaer token,
// extracts the subject from the token, and attaches it to teh request context.
func ExactSubjectAccess(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	var token string
	// first exact defaultauthheader
	authInfo := req.Request.Header.Get(defaultAuthHeader)
	if authInfo == "" {
		// second exact openFuyaoAuthHeader
		authInfo = req.Request.Header.Get(openFuyaoAuthHeader)
	}

	if authInfo != "" {
		token = strings.TrimPrefix(authInfo, "Bearer ")
		subject, err := getSubject(token)
		if err != nil {
			return
		}

		ctx := context.WithValue(req.Request.Context(), "user", subject)
		req.Request = req.Request.WithContext(ctx)
	}

	chain.ProcessFilter(req, resp)
}

func getSubject(token string) (string, error) {
	// parse JWT
	var claims = JWTAccessClaims{
		StandardClaims: jwt.StandardClaims{},
	}
	_, _, err := jwt.NewParser().ParseUnverified(token, &claims)
	if err != nil {
		zlog.LogError("Error Paring tokenJWT: %v", err)
		return "", err
	}
	return claims.Subject, nil
}
