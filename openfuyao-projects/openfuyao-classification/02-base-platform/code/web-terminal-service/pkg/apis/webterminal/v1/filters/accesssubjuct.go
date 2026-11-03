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

package filters

import (
	"context"
	"strings"

	"github.com/emicklei/go-restful/v3"
	"github.com/golang-jwt/jwt/v4"

	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/responsehandlers"
	"openfuyao.com/web-terminal-service/pkg/zlog"
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

		subject, err := getSubject(resp, token)
		if err != nil {
			return
		}

		ctx := context.WithValue(req.Request.Context(), "user", subject)
		zlog.LogInfof("User subject: %v", subject)
		req.Request = req.Request.WithContext(ctx)
	} else {
		zlog.LogInfof("authInfo is nil! ")
	}
	chain.ProcessFilter(req, resp) // 继续调用链中的下一个过滤器
}

func getSubject(resp *restful.Response, token string) (string, error) {
	// parse JWT
	var claims = JWTAccessClaims{
		StandardClaims: jwt.StandardClaims{},
	}
	_, _, err := jwt.NewParser().ParseUnverified(token, &claims) // 是否需要验证token的签名？
	if err != nil {
		zlog.LogErrorf("LogError Paring tokenJWT: %v", err)
		responsehandlers.SendStatusServerError(resp, "LogError Paring token", err)
		return "", err
	}
	return claims.Subject, nil
}
