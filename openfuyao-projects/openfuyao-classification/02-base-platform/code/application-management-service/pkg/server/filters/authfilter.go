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

// Package filters do pre-requisite checks before passing requests to webservices
package filters

import (
	"context"
	"net/http"
	"strings"

	"github.com/emicklei/go-restful/v3"
	"github.com/golang-jwt/jwt/v4"
	"k8s.io/apiserver/pkg/authentication/user"

	"application-management-service/pkg/constant"
	"application-management-service/pkg/zlog"
)

// JWTAccessClaims structure
type JWTAccessClaims struct {
	jwt.RegisteredClaims
}

// ExtractUserFromJWT extract user from JWT
func ExtractUserFromJWT(token string) (user.Info, error) {
	var claims = JWTAccessClaims{
		RegisteredClaims: jwt.RegisteredClaims{},
	}
	_, _, err := jwt.NewParser().ParseUnverified(token, &claims)
	if err != nil {
		zlog.Errorf("Fail to parse tokenJWT: %v", err)
		return nil, err
	}

	var extractedUser user.DefaultInfo
	extractedUser.Name = claims.Subject

	return &extractedUser, nil
}

// AuthenticateOpenFuyaoUser authenticate openfuyao user
func AuthenticateOpenFuyaoUser(req *restful.Request, resp *restful.Response, chain *restful.FilterChain) {
	// get access token
	token := req.HeaderParameter(constant.OpenFuyaoAuthHeaderKey)
	if !strings.HasPrefix(token, "Bearer ") {
		zlog.Warnf("the request does not have openFuyao token, reqURL: %s", req.Request.URL.Path)
		chain.ProcessFilter(req, resp)
		return
	}

	// decode the token info
	token = strings.TrimPrefix(token, "Bearer ")
	extractedUser, err := ExtractUserFromJWT(token)
	if err != nil {
		zlog.Errorf("error extract user from jwt token: %v", err)
		_ = resp.WriteHeaderAndEntity(http.StatusUnauthorized, "unauthorized")
		return
	}

	// add user to child context
	ctx := context.WithValue(req.Request.Context(), constant.UserKey, extractedUser)
	req.Request = req.Request.WithContext(ctx)
	zlog.Infof("add current user %s to req.context", extractedUser.GetName())

	chain.ProcessFilter(req, resp)

	return
}
