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

// Package fuyao defines webhook handlers for fuyao internal oauth2server
package fuyao

import (
	"time"
)

import (
	"github.com/golang-jwt/jwt/v4"
	"k8s.io/api/authentication/v1"
)

import (
	"openfuyao/oauth-webhook/pkg/fuyaoerrors"
)

// JWTAccessClaims jwt claims
type JWTAccessClaims struct {
	jwt.StandardClaims
}

// JWTValidator is the token validate processor for jwt bearer token
type JWTValidator struct {
	signingKey []byte
}

// NewJWTValidator is the init function for JWTValidator
func NewJWTValidator(key []byte) *JWTValidator {
	return &JWTValidator{
		signingKey: key,
	}
}

// Validate implements JWTValidate for fuyao webhook check
func (v *JWTValidator) Validate(req v1.TokenReview) (*v1.TokenReview, bool) {
	// first set default false response
	response := req.DeepCopy()
	response.Status = v1.TokenReviewStatus{
		Authenticated: false,
		User:          v1.UserInfo{},
		Audiences:     nil,
		Error:         "",
	}

	// decode the JWT token
	JWTToken := req.Spec.Token
	var claims = JWTAccessClaims{
		StandardClaims: jwt.StandardClaims{},
	}
	token, err := jwt.ParseWithClaims(JWTToken, &claims, func(token *jwt.Token) (interface{}, error) {
		return v.signingKey, nil
	})
	if err != nil {
		response.Status.Error = err.Error()
		return response, false
	}

	// verify expiration time
	if !token.Valid || time.Now().Unix() > claims.ExpiresAt {
		response.Status.Error = fuyaoerrors.ErrStrJWTValidationFailed
		return response, false
	}

	// pass the validation process, set userinfo
	response.Status = v1.TokenReviewStatus{
		Authenticated: true,
		User: v1.UserInfo{
			Username: claims.Subject,
			UID:      "",
			Groups:   nil,
			Extra:    nil,
		},
		Audiences: nil,
		Error:     "",
	}

	return response, true
}
