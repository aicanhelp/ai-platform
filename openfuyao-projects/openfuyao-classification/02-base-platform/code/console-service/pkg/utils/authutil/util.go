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

package authutil

import (
	"encoding/base64"

	"github.com/golang-jwt/jwt/v4"
	"k8s.io/apiserver/pkg/authentication/user"

	"console-service/pkg/zlog"
)

// JWTAccessClaims structure
type JWTAccessClaims struct {
	jwt.StandardClaims
}

// ExtractUserFromJWT extracts userinfo from token
func ExtractUserFromJWT(token string) (user.Info, error) {
	var claims = JWTAccessClaims{
		StandardClaims: jwt.StandardClaims{},
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

func GenerateToken(sub string) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	payload := base64.RawURLEncoding.EncodeToString([]byte(`{"sub":"` + sub + `"}`))

	return header + "." + payload + "."
}
