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

// Package generators generates auth codes / access tokens
package generators

import (
	"crypto/rand"
	"encoding/hex"
	"io"

	"gopkg.in/oauth2.v3"
	"gopkg.in/oauth2.v3/generates"

	"openfuyao/oauth-server/pkg/constants"
)

// NewFuyaoAuthorizeGenerate create to generate the authorize code instance
func NewFuyaoAuthorizeGenerate() *FuyaoAuthorizeGenerate {
	return &FuyaoAuthorizeGenerate{}
}

// FuyaoAuthorizeGenerate generate the authorize code
type FuyaoAuthorizeGenerate struct {
	generates.AuthorizeGenerate
}

// Token generate token according to the UUID token, returns lowercase letters
func (ag *FuyaoAuthorizeGenerate) Token(basicInfo *oauth2.GenerateBasic) (string, error) {
	var authCodeBytes [constants.AuthCodeByteLength]byte
	_, err := io.ReadFull(rand.Reader, authCodeBytes[:])
	if err != nil {
		return "", err
	}
	authCode := hex.EncodeToString(authCodeBytes[:])
	return authCode, nil
}
