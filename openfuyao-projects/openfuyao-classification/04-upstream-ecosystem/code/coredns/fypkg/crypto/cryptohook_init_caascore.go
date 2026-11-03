//go:build caascore
// +build caascore

// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package crypto provide hook function for customized features
package crypto

import (
	"github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption"
)

// init register the crypto function
func init() {
	SetDecryptKeyHookFunc(decryption.FyTryDecryptKeyFile)
	SetDecryptKeyBytesHookFunc(decryption.FYTryDecryptcert)
}
