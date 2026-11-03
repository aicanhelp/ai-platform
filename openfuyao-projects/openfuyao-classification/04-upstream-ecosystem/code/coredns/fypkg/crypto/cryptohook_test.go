// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package crypto provide hook function for customized features
package crypto

import (
	"io/ioutil"
	"testing"
)

type fakeCrypto struct{}

func TestSetCryptoKeyHookFunc(t *testing.T) {
	SetDecryptKeyHookFunc(nil)
	if GetDecryptKeyHookFunc() != nil {
		t.Fatalf("Unexpected Hook State")
	}
	fakeCrypto := fakeCrypto{}
	SetDecryptKeyHookFunc(fakeCrypto.DecryptFromKeyPath)
	if GetDecryptKeyHookFunc() == nil {
		t.Fatalf("Unexpected Hook State")
	}
}

func (c *fakeCrypto) DecryptFromKeyPath(keyPath string) ([]byte, error) {
	return ioutil.ReadFile(keyPath)
}
