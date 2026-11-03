// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package crypto provide hook function for customized features
package crypto

type DecryptKeyFileFunc func(string) ([]byte, error)

var decryptKeyFile DecryptKeyFileFunc

func GetDecryptKeyHookFunc() DecryptKeyFileFunc {
	return decryptKeyFile
}

func SetDecryptKeyHookFunc(hook DecryptKeyFileFunc) {
	decryptKeyFile = hook
}

type DecryptKeyBytesFunc func([]byte) ([]byte, error)

var decryptKey DecryptKeyBytesFunc

func GetDecryptKeyBytesHookFunc() DecryptKeyBytesFunc {
	return decryptKey
}

func SetDecryptKeyBytesHookFunc(hook DecryptKeyBytesFunc) {
	decryptKey = hook
}
