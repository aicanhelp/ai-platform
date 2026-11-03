// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package decryption

import (
	"errors"
	"os"
)

var (
	isEnable bool

	decryptKey = FYTryDecryptCert
)

func init() {
	isEnable = grpcEnable()
}

func FYTryDecryptKeyFile(keyFile string) ([]byte, error) {
	if !isEnable {
		return nil, nil
	}

	if keyFile == "" {
		return nil, errors.New("invalid for empty path")
	}

	encryptedData, err := os.ReadFile(keyFile)
	if err != nil {
		return nil, err
	}

	plainKey, err := decryptKey(encryptedData)
	return plainKey, err
}
