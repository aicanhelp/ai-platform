// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package decryption

import (
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"path"
	"reflect"
	"testing"
)

var (
	plainKeyFileContent = []byte(`
-----BEGIN RSA PRIVATE KEY-----
... mock Private key data ...
-----END RSA PRIVATE KEY-----
`)

	encryptedKeyFileContent = []byte(base64.StdEncoding.EncodeToString(plainKeyFileContent))
)

func Test_should_return_nil_with_false_while_unable(t *testing.T) {
	mockEnable(t, false)
	keyData, err := FYTryDecryptKeyFile("mock_file_path")
	if keyData != nil {
		t.Errorf("FyTryDecryptKeyFile() should not return cert while not enabled")
	}
	if err != nil {
		t.Errorf("FYTryDecryptKeyFile() should not be failed while not enabled")
	}
}

func Test_should_error_while_empty_file_param(t *testing.T) {
	mockEnable(t, true)
	keyData, err := FYTryDecryptKeyFile("")
	if keyData != nil {
		t.Errorf("FyTryDecryptKeyFile() should not return cert while path is empty")
	}
	if err == nil {
		t.Errorf("FYTryDecryptKeyFile() should be failed while path is empty")
	}
}

func Test_should_error_while_read_file_failed(t *testing.T) {
	mockEnable(t, true)
	keyData, err := FYTryDecryptKeyFile("not_exist_path_xxx")
	if keyData != nil {
		t.Errorf("FyTryDecryptKeyFile() should not return cert while read file failed")
	}
	if err == nil {
		t.Errorf("FYTryDecryptKeyFile() should be failed while read file failed")
	}
}

func Test_should_be_error_while_decrypt_failed(t *testing.T) {
	mockEnable(t, true)
	mockFilePath, failed := mockDecryptKeyFunc(t, func(keyBlock []byte) ([]byte, error) {
		return nil, errors.New("mock for decrypt failed")
	})
	if failed != nil {
		t.Errorf("FYTryDecryptKeyFile() mock failed for %v", failed)
	}
	keyData, err := FYTryDecryptKeyFile(mockFilePath)
	if keyData != nil {
		t.Errorf("FyTryDecryptKeyFile() should not return cert while decrypt failed")
	}
	if err == nil {
		t.Errorf("FYTryDecryptKeyFile() should be failed while decrypt failed")
	}
}

func Test_should_be_plain_key_while_given_encrypt_file(t *testing.T) {
	mockEnable(t, true)
	mockFilePath, failed := mockDecryptKeyFunc(t, func(keyBlock []byte) ([]byte, error) {
		return base64.StdEncoding.DecodeString(string(keyBlock))
	})
	if failed != nil {
		t.Errorf("FYTryDecryptKeyFile() mock failed for %v", failed)
	}

	keyData, err := FYTryDecryptKeyFile(mockFilePath)
	if !reflect.DeepEqual(plainKeyFileContent, keyData) {
		t.Errorf("FyTryDecryptKeyFile() should return plain data while decrypt success")
	}
	if err != nil {
		t.Errorf("FYTryDecryptKeyFile() should be no error while decrypt success, got: %v", err)
	}
}

func mockEnable(t *testing.T, enable bool) {
	origin := isEnable
	t.Cleanup(func() {
		isEnable = origin
	})
	isEnable = enable
}

func mockDecryptKeyFunc(t *testing.T, mockDecryptKey func(keyBlock []byte) ([]byte, error)) (string, error) {
	old := decryptKey
	var mockDir = new(string)
	t.Cleanup(func() {
		decryptKey = old
		_ = os.RemoveAll(*mockDir)
	})
	decryptKey = mockDecryptKey
	tempDir, err := os.MkdirTemp("", "decrypted_key_dir-*")
	if err != nil {
		return "", err
	}
	*mockDir = tempDir
	keyFilePath := path.Join(tempDir, "key.pem")
	if err := os.WriteFile(keyFilePath, encryptedKeyFileContent, 0644); err != nil {
		return "", fmt.Errorf("error while writing data: %v", err)
	}
	return keyFilePath, nil
}
