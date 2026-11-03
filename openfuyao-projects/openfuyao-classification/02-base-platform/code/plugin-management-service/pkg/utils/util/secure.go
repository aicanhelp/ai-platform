/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

/*
Package util include consoleplugin-management-service level util function
*/
package util

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"io"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"plugin-management-service/pkg/constant"
	"plugin-management-service/pkg/errors"
	"plugin-management-service/pkg/zlog"
)

// DecodeBase64String decode base64 string
func DecodeBase64String(input string) (string, error) {
	decoded, err := base64.StdEncoding.DecodeString(input)
	if err != nil {
		return "", err
	}
	return string(decoded), nil
}

// GetSecret looks up secret by its name and namespace
func GetSecret(clientset kubernetes.Interface, secretName, namespace string) (*v1.Secret, error) {
	secret, err := clientset.CoreV1().Secrets(namespace).
		Get(context.Background(), secretName, metav1.GetOptions{})
	if err != nil {
		zlog.Errorf("Secret %s lookup failed, err: %v", secretName, err)
		return nil, err
	}
	zlog.Debugf("Secret %s found in namespace %s", secret.Name, secret.Namespace)
	return secret, nil
}

// GetSecretSymmetricEncryptKey get plugin-management-service symmetric encrypt key
func GetSecretSymmetricEncryptKey(clientset kubernetes.Interface, secretName string) ([]byte, error) {
	var decryptKey *v1.Secret
	var field []byte
	var exist bool
	var err error

	if decryptKey, err = GetSecret(clientset, secretName,
		constant.PluginManagementServiceDefaultNamespace); decryptKey == nil || err != nil {
		return nil, err
	}
	if field, exist = decryptKey.Data[constant.SymmetricKey]; !exist {
		return nil, &errors.FieldNotFoundError{
			Message: fmt.Sprintf("%s not found", constant.SymmetricKey),
			Field:   constant.SymmetricKey,
		}
	}
	return field, nil
}

func getAEAD(key []byte) (cipher.AEAD, error) {
	// Create a new AES cipher block
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}

	// Create a GCM cipher mode instance
	aesGCM, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}
	return aesGCM, nil
}

// Encrypt encrypts the given plaintext using AES-GCM with the provided key.
func Encrypt(plainText, key []byte) ([]byte, error) {
	aesGCM, err := getAEAD(key)
	if err != nil {
		return nil, err
	}

	// Create a nonce. Nonce size should be aesGCM.NonceSize().
	nonce := make([]byte, aesGCM.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	// Encrypt the plaintext using AES-GCM
	cipherText := aesGCM.Seal(nil, nonce, plainText, nil)

	// Return the nonce and ciphertext concatenated
	return append(nonce, cipherText...), nil
}

// Decrypt decrypts the given ciphertext using AES-GCM with the provided key.
func Decrypt(cipherText, key []byte) ([]byte, error) {
	aesGCM, err := getAEAD(key)
	if err != nil {
		return nil, err
	}

	// Separate the nonce and ciphertext
	nonceSize := aesGCM.NonceSize()
	nonce, splitCipherText := cipherText[:nonceSize], cipherText[nonceSize:]

	// Decrypt the ciphertext using AES-GCM
	plainText, err := aesGCM.Open(nil, nonce, splitCipherText, nil)
	if err != nil {
		return nil, err
	}

	return plainText, nil
}
