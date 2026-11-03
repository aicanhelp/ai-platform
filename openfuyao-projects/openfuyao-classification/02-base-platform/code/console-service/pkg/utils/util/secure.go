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
Package util include console-service level util function
*/
package util

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"io"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"

	"console-service/pkg/constant"
	"console-service/pkg/errors"
	"console-service/pkg/utils/k8sutil"
)

// GetSecretSymmetricEncryptKey get console-service symmetric encrypt key
func GetSecretSymmetricEncryptKey(clientset kubernetes.Interface, secretName string) ([]byte, error) {
	var decryptKey *v1.Secret
	var field []byte
	var exist bool
	var err error

	if decryptKey, err = k8sutil.GetSecret(clientset, secretName,
		constant.ConsoleServiceDefaultNamespace); decryptKey == nil || err != nil {
		return nil, err
	}
	if field, exist = decryptKey.Data[constant.SymmetricKey]; !exist {
		return nil, &errors.FieldNotFoundError{
			Message: "SymmetricKey not found",
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
