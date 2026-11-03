/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package security

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"encoding/base64"
	"fmt"
)

const (
	signature = "c02ba0b582501608"
)

// AesEncrypt 加密
func AesEncrypt(password string) (string, error) {
	var cryptedPassword string
	var err error
	func() {
		defer func() {
			if exception := recover(); exception != nil {
				cryptedPassword = ""
				err = fmt.Errorf("%v", exception)
			}
		}()
		originData := []byte(password)
		key := []byte(signature)
		block, blockErr := aes.NewCipher(key)
		if blockErr != nil {
			err = blockErr
			return
		}
		blockSize := block.BlockSize()
		originData = pkcs7Padding(originData, blockSize)
		blockMode := cipher.NewCBCEncrypter(block, key[:blockSize])
		crypted := make([]byte, len(originData))
		blockMode.CryptBlocks(crypted, originData)
		cryptedPassword = base64.StdEncoding.EncodeToString(crypted)
	}()
	return cryptedPassword, err
}

// AesDecrypt 解密
func AesDecrypt(cryptedPassword string) (string, error) {
	var password string
	var err error
	func() {
		defer func() {
			if exception := recover(); exception != nil {
				password = ""
				err = fmt.Errorf("%v", exception)
			}
		}()
		crypted, decodeErr := base64.StdEncoding.DecodeString(cryptedPassword)
		if decodeErr != nil {
			panic(decodeErr)
		}
		key := []byte(signature)
		block, blockErr := aes.NewCipher(key)
		if blockErr != nil {
			err = blockErr
			return
		}
		blockMode := cipher.NewCBCDecrypter(block, key[:block.BlockSize()])
		origData := make([]byte, len(crypted))
		blockMode.CryptBlocks(origData, crypted)
		originData := pkcs7UnPadding(origData)
		password = string(originData)
	}()
	return password, err
}

func pkcs7Padding(ciphertext []byte, blockSize int) []byte {
	padding := blockSize - len(ciphertext)%blockSize
	padtext := bytes.Repeat([]byte{byte(padding)}, padding)
	return append(ciphertext, padtext...)
}

func pkcs7UnPadding(originData []byte) []byte {
	length := len(originData)
	unpadding := int(originData[length-1])
	return originData[:(length - unpadding)]
}
