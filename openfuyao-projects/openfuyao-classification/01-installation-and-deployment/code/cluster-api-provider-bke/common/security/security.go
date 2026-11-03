/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
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
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/base64"
	"errors"
	"fmt"
	"sync"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	// legacySignature is kept solely for decrypting existing BKENode passwords
	// that were encrypted with the old hardcoded key. It must never be used for
	// new encryption.
	legacySignature = "c02ba0b582501608"

	ivLen = aes.BlockSize // 16 bytes

	// magicPrefix is a 4-byte marker prepended to new-format ciphertext so
	// AesDecrypt can deterministically distinguish new format
	// (base64(magic || IV || ciphertext)) from legacy format
	// (base64(ciphertext) with IV = legacy key).
	magicPrefix = "OFMK"

	// DefaultSecretNamespace is the namespace of the Secret holding the AES key.
	DefaultSecretNamespace = "cluster-system"

	// DefaultSecretName is the name of the Secret holding the AES key.
	DefaultSecretName = "bke-password-encryption-key"

	// SecretDataKey is the Secret data field holding the base64-encoded key.
	SecretDataKey = "encryption-key"
)

// KeyProvider supplies the AES-256 key (32 bytes) used for BKENode password
// encryption. Implementations are responsible for caching; Key may be called
// concurrently (webhook and reconcilers share the same provider).
type KeyProvider interface {
	Key(ctx context.Context) ([]byte, error)
}

var (
	providerMu  sync.RWMutex
	keyProvider KeyProvider
)

// SetKeyProvider registers the key source. It must be called before the first
// AesEncrypt/AesDecrypt call. Re-registration is allowed (tests override the
// provider).
func SetKeyProvider(p KeyProvider) {
	providerMu.Lock()
	defer providerMu.Unlock()
	keyProvider = p
}

// currentProvider returns the registered provider. When none is registered it
// fails fast instead of silently falling back to the legacy key.
func currentProvider() (KeyProvider, error) {
	providerMu.RLock()
	defer providerMu.RUnlock()
	if keyProvider == nil {
		return nil, errors.New("password encryption key provider is not initialized, " +
			"call security.SetKeyProvider before encrypting/decrypting passwords")
	}
	return keyProvider, nil
}

// loadKey fetches the key via the registered provider.
func loadKey(ctx context.Context) ([]byte, error) {
	p, err := currentProvider()
	if err != nil {
		return nil, err
	}
	k, err := p.Key(ctx)
	if err != nil {
		return nil, err
	}
	if len(k) != 32 {
		return nil, fmt.Errorf("password encryption key: expected 32 bytes, got %d", len(k))
	}
	if string(k) == legacySignature {
		return nil, errors.New("refusing to use the compromised legacy key")
	}
	return k, nil
}

// CurrentKeyBase64 returns the current key in base64 form. It is used to copy
// the same key to target (child management) clusters.
func CurrentKeyBase64(ctx context.Context) (string, error) {
	k, err := loadKey(ctx)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(k), nil
}

// IsNewFormatCiphertext reports whether the value is new-format ciphertext
// (OFMK prefix). It is used to detect whether a lost key would make existing
// node passwords unrecoverable.
func IsNewFormatCiphertext(s string) bool {
	data, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return false
	}
	return bytes.HasPrefix(data, []byte(magicPrefix))
}

// AesEncrypt encrypts a password using AES-256-CBC with a random IV.
// The ciphertext format is base64(magic || IV || ciphertext).
func AesEncrypt(password string) (string, error) {
	k, err := loadKey(context.Background())
	if err != nil {
		log.Errorf("AesEncrypt: key load failed: %v", err)
		return "", err
	}

	block, err := aes.NewCipher(k)
	if err != nil {
		log.Errorf("AesEncrypt: create cipher failed: %v", err)
		return "", err
	}
	blockSize := block.BlockSize()

	iv := make([]byte, ivLen)
	if _, err := rand.Read(iv); err != nil {
		log.Errorf("AesEncrypt: generate IV failed: %v", err)
		return "", err
	}

	padded := pkcs7Padding([]byte(password), blockSize)
	crypted := make([]byte, len(padded))
	cipher.NewCBCEncrypter(block, iv).CryptBlocks(crypted, padded)

	// Prepend magic prefix so decrypt can distinguish new from legacy format.
	output := append([]byte(magicPrefix), iv...)
	output = append(output, crypted...)
	encoded := base64.StdEncoding.EncodeToString(output)
	log.Info("AesEncrypt: password encrypted successfully with new format")
	return encoded, nil
}

// AesDecrypt decrypts a password, automatically distinguishing the new format
// (base64(magic || IV || ciphertext)) from the legacy format (base64(ciphertext)
// with IV = legacy key). The magic prefix makes detection deterministic so
// there is no risk of misidentifying legacy ciphertext as new format.
// This ensures existing BKENode passwords remain readable after upgrade.
func AesDecrypt(cryptedPassword string) (string, error) {
	// The key provider is loaded lazily inside the new-format branch below.
	// The legacy branch uses legacySignature directly, so no key load is
	// needed there.

	data, err := base64.StdEncoding.DecodeString(cryptedPassword)
	if err != nil {
		log.Debugf("AesDecrypt: base64 decode failed: %v", err)
		return "", err
	}

	// New format: magic || IV || ciphertext.
	if bytes.HasPrefix(data, []byte(magicPrefix)) {
		k, kErr := loadKey(context.Background())
		if kErr != nil {
			log.Errorf("AesDecrypt: new-format ciphertext but key load failed: %v", kErr)
			return "", fmt.Errorf("decrypt new-format ciphertext: %w", kErr)
		}

		remainder := data[len(magicPrefix):]
		if len(remainder) <= ivLen || len(remainder)%ivLen != 0 {
			log.Errorf("AesDecrypt: invalid new-format ciphertext length: %d", len(remainder))
			return "", errors.New("invalid new-format ciphertext length")
		}

		block, err := aes.NewCipher(k)
		if err != nil {
			log.Debugf("AesDecrypt: create cipher failed: %v", err)
			return "", err
		}
		iv := remainder[:ivLen]
		crypted := remainder[ivLen:]
		origData := make([]byte, len(crypted))
		cipher.NewCBCDecrypter(block, iv).CryptBlocks(origData, crypted)
		result, err := pkcs7UnPaddingSafe(origData)
		if err != nil {
			log.Errorf("AesDecrypt: unpadding new-format ciphertext failed: %v", err)
			return "", err
		}
		log.Info("AesDecrypt: password decrypted successfully (new format)")
		return string(result), nil
	}

	// Legacy format: ciphertext only, IV = legacy key[:blockSize].
	log.Info("AesDecrypt: attempting legacy format decryption")
	if len(data) == 0 || len(data)%ivLen != 0 {
		log.Debugf("AesDecrypt: ciphertext matches neither new-format nor legacy (length=%d)", len(data))
		return "", errors.New("invalid ciphertext length")
	}
	legacyKey := []byte(legacySignature)
	legacyBlock, err := aes.NewCipher(legacyKey)
	if err != nil {
		log.Debugf("AesDecrypt: create legacy cipher failed: %v", err)
		return "", err
	}
	origData := make([]byte, len(data))
	cipher.NewCBCDecrypter(legacyBlock, legacyKey[:ivLen]).CryptBlocks(origData, data)
	result, err := pkcs7UnPaddingSafe(origData)
	if err != nil {
		log.Debugf("AesDecrypt: unpadding legacy ciphertext failed: %v", err)
		return "", err
	}
	log.Info("AesDecrypt: password decrypted successfully (legacy format)")
	return string(result), nil
}

// pkcs7UnPaddingSafe removes PKCS#7 padding with bounds checking.
func pkcs7UnPaddingSafe(originData []byte) ([]byte, error) {
	length := len(originData)
	if length == 0 {
		return nil, errors.New("empty data")
	}
	unpadding := int(originData[length-1])
	if unpadding > length || unpadding == 0 || unpadding > aes.BlockSize {
		return nil, errors.New("invalid padding")
	}
	return originData[:length-unpadding], nil
}

// pkcs7Padding applies PKCS#7 padding.
func pkcs7Padding(ciphertext []byte, blockSize int) []byte {
	padding := blockSize - len(ciphertext)%blockSize
	padtext := bytes.Repeat([]byte{byte(padding)}, padding)
	return append(ciphertext, padtext...)
}

// pkcs7UnPadding removes PKCS#7 padding without bounds checking.
// Retained for backward compatibility with any external callers.
func pkcs7UnPadding(originData []byte) []byte {
	length := len(originData)
	unpadding := int(originData[length-1])
	return originData[:length-unpadding]
}
