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
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// staticKeyProvider is a trivial KeyProvider for tests.
type staticKeyProvider struct {
	key  []byte
	err  error
	calls int
}

func (p *staticKeyProvider) Key(_ context.Context) ([]byte, error) {
	p.calls++
	if p.err != nil {
		return nil, p.err
	}
	return p.key, nil
}

// setTestKey registers a random 32-byte AES-256 key provider and restores the
// previous provider (usually nil) on cleanup.
func setTestKey(t *testing.T) {
	t.Helper()
	key := make([]byte, 32)
	if _, err := rand.Read(key); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}
	setTestKeyBytes(t, key)
}

// setTestKeyBytes registers a fixed key provider (exposed for secret_test.go).
func setTestKeyBytes(t *testing.T, key []byte) {
	t.Helper()
	prev := currentKeyProvider()
	SetKeyProvider(&staticKeyProvider{key: key})
	t.Cleanup(func() { SetKeyProvider(prev) })
}

// unsetTestKey unregisters the provider (no key source).
func unsetTestKey(t *testing.T) {
	t.Helper()
	prev := currentKeyProvider()
	SetKeyProvider(nil)
	t.Cleanup(func() { SetKeyProvider(prev) })
}

// currentKeyProvider returns the currently registered provider (may be nil).
func currentKeyProvider() KeyProvider {
	providerMu.RLock()
	defer providerMu.RUnlock()
	return keyProvider
}

// randomHex returns a short random hex string for test-only secrets.
func randomHex(t *testing.T, n int) string {
	t.Helper()
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}
	return hex.EncodeToString(buf)
}

// legacyEncrypt encrypts a password using the old hardcoded key + IV=key
// format (no magic prefix). Used to produce legacy ciphertext for
// backward-compat tests.
func legacyEncrypt(t *testing.T, password string) string {
	t.Helper()
	key := []byte(legacySignature)
	block, err := aes.NewCipher(key)
	assert.NoError(t, err)
	padded := pkcs7Padding([]byte(password), block.BlockSize())
	crypted := make([]byte, len(padded))
	cipher.NewCBCEncrypter(block, key[:block.BlockSize()]).CryptBlocks(crypted, padded)
	return base64.StdEncoding.EncodeToString(crypted)
}

func invalidLegacyCiphertext(t *testing.T) string {
	t.Helper()
	key := []byte(legacySignature)
	block, err := aes.NewCipher(key)
	assert.NoError(t, err)

	invalidPlaintext := make([]byte, aes.BlockSize)
	cipherInput := make([]byte, aes.BlockSize)
	for i := range cipherInput {
		cipherInput[i] = invalidPlaintext[i] ^ key[i]
	}

	crypted := make([]byte, aes.BlockSize)
	block.Encrypt(crypted, cipherInput)
	return base64.StdEncoding.EncodeToString(crypted)
}

func invalidNewFormatCiphertext(t *testing.T) string {
	t.Helper()
	key, err := loadKey(context.Background())
	assert.NoError(t, err)
	block, err := aes.NewCipher(key)
	assert.NoError(t, err)

	iv := make([]byte, aes.BlockSize)
	invalidPlaintext := make([]byte, aes.BlockSize)
	cipherInput := make([]byte, aes.BlockSize)
	for i := range cipherInput {
		cipherInput[i] = invalidPlaintext[i] ^ iv[i]
	}

	crypted := make([]byte, aes.BlockSize)
	block.Encrypt(crypted, cipherInput)
	data := append([]byte(magicPrefix), iv...)
	data = append(data, crypted...)
	return base64.StdEncoding.EncodeToString(data)
}

func TestAesEncrypt_NewFormat(t *testing.T) {
	setTestKey(t)
	password := randomHex(t, 8)

	result, err := AesEncrypt(password)
	assert.NoError(t, err)

	// New format: base64(magic || IV || ciphertext).
	// Decoded must start with magic prefix.
	raw, err := base64.StdEncoding.DecodeString(result)
	assert.NoError(t, err)
	assert.True(t, bytes.HasPrefix(raw, []byte(magicPrefix)),
		"new-format ciphertext must start with magic prefix")

	// After magic (4 bytes) and IV (16 bytes), ciphertext is a multiple of 16.
	remainder := raw[len(magicPrefix):]
	assert.True(t, len(remainder) > aes.BlockSize)
	assert.Equal(t, 0, len(remainder)%aes.BlockSize)
}

func TestAesDecrypt_NewFormat(t *testing.T) {
	setTestKey(t)
	password := "test-password-12345"

	encrypted, err := AesEncrypt(password)
	assert.NoError(t, err)

	decrypted, err := AesDecrypt(encrypted)
	assert.NoError(t, err)
	assert.Equal(t, password, decrypted)
}

func TestAesDecrypt_LegacyFormat(t *testing.T) {
	unsetTestKey(t)
	password := "legacy-password-456"

	legacyCiphertext := legacyEncrypt(t, password)

	// With a new key configured, legacy ciphertext should still decrypt.
	setTestKey(t)
	decrypted, err := AesDecrypt(legacyCiphertext)
	assert.NoError(t, err)
	assert.Equal(t, password, decrypted)
}

func TestAesEncrypt_NoProvider(t *testing.T) {
	unsetTestKey(t)
	password := randomHex(t, 8)

	_, err := AesEncrypt(password)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "key provider is not initialized")
}

func TestAesDecrypt_NoProvider_Legacy(t *testing.T) {
	unsetTestKey(t)
	password := "no-key-legacy-789"

	legacyCiphertext := legacyEncrypt(t, password)

	// Legacy ciphertext decrypts without any provider registered.
	decrypted, err := AesDecrypt(legacyCiphertext)
	assert.NoError(t, err)
	assert.Equal(t, password, decrypted)
}

func TestAesDecrypt_NoProvider_NewFormat_Error(t *testing.T) {
	// Encrypt with a real key first.
	setTestKey(t)
	password := "new-format-then-no-key"
	encrypted, err := AesEncrypt(password)
	assert.NoError(t, err)

	// Now unregister the provider — decrypting new-format ciphertext must
	// fail (not silently fall back to legacy).
	unsetTestKey(t)
	_, err = AesDecrypt(encrypted)
	assert.Error(t, err)
}

func TestAesEncrypt_ProviderError(t *testing.T) {
	prev := currentKeyProvider()
	SetKeyProvider(&staticKeyProvider{err: assert.AnError})
	t.Cleanup(func() { SetKeyProvider(prev) })

	_, err := AesEncrypt(randomHex(t, 8))
	assert.Error(t, err)
}

func TestAesEncrypt_RefusesLegacyKeyFromProvider(t *testing.T) {
	prev := currentKeyProvider()
	SetKeyProvider(&staticKeyProvider{key: []byte(legacySignature)})
	t.Cleanup(func() { SetKeyProvider(prev) })

	_, err := AesEncrypt(randomHex(t, 8))
	assert.Error(t, err)
}

func TestCurrentKeyBase64(t *testing.T) {
	setTestKey(t)
	encoded, err := CurrentKeyBase64(context.Background())
	assert.NoError(t, err)

	raw, err := base64.StdEncoding.DecodeString(encoded)
	assert.NoError(t, err)
	assert.Len(t, raw, 32)
}

func TestIsNewFormatCiphertext(t *testing.T) {
	setTestKey(t)

	// New-format ciphertext.
	enc, err := AesEncrypt("some-password")
	require.NoError(t, err)
	assert.True(t, IsNewFormatCiphertext(enc))

	// Legacy ciphertext is not new-format.
	assert.False(t, IsNewFormatCiphertext(legacyEncrypt(t, "legacy-pw")))

	// Plaintext / non-base64 values are not new-format.
	assert.False(t, IsNewFormatCiphertext("plaintext-password"))
	assert.False(t, IsNewFormatCiphertext("!!!not-base64!!!"))
}

func TestAesEncrypt_RandomIV(t *testing.T) {
	setTestKey(t)
	password := "same-plaintext-for-iv-test"

	enc1, err := AesEncrypt(password)
	assert.NoError(t, err)

	enc2, err := AesEncrypt(password)
	assert.NoError(t, err)

	// Same plaintext, different IVs => different ciphertexts.
	assert.NotEqual(t, enc1, enc2)

	// Both should decrypt back to the original.
	dec1, err := AesDecrypt(enc1)
	assert.NoError(t, err)
	assert.Equal(t, password, dec1)

	dec2, err := AesDecrypt(enc2)
	assert.NoError(t, err)
	assert.Equal(t, password, dec2)
}

func TestAesDecrypt_InvalidCiphertext(t *testing.T) {
	setTestKey(t)

	// Completely invalid base64.
	_, err := AesDecrypt("!!!not-valid-base64!!!")
	assert.Error(t, err)

	// Valid base64 but too short to be a real ciphertext.
	short := base64.StdEncoding.EncodeToString([]byte("short"))
	_, err = AesDecrypt(short)
	assert.Error(t, err)

	// Valid base64, correct length, but decrypts to invalid legacy padding.
	_, err = AesDecrypt(invalidLegacyCiphertext(t))
	assert.Error(t, err)
}

func TestAesDecrypt_MagicPrefixButGarbage(t *testing.T) {
	setTestKey(t)

	// Data that starts with magic prefix but decrypts to invalid padding.
	_, err := AesDecrypt(invalidNewFormatCiphertext(t))
	assert.Error(t, err)
}

func TestAesEncryptDecrypt_RoundTrip(t *testing.T) {
	setTestKey(t)
	samples := []string{
		"",
		"a",
		"short",
		"exactly16bytesok!",
		"this is a longer password with spaces and special chars: !@#$%^&*()",
	}

	for _, password := range samples {
		encrypted, err := AesEncrypt(password)
		assert.NoError(t, err)

		decrypted, err := AesDecrypt(encrypted)
		assert.NoError(t, err)
		assert.Equal(t, password, decrypted)
	}
}
