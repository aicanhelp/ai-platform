/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package pkiutil

import (
	"crypto/x509"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParsePublicKeyAlgorithm(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected x509.PublicKeyAlgorithm
	}{
		{"RSA lowercase", "rsa", x509.RSA},
		{"RSA uppercase", "RSA", x509.RSA},
		{"ECDSA", "ecdsa", x509.ECDSA},
		{"EC", "ec", x509.ECDSA},
		{"Ed25519", "ed25519", x509.Ed25519},
		{"Unknown", "unknown", x509.RSA},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ParsePublicKeyAlgorithm(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestParseKeyUsages(t *testing.T) {
	usages := []string{"digital signature", "key encipherment", "cert sign"}
	result := ParseKeyUsages(usages)

	assert.Contains(t, result, x509.KeyUsageDigitalSignature)
	assert.Contains(t, result, x509.KeyUsageKeyEncipherment)
	assert.Contains(t, result, x509.KeyUsageCertSign)
}

func TestParseExtKeyUsages(t *testing.T) {
	usages := []string{"client auth", "server auth", "any"}
	result := ParseExtKeyUsages(usages)

	assert.Contains(t, result, x509.ExtKeyUsageClientAuth)
	assert.Contains(t, result, x509.ExtKeyUsageServerAuth)
	assert.Contains(t, result, x509.ExtKeyUsageAny)
}

func TestParseExtKeyUsagesEmpty(t *testing.T) {
	result := ParseExtKeyUsages([]string{})
	assert.Empty(t, result)
}

func TestParseKeyUsagesEmpty(t *testing.T) {
	result := ParseKeyUsages([]string{})
	assert.Empty(t, result)
}
