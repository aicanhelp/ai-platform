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
	"crypto/rand"
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAesEncrypt(t *testing.T) {
	const testPasswordBytes = 7
	password := randomHex(t, testPasswordBytes)
	result, err := AesEncrypt(password)
	if err != nil {
		t.Fatalf("AesEncrypt error: %v", err)
	}

	origData, err := AesDecrypt(result)
	if err != nil {
		t.Fatalf("AesDecrypt error: %v", err)
	}

	assert.New(t).Equal(password, origData)
}

// randomHex returns a short random hex string for test-only secrets to avoid
// hard-coded credentials.
func randomHex(t *testing.T, n int) string {
	t.Helper()
	buf := make([]byte, n)
	if _, err := rand.Read(buf); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}
	return hex.EncodeToString(buf)
}
