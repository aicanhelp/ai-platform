/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package params

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// TestVerifyPasswd tests the VerifyPasswd function
func TestVerifyPasswd(t *testing.T) {
	// Positive test cases
	t.Run("ValidPasswd", func(t *testing.T) {
		value := []byte("Strong123456!")
		result, err := VerifyPasswd(value)
		assert.NoError(t, err)
		assert.Equal(t, result, true)
	})

	// Password length is less than 8
	t.Run("ShortPasswd", func(t *testing.T) {
		value := []byte("Short12")
		result, err := VerifyPasswd(value)
		assert.Error(t, err)
		assert.Equal(t, result, false)
	})

	// Password doesn't contain special character
	t.Run("NoSpecialChar", func(t *testing.T) {
		value := []byte("NoSpecial123456")
		result, err := VerifyPasswd(value)
		assert.Error(t, err)
		assert.Equal(t, result, false)
	})

	// Password doesn't contain a digit
	t.Run("NoDigit", func(t *testing.T) {
		value := []byte("NoDigitdddd!")
		result, err := VerifyPasswd(value)
		assert.Error(t, err)
		assert.Equal(t, result, false)
	})
}

// TestVerifyZone tests the VerifyZone function
func TestVerifyZone(t *testing.T) {
	// Positive test case
	result, err := VerifyZone("country", "cn")
	assert.NoError(t, err)
	assert.Equal(t, result, true)

	// Negative test case
	result, err = VerifyZone("country", "test")
	fmt.Println(result, err)
	assert.Equal(t, result, false)
}

// TestVerifyDomain tests the VerifyDomain function
func TestVerifyDomain(t *testing.T) {
	// Positive test case
	result, err := VerifyDomain("url", "https://example.com")
	assert.NoError(t, err)
	assert.Equal(t, result, true)

	// Negative test case
	result, err = VerifyDomain("url", "my.example.com")
	assert.NoError(t, err)
	assert.Equal(t, result, true)
}

func TestVerifyRegistryCertExpiryDays(t *testing.T) {
	err := VerifyRegistryCertExpiryDays("30")
	assert.Error(t, err)
	err = VerifyRegistryCertExpiryDays("")
	assert.Error(t, err)
	err = VerifyRegistryCertExpiryDays("-30")
	assert.Error(t, err)
	err = VerifyRegistryCertExpiryDays("0")
	assert.Error(t, err)
	err = VerifyRegistryCertExpiryDays("3650")
	assert.NoError(t, err)
}

func TestVerifyArtifactPath(t *testing.T) {
	artifactPath := ""
	err := VerifyArtifactPath(artifactPath)
	assert.Error(t, err)

	artifactPath = filepath.Join(os.TempDir(), "test.tar.gz")
	err = VerifyArtifactPath(artifactPath)
	assert.Error(t, err)

	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-artifact-path-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	artifactPath = filepath.Join(tempDir, "test.tar.gz")
	_, err = os.Create(artifactPath)
	assert.NoError(t, err)

	err = VerifyArtifactPath(artifactPath)
	assert.NoError(t, err)
}
