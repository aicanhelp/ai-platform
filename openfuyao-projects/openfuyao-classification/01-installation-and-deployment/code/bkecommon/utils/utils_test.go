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
package utils

import (
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSliceContainsString(t *testing.T) {
	assert.True(t, SliceContainsString([]string{"a", "b\n"}, "b"))
	assert.False(t, SliceContainsString([]string{"a", "b"}, "c"))
	assert.False(t, SliceContainsString(nil, "a"))
}

func TestSliceContainsSlice(t *testing.T) {
	src := []string{"a", "b", "c"}
	dst := []string{"a", "c"}
	assert.True(t, SliceContainsSlice(src, dst))
	assert.False(t, SliceContainsSlice(src, []string{"a", "d"}))
	assert.False(t, SliceContainsSlice(nil, dst))
}

func TestSliceEqualString(t *testing.T) {
	s1 := []string{"a", "b"}
	s2 := []string{"b", "a"}
	s3 := []string{"a", "c"}
	s4 := []string{"a", "b", "c"}
	assert.True(t, SliceEqualString(s1, s2))
	assert.False(t, SliceEqualString(s1, s3))
	assert.False(t, SliceEqualString(s1, s4))
}

func TestExists(t *testing.T) {
	// 临时创建一个文件
	tempFile, err := os.CreateTemp("", "testfile")
	require.NoError(t, err)
	defer os.Remove(tempFile.Name())
	assert.True(t, Exists(tempFile.Name()))
	assert.False(t, Exists("/nonexistent/path/file"))
}

func TestB64Encode(t *testing.T) {
	result := B64Encode("hello")
	assert.Equal(t, "aGVsbG8=", result)
}

func TestGetManifestsBuildInfoNotExist(t *testing.T) {
	p := "/manifests/BUILD_INFO"
	_, err := os.Stat(p)
	if err == nil {
		err := os.Remove(p) // 确保不存在
		if err != nil {
			t.Error(err)
		}
	}

	_, err = GetManifestsBuildInfo()
	assert.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "not found manifests build info"))
}
