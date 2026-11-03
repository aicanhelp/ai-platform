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

package utils

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestExpandPathsEmptyList(t *testing.T) {
	expander := NewPathExpander()

	result, err := expander.ExpandPaths([]string{})

	assert.NoError(t, err)
	assert.Equal(t, 0, len(result))
}

func TestExpandPathsSinglePath(t *testing.T) {
	expander := NewPathExpander()

	result, err := expander.ExpandPaths([]string{"/tmp"})

	assert.NoError(t, err)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "/tmp", result[0])
}

func TestExpandPathNoWildcards(t *testing.T) {
	expander := NewPathExpander()

	result, err := expander.ExpandPath("/tmp/test")

	assert.NoError(t, err)
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "/tmp/test", result[0])
}

func TestExpandPathWithWildcards(t *testing.T) {
	tempDir := createTestFiles(t)
	defer cleanupTestFiles(t, tempDir)

	expander := NewPathExpander()

	pattern := filepath.Join(tempDir, "*.txt")
	result, err := expander.ExpandPath(pattern)

	assert.NoError(t, err)
	assert.Equal(t, 2, len(result)) // file1.txt and file2.txt
}

func TestExpandPathWildcardNoMatches(t *testing.T) {
	expander := NewPathExpander()

	result, err := expander.ExpandPath("/nonexistent/*.txt")

	assert.NoError(t, err)
	assert.Equal(t, 0, len(result))
}

func TestExpandEnvironmentVariablesHomeDirectory(t *testing.T) {
	expander := NewPathExpander()

	// 测试家目录展开
	result := expander.expandEnvironmentVariables("~")
	assert.NotEqual(t, "~", result)
	assert.True(t, filepath.IsAbs(result), "Result should be an absolute path")

	result = expander.expandEnvironmentVariables("~/test")
	assert.True(t, filepath.IsAbs(result), "Result should be an absolute path")
	assert.True(t, strings.HasSuffix(result, "test"), "Result should end with 'test'")
}

func TestExpandEnvironmentVariablesEnvVars(t *testing.T) {
	expander := NewPathExpander()

	// 设置测试环境变量
	err := os.Setenv("TEST_VAR", "test_value")
	if err != nil {
		return
	}
	defer os.Unsetenv("TEST_VAR")

	result := expander.expandEnvironmentVariables("$TEST_VAR/path")
	assert.NoError(t, err)
	assert.Equal(t, "test_value/path", result)

	result = expander.expandEnvironmentVariables("${TEST_VAR}/path")
	assert.Equal(t, "test_value/path", result)
}

func TestContainsWildcard(t *testing.T) {
	testCases := []struct {
		path     string
		expected bool
	}{
		{"/tmp/*.txt", true},
		{"/tmp/file?.txt", true},
		{"/tmp/file[0-9].txt", true},
		{"/tmp/normal.txt", false},
		{"/tmp/directory/", false},
		{"", false},
	}

	for _, tc := range testCases {
		t.Run(tc.path, func(t *testing.T) {
			result := containsWildcard(tc.path)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// createTestFiles 创建测试文件
func createTestFiles(t *testing.T) string {
	tempDir, err := os.MkdirTemp("", "path-expander-test-*")
	assert.NoError(t, err)

	// 创建测试文件
	createTestFile(t, tempDir, "file1.txt")
	createTestFile(t, tempDir, "file2.txt")
	createTestFile(t, tempDir, "file3.go")
	createTestFile(t, tempDir, "data.json")

	return tempDir
}

// createTestFile 创建单个测试文件
func createTestFile(t *testing.T, dir, filename string) {
	filePath := filepath.Join(dir, filename)
	err := os.WriteFile(filePath, []byte("test content"), 0644)
	assert.NoError(t, err)
}

// cleanupTestFiles 清理测试文件
func cleanupTestFiles(t *testing.T, dir string) {
	err := os.RemoveAll(dir)
	if err != nil {
		t.Logf("Warning: Failed to cleanup test files: %v", err)
	}
}
