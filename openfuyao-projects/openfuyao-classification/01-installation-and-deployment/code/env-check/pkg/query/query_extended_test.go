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

package query

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
	"env-check/pkg/utils"
)

func TestGetFileOwnerCrossPlatformExistingFile(t *testing.T) {
	tempFile, err := os.CreateTemp("", "test-file-*")
	assert.NoError(t, err)
	defer os.Remove(tempFile.Name())
	tempFile.Close()

	assert.NotPanics(t, func() {
		_, _, _ = getFileOwnerCrossPlatform(tempFile.Name())
	})
}

func TestGetFileOwnerCrossPlatformNonExistent(t *testing.T) {
	_, _, err := getFileOwnerCrossPlatform("/nonexistent/path/file.txt")
	assert.Error(t, err)
}

func TestGetUnixFileOwnerExistingFile(t *testing.T) {
	tempFile, err := os.CreateTemp("", "test-file-*")
	assert.NoError(t, err)
	defer os.Remove(tempFile.Name())
	tempFile.Close()

	assert.NotPanics(t, func() {
		_, _, _ = getUnixFileOwner(tempFile.Name())
	})
}

func TestQueryPathPermissionDenied(t *testing.T) {
	cfg := createTestConfig("")
	log := logger.NewLogger("")
	q := NewQuery(cfg, log)

	assert.NotPanics(t, func() {
		fileInfo := q.queryPath("bad\x00path")
		assert.False(t, fileInfo.Exists)
	})
}

func TestQueryPathNonExistentPath(t *testing.T) {
	cfg := createTestConfig("")
	log := logger.NewLogger("")
	q := NewQuery(cfg, log)

	fileInfo := q.queryPath("/nonexistent/path/file.txt")

	assert.False(t, fileInfo.Exists)
}

func TestQueryExpandedPathError(t *testing.T) {
	tempDir := setupTestDir(t)
	defer os.RemoveAll(tempDir)

	cfg := createTestConfig(tempDir)
	log := logger.NewLogger("")
	q := NewQuery(cfg, log)

	expander := utils.NewPathExpander()
	result := &config.QueryResult{
		Files:   make([]config.FileInfo, 0),
		Summary: config.QuerySummary{},
	}

	q.queryExpandedPath(tempDir, expander, result)

	assert.Equal(t, 1, len(result.Files))
	assert.True(t, result.Files[0].Exists)
}
