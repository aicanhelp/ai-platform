/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package repository

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	testFileModeExec = 0755
)

func TestRemoveArchiveExtensions_TarGz(t *testing.T) {
	result := removeArchiveExtensions("file.tar.gz")
	assert.Equal(t, "file", result)
}

func TestRemoveArchiveExtensions_Tgz(t *testing.T) {
	result := removeArchiveExtensions("file.tgz")
	assert.Equal(t, "file", result)
}

func TestRemoveArchiveExtensions_NoExtension(t *testing.T) {
	result := removeArchiveExtensions("file")
	assert.Equal(t, "file", result)
}

func TestDecompressDataPackage_AlreadyExists(t *testing.T) {
	tempDir := t.TempDir()
	testDir := tempDir + "/testdata"
	err := os.MkdirAll(testDir, testFileModeExec)
	assert.NoError(t, err)

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == testDir {
			return true
		}
		return false
	})

	cfg := decompressConfig{
		dataFile:      tempDir + "/test.tar.gz",
		dataDirectory: testDir,
		name:          "test",
		logMessage:    "test",
		skipMessage:   "skip",
	}

	err = decompressDataPackage(cfg)
	assert.NoError(t, err)
}

func TestDecompressDataPackage_RemoveError(t *testing.T) {
	tempDir := t.TempDir()
	testDir := tempDir + "/testdata"
	dataFile := tempDir + "/test.tar.gz"

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(utils.Exists, func(path string) bool {
		if path == dataFile {
			return true
		}
		return false
	})

	patches.ApplyFunc(os.Remove, func(string) error {
		return assert.AnError
	})

	cfg := decompressConfig{
		dataFile:      dataFile,
		dataDirectory: testDir,
		name:          "test",
		logMessage:    "test",
		skipMessage:   "skip",
	}

	err := decompressDataPackage(cfg)
	assert.Error(t, err)
}

func TestPrepareTempDirectory_Success(t *testing.T) {
	tempDir := t.TempDir()
	targetTemp := tempDir + "/temp"

	err := prepareTempDirectory(targetTemp)
	assert.NoError(t, err)
}

func TestPrepareTempDirectory_RemoveExisting(t *testing.T) {
	tempDir := t.TempDir()
	targetTemp := tempDir + "/temp"

	err := os.MkdirAll(targetTemp, testFileModeExec)
	assert.NoError(t, err)

	err = prepareTempDirectory(targetTemp)
	assert.NoError(t, err)
}

func TestPrepareTempDirectory_RemoveError(t *testing.T) {
	tempDir := t.TempDir()
	targetTemp := tempDir + "/temp"

	err := os.MkdirAll(targetTemp, testFileModeExec)
	assert.NoError(t, err)

	patches := gomonkey.NewPatches()

	defer patches.Reset()

	patches.ApplyFunc(os.RemoveAll, func(string) error {
		return assert.AnError
	})

	err = prepareTempDirectory(targetTemp)
	assert.Error(t, err)
}

func TestVerifyDirectoryExists_Exists(t *testing.T) {
	tempDir := t.TempDir()
	dir := tempDir + "/testdir"

	err := os.MkdirAll(dir, testFileModeExec)
	assert.NoError(t, err)

	err = verifyDirectoryExists(dir)
	assert.NoError(t, err)
}

func TestVerifyDirectoryExists_NotExists(t *testing.T) {
	err := verifyDirectoryExists("/nonexistent/directory")
	assert.Error(t, err)
}

func TestCleanYumDataDirectory_Success(t *testing.T) {
	err := cleanYumDataDirectory()
	assert.NoError(t, err)
}

func TestCleanYumDataDirectory_RemoveError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(os.Remove, func(string) error {
		return assert.AnError
	})

	err := cleanYumDataDirectory()
	assert.Error(t, err)
}

func TestDecompressYumDataFile_NotExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(string) bool {
		return false
	})

	patches.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
		return assert.AnError
	})

	err := decompressYumDataFile()
	assert.Error(t, err)
}
