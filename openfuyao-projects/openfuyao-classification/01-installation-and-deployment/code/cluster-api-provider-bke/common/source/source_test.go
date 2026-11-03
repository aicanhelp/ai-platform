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
package source

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	testFilePermission = 0644
	testDirPermission  = 0755
)

func TestGetCustomDownloadPath(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "URL with trailing slash",
			input:    "http://example.com/repo/",
			expected: "http://example.com/repo/files",
		},
		{
			name:     "URL without trailing slash",
			input:    "http://example.com/repo",
			expected: "http://example.com/repo/files",
		},
		{
			name:     "URL with query parameters",
			input:    "http://example.com/repo?param=v",
			expected: "http://example.com/repo?param=v/files",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetCustomDownloadPath(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestWriteYumSource(t *testing.T) {
	tempDir := t.TempDir()
	originalYumRepos := yumRepos
	yumRepos = tempDir
	defer func() { yumRepos = originalYumRepos }()

	testURL := "http://test.repo.com/centos"
	err := writeYumSource(testURL)
	assert.NoError(t, err)

	content, err := os.ReadFile(filepath.Join(yumRepos, bkeRepoFile))
	assert.NoError(t, err)
	assert.Contains(t, string(content), testURL)
	assert.Contains(t, string(content), "priority = 1")
}

func TestWriteAptSource(t *testing.T) {
	tempDir := t.TempDir()
	originalAptSourcesDir := aptSourcesDir
	originalAptPrefsDir := aptPrefsDir
	aptSourcesDir = filepath.Join(tempDir, "sources.list.d")
	aptPrefsDir = filepath.Join(tempDir, "preferences.d")
	defer func() {
		aptSourcesDir = originalAptSourcesDir
		aptPrefsDir = originalAptPrefsDir
	}()
	require.NoError(t, os.MkdirAll(aptSourcesDir, testDirPermission))
	require.NoError(t, os.MkdirAll(aptPrefsDir, testDirPermission))

	testURL := "http://test.repo.com/ubuntu"
	err := writeAptSource(testURL)
	assert.NoError(t, err)

	content, err := os.ReadFile(filepath.Join(aptSourcesDir, bkeListFile))
	assert.NoError(t, err)
	assert.Contains(t, string(content), testURL)

	prefsContent, err := os.ReadFile(filepath.Join(aptPrefsDir, bkePrefsFile))
	assert.NoError(t, err)
	assert.Contains(t, string(prefsContent), "Pin: origin test.repo.com")
	assert.Contains(t, string(prefsContent), "Pin-Priority: 1001")
}

func TestResetAptSource_NoBackup(t *testing.T) {
	tempDir := t.TempDir()
	originalAptSourcesDir := aptSourcesDir
	originalAptPrefsDir := aptPrefsDir
	aptSourcesDir = filepath.Join(tempDir, "sources.list.d")
	aptPrefsDir = filepath.Join(tempDir, "preferences.d")
	defer func() {
		aptSourcesDir = originalAptSourcesDir
		aptPrefsDir = originalAptPrefsDir
	}()

	err := resetAptSource()
	assert.NoError(t, err)
}

func TestResetYumSource_Success(t *testing.T) {
	tempDir := t.TempDir()
	originalYumRepos := yumRepos
	yumRepos = tempDir
	defer func() { yumRepos = originalYumRepos }()

	err := os.WriteFile(filepath.Join(tempDir, bkeRepoFile), []byte("test"), testFilePermission)
	assert.NoError(t, err)

	err = resetYumSource()
	assert.NoError(t, err)

	_, err = os.Stat(filepath.Join(tempDir, bkeRepoFile))
	assert.True(t, os.IsNotExist(err))
}

func TestResetYumSource_NoBakDir(t *testing.T) {
	tempDir := t.TempDir()
	originalYumRepos := yumRepos
	yumRepos = tempDir
	defer func() { yumRepos = originalYumRepos }()

	err := resetYumSource()
	assert.NoError(t, err)
}

func TestResetYumSource_NoBkeRepo(t *testing.T) {
	tempDir := t.TempDir()
	originalYumRepos := yumRepos
	yumRepos = tempDir
	defer func() { yumRepos = originalYumRepos }()

	err := resetYumSource()
	assert.NoError(t, err)
}

func TestGetRPMDownloadPath(t *testing.T) {
	_, err := GetRPMDownloadPath("http://test.com")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestSetSource(t *testing.T) {
	err := SetSource("http://test.com")
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestResetSource(t *testing.T) {
	err := ResetSource()
	if err != nil {
		t.Logf("Expected error: %v", err)
	}
}

func TestResetAptSourceWithBackup(t *testing.T) {
	tempDir := t.TempDir()
	originalAptSourcesDir := aptSourcesDir
	originalAptPrefsDir := aptPrefsDir
	aptSourcesDir = filepath.Join(tempDir, "sources.list.d")
	aptPrefsDir = filepath.Join(tempDir, "preferences.d")
	defer func() {
		aptSourcesDir = originalAptSourcesDir
		aptPrefsDir = originalAptPrefsDir
	}()
	require.NoError(t, os.MkdirAll(aptSourcesDir, testDirPermission))
	require.NoError(t, os.MkdirAll(aptPrefsDir, testDirPermission))

	listPath := filepath.Join(aptSourcesDir, bkeListFile)
	prefsPath := filepath.Join(aptPrefsDir, bkePrefsFile)
	err := os.WriteFile(listPath, []byte("test"), testFilePermission)
	assert.NoError(t, err)
	err = os.WriteFile(prefsPath, []byte("backup"), testFilePermission)
	assert.NoError(t, err)

	err = resetAptSource()
	assert.NoError(t, err)
	_, err = os.Stat(listPath)
	assert.True(t, os.IsNotExist(err))
	_, err = os.Stat(prefsPath)
	assert.True(t, os.IsNotExist(err))
}

func TestOriginFromURL(t *testing.T) {
	assert.Equal(t, "test.repo.com", originFromURL("http://test.repo.com/ubuntu"))
	assert.Equal(t, "raw-value", originFromURL("raw-value"))
}

func TestResetYumSourceKeepsOtherFiles(t *testing.T) {
	tempDir := t.TempDir()
	originalYumRepos := yumRepos
	yumRepos = tempDir
	defer func() { yumRepos = originalYumRepos }()

	err := os.WriteFile(filepath.Join(tempDir, bkeRepoFile), []byte("test"), testFilePermission)
	assert.NoError(t, err)
	err = os.WriteFile(filepath.Join(tempDir, "system.repo"), []byte("system"), testFilePermission)
	assert.NoError(t, err)

	err = resetYumSource()
	assert.NoError(t, err)
	_, err = os.Stat(filepath.Join(tempDir, bkeRepoFile))
	assert.True(t, os.IsNotExist(err))
	_, err = os.Stat(filepath.Join(tempDir, "system.repo"))
	assert.NoError(t, err)
}
