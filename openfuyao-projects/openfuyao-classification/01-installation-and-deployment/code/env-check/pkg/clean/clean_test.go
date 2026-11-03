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

package clean

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func TestCleanerExecuteActualDelete(t *testing.T) {
	// 注意：这里不defer删除，因为要测试删除操作
	tempDir := setupTestDir(t)

	cfg := &config.AppConfig{
		Mode:       config.ModeClean,
		Paths:      []string{tempDir},
		CleanForce: true,
	}
	log := logger.NewLogger("")
	cleaner := NewCleaner(cfg, log)

	result, err := cleaner.Execute()
	assert.NoError(t, err)

	// 验证文件已被删除
	_, err = os.Stat(tempDir)
	assert.Error(t, err) // 应该出错，因为目录已被删除

	assert.Equal(t, true, len(result.Deleted) > 0)
}

func TestCleanPathFile(t *testing.T) {
	tempDir := setupTestDir(t)
	defer os.RemoveAll(tempDir)

	cleaner, result := createDefaultCleaner()

	cleaner.cleanPath(tempDir, result)
	assert.Equal(t, true, len(result.Skipped) > 0)
}

func TestCleanPathNonExistent(t *testing.T) {
	cleaner, result := createDefaultCleaner()

	cleaner.cleanPath("/non/existent/path", result)
	assert.Equal(t, true, len(result.Skipped) > 0)
}

func TestShouldDeleteAutoMode(t *testing.T) {
	cfg := &config.AppConfig{
		CleanForce: true,
	}
	log := logger.NewLogger("")
	cleaner := NewCleaner(cfg, log)

	fileInfo := config.FileInfo{Path: "/test"}
	result := cleaner.shouldDelete(fileInfo)
	assert.Equal(t, true, result)
}

func TestFileCleanerAskForDeletion(t *testing.T) {
	tests := setupCleanResponse()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 创建logger
			log := logger.NewLogger("")
			// 创建cleaner实例
			cleaner := &FileCleaner{
				logger: log,
			}
			// 创建管道来模拟输入
			stdin, stdinWriter, err := os.Pipe()
			if err != nil {
				t.Fatal(err)
			}
			defer stdin.Close()
			defer stdinWriter.Close()
			// 保存原始stdin
			oldStdin := os.Stdin
			defer func() { os.Stdin = oldStdin }()
			// 替换stdin
			os.Stdin = stdin
			// 在新的goroutine中写入测试输入
			go func() {
				defer stdinWriter.Close()
				_, err := stdinWriter.WriteString(tt.input)
				if err != nil {
					t.Error(err)
				}
			}()
			// 调用被测试的方法
			result := cleaner.askForDeletion(tt.path)
			// 验证结果
			assert.Equal(t, tt.expected, result)
		})
	}
}

func setupCleanResponse() []struct {
	name           string
	input          string
	expected       bool
	expectErrorLog bool
	path           string
} {
	tests := []struct {
		name           string
		input          string
		expected       bool
		expectErrorLog bool
		path           string
	}{
		{
			name:     "input y should return true",
			input:    "y\n",
			expected: true,
			path:     "/tmp/test.txt",
		},
		{
			name:     "input n should return false",
			input:    "n\n",
			expected: false,
			path:     "/tmp/test3.txt",
		},
		{
			name:     "input empty should return false",
			input:    "\n",
			expected: false,
			path:     "/tmp/test5.txt",
		},
		{
			name:     "invalid input then valid",
			input:    "maybe\ny\n",
			expected: true,
			path:     "/tmp/test8.txt",
		},
	}
	return tests
}

// 辅助函数
func setupTestDir(t *testing.T) string {
	tempDir, err := os.MkdirTemp("", "clean-test-*")
	assert.NoError(t, err)

	testFile := filepath.Join(tempDir, "test.txt")
	err = os.WriteFile(testFile, []byte("test content"), config.FileMode)
	assert.NoError(t, err)

	return tempDir
}

func createDefaultCleaner() (*FileCleaner, *config.CleanResult) {
	cfg := &config.AppConfig{
		Mode:       config.ModeClean,
		CleanForce: false,
	}
	log := logger.NewLogger("")
	cleaner := NewCleaner(cfg, log)

	result := &config.CleanResult{
		Deleted: make([]config.FileInfo, 0),
		Failed:  make([]config.FileInfo, 0),
		Skipped: make([]config.FileInfo, 0),
	}
	return cleaner, result
}

func TestCleanerExecuteWildcardExpansion(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "clean-wildcard-*")
	assert.NoError(t, err)
	for _, name := range []string{"a.txt", "b.txt", "c.log"} {
		assert.NoError(t, os.WriteFile(filepath.Join(tempDir, name), []byte("x"), 0644))
	}

	cfg := &config.AppConfig{
		Mode:       config.ModeClean,
		Paths:      []string{filepath.Join(tempDir, "*.txt")},
		CleanForce: true,
	}
	cleaner := NewCleaner(cfg, logger.NewLogger(""))

	result, err := cleaner.Execute()
	assert.NoError(t, err)
	// *.txt matches a.txt, b.txt (2 deleted); c.log untouched
	assert.Equal(t, 2, len(result.Deleted))
	assert.False(t, fileExists(filepath.Join(tempDir, "a.txt")))
	assert.False(t, fileExists(filepath.Join(tempDir, "b.txt")))
	assert.True(t, fileExists(filepath.Join(tempDir, "c.log")))

	os.RemoveAll(tempDir) // cleanup remaining c.log
}

func TestCleanerExecuteEnvVarExpansion(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "clean-env-*")
	assert.NoError(t, err)
	target := filepath.Join(tempDir, "test.txt")
	assert.NoError(t, os.WriteFile(target, []byte("x"), 0644))

	assert.NoError(t, os.Setenv("CLEAN_TEST_DIR", tempDir))
	defer os.Unsetenv("CLEAN_TEST_DIR")

	cfg := &config.AppConfig{
		Mode:       config.ModeClean,
		Paths:      []string{"$CLEAN_TEST_DIR/test.txt"},
		CleanForce: true,
	}
	cleaner := NewCleaner(cfg, logger.NewLogger(""))

	result, err := cleaner.Execute()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(result.Deleted))
	assert.False(t, fileExists(target))

	os.RemoveAll(tempDir)
}

func TestCleanerExecuteWildcardNoMatch(t *testing.T) {
	cfg := &config.AppConfig{
		Mode:       config.ModeClean,
		Paths:      []string{filepath.Join(os.TempDir(), "nonexistent-clean-*.txt")},
		CleanForce: true,
	}
	cleaner := NewCleaner(cfg, logger.NewLogger(""))

	result, err := cleaner.Execute()
	assert.NoError(t, err)
	assert.Equal(t, 1, len(result.Skipped))
	assert.Contains(t, result.Skipped[0].Error, "no files matched")
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
