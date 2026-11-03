/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.cn.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package logger

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// tempLogPath returns a log file path inside a per-test temporary directory so
// that file-backed logging never leaves stray artifacts in the package dir.
//
// A manually-created temp directory (rather than t.TempDir) is used because
// ologger's underlying lumberjack writer keeps the log file handle open for the
// lifetime of the process. On Windows an open file cannot be deleted, so
// t.TempDir's automatic cleanup would fail the test. The cleanup here discards
// the removal error; leftover files live under the OS temp dir and are reaped
// by the system.
func tempLogPath(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "envcheck-log-")
	require.NoError(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	return filepath.Join(dir, "test.log")
}

func TestNewLoggerNotNil(t *testing.T) {
	l := NewLogger(tempLogPath(t))
	assert.NotNil(t, l)
}

func TestLoggerMethodsNoPanic(t *testing.T) {
	l := NewLogger(tempLogPath(t))
	assert.NotNil(t, l)

	assert.NotPanics(t, func() { l.Info("hello-info") })
	assert.NotPanics(t, func() { l.Warning("hello-warning") })
	assert.NotPanics(t, func() { l.Error("hello-error") })
	assert.NotPanics(t, func() { l.Success("hello-success") })
}

// TestNewLoggerWithCustomLogFileNoPanic ensures a caller-supplied log_file path
// is accepted without panicking.
func TestNewLoggerWithCustomLogFileNoPanic(t *testing.T) {
	l := NewLogger(tempLogPath(t))
	assert.NotNil(t, l)
	assert.NotPanics(t, func() { l.Info("custom path compatibility") })
}

// TestNewLoggerWritesToConfiguredFile verifies the core fix: the log_file
// argument now actually controls the runtime output path. Previously log_file
// was silently ignored and logs only reached ologger's global default logger;
// now the dedicated instance writes to the configured file.
func TestNewLoggerWritesToConfiguredFile(t *testing.T) {
	logPath := tempLogPath(t)
	l := NewLogger(logPath)

	messages := []string{
		"file-activation-info",
		"file-activation-warning",
		"file-activation-error",
		"file-activation-success",
	}
	l.Info(messages[0])
	l.Warning(messages[1])
	l.Error(messages[2])
	l.Success(messages[3])

	// The configured log file must exist and be non-empty.
	info, err := os.Stat(logPath)
	require.NoError(t, err, "log file should be created at the configured path")
	assert.Greater(t, info.Size(), int64(0), "log file should not be empty")

	data, err := os.ReadFile(logPath)
	require.NoError(t, err)
	content := string(data)

	// Every emitted message must be present in the file.
	for _, msg := range messages {
		assert.True(t, strings.Contains(content, msg),
			"expected log file to contain %q, got:\n%s", msg, content)
	}
}

// TestNewLoggerEmptyPathDoesNotCreateFile verifies that an empty log_file keeps
// logging console-only: no file handle is opened. Production always supplies a
// path via config, while tests pass "" so they neither create stray files nor
// hold open handles that would break t.TempDir cleanup on Windows.
func TestNewLoggerEmptyPathDoesNotCreateFile(t *testing.T) {
	dir, err := os.MkdirTemp("", "envcheck-empty-")
	require.NoError(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	prevWD, err := os.Getwd()
	require.NoError(t, err)
	require.NoError(t, os.Chdir(dir))
	defer os.Chdir(prevWD)

	l := NewLogger("")
	assert.NotNil(t, l)
	l.Info("console-only-message")

	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	assert.Empty(t, entries, "expected no log file to be created for an empty log_file path")
}
