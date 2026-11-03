/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package watch

import (
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFiles(t *testing.T) {
	tmpDir := t.TempDir()

	w, err := Files(tmpDir)
	assert.NoError(t, err)
	assert.NotNil(t, w)
	w.Close()
}

func TestFilesNonExistentDir(t *testing.T) {
	_, err := Files("/non/existent/directory")
	assert.Error(t, err)
}

func TestSignals(t *testing.T) {
	ch := Signals(syscall.SIGINT, syscall.SIGTERM)
	assert.NotNil(t, ch)

	signal.Reset(syscall.SIGINT, syscall.SIGTERM)
}

func TestFilesMultiplePaths(t *testing.T) {
	tmpDir := t.TempDir()

	subDir := filepath.Join(tmpDir, "subdir")
	err := os.Mkdir(subDir, 0755)
	assert.NoError(t, err)

	w, err := Files(tmpDir, subDir)
	assert.NoError(t, err)
	assert.NotNil(t, w)
	w.Close()
}

func TestSignalsMultiple(t *testing.T) {
	ch := Signals(syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP, syscall.SIGQUIT)
	assert.NotNil(t, ch)

	signal.Reset(syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP, syscall.SIGQUIT)
}
