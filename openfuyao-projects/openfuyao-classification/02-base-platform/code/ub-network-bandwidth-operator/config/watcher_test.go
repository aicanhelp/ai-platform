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

package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/fsnotify/fsnotify"
	"github.com/stretchr/testify/assert"
)

func TestNewWatcher(t *testing.T) {
	tmpDir := t.TempDir()

	w, err := NewWatcher(filepath.Join(tmpDir, "config.yaml"))
	assert.NoError(t, err)
	assert.NotNil(t, w)
	defer w.Close()

	assert.NotNil(t, w.Events())
	assert.NotNil(t, w.Errors())
}

func TestNewWatcherNonExistentDir(t *testing.T) {
	_, err := NewWatcher("/non/existent/directory/config.yaml")
	assert.Error(t, err)
}

func TestWatcherClose(t *testing.T) {
	tmpDir := t.TempDir()

	w, err := NewWatcher(filepath.Join(tmpDir, "config.yaml"))
	assert.NoError(t, err)

	err = w.Close()
	assert.NoError(t, err)
}

func TestWatcherDetectsFileChange(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.yaml")

	err := os.WriteFile(configPath, []byte("bandwidths: []"), 0644)
	assert.NoError(t, err)

	w, err := NewWatcher(configPath)
	assert.NoError(t, err)
	defer w.Close()

	err = os.WriteFile(configPath, []byte("bandwidths:\n  - priority: 0\n    bandwidth: 10"), 0644)
	assert.NoError(t, err)

	select {
	case event := <-w.Events():
		assert.NotEqual(t, fsnotify.Remove, event.Op)
	case <-w.Errors():
		t.Fatal("unexpected watcher error")
	default:
	}
}

func TestWatcherErrorsChannel(t *testing.T) {
	tmpDir := t.TempDir()

	w, err := NewWatcher(filepath.Join(tmpDir, "config.yaml"))
	assert.NoError(t, err)
	defer w.Close()

	assert.NotNil(t, w.Errors())

	select {
	case <-w.Errors():
		t.Fatal("unexpected error on watcher errors channel")
	default:
	}
}

func TestWatcherMultipleClose(t *testing.T) {
	tmpDir := t.TempDir()

	w, err := NewWatcher(filepath.Join(tmpDir, "config.yaml"))
	assert.NoError(t, err)

	err = w.Close()
	assert.NoError(t, err)

	err = w.Close()
	assert.Error(t, err)
}
