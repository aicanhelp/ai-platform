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
	"errors"
	"fmt"
	"path/filepath"
	"sync"

	"github.com/fsnotify/fsnotify"
	"k8s.io/klog/v2"
)

// Watcher watches the directory that contains the mounted config file.
type Watcher struct {
	watcher   *fsnotify.Watcher
	configDir string
	mu        sync.Mutex
	closed    bool
}

// NewWatcher creates a filesystem watcher for configPath's parent directory.
func NewWatcher(configPath string) (*Watcher, error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("create fsnotify watcher: %w", err)
	}

	configDir := filepath.Dir(configPath)
	if err := w.Add(configDir); err != nil {
		w.Close()
		return nil, fmt.Errorf("add watch path %s: %w", configDir, err)
	}

	klog.Infof("config watcher started on directory: %s", configDir)

	return &Watcher{
		watcher:   w,
		configDir: configDir,
	}, nil
}

// Events returns filesystem events reported by the underlying watcher.
func (w *Watcher) Events() <-chan fsnotify.Event {
	return w.watcher.Events
}

// Errors returns asynchronous errors reported by the underlying watcher.
func (w *Watcher) Errors() <-chan error {
	return w.watcher.Errors
}

// Close releases the watcher resources.
func (w *Watcher) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return errors.New("config watcher already closed")
	}
	w.closed = true
	return w.watcher.Close()
}
