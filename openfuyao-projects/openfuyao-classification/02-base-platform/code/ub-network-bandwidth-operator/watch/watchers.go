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

// Package watch wraps filesystem and OS signal watchers.
package watch

import (
	"os"
	"os/signal"

	"github.com/fsnotify/fsnotify"
)

// Files creates an fsnotify watcher for the supplied paths.
func Files(files ...string) (*fsnotify.Watcher, error) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}

	for _, file := range files {
		if err = watcher.Add(file); err != nil {
			watcher.Close()
			return nil, err
		}
	}

	return watcher, nil
}

// Signals creates a channel subscribed to the supplied OS signals.
func Signals(signals ...os.Signal) chan os.Signal {
	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, signals...)

	return signalCh
}
