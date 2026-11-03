/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package zlog

import (
	"testing"
)

func TestWatchConfigExtra(t *testing.T) {
	// watchConfig registers a viper watcher; without a config file it must not panic
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("watchConfig() panicked: %v", r)
		}
	}()
	watchConfig()
}

func TestLogWithExtra(t *testing.T) {
	logger := LogWith("component", "test")
	if logger == nil {
		t.Fatal("LogWith() = nil, want non-nil logger")
	}
	if Logger == nil {
		t.Fatal("global Logger is nil")
	}
}

func TestSyncExtra(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Sync() panicked: %v", r)
		}
	}()
	_ = Sync()
}
