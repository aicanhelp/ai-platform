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

package utils

import (
	"testing"
	"time"
)

func TestDurationMillisecondsSinceClampsFutureStart(t *testing.T) {
	start := time.Now().Add(2 * time.Second)

	if got := DurationMillisecondsSince(start); got != 0 {
		t.Fatalf("expected future start to clamp to 0ms, got %d", got)
	}
}

func TestDurationMillisecondsSinceReportsElapsedMilliseconds(t *testing.T) {
	start := time.Now().Add(-1100 * time.Millisecond)

	if got := DurationMillisecondsSince(start); got < 1000 {
		t.Fatalf("expected elapsed milliseconds to be at least 1000, got %d", got)
	}
}
