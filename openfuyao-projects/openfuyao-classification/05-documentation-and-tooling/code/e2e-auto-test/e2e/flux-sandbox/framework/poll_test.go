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

package framework

import (
	"testing"
	"time"
)

func TestWaitForSucceeds(t *testing.T) {
	pollInterval = time.Millisecond
	calls := 0
	err := WaitFor("condition", 10*time.Second, func() bool {
		calls++
		return calls >= 2
	})
	if err != nil {
		t.Fatalf("WaitFor: %v", err)
	}
}

func TestWaitForTimesOut(t *testing.T) {
	pollInterval = time.Millisecond
	err := WaitFor("never true", 10*time.Millisecond, func() bool { return false })
	if err == nil {
		t.Fatal("expected timeout error")
	}
}

func TestWaitForNotTrips(t *testing.T) {
	pollInterval = time.Millisecond
	err := WaitForNot("stays false", 10*time.Second, func() bool { return true })
	if err == nil {
		t.Fatal("expected immediate trip error")
	}
}

func TestWaitForNotHolds(t *testing.T) {
	pollInterval = time.Millisecond
	err := WaitForNot("stays false", 10*time.Millisecond, func() bool { return false })
	if err != nil {
		t.Fatalf("WaitForNot: %v", err)
	}
}
