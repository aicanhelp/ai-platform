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
	"fmt"
	"time"
)

// WaitFor polls check every pollInterval until it returns true or timeout
// elapses, returning an error on timeout.
func WaitFor(desc string, timeout time.Duration, check func() bool) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if check() {
			return nil
		}
		time.Sleep(pollInterval)
	}
	return fmt.Errorf("timed out after %v waiting for %s", timeout, desc)
}

// WaitForNot asserts that check stays false for the whole window; it returns
// an error as soon as check becomes true.
func WaitForNot(desc string, window time.Duration, check func() bool) error {
	deadline := time.Now().Add(window)
	for time.Now().Before(deadline) {
		if check() {
			return fmt.Errorf("unexpected condition met within %v: %s", window, desc)
		}
		time.Sleep(pollInterval)
	}
	return nil
}
