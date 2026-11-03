/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package mutx

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewGlobalLocks(t *testing.T) {
	locks := NewGlobalLocks()
	assert.NotNil(t, locks)
}

func TestGlobalLocksTryAcquire(t *testing.T) {
	locks := NewGlobalLocks()

	result := locks.TryAcquire("test-id")
	assert.True(t, result, "first acquire should succeed")

	result = locks.TryAcquire("test-id")
	assert.False(t, result, "second acquire should fail")

	result = locks.TryAcquire("another-id")
	assert.True(t, result, "acquire different id should succeed")
}

func TestGlobalLocksRelease(t *testing.T) {
	locks := NewGlobalLocks()

	result := locks.TryAcquire("test-id")
	assert.True(t, result)

	locks.Release("test-id")

	result = locks.TryAcquire("test-id")
	assert.True(t, result, "should be able to acquire after release")
}

func TestGlobalLocksReleaseNonExistent(t *testing.T) {
	locks := NewGlobalLocks()
	// Should not panic
	locks.Release("non-existent")
}
