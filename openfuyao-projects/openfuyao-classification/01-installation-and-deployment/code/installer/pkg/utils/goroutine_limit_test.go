/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package utils

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewGoRoutineLimits(t *testing.T) {
	tests := []struct {
		name    string
		wantCap int
	}{
		{
			name:    "creates limiter with correct capacity",
			wantCap: runtime.NumCPU() * LogicalCPUMultiple,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			limiter := NewGoRoutineLimits()
			assert.NotNil(t, limiter, "limiter should not be nil")
			assert.Equal(t, tt.wantCap, cap(limiter.ch), "channel capacity mismatch")
		})
	}
}

func TestGoRoutineLimit_Add(t *testing.T) {
	const len0, len1, len2 = 0, 1, 2
	t.Run("adds token to channel", func(t *testing.T) {
		limiter := &GoRoutineLimit{ch: make(chan int, 2)}

		// Before Add
		assert.Equal(t, len0, len(limiter.ch), "channel should be empty")

		limiter.Add()

		// After Add
		assert.Equal(t, len1, len(limiter.ch), "channel should have one token")

		limiter.Add()
		assert.Equal(t, len2, len(limiter.ch), "channel should be full")
	})
}
