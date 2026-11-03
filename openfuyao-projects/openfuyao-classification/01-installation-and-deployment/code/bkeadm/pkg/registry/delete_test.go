/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package registry

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDelete(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		expectPanic bool
	}{
		{
			name:        "delete with valid args",
			args:        []string{"test-image:latest"},
			expectPanic: false,
		},
		{
			name:        "delete with no args",
			args:        []string{},
			expectPanic: false,
		},
		{
			name:        "delete with multiple args",
			args:        []string{"image1:latest", "image2:latest"},
			expectPanic: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{
				Args: tt.args,
			}

			assert.NotPanics(t, func() {
				op.Delete()
			})
		})
	}
}

func TestDeleteWithDockerPrefix(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		expectPanic bool
	}{
		{
			name:        "delete with docker prefix",
			args:        []string{"docker://test-image:latest"},
			expectPanic: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := &Options{
				Args: tt.args,
			}

			assert.NotPanics(t, func() {
				op.Delete()
			})
		})
	}
}
