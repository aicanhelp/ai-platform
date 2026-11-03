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

package utils

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetNodeName(t *testing.T) {
	tests := []struct {
		name    string
		envVal  string
		want    string
		wantErr bool
	}{
		{
			name:    "valid node name",
			envVal:  "node-1",
			want:    "node-1",
			wantErr: false,
		},
		{
			name:    "empty node name",
			envVal:  "",
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envVal != "" {
				t.Setenv("NODE_NAME", tt.envVal)
			} else {
				os.Unsetenv("NODE_NAME")
			}

			got, err := GetNodeName()
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestIntSliceToFormattedString(t *testing.T) {
	tests := []struct {
		name  string
		slice []int
		want  string
	}{
		{
			name:  "multiple elements",
			slice: []int{1, 2, 3},
			want:  "1, 2, 3",
		},
		{
			name:  "single element",
			slice: []int{42},
			want:  "42",
		},
		{
			name:  "empty slice",
			slice: []int{},
			want:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IntSliceToFormattedString(tt.slice)
			assert.Equal(t, tt.want, got)
		})
	}
}
