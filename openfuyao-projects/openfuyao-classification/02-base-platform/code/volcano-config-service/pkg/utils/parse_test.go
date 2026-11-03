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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseRanges(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    []int
		wantErr bool
	}{
		{
			name:    "single number",
			input:   "3",
			want:    []int{3},
			wantErr: false,
		},
		{
			name:    "range",
			input:   "1-3",
			want:    []int{1, 2, 3},
			wantErr: false,
		},
		{
			name:    "mixed",
			input:   "0,2-4,6",
			want:    []int{0, 2, 3, 4, 6},
			wantErr: false,
		},
		{
			name:    "empty string",
			input:   "",
			want:    []int{},
			wantErr: false,
		},
		{
			name:    "whitespace string",
			input:   "  ",
			want:    []int{},
			wantErr: false,
		},
		{
			name:    "invalid format",
			input:   "abc",
			want:    []int{},
			wantErr: true,
		},
		{
			name:    "invalid range start",
			input:   "a-3",
			want:    []int{},
			wantErr: true,
		},
		{
			name:    "invalid range end",
			input:   "1-b",
			want:    []int{},
			wantErr: true,
		},
		{
			name:    "too many dashes",
			input:   "1-2-3",
			want:    []int{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseRanges(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}
