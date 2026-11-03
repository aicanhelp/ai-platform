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
)

func TestEscapeSpecialChars(t *testing.T) {
	tests := []struct {
		name     string
		arg      string
		expected string
	}{
		{
			name:     "double quote",
			arg:      `hello "world"`,
			expected: `hello "world"`,
		},
		{
			name:     "backspace",
			arg:      "hello\bworld",
			expected: `hello\bworld`,
		},
		{
			name:     "form feed",
			arg:      "hello\fworld",
			expected: `hello\fworld`,
		},
		{
			name:     "vertical tab",
			arg:      "hello\vworld",
			expected: `hello\vworld`,
		},
		{
			name:     "mixed special chars",
			arg:      "\n\r\t\b\f\v",
			expected: `\n\r\t\b\f\v`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EscapeSpecialChars(tt.arg)
			if got != tt.expected {
				t.Errorf("EscapeSpecialChars(%q) = %q, want %q", tt.arg, got, tt.expected)
			}
		})
	}
}
