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
		input, want string
	}{
		{"", ""},
		{"Normal text", "Normal text"},
		{"Line1\nLine2", "Line1\\nLine2"},
		{"Text\rReturn", "Text\\rReturn"},
		{"Tab\tHere", "Tab\\tHere"},
		{`Quote "me"`, `Quote "me"`},
		{"Back\bspace", "Back\\bspace"},
		{"Page\fBreak", "Page\\fBreak"},
		{"Vert\vTab", "Vert\\vTab"},
		{"中文\n日本", "中文\\n日本"},
		{`C:\Path`, `C:\Path`},
		{"\n\r\t\"\b\f\v", `\n\r\t"\b\f\v`},
	}

	for _, tt := range tests {
		got := EscapeSpecialChars(tt.input)
		if got != tt.want {
			t.Errorf("EscapeSpecialChars(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}
