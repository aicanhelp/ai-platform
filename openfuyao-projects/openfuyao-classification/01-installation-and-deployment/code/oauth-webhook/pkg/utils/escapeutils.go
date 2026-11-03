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

// Package utils provide utils functions
package utils

import "bytes"

var escapeMap = map[string]string{
	"\n": `\n`,
	"\r": `\r`,
	"\t": `\t`,
	`\"`: `"`,
	"\b": `\b`,
	"\f": `\f`,
	"\v": `\v`,
}

// EscapeSpecialChars escape special characters
func EscapeSpecialChars(input string) string {
	var buffer bytes.Buffer

	for _, char := range input {
		str := string(char)
		if escaped, exists := escapeMap[str]; exists {
			buffer.WriteString(escaped)
		} else {
			buffer.WriteString(str)
		}
	}

	return buffer.String()
}
