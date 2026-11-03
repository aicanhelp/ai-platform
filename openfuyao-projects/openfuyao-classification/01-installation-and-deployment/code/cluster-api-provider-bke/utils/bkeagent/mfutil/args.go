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

package mfutil

import "strings"

func DedupFlags(args []string) []string {
	if len(args) == 0 {
		return args
	}

	out := make([]string, 0, len(args))
	pos := make(map[string]int)
	for i, arg := range args {
		if i == 0 || !strings.HasPrefix(arg, "--") {
			out = append(out, arg)
			continue
		}
		key := flagKey(arg)
		if idx, ok := pos[key]; ok {
			out[idx] = arg
			continue
		}
		pos[key] = len(out)
		out = append(out, arg)
	}
	return out
}

func flagKey(arg string) string {
	arg = strings.TrimPrefix(arg, "--")
	if i := strings.IndexByte(arg, '='); i >= 0 {
		return arg[:i]
	}
	return arg
}
