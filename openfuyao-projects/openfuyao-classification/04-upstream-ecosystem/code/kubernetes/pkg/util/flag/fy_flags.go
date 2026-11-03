/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package flag

import (
	"fmt"
	"sort"
	"strings"
)

const stringLen = 2

type MapStringMapStringString struct {
	Map         *map[string]map[string]string
	initialized bool
}

func NewMapStringMapStringString(m *map[string]map[string]string) *MapStringMapStringString {
	return &MapStringMapStringString{Map: m}
}

func (m *MapStringMapStringString) Set(value string) error {
	if m == nil || m.Map == nil {
		return fmt.Errorf("MapStringMapStringString initialized failed")
	}
	if !m.initialized || *m.Map == nil {
		m.initialized = true
		*m.Map = make(map[string]map[string]string)
	}

	for _, s := range strings.Split(value, ";") {
		if len(s) == 0 {
			continue
		}

		subStr := strings.Split(s, ":")
		if len(subStr) != stringLen {
			return fmt.Errorf("invalid format for %s, expect string:string", s)
		}
		(*m.Map)[subStr[0]] = make(map[string]string)
		for _, ss := range splitAtCommas(subStr[1]) {
			if len(ss) == 0 {
				continue
			}
			arr := strings.SplitN(ss, "=", stringLen)
			if len(arr) < stringLen {
				continue
			}
			k := strings.TrimSpace(arr[0])
			v := strings.Trim(strings.TrimSpace(arr[1]), "[]")
			(*m.Map)[subStr[0]][k] = v
		}
	}
	return nil
}

func (m *MapStringMapStringString) String() string {
	if m == nil || m.Map == nil {
		return ""
	}
	var res []string
	for k1, v1 := range *m.Map {
		tmpStr := k1 + ":"
		var pairs []string
		for k2, v2 := range v1 {
			if strings.Contains(v2, ",") {
				pairs = append(pairs, fmt.Sprintf("%s=[%s]", k2, v2))
			} else {
				pairs = append(pairs, fmt.Sprintf("%s=%s", k2, v2))
			}
		}
		sort.Strings(pairs)
		tmpStr += strings.Join(pairs, ",")
		res = append(res, tmpStr)
	}
	sort.Strings(res)
	return strings.Join(res, ";")
}

// Type implements github.com/spf13/pflag.Value
func (m *MapStringMapStringString) Type() string {
	return "mapStringMapStringString"
}

// Empty implements OmitEmpty
func (m *MapStringMapStringString) Empty() bool {
	return len(*m.Map) == 0
}

// splitAtCommas split a string at commas, ignoring commas in brackets.
func splitAtCommas(s string) []string {
	var (
		res        []string
		beg        int
		inBrackets bool
	)

	for i := 0; i < len(s); i++ {
		if s[i] == ',' && !inBrackets {
			res = append(res, s[beg:i])
			beg = i + 1
		} else if s[i] == '[' || s[i] == ']' {
			if !inBrackets {
				inBrackets = true
			} else if i > 0 && s[i-1] != '\\' {
				inBrackets = false
			}
		}
	}
	return append(res, s[beg:])
}
