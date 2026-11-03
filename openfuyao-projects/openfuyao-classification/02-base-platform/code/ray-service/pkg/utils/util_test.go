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

package utils

import (
	"reflect"
	"testing"
)

// Common test structures
type filterTestCase struct {
	name    string
	ori     []string
	f       func(string) bool
	wantRes []string
}
type mapTestCase struct {
	name    string
	ori     []string
	f       func(string) string
	wantRes []string
}
type mapInPlaceTestCase struct {
	name string
	ori  []string
	f    func(*string)
}
type reduceTestCase struct {
	name    string
	ori     []string
	f       func(string, string) string
	init    string
	wantRes string
}

func TestFilter(t *testing.T) {
	tests := []filterTestCase{
		{
			name:    "test1",
			ori:     []string{"a", "b", "c"},
			f:       func(s string) bool { return s == "a" },
			wantRes: []string{"a"},
		},
		// Add more test cases here
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRes := Filter(tt.ori, tt.f); !reflect.DeepEqual(gotRes, tt.wantRes) {
				t.Errorf("Filter() = %v, want %v", gotRes, tt.wantRes)
			}
		})
	}
}
func TestMap(t *testing.T) {
	tests := []mapTestCase{
		{
			name: "test1",
			ori:  []string{"a", "b", "c"},
			f: func(s string) string {
				return s + "1"
			},
			wantRes: []string{"a1", "b1", "c1"},
		},
		// Add more test cases here
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRes := Map(tt.ori, tt.f); !reflect.DeepEqual(gotRes, tt.wantRes) {
				t.Errorf("Map() = %v, want %v", gotRes, tt.wantRes)
			}
		})
	}
}
func TestMapInPlace(t *testing.T) {
	tests := []mapInPlaceTestCase{
		{
			name: "test1",
			ori:  []string{"a", "b", "c"},
			f: func(s *string) {
				*s = *s + "1"
			},
		},
		// Add more test cases here
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := make([]string, len(tt.ori))
			copy(original, tt.ori)
			MapInPlace(tt.ori, tt.f)
			expected := []string{"a1", "b1", "c1"}
			if !reflect.DeepEqual(tt.ori, expected) {
				t.Errorf("MapInPlace() changed %v to %v, want %v", original, tt.ori, expected)
			}
		})
	}
}
func TestReduce(t *testing.T) {
	tests := []reduceTestCase{
		{
			name:    "test1",
			ori:     []string{"a", "b", "c"},
			f:       func(a, b string) string { return a + b },
			init:    "",
			wantRes: "abc",
		},
		// Add more test cases here
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRes := Reduce(tt.ori, tt.f, tt.init); gotRes != tt.wantRes {
				t.Errorf("Reduce() = %v, want %v", gotRes, tt.wantRes)
			}
		})
	}
}
