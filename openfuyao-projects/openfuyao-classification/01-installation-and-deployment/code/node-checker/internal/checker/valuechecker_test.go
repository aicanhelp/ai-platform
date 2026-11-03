/*
Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/

// check  package provides check functions for oschecktool
package checker

import (
	"fmt"
	"testing"
)

func TestValueChecker(t *testing.T) {

	type initCase struct {
		valueType string
		expect    string
		hasErr    bool
	}
	type checkCase struct {
		value string
		rslt  bool
	}

	type testCase struct {
		init  initCase
		check []checkCase
	}

	cases := []testCase{
		{init: initCase{"int", "abc", true}},
		{init: initCase{"int", "", true}},
		{init: initCase{"int", "12345678901234567890123456789012345678901234567890", true}},
		{init: initCase{"int", " 123", false},
			check: []checkCase{
				{"1", false},
				{"abc", false},
				{"123", true},
				{"1.3", false},
			}},
		{init: initCase{"integer", " 123 ", false},
			check: []checkCase{
				{"1", false},
				{"abc", false},
				{"123", true},
				{" 123", true},
				{"1.3", false},
			}},
		{init: initCase{"int-range", "abc", true}},
		{init: initCase{"int-range", "[,]", false}},
		{init: initCase{"int-range", "[a,]", true}},
		{init: initCase{"int-range", "[1.1,]", true}},
		{init: initCase{"int-range", "[2,1]", true}},
		{init: initCase{"int-range", "[1,3]", false},
			check: []checkCase{
				{"0", false},
				{"1", true},
				{"2", true},
				{"3", true},
				{"4", false},
			}},
		{init: initCase{"int-range", "[1,3)", false},
			check: []checkCase{
				{"0", false},
				{"1", true},
				{"2", true},
				{"3", false},
				{"4", false},
			}},
		{init: initCase{"int-range", "(1,3)", false},
			check: []checkCase{
				{"0", false},
				{"1", false},
				{"2", true},
				{"3", false},
				{"4", false},
				{"1.2", false},
			}},
		{init: initCase{"int-range", "(,3)", false},
			check: []checkCase{
				{"0", true},
				{"1", true},
				{"2", true},
				{"3", false},
				{"4", false},
			}},
		{init: initCase{"int-range", "[,3)", false},
			check: []checkCase{
				{"0", true},
				{"1", true},
				{"2", true},
				{"3", false},
				{"4", false},
			}},
		{init: initCase{"int-range", "[,)", false},
			check: []checkCase{
				{"0", true},
				{"100", true},
			}},
		{init: initCase{"int-range", "(,)", false},
			check: []checkCase{
				{"0", true},
				{"100", true},
			}},
		{init: initCase{"int-range", "(2,)", false},
			check: []checkCase{
				{"0", false},
				{"2", false},
				{"3", true},
				{"a", false},
			}},
		{init: initCase{"str", "", false},
			check: []checkCase{
				{"", true},
				{" ", true},
			}},
		{init: initCase{"str", " ", false},
			check: []checkCase{
				{"", true},
				{" ", true},
			}},
		{init: initCase{"str", "abc", false},
			check: []checkCase{
				{"abc", true},
				{" abc  ", true},
				{"a", false},
			}},
		{init: initCase{"re", "abc[.*", true}},
		{init: initCase{"re", "^abc.*$", false},
			check: []checkCase{
				{"abcde", true},
				{" abc  ", true},
				{" abc d", true},
				{"aabc d", false},
			}},
		{init: initCase{"version-range", "abc", true}},
		{init: initCase{"version-range", "1.2.3", true}},
		{init: initCase{"version-range", "(1.2.3)", true}},
		{init: initCase{"version-range", "(1.2.3,)", false},
			check: []checkCase{
				{"abc", false},
				{"1.2.3", false},
				{"1.2.4", true},
				{"1.3", true},
				{"9.9", true},
			}},
		{init: initCase{"version-range", "[1.2.3,)", false},
			check: []checkCase{
				{"abc", false},
				{"1.2.3", true},
				{"1.2", true},
				{"1.3", true},
				{"9.9", true},
			}},
		{init: initCase{"version-range", "[1.2.3,3]", false},
			check: []checkCase{
				{"1.2.3", true},
				{"3", true},
				{"3.1", true},
				{"4", false},
			}},
		{init: initCase{"version-range", "[1.2.3,3)", false},
			check: []checkCase{
				{"1.2.3", true},
				{"1.3.3", true},
				{"3", false},
				{"3.1", false},
				{"4", false},
			}},
		{init: initCase{"version-range", "[,3)", false},
			check: []checkCase{
				{"1.2.3", true},
				{"1.3.3", true},
				{"3", false},
				{"3.1", false},
			}},
		{init: initCase{"float", "1.1", true}},
	}

	for _, testCase := range cases {
		if len(testCase.check) == 0 {
			t.Run(fmt.Sprintf(`GetValueChecker("%s", "%s")`, testCase.init.valueType, testCase.init.expect),
				func(t *testing.T) {
					_, err := GetValueChecker(testCase.init.valueType, testCase.init.expect)
					if testCase.init.hasErr != (err != nil) {
						t.Errorf("GetValueChecker(%s, %s) error=%v, wantErr %v",
							testCase.init.valueType, testCase.init.expect, err, testCase.init.hasErr)
						return
					}
				})
		} else {
			for _, checkItem := range testCase.check {
				t.Run(fmt.Sprintf(`GetValueChecker("%s", "%s").Check("%s")`,
					testCase.init.valueType, testCase.init.expect, checkItem.value),
					func(t *testing.T) {
						checker, err := GetValueChecker(testCase.init.valueType, testCase.init.expect)
						if testCase.init.hasErr != (err != nil) {
							t.Errorf("GetValueChecker(%s, %s) error=%v, wantErr %v",
								testCase.init.valueType, testCase.init.expect, err, testCase.init.hasErr)
							return
						}
						if rslt := checker.Check(checkItem.value); rslt != checkItem.rslt {
							t.Errorf(`GetValueChecker("%s", "%s").check("%s") = %v, want: %v`,
								testCase.init.valueType, testCase.init.expect, checkItem.value, rslt, checkItem.rslt)
						}
					})
			}

		}
	}
}
