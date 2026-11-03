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
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"
)

func TestMapStringMapStringString(t *testing.T) {
	testCases := []struct {
		name      string
		argc      string
		expectErr bool
		expectVal map[string]map[string]string
	}{
		{
			name: "normal por-reserved config",
			argc: "omu:cpu=60m,memory=60Mi;pod-reserved:cpu=100m,memory=100Mi",
			expectVal: map[string]map[string]string{
				"pod-reserved": {
					"cpu":    "100m",
					"memory": "100Mi",
				},
				"omu": {
					"cpu":    "60m",
					"memory": "60Mi",
				},
			},
		},
		{
			name: "nested por-reserved config",
			argc: "pod-reserved:cpu=100m,memory=100Mi,omu=[cpu=60m,mem=60Mi]",
			expectVal: map[string]map[string]string{
				"pod-reserved": {
					"cpu":    "100m",
					"memory": "100Mi",
					"omu":    "cpu=60m,mem=60Mi",
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fs := pflag.NewFlagSet("blah", pflag.PanicOnError)
			podReserved := map[string]map[string]string{}
			msmss := NewMapStringMapStringString(&podReserved)
			fs.Var(msmss, "pod-reserved", "A set of ResourceName=ResourceQuantity")

			var err error
			func() {
				defer func() {
					if r := recover(); r != nil {
						err = r.(error)
					}
				}()
				fs.Parse(strings.Split("blah --pod-reserved="+tc.argc, " "))
			}()

			if tc.expectErr && err == nil {
				t.Errorf("%v, did not observe an expected error", tc.name)
				return
			}
			if !tc.expectErr && err != nil {
				t.Errorf("%v, observe an unexpected error: %v", tc.name, err)
				return
			}

			if !reflect.DeepEqual(tc.expectVal, podReserved) {
				t.Errorf("%v, unexpected ip: expected %q, saw %q, tc.e", tc.name, tc.expectVal, podReserved)
			}

			if msmss.String() != tc.argc {
				t.Errorf("%v, unexpected result for pof reservred2, expected '%v', actual: '%v'", tc.name, msmss.String(), tc.argc)
			}
		})
	}
}

func TestMapIsNil(t *testing.T) {
	msmss := NewMapStringMapStringString(nil)
	str := msmss.String()
	if str != "" {
		t.Errorf("unexpected result: %v", str)
	}
}

func TestMapSet(t *testing.T) {
	msmss := NewMapStringMapStringString(nil)
	err := msmss.Set("test")
	if err == nil {
		t.Errorf("set should be error")
	}
	msmss = buildMap()
	err = msmss.Set("")
	if err != nil {
		t.Errorf("set should not be error %s", err)
	}
	err = msmss.Set("test:test:test:test")
	if err == nil {
		t.Errorf("set should be error")
	}
}

func TestType(t *testing.T) {
	msmss := buildMap()
	tp := msmss.Type()
	if tp != "mapStringMapStringString" {
		t.Errorf("type should be mapStringMapStringString")
	}
}

func TestEmpty(t *testing.T) {
	msmss := buildMap()
	empty := msmss.Empty()
	if !empty {
		t.Errorf("map should be empty")
	}
}

func buildMap() *MapStringMapStringString {
	podReserved := make(map[string]map[string]string)
	return NewMapStringMapStringString(&podReserved)
}
