/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package common

import (
	"fmt"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
)

func AssertEquals(expect, act interface{}, t testing.TB) {
	equals(expect, act, t, 1)
}

func AssertIsNil(obtained interface{}, t testing.TB) {
	isNil(obtained, t, 1)
}

func AssertNotNil(obtained interface{}, t testing.TB) {
	notNil(obtained, t, 1)
}

func equals(expect, act interface{}, t testing.TB, caller int) {
	if !reflect.DeepEqual(expect, act) {
		_, file, line, _ := runtime.Caller(caller + 1)
		fmt.Printf("%s:%d: expect: %v (%T), got: %v (%T)\n",
			filepath.Base(file), line, expect, expect, act, act)
		t.FailNow()
	}
}

func isNil(obj interface{}, t testing.TB, caller int) {
	if !_isNil(obj) {
		_, file, line, _ := runtime.Caller(caller + 1)
		fmt.Printf("%s:%d: expected nil, got: %v\n", filepath.Base(file), line, obj)
		t.FailNow()
	}
}

func notNil(obj interface{}, t testing.TB, caller int) {
	if _isNil(obj) {
		_, file, line, _ := runtime.Caller(caller + 1)
		fmt.Printf("%s:%d: expected non-nil, got: %v\n", filepath.Base(file), line, obj)
		t.FailNow()
	}
}

func _isNil(obj interface{}) bool {
	if obj == nil {
		return true
	}

	switch v := reflect.ValueOf(obj); v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	default:
		return false
	}
}
