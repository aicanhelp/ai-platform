/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package template

import (
	"fmt"
	"reflect"
	"testing"
	"text/template"

	"github.com/blang/semver"
)

var initGot = UtilFuncMap()
var initGotK8sVersionFuncMap = K8sVersionFuncMap()

func TestCommonFuncMap(t *testing.T) {
	t.Run("test", func(t *testing.T) {
		if got := CommonFuncMap(); len(got) == 0 {
			t.Errorf("CommonFuncMap() = %v", got)
		}
	})
}

func TestDefaultFuncMap(t *testing.T) {
	t.Run("test1", func(t *testing.T) {
		if got := DefaultFuncMap(); len(got) == 0 {
			t.Errorf("DefaultFuncMap() = %v", got)
		}
	})

	got := DefaultFuncMap()
	fn, ok := got["defaultFalse"].(func(interface{}) string)
	if !ok {
		t.Error()
	}
	t.Run("test2", func(t *testing.T) {
		if r := fn(true); r != "true" {
			t.Errorf("DefaultFuncMap() defaultFalse(true) = %v", r)
		}
	})

	t.Run("test3", func(t *testing.T) {
		if r := fn(false); r != "false" {
			t.Errorf("DefaultFuncMap() defaultFalse(false) = %v", r)
		}
	})
}
func TestK8sVersionFuncMap(t *testing.T) {
	t.Run("K8sVersionFuncMap", func(t *testing.T) {
		if gotK8sVersionFuncMap := K8sVersionFuncMap(); len(gotK8sVersionFuncMap) == 0 {
			t.Errorf("K8sVersionFuncMap() = %v", gotK8sVersionFuncMap)
		}
	})
}
func TestK8sVersionFuncMapVgt(t *testing.T) {
	t.Run("Vgt", func(t *testing.T) {
		fn, err := getVersionComparisonFunc("vgt")
		if err != nil {
			t.Error(err)
		}
		if r := fn("1.1.1", "1.1.2"); r {
			t.Errorf("DefaultFuncMap() defaultFalse(true) = %v", r)
		}
	})
}
func TestK8sVersionFuncMapVlt(t *testing.T) {
	t.Run("Vlt", func(t *testing.T) {
		fn, err := getVersionComparisonFunc("vlt")
		if err != nil {
			t.Error(err)
		}
		if r := fn("1.1.1", "1.1.2"); !r {
			t.Errorf("DefaultFuncMap() defaultFalse(true) = %v", r)
		}
	})
}
func TestK8sVersionFuncMapVeq(t *testing.T) {
	t.Run("Veq", func(t *testing.T) {

		fn, err := getVersionComparisonFunc("veq")
		if err != nil {
			t.Error(err)
		}
		if r := fn("1.1.1", "1.1.2"); r {
			t.Errorf("DefaultFuncMap() defaultFalse(true) = %v", r)
		}
	})
}
func TestK8sVersionFuncMapVgte(t *testing.T) {
	t.Run("Vgte", func(t *testing.T) {
		fn, err := getVersionComparisonFunc("vgte")
		if err != nil {
			t.Error(err)
		}
		if r := fn("1.1.1", "1.1.2"); r {
			t.Errorf("DefaultFuncMap() defaultFalse(true) = %v", r)
		}
	})
}
func TestK8sVersionFuncMapVlte(t *testing.T) {
	t.Run("Vlte", func(t *testing.T) {
		fn, err := getVersionComparisonFunc("vlte")
		if err != nil {
			t.Error(err)
		}
		if r := fn("1.1.1", "1.1.2"); !r {
			t.Errorf("DefaultFuncMap() defaultFalse(true) = %v", r)
		}
	})
}
func TestK8sVersionFuncMapVne(t *testing.T) {
	t.Run("Vne", func(t *testing.T) {
		fn, err := getVersionComparisonFunc("vne")
		if err != nil {
			t.Error(err)
		}
		if r := fn("1.1.1", "1.1.2"); !r {
			t.Errorf("DefaultFuncMap() defaultFalse(true) = %v", r)
		}
	})
}

func TestMergeFuncMap(t *testing.T) {
	type args struct {
		f1 template.FuncMap
		f2 template.FuncMap
	}
	tests := []struct {
		name string
		args args
		want template.FuncMap
	}{
		{
			name: "test",
			args: args{
				f1: template.FuncMap{
					"aa": "aa",
				},
				f2: template.FuncMap{
					"bb": "bb",
				},
			},
			want: template.FuncMap{
				"aa": "aa",
				"bb": "bb",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MergeFuncMap(tt.args.f1, tt.args.f2); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MergeFuncMap() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMergeFuncMapList(t *testing.T) {
	templateMapa := template.FuncMap{
		"aa": "aa",
	}
	templateMapb := template.FuncMap{
		"bb": "bb",
	}
	templateMapc := template.FuncMap{
		"aa": "aa",
		"bb": "bb",
	}
	type args struct {
		funcMaps []template.FuncMap
	}
	tests := []struct {
		name string
		args args
		want template.FuncMap
	}{
		{
			name: "test",
			args: args{
				funcMaps: []template.FuncMap{
					templateMapa,
					templateMapb,
				},
			},
			want: templateMapc,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MergeFuncMapList(tt.args.funcMaps...); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MergeFuncMapList() = %v, want %v", got, tt.want)
			}
		})
	}
}
func TestUtilFuncMap(t *testing.T) {
	t.Run("UtilFuncMap", func(t *testing.T) {
		if got := UtilFuncMap(); len(got) == 0 {
			t.Errorf("UtilFuncMap() = %v", got)
		}
	})
}
func TestUtilFuncMapB64encode(t *testing.T) {
	t.Run("B64encode", func(t *testing.T) {
		fn, err := getUtilFuncMapB64encodeFunc("b64encode")
		if err != nil {
			t.Error()
		}

		if r := fn("1.1.1"); len(r) == 0 {
			t.Errorf("%s = %v", `UtilFuncMap() b64encode("1.1.1")`, r)
		}
	})
}
func TestUtilFuncMapSplit(t *testing.T) {
	t.Run("Split", func(t *testing.T) {
		splitFn, err := getUtilFuncMapArrayStringFunc("split")
		if err != nil {
			t.Error()
		}
		if r := splitFn("1.1.1", "."); !reflect.DeepEqual(r, []string{"1", "1", "1"}) {
			t.Errorf("%s = %v", `UtilFuncMap() split("1.1.1")`, r)
		}
	})
}
func TestUtilFuncMapStringToSliceString(t *testing.T) {
	t.Run("StringToSliceString", func(t *testing.T) {
		stringToSliceStringFn, err := getUtilFuncMapStringFunc("stringToSliceString")
		if err != nil {
			t.Error()
		}
		if r := stringToSliceStringFn("1.1.1", "."); r != `["1","1","1"]` {
			t.Errorf("%s = %v", `UtilFuncMap() stringToSliceString("1.1.1")`, r)
		}
	})
}
func TestUtilFuncMapInt(t *testing.T) {

	initFnArgs := "1111"
	returnValue := 1111
	t.Run("Int", func(t *testing.T) {
		intFn, err := getUtilFuncMapIntFunc("int")
		if err != nil {
			t.Error()
		}
		if r := intFn(initFnArgs); r != returnValue {
			t.Errorf("%s = %v", `UtilFuncMap() int("1111")`, r)
		}
	})
}
func TestUtilFuncMapIndent(t *testing.T) {
	t.Run("Indent", func(t *testing.T) {
		indentFn, err := getUtilFuncMapStringFunc("indent")
		if err != nil {
			t.Error()
		}
		if r := indentFn("1.1.1", "1.2"); len(r) == 0 {
			t.Errorf("%s = %v", `UtilFuncMap() indent("1.1.1", "1.1")`, r)
		}
	})
}

func TestParseVersion(t *testing.T) {
	type args struct {
		version string
	}
	tests := []struct {
		name string
		args args
		want semver.Version
	}{
		{
			name: "test1",
			args: args{
				version: "1.2.0",
			},
			want: semver.Version{
				Major: 1,
				Minor: 2,
				Patch: 0,
			},
		},
		{
			name: "test2",
			args: args{
				version: "v1.2.0",
			},
			want: semver.Version{
				Major: 1,
				Minor: 2,
				Patch: 0,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := parseVersion(tt.args.version); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseVersion() = %v, want %v", got, tt.want)
			}
		})
	}
}

// 辅助函数：提取类型断言逻辑
func getVersionComparisonFunc(key string) (func(string, string) bool, error) {
	fn, ok := initGotK8sVersionFuncMap[key]
	if !ok {
		return nil, fmt.Errorf("function key %q not found", key)
	}

	comparisonFn, ok := fn.(func(string, string) bool)
	if !ok {
		return nil, fmt.Errorf("invalid function type for key %q", key)
	}

	return comparisonFn, nil
}

// 辅助函数：提取类型断言逻辑
func getUtilFuncMapArrayStringFunc(key string) (func(string, string) []string, error) {
	fn, ok := initGot[key]
	if !ok {
		return nil, fmt.Errorf("function key %q not found", key)
	}

	comparisonFn, ok := fn.(func(string, string) []string)
	if !ok {
		return nil, fmt.Errorf("invalid function type for key %q", key)
	}

	return comparisonFn, nil
}
func getUtilFuncMapStringFunc(key string) (func(string, string) string, error) {
	fn, ok := initGot[key]
	if !ok {
		return nil, fmt.Errorf("function key %q not found", key)
	}

	comparisonFn, ok := fn.(func(string, string) string)
	if !ok {
		return nil, fmt.Errorf("invalid function type for key %q", key)
	}

	return comparisonFn, nil
}
func getUtilFuncMapB64encodeFunc(key string) (func(s interface{}) string, error) {
	fn, ok := initGot[key]
	if !ok {
		return nil, fmt.Errorf("function key %q not found", key)
	}

	comparisonFn, ok := fn.(func(s interface{}) string)
	if !ok {
		return nil, fmt.Errorf("invalid function type for key %q", key)
	}

	return comparisonFn, nil
}
func getUtilFuncMapIntFunc(key string) (func(s string) int, error) {
	fn, ok := initGot[key]
	if !ok {
		return nil, fmt.Errorf("function key %q not found", key)
	}

	comparisonFn, ok := fn.(func(s string) int)
	if !ok {
		return nil, fmt.Errorf("invalid function type for key %q", key)
	}

	return comparisonFn, nil
}
