/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package template

import (
	"fmt"
	"strconv"
	"strings"
	"text/template"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/versionutil"
)

// MergeFuncMap merge two funcMap
func MergeFuncMap(f1, f2 template.FuncMap) template.FuncMap {
	for k, v := range f2 {
		f1[k] = v
	}
	return f1
}

func MergeFuncMapList(funcMaps ...template.FuncMap) template.FuncMap {
	f := funcMaps[0]
	for _, funcMap := range funcMaps[1:] {
		fm := funcMap
		MergeFuncMap(f, fm)
	}
	return f
}

func CommonFuncMap() template.FuncMap {
	return MergeFuncMapList(K8sVersionFuncMap(), DefaultFuncMap(), UtilFuncMap())
}

// K8sVersionFuncMap return k8s version func map
func K8sVersionFuncMap() template.FuncMap {
	return *versionutil.K8sVersionFuncMap()
}

func DefaultFuncMap() template.FuncMap {
	return template.FuncMap{
		"defaultFalse": func(b interface{}) string {
			//转为字符串
			t := fmt.Sprintf("%v", b)
			if t != "true" && t != "false" {
				return "false"
			}
			return t
		},
		// etc
	}
}

func UtilFuncMap() template.FuncMap {
	return template.FuncMap{
		"b64encode": func(s interface{}) string {
			// convert to string
			return utils.B64Encode(fmt.Sprintf("%v", s))
		},
		"split": func(s string, sep string) []string {
			return strings.Split(s, sep)
		},
		// stringToSliceString 将字符串转换为切片字符串 如输入字符串 "1,2,3" 转换为字符串 ["1","2","3"]
		"stringToSliceString": func(s string, sep string) string {
			if s == "" {
				return "[]"
			}
			srcString := strings.Split(s, sep)
			dstString := ""
			for str := range srcString {
				dstString += "\"" + srcString[str] + "\","
			}
			return "[" + dstString[:len(dstString)-1] + "]"
		},
		"int": func(s string) int {
			i, err := strconv.Atoi(s)
			if err != nil {
				// Return 0 if conversion fails
				return 0
			}
			return i
		},
		"indent": func(s, prefix string) string {
			lines := strings.Split(s, "\n")
			indentedLines := make([]string, len(lines))

			// 添加前缀，但跳过第一行
			indentedLines[0] = lines[0]
			for i := 1; i < len(lines); i++ {
				indentedLines[i] = prefix + lines[i]
			}

			return strings.Join(indentedLines, "\n")
		},
	}
}
