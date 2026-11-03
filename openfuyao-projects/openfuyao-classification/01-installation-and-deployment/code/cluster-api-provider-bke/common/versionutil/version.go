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

package versionutil

import (
	"text/template"

	"github.com/blang/semver/v4"
)

// K8sVersionFuncMap returns k8s version comparison functions for template usage
func K8sVersionFuncMap() *template.FuncMap {
	return &template.FuncMap{
		"vgt": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.GT(dv)
		},
		"vlt": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.LT(dv)
		},
		"veq": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.EQ(dv)
		},
		"vgte": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.GTE(dv)
		},
		"vlte": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.LTE(dv)
		},
		"vne": func(src, dst string) bool {
			sv := parseVersion(src)
			dv := parseVersion(dst)
			return sv.NE(dv)
		},
	}
}

func parseVersion(version string) semver.Version {
	v, _ := semver.ParseTolerant(version)
	return v
}
