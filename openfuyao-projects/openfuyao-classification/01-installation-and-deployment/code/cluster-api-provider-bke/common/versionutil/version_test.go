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
	"bytes"
	"testing"
	"text/template"

	"github.com/stretchr/testify/assert"
)

func TestParseVersion(t *testing.T) {
	tests := []struct {
		name    string
		version string
	}{
		{"valid semver", "v1.23.0"},
		{"without v prefix", "1.23.0"},
		{"with patch", "1.23.5"},
		{"invalid version", "invalid"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := parseVersion(tt.version)
			assert.NotNil(t, v)
		})
	}
}

func TestK8sVersionFuncMap_vgt(t *testing.T) {
	funcMap := K8sVersionFuncMap()
	tmpl := template.Must(template.New("test").Funcs(*funcMap).Parse("{{vgt .src .dst}}"))

	tests := []struct {
		name     string
		src      string
		dst      string
		expected string
	}{
		{"greater", "v1.24.0", "v1.23.0", "true"},
		{"not greater", "v1.23.0", "v1.24.0", "false"},
		{"equal", "v1.23.0", "v1.23.0", "false"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := tmpl.Execute(&buf, map[string]string{"src": tt.src, "dst": tt.dst})
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, buf.String())
		})
	}
}

func TestK8sVersionFuncMap_vlt(t *testing.T) {
	funcMap := K8sVersionFuncMap()
	tmpl := template.Must(template.New("test").Funcs(*funcMap).Parse("{{vlt .src .dst}}"))

	tests := []struct {
		name     string
		src      string
		dst      string
		expected string
	}{
		{"less", "v1.23.0", "v1.24.0", "true"},
		{"not less", "v1.24.0", "v1.23.0", "false"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := tmpl.Execute(&buf, map[string]string{"src": tt.src, "dst": tt.dst})
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, buf.String())
		})
	}
}

func TestK8sVersionFuncMap_veq(t *testing.T) {
	funcMap := K8sVersionFuncMap()
	tmpl := template.Must(template.New("test").Funcs(*funcMap).Parse("{{veq .src .dst}}"))

	tests := []struct {
		name     string
		src      string
		dst      string
		expected string
	}{
		{"equal", "v1.23.0", "v1.23.0", "true"},
		{"not equal", "v1.23.0", "v1.24.0", "false"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := tmpl.Execute(&buf, map[string]string{"src": tt.src, "dst": tt.dst})
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, buf.String())
		})
	}
}

func TestK8sVersionFuncMap_vgte(t *testing.T) {
	funcMap := K8sVersionFuncMap()
	tmpl := template.Must(template.New("test").Funcs(*funcMap).Parse("{{vgte .src .dst}}"))

	tests := []struct {
		name     string
		src      string
		dst      string
		expected string
	}{
		{"greater", "v1.24.0", "v1.23.0", "true"},
		{"equal", "v1.23.0", "v1.23.0", "true"},
		{"less", "v1.22.0", "v1.23.0", "false"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := tmpl.Execute(&buf, map[string]string{"src": tt.src, "dst": tt.dst})
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, buf.String())
		})
	}
}

func TestK8sVersionFuncMap_vlte(t *testing.T) {
	funcMap := K8sVersionFuncMap()
	tmpl := template.Must(template.New("test").Funcs(*funcMap).Parse("{{vlte .src .dst}}"))

	tests := []struct {
		name     string
		src      string
		dst      string
		expected string
	}{
		{"less", "v1.22.0", "v1.23.0", "true"},
		{"equal", "v1.23.0", "v1.23.0", "true"},
		{"greater", "v1.24.0", "v1.23.0", "false"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := tmpl.Execute(&buf, map[string]string{"src": tt.src, "dst": tt.dst})
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, buf.String())
		})
	}
}

func TestK8sVersionFuncMap_vne(t *testing.T) {
	funcMap := K8sVersionFuncMap()
	tmpl := template.Must(template.New("test").Funcs(*funcMap).Parse("{{vne .src .dst}}"))

	tests := []struct {
		name     string
		src      string
		dst      string
		expected string
	}{
		{"not equal", "v1.23.0", "v1.24.0", "true"},
		{"equal", "v1.23.0", "v1.23.0", "false"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := tmpl.Execute(&buf, map[string]string{"src": tt.src, "dst": tt.dst})
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, buf.String())
		})
	}
}
