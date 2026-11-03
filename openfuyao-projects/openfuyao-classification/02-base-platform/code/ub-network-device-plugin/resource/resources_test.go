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

package resource

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInitDefaultResources(t *testing.T) {
	res, err := InitDefaultResources()

	assert.NoError(t, err)
	assert.NotNil(t, res)
	assert.Len(t, res.SHAREDs, 1)
}

func TestNewResourceName(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    string
		wantErr bool
	}{
		{
			name:    "完整资源名",
			input:   "unifiedbus.com/test",
			want:    "unifiedbus.com/test",
			wantErr: false,
		},
		{
			name:    "添加前缀",
			input:   "test",
			want:    "unifiedbus.com/test",
			wantErr: false,
		},
		{
			name:    "资源名过长",
			input:   "test-" + string(make([]byte, 100)),
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewResourceName(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, Name(tt.want), got)
			}
		})
	}
}

func TestNewResource(t *testing.T) {
	tests := []struct {
		name    string
		pattern string
		nameStr string
		wantErr bool
	}{
		{
			name:    "成功创建资源",
			pattern: "*",
			nameStr: "test",
			wantErr: false,
		},
		{
			name:    "资源名无效",
			pattern: "*",
			nameStr: string(make([]byte, 100)),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewResource(tt.pattern, tt.nameStr)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, got)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, got)
				assert.Equal(t, Pattern(tt.pattern), got.Pattern)
			}
		})
	}
}

func TestName_Split(t *testing.T) {
	tests := []struct {
		name     string
		input    Name
		wantPart string
		wantName string
	}{
		{
			name:     "完整资源名",
			input:    "unifiedbus.com/test",
			wantPart: "unifiedbus.com",
			wantName: "test",
		},
		{
			name:     "无分隔符",
			input:    "test",
			wantPart: "",
			wantName: "test",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			part, name := tt.input.Split()
			assert.Equal(t, tt.wantPart, part)
			assert.Equal(t, tt.wantName, name)
		})
	}
}

func TestResources_AddSHAREResource(t *testing.T) {
	res := &Resources{}

	err := res.AddSHAREResource("*", "test")
	assert.NoError(t, err)
	assert.Len(t, res.SHAREDs, 1)

	err = res.AddSHAREResource("test*", "test2")
	assert.NoError(t, err)
	assert.Len(t, res.SHAREDs, 2)
}

func TestPattern_Matches(t *testing.T) {
	tests := []struct {
		name    string
		pattern Pattern
		input   string
		want    bool
	}{
		{
			name:    "通配符匹配",
			pattern: "*",
			input:   "test",
			want:    true,
		},
		{
			name:    "前缀匹配",
			pattern: "test*",
			input:   "test123",
			want:    true,
		},
		{
			name:    "不匹配",
			pattern: "test*",
			input:   "other",
			want:    false,
		},
		{
			name:    "精确匹配",
			pattern: "test",
			input:   "test",
			want:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.pattern.Matches(tt.input)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestWildCardToRegexp(t *testing.T) {
	tests := []struct {
		name     string
		pattern  string
		expected string
	}{
		{
			name:     "通配符",
			pattern:  "*",
			expected: ".*",
		},
		{
			name:     "前缀通配符",
			pattern:  "test*",
			expected: "test.*",
		},
		{
			name:     "中间通配符",
			pattern:  "test*123",
			expected: "test.*123",
		},
		{
			name:     "多个通配符",
			pattern:  "*test*",
			expected: ".*test.*",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := wildCardToRegexp(tt.pattern)
			assert.Equal(t, tt.expected, got)
		})
	}
}
