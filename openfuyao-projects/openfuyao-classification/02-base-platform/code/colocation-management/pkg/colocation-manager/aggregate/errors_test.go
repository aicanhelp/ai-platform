/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-08
 */

package aggregate

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestNewKeyError test NewKeyError
func TestNewKeyError(t *testing.T) {
	tests := []struct {
		name     string
		key      interface{}
		expected KeyError
	}{
		{
			name:     "string key",
			key:      "test-key",
			expected: KeyError{key: "test-key"},
		},
		{
			name:     "integer key",
			key:      123,
			expected: KeyError{key: 123},
		},
		{
			name:     "float key",
			key:      45.67,
			expected: KeyError{key: 45.67},
		},
		{
			name:     "nil key",
			key:      nil,
			expected: KeyError{key: nil},
		},
		{
			name:     "struct key",
			key:      struct{ name string }{name: "test"},
			expected: KeyError{key: struct{ name string }{name: "test"}},
		},
		{
			name:     "slice key",
			key:      []string{"a", "b", "c"},
			expected: KeyError{key: []string{"a", "b", "c"}},
		},
		{
			name:     "map key",
			key:      map[string]int{"a": 1},
			expected: KeyError{key: map[string]int{"a": 1}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewKeyError(tt.key)
			assert.Equal(t, tt.expected, result)
			assert.Equal(t, tt.key, result.key)
		})
	}
}

// TestKeyError test KeyError
func TestKeyError(t *testing.T) {
	tests := []struct {
		name     string
		key      interface{}
		expected string
	}{
		{
			name:     "string key",
			key:      "missing-key",
			expected: "KeyError: missing-key",
		},
		{
			name:     "integer key",
			key:      42,
			expected: "KeyError: 42",
		},
		{
			name:     "float key",
			key:      3.14,
			expected: "KeyError: 3.14",
		},
		{
			name:     "boolean key",
			key:      true,
			expected: "KeyError: true",
		},
		{
			name:     "nil key",
			key:      nil,
			expected: "KeyError: <nil>", // 修改为更友好的格式
		},
		{
			name:     "array key",
			key:      [2]int{1, 2},
			expected: "KeyError: [1 2]",
		},
		{
			name:     "pointer key",
			key:      &struct{}{},
			expected: "KeyError: &{}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := KeyError{key: tt.key}
			result := err.Error()
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestImplementsErrorInterface test error interface
func TestImplementsErrorInterface(t *testing.T) {
	var err error = KeyError{key: "test"}
	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "KeyError: test")
}

// TestKeyAccess test key access
func TestKeyAccess(t *testing.T) {
	testKey := "custom-key"
	err := NewKeyError(testKey)

	assert.Equal(t, testKey, err.key)

	err.key = "modified-key"
	assert.Equal(t, "modified-key", err.key)
}

// TestEquality test equality
func TestEquality(t *testing.T) {
	key1 := "same-key"
	key2 := "different-key"

	err1 := NewKeyError(key1)
	err2 := NewKeyError(key1)
	err3 := NewKeyError(key2)

	assert.Equal(t, err1, err2)

	assert.NotEqual(t, err1, err3)
}

// TestComplexTypes test complex types
func TestComplexTypes(t *testing.T) {
	type CustomType struct {
		ID   int
		Name string
	}

	custom := CustomType{ID: 1, Name: "test"}
	err := NewKeyError(custom)

	expected := fmt.Sprintf("KeyError: {%d %s}", custom.ID, custom.Name)
	assert.Equal(t, expected, err.Error())
}

// TestEmptyStringKey test empty string
func TestEmptyStringKey(t *testing.T) {
	err := NewKeyError("")
	assert.Equal(t, "KeyError: ", err.Error())
}

// TestKSpecialCharacters test special characters
func TestKSpecialCharacters(t *testing.T) {
	specialKey := "key-with-special-chars!@#$%^&*()_+-=[]{}|;:',.<>/?`~"
	err := NewKeyError(specialKey)
	expected := "KeyError: " + specialKey
	assert.Equal(t, expected, err.Error())
}

// TestUnicodeCharacters test unicode
func TestUnicodeCharacters(t *testing.T) {
	unicodeKey := "测试-中文-😊-unicode"
	err := NewKeyError(unicodeKey)
	expected := "KeyError: " + unicodeKey
	assert.Equal(t, expected, err.Error())
}

// TestAsFunction
func TestAsFunction(t *testing.T) {
	var err error = NewKeyError("function-test")
	assert.Contains(t, err.Error(), "function-test")
}

// BenchmarkNewKeyError Benchmark tests
func BenchmarkNewKeyError(b *testing.B) {
	testKey := "benchmark-key"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NewKeyError(testKey)
	}
}

// BenchmarkComplexKey test ComplexKey
func BenchmarkComplexKey(b *testing.B) {
	complexKey := map[string]interface{}{
		"id":   123,
		"name": "benchmark",
		"data": []int{1, 2, 3, 4, 5},
	}
	err := NewKeyError(complexKey)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = err.Error()
	}
}
