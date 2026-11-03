/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package util

import (
	"errors"
	"io"
	"testing"
)

// TestClearByte checks if all bytes in the slice are set to zero.
func TestClearByte(t *testing.T) {
	// Create a byte slice with non-zero values.
	data := []byte{1, 2, 3, 4, 5}

	// Call ClearByte to zero out the slice.
	ClearByte(data)

	// Check each byte to ensure it's been set to zero.
	for i, b := range data {
		if b != 0 {
			t.Errorf("byte at index %d is not zero, got %d", i, b)
		}
	}

	var emptyData []byte
	ClearByte(emptyData) // This should not cause any issue or panic.
}

// MockFile implement multipart.File interface for testing
type MockFile struct {
	Data   []byte
	Offset int
	Err    error // 可以被设置为模拟读取错误
}

func (m *MockFile) Read(p []byte) (int, error) {
	if m.Err != nil {
		return 0, m.Err
	}
	if m.Offset >= len(m.Data) {
		return 0, io.EOF
	}
	n := copy(p, m.Data[m.Offset:])
	m.Offset += n
	return n, nil
}

func (m *MockFile) Close() error {
	return nil
}

func (m *MockFile) Seek(offset int64, whence int) (int64, error) {
	return 0, nil
}

func (m *MockFile) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, errors.New("negative offset")
	}
	if int(off) >= len(m.Data) {
		return 0, io.EOF
	}
	n := copy(p, m.Data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func TestCheckFileSize(t *testing.T) {
	tests := []struct {
		name        string
		data        []byte
		bufferSize  int64
		maxFileSize int64
		expected    bool
		expectError bool
		mockError   error
	}{
		{"Under Limit", []byte("hello"), 1024, 10, true, false, nil},
		{"Exact Limit", []byte("hello"), 1024, 5, true, false, nil},
		{"Over Limit", []byte("hello world"), 1024, 5, false, false, nil},
		{"With Error", []byte("hello"), 1024, 5, false, true, io.ErrUnexpectedEOF},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockFile := &MockFile{Data: tt.data, Err: tt.mockError}
			result, err := CheckFileSize(mockFile, tt.bufferSize, tt.maxFileSize)

			if (err != nil) != tt.expectError {
				t.Errorf("%s: expected error %v, got %v", tt.name, tt.expectError, err != nil)
			}

			if result != tt.expected {
				t.Errorf("%s: expected result %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}

func TestContains(t *testing.T) {
	// 定义一组测试用例
	tests := []struct {
		name     string
		slice    []string
		str      string
		expected bool
	}{
		{"Found", []string{"apple", "banana", "cherry"}, "banana", true},
		{"Not Found", []string{"apple", "banana", "cherry"}, "mango", false},
		{"Empty Slice", []string{}, "banana", false},
		{"Empty String", []string{"apple", "banana", "cherry"}, "", false},
		{"Nil Slice", nil, "banana", false},
		{"Looking for Nil", []string{"apple", "banana", "cherry"}, "", false},
	}

	// 循环执行每个测试用例
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Contains(tt.slice, tt.str)
			if result != tt.expected {
				t.Errorf("Contains(%v, %q) = %v, expected %v", tt.slice, tt.str, result, tt.expected)
			}
		})
	}
}

func TestContainsOne(t *testing.T) {
	tests := []struct {
		name     string
		leftArr  []string
		rightArr []string
		expected bool
	}{
		{"Contains One", []string{"apple", "banana", "cherry"}, []string{"banana", "mango"}, true},
		{"Contains Multiple", []string{"apple", "banana", "cherry"}, []string{"banana", "cherry"}, true},
		{"Contains None", []string{"apple", "banana", "cherry"}, []string{"mango", "orange"}, false},
		{"Left Empty", []string{}, []string{"mango", "orange"}, false},
		{"Right Empty", []string{"apple", "banana", "cherry"}, []string{}, false},
		{"Both Empty", []string{}, []string{}, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsOne(tt.leftArr, tt.rightArr)
			if result != tt.expected {
				t.Errorf("%s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}

func TestContainsAll(t *testing.T) {
	tests := []struct {
		name     string
		outer    []string
		inner    []string
		expected bool
	}{
		{"All Contained", []string{"apple", "banana", "cherry"}, []string{"banana", "apple"}, true},
		{"Missing Some", []string{"apple", "banana", "cherry"}, []string{"banana", "mango"}, false},
		{"Inner Empty", []string{"apple", "banana", "cherry"}, []string{}, true},
		{"Outer Empty", []string{}, []string{"banana", "apple"}, false},
		{"Both Empty", []string{}, []string{}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsAll(tt.outer, tt.inner)
			if result != tt.expected {
				t.Errorf("%s failed: expected %v, got %v", tt.name, tt.expected, result)
			}
		})
	}
}
