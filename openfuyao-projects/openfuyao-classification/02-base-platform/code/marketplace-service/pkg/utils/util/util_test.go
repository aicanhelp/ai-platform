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
	"archive/tar"
	"bytes"
	"compress/gzip"
	"errors"
	"io"
	"mime/multipart"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"marketplace-service/pkg/constant"
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

	emptyData := []byte{}
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

// TestNotContains 测试非包含逻辑
func TestNotContains(t *testing.T) {
	tests := []struct {
		name     string
		slice    []string
		str      string
		expected bool
	}{
		{"Not Found", []string{"a", "b", "c"}, "d", true},
		{"Found", []string{"a", "b", "c"}, "b", false},
		{"Empty Slice", []string{}, "a", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if res := NotContains(tt.slice, tt.str); res != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, res)
			}
		})
	}
}

// TestIsValidSearchParam 测试搜索参数合法性
func TestIsValidSearchParam(t *testing.T) {
	tests := []struct {
		name     string
		param    string
		expected bool
	}{
		{"Valid Length", "test", true},
		{"Exceed Limit", strings.Repeat("a", constant.SearchParamLengthLimit+1), false},
		{"Empty Param", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if res := IsValidSearchParam(tt.param); res != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, res)
			}
		})
	}
}

// TestIsHTTPClientError 测试4xx状态码判断
func TestIsHTTPClientError(t *testing.T) {
	tests := []struct {
		name     string
		code     int
		expected bool
	}{
		{"400 Error", 400, true},
		{"499 Error", 499, true},
		{"500 Error", 500, false},
		{"200 OK", 200, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if res := IsHTTPClientError(tt.code); res != tt.expected {
				t.Errorf("code %d: expected %v, got %v", tt.code, tt.expected, res)
			}
		})
	}
}

// TestEscapeSpecialChars 测试特殊字符转义
func TestEscapeSpecialChars(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"Newline", "a\nb", `a\nb`},
		{"Quote", `"test"`, `\"test\"`},
		{"Normal", "hello", "hello"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if res := EscapeSpecialChars(tt.input); res != tt.expected {
				t.Errorf("input %q: expected %q, got %q", tt.input, tt.expected, res)
			}
		})
	}
}

// TestSanitizeArray 测试数组特殊字符清洗
func TestSanitizeArray(t *testing.T) {
	input := []string{"a\nb", `"test"`}
	expected := []string{`a\nb`, `\"test\"`}
	res := SanitizeArray(input)
	if !reflect.DeepEqual(res, expected) {
		t.Errorf("expected %v, got %v", expected, res)
	}
}

// TestIsValidPath 测试路径合法性（防路径穿越）
func TestIsValidPath(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{"Normal Path", "test/file.txt", true},
		{"Parent Path", "../file.txt", false},
		{"Clean Parent", filepath.Clean("../file.txt"), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if res := isValidPath(tt.path); res != tt.expected {
				t.Errorf("path %q: expected %v, got %v", tt.path, tt.expected, res)
			}
		})
	}
}

// TestParseScenesConfig 测试场景配置解析
func TestParseScenesConfig(t *testing.T) {
	raw := "- scene1\n- scene2\n# comment\n- scene3\\"
	expected := []string{"scene1", "scene2", "scene3"}
	res := parseScenesConfig(raw)
	if !reflect.DeepEqual(res, expected) {
		t.Errorf("expected %v, got %v", expected, res)
	}
}

// TestValidateTarGzFile_Valid 测试合法tar.gz文件校验
func TestValidateTarGzFile_Valid(t *testing.T) {
	// 构造合法tar.gz数据
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)

	// 1. 校验WriteHeader错误
	err := tw.WriteHeader(&tar.Header{Name: "test.txt", Size: 5})
	assert.NoError(t, err, "tar WriteHeader failed")

	// 2. 校验写入数据长度和错误
	const len5 = 5
	n, err := tw.Write([]byte("hello"))
	assert.NoError(t, err, "tar Write failed")
	assert.Equal(t, len5, n, "写入字节数与Header.Size不匹配")

	// 3. 确保关闭顺序并校验错误
	err = tw.Close()
	assert.NoError(t, err, "tar Close failed")
	err = gw.Close()
	assert.NoError(t, err, "gzip Close failed")

	// 4. 显式初始化Offset，规范MockFile
	mockFile := &MockFile{Data: buf.Bytes(), Offset: 0}
	fileHeader := &multipart.FileHeader{Filename: "test.tgz"}
	ok, err := ValidateTarGzFile(mockFile, fileHeader, 1024)
	assert.True(t, ok, "合法文件校验失败")
	assert.NoError(t, err, "合法文件校验返回错误")
}

// TestValidateTarGzFile_TooLarge 测试超大文件校验
func TestValidateTarGzFile_TooLarge(t *testing.T) {
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)

	// 1. 写入超大文件头并校验
	const len1025 = 1025
	err := tw.WriteHeader(&tar.Header{Name: "big.txt", Size: len1025})
	assert.NoError(t, err, "tar WriteHeader failed")

	// 2. 写入指定长度数据并校验
	data := make([]byte, len1025)
	n, err := tw.Write(data)
	assert.NoError(t, err, "tar Write failed")
	assert.Equal(t, len1025, n, "写入字节数与Header.Size不匹配")

	// 3. 关闭资源并校验
	err = tw.Close()
	assert.NoError(t, err, "tar Close failed")
	err = gw.Close()
	assert.NoError(t, err, "gzip Close failed")

	// 4. 执行校验并精准断言错误
	mockFile := &MockFile{Data: buf.Bytes(), Offset: 0}
	fileHeader := &multipart.FileHeader{Filename: "test.tgz"}
	ok, err := ValidateTarGzFile(mockFile, fileHeader, 1024)
	assert.False(t, ok, "超大文件校验通过")
	assert.ErrorContains(t, err, "too large", "错误信息未包含文件过大关键词")
}

// TestParseConfig 测试配置解析逻辑
func TestParseConfig(t *testing.T) {
	configMap := map[string]string{
		marketplaceScenes: "- scene1\n- scene2",
		localHarborHost:   "localhost",
		chartLimit:        "100",
	}
	config := parseConfig(configMap)
	assert.Equal(t, "localhost", config.LocalHarborHost)
	assert.Equal(t, []string{"scene1", "scene2"}, config.MarketplaceScenes)
	assert.Equal(t, "100", config.ChartLimit)
}
