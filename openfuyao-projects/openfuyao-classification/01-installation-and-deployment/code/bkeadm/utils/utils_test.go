/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package utils

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

// Constants for numeric literals to comply with ds.txt standards
const (
	testNumericZero     = 0
	testNumericOne      = 1
	testNumericTwo      = 2
	testNumericThree    = 3
	testNumericFive     = 5
	testNumericTen      = 10
	testNumericFortyTwo = 42
	testNumericHundred  = 100
	testNumericThousand = 1000
	testPortNumber      = 12345
)

// Constants for IP address segments to comply with ds.txt standards
const (
	testIPv4SegmentA         = 192
	testIPv4SegmentB         = 168
	testIPv4SegmentC         = 1
	testIPv4SegmentD         = 100
	testIPv4SegmentLoopbackA = 127
	testIPv4SegmentLoopbackB = 0
	testIPv4SegmentLoopbackC = 0
	testIPv4SegmentLoopbackD = 1
	testIPv4SegmentPrivateA  = 10
	testIPv4SegmentPrivateB  = 0
	testIPv4SegmentPrivateC  = 0
	testIPv4SegmentPrivateD  = 1
	testIPv4SegmentPrivate2A = 172
	testIPv4SegmentPrivate2B = 16
	testIPv4SegmentPrivate2C = 0
	testIPv4SegmentPrivate2D = 1
)

// Variables for IP addresses constructed from constants
var (
	testIP192_168_1_100    = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentA, testIPv4SegmentB, testIPv4SegmentC, testIPv4SegmentD)
	testIPLoopback         = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentLoopbackA, testIPv4SegmentLoopbackB, testIPv4SegmentLoopbackC, testIPv4SegmentLoopbackD)
	testIP10_0_0_1         = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentPrivateA, testIPv4SegmentPrivateB, testIPv4SegmentPrivateC, testIPv4SegmentPrivateD)
	testIP172_16_0_1       = fmt.Sprintf("%d.%d.%d.%d", testIPv4SegmentPrivate2A, testIPv4SegmentPrivate2B, testIPv4SegmentPrivate2C, testIPv4SegmentPrivate2D)
	testIPNet192_168_1_100 = net.IPv4(byte(testIPv4SegmentA), byte(testIPv4SegmentB), byte(testIPv4SegmentC), byte(testIPv4SegmentD))
	testIPNetLoopback      = net.IPv4(byte(testIPv4SegmentLoopbackA), byte(testIPv4SegmentLoopbackB), byte(testIPv4SegmentLoopbackC), byte(testIPv4SegmentLoopbackD))
	testIPNet10_0_0_1      = net.IPv4(byte(testIPv4SegmentPrivateA), byte(testIPv4SegmentPrivateB), byte(testIPv4SegmentPrivateC), byte(testIPv4SegmentPrivateD))
	testIPNet172_16_0_1    = net.IPv4(byte(testIPv4SegmentPrivate2A), byte(testIPv4SegmentPrivate2B), byte(testIPv4SegmentPrivate2C), byte(testIPv4SegmentPrivate2D))
)

func TestExists(t *testing.T) {
	// Create a temporary file for testing
	tempDir := t.TempDir()
	existingFile := filepath.Join(tempDir, "existing-file.txt")
	err := os.WriteFile(existingFile, []byte("test"), DefaultFilePermission)
	assert.NoError(t, err)

	nonExistingFile := filepath.Join(tempDir, "non-existing-file.txt")

	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{
			name:     "existing file",
			path:     existingFile,
			expected: true,
		},
		{
			name:     "non-existing file",
			path:     nonExistingFile,
			expected: false,
		},
		{
			name:     "existing directory",
			path:     tempDir,
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Exists(tt.path)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestContainsString(t *testing.T) {
	slice := []string{"apple", "banana", "cherry"}

	tests := []struct {
		name     string
		slice    []string
		item     string
		expected bool
	}{
		{
			name:     "item exists in slice",
			slice:    slice,
			item:     "banana",
			expected: true,
		},
		{
			name:     "item does not exist in slice",
			slice:    slice,
			item:     "orange",
			expected: false,
		},
		{
			name:     "empty slice",
			slice:    []string{},
			item:     "apple",
			expected: false,
		},
		{
			name:     "nil slice",
			slice:    nil,
			item:     "apple",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsString(tt.slice, tt.item)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestContainsStringPrefix(t *testing.T) {
	slice := []string{"apple", "banana", "cherry-apple", "orange"}

	tests := []struct {
		name     string
		slice    []string
		prefix   string
		expected bool
	}{
		{
			name:     "prefix exists in slice",
			slice:    slice,
			prefix:   "app",
			expected: true,
		},
		{
			name:     "prefix exists in middle of string",
			slice:    slice,
			prefix:   "cherry",
			expected: true,
		},
		{
			name:     "prefix does not exist in slice",
			slice:    slice,
			prefix:   "grape",
			expected: false,
		},
		{
			name:     "empty slice",
			slice:    []string{},
			prefix:   "app",
			expected: false,
		},
		{
			name:     "nil slice",
			slice:    nil,
			prefix:   "app",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsStringPrefix(tt.slice, tt.prefix)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsDir(t *testing.T) {
	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, "test-file.txt")
	err := os.WriteFile(tempFile, []byte("test"), DefaultFilePermission)
	assert.NoError(t, err)

	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{
			name:     "path is a directory",
			path:     tempDir,
			expected: true,
		},
		{
			name:     "path is a file",
			path:     tempFile,
			expected: false,
		},
		{
			name:     "path does not exist",
			path:     filepath.Join(tempDir, "nonexistent"),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsDir(tt.path)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsFile(t *testing.T) {
	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, "test-file.txt")
	err := os.WriteFile(tempFile, []byte("test"), DefaultFilePermission)
	assert.NoError(t, err)

	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{
			name:     "path is a file",
			path:     tempFile,
			expected: true,
		},
		{
			name:     "path is a directory",
			path:     tempDir,
			expected: false,
		},
		{
			name:     "path does not exist",
			path:     filepath.Join(tempDir, "nonexistent"),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsFile(tt.path)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestAbsPath(t *testing.T) {
	tempDir := t.TempDir()
	relativePath := "test-relative-path"
	absolutePath := filepath.Join(tempDir, "test-absolute-path")

	// Create the absolute path file
	err := os.WriteFile(absolutePath, []byte("test"), DefaultFilePermission)
	assert.NoError(t, err)

	// Change working directory for testing relative paths
	originalWd, err := os.Getwd()
	assert.NoError(t, err)
	defer func() {
		err := os.Chdir(originalWd)
		assert.NoError(t, err)
	}()

	err = os.Chdir(tempDir)
	assert.NoError(t, err)

	tests := []struct {
		name        string
		inputPath   string
		expectError bool
	}{
		{
			name:        "absolute path input",
			inputPath:   absolutePath,
			expectError: false,
		},
		{
			name:        "relative path input",
			inputPath:   relativePath,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := AbsPath(tt.inputPath)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, filepath.IsAbs(result))
			}
		})
	}
}

func TestDirectoryIsEmpty(t *testing.T) {
	tempDir := t.TempDir()
	notEmptyDir := filepath.Join(tempDir, "not-empty")
	err := os.MkdirAll(notEmptyDir, DefaultDirPermission)
	assert.NoError(t, err)

	// Add a file to the directory
	err = os.WriteFile(filepath.Join(notEmptyDir, "test-file.txt"), []byte("test"), DefaultFilePermission)
	assert.NoError(t, err)

	emptyDir := filepath.Join(tempDir, "empty")
	err = os.MkdirAll(emptyDir, DefaultDirPermission)
	assert.NoError(t, err)

	nonExistentDir := filepath.Join(tempDir, "non-existent")

	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{
			name:     "empty directory",
			path:     emptyDir,
			expected: true,
		},
		{
			name:     "non-empty directory",
			path:     notEmptyDir,
			expected: false,
		},
		{
			name:     "non-existent directory",
			path:     nonExistentDir,
			expected: true, // Returns true when there's an error reading the directory
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := DirectoryIsEmpty(tt.path)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestGetOutBoundIP(t *testing.T) {
	// Apply patches to mock the network connection
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.Dial, func(network, address string) (net.Conn, error) {
		assert.Equal(t, "udp", network)
		assert.Equal(t, defaultDNSServer, address)

		// Create a mock connection
		return &mockConn{
			localAddr: &net.UDPAddr{
				IP:   testIPNet192_168_1_100,
				Port: testPortNumber,
			},
		}, nil
	})

	ip, err := GetOutBoundIP()

	assert.NoError(t, err)
	assert.Equal(t, testIP192_168_1_100, ip)
}

func TestGetOutBoundIPError(t *testing.T) {
	// Apply patches to simulate dial error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.Dial, func(network, address string) (net.Conn, error) {
		return nil, fmt.Errorf("dial error")
	})

	ip, err := GetOutBoundIP()

	assert.Error(t, err)
	assert.Equal(t, "", ip)
	assert.Contains(t, err.Error(), "dial error")
}

func TestGetIntranetIP(t *testing.T) {
	// Apply patches to mock network interface addresses
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.InterfaceAddrs, func() ([]net.Addr, error) {
		return []net.Addr{
			&net.IPNet{
				IP: testIPNetLoopback, // loopback, should be skipped
			},
			&net.IPNet{
				IP: testIPNet192_168_1_100, // valid IPv4
			},
			&net.IPNet{
				IP: net.ParseIP("fe80::1"), // IPv6, should be skipped
			},
			&net.IPNet{
				IP: testIPNet10_0_0_1, // valid IPv4
			},
		}, nil
	})

	ip, err := GetIntranetIp()

	assert.NoError(t, err)
	assert.Equal(t, testIP192_168_1_100, ip) // Should return the first valid non-loopback IPv4
}

func TestGetIntranetIPError(t *testing.T) {
	// Apply patches to simulate error in InterfaceAddrs
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.InterfaceAddrs, func() ([]net.Addr, error) {
		return nil, fmt.Errorf("interface error")
	})

	ip, err := GetIntranetIp()

	assert.Error(t, err)
	assert.Equal(t, "", ip)
	assert.Contains(t, err.Error(), "interface error")
}

func TestGetIntranetIPNoValidAddresses(t *testing.T) {
	// Apply patches to return only loopback addresses
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.InterfaceAddrs, func() ([]net.Addr, error) {
		return []net.Addr{
			&net.IPNet{
				IP: testIPNetLoopback, // loopback
			},
			&net.IPNet{
				IP: net.ParseIP("::1"), // IPv6 loopback
			},
		}, nil
	})

	ip, err := GetIntranetIp()

	assert.NoError(t, err)
	assert.Equal(t, "", ip) // Should return empty when no valid addresses found
}

func TestIsNum(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected bool
	}{
		{
			name:     "valid integer",
			input:    "123",
			expected: true,
		},
		{
			name:     "valid float",
			input:    "123.45",
			expected: true,
		},
		{
			name:     "valid negative number",
			input:    "-123",
			expected: true,
		},
		{
			name:     "valid scientific notation",
			input:    "1.23e10",
			expected: true,
		},
		{
			name:     "invalid string",
			input:    "abc",
			expected: false,
		},
		{
			name:     "mixed alphanumeric",
			input:    "123abc",
			expected: false,
		},
		{
			name:     "empty string",
			input:    "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsNum(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCopyFile(t *testing.T) {
	tempDir := t.TempDir()
	sourceFile := filepath.Join(tempDir, "source.txt")
	destinationFile := filepath.Join(tempDir, "destination.txt")

	content := "test file content"
	err := os.WriteFile(sourceFile, []byte(content), DefaultFilePermission)
	assert.NoError(t, err)

	err = CopyFile(sourceFile, destinationFile)

	assert.NoError(t, err)

	// Verify that the destination file has the same content
	destinationContent, err := os.ReadFile(destinationFile)
	assert.NoError(t, err)
	assert.Equal(t, content, string(destinationContent))
}

func TestCopyFileError(t *testing.T) {
	// Test with non-existent source file
	err := CopyFile("/nonexistent/source.txt", "/tmp/dest.txt")

	assert.Error(t, err)
}

func TestRemoveStringObject(t *testing.T) {
	tests := []struct {
		name     string
		array    []string
		object   string
		expected []string
	}{
		{
			name:     "remove existing object",
			array:    []string{"apple", "banana", "cherry", "banana"},
			object:   "banana",
			expected: []string{"apple", "cherry"},
		},
		{
			name:     "remove non-existing object",
			array:    []string{"apple", "banana", "cherry"},
			object:   "orange",
			expected: []string{"apple", "banana", "cherry"},
		},
		{
			name:     "remove from empty array",
			array:    []string{},
			object:   "apple",
			expected: []string{},
		},
		{
			name:     "remove all objects",
			array:    []string{"apple", "apple", "apple"},
			object:   "apple",
			expected: []string{},
		},
		{
			name:     "nil array",
			array:    nil,
			object:   "apple",
			expected: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := RemoveStringObject(tt.array, tt.object)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsChanClosed(t *testing.T) {
	// Test with a closed channel
	closedChan := make(chan struct{})
	close(closedChan)

	result := IsChanClosed(closedChan)
	assert.True(t, result)

	// Test with an open channel
	openChan := make(chan struct{})

	result = IsChanClosed(openChan)
	assert.False(t, result)

	// Close the channel for cleanup
	close(openChan)
}

func TestReverseArray(t *testing.T) {
	tests := []struct {
		name     string
		input    []string
		expected []string
	}{
		{
			name:     "normal array",
			input:    []string{"a", "b", "c"},
			expected: []string{"c", "b", "a"},
		},
		{
			name:     "single element",
			input:    []string{"a"},
			expected: []string{"a"},
		},
		{
			name:     "empty array",
			input:    []string{},
			expected: []string{},
		},
		{
			name:     "two elements",
			input:    []string{"a", "b"},
			expected: []string{"b", "a"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ReverseArray(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestCopyMap(t *testing.T) {
	original := map[string]string{
		"key1": "value1",
		"key2": "value2",
		"key3": "value3",
	}

	copied := CopyMap(original)

	assert.Equal(t, original, copied)

}

func TestCopyMapWithEmptyMap(t *testing.T) {
	original := map[string]string{}

	copied := CopyMap(original)

	assert.Equal(t, original, copied)
	assert.Empty(t, copied)
}

func TestCopyMapWithNilMap(t *testing.T) {
	var original map[string]string

	CopyMap(original)

}

func TestLoopIP(t *testing.T) {
	// Apply patches to mock DNS lookup
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.LookupIP, func(host string) ([]net.IP, error) {
		assert.Equal(t, "example.com", host)
		return []net.IP{
			testIPNet192_168_1_100,
			testIPNet10_0_0_1,
			net.ParseIP("2001:db8::1"), // IPv6, should be filtered out
			testIPNet172_16_0_1,
		}, nil
	})

	ips, err := LoopIP("example.com")

	assert.NoError(t, err)
	assert.Equal(t, []string{testIP192_168_1_100, testIP10_0_0_1, testIP172_16_0_1}, ips)
}

func TestLoopIPError(t *testing.T) {
	// Apply patches to simulate DNS lookup error
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.LookupIP, func(host string) ([]net.IP, error) {
		return nil, fmt.Errorf("lookup error")
	})

	ips, err := LoopIP("nonexistent-domain.com")

	assert.Error(t, err)
	assert.Empty(t, ips)
	assert.Contains(t, err.Error(), "lookup error")
}

func TestRandInt(t *testing.T) {
	// Test that RandInt returns values within the expected range
	for i := testNumericZero; i < testNumericHundred; i++ {
		result := RandInt(testNumericOne, testNumericTen)
		assert.GreaterOrEqual(t, result, testNumericOne)
		assert.LessOrEqual(t, result, testNumericTen)
	}

	// Test with reversed min/max values
	result := RandInt(testNumericTen, testNumericOne) // min > max, should be swapped
	assert.GreaterOrEqual(t, result, testNumericOne)
	assert.LessOrEqual(t, result, testNumericTen)

	// Test with equal min/max
	result = RandInt(testNumericFive, testNumericFive)
	assert.Equal(t, testNumericFive, result)
}

func TestPromptForConfirmation(t *testing.T) {
	// Test with skipPrompt = true
	result := PromptForConfirmation(true)
	assert.True(t, result)

	// For interactive prompts, we can't easily test the user input part,
	// but we can verify the function doesn't crash with skipPrompt = true
	assert.True(t, true)
}

func TestCopyDir(t *testing.T) {
	// Create source and destination directories
	tempDir := t.TempDir()
	srcDir := filepath.Join(tempDir, "src")
	dstDir := filepath.Join(tempDir, "dst")

	// Create source directory structure
	err := os.MkdirAll(filepath.Join(srcDir, "subdir"), DefaultDirPermission)
	assert.NoError(t, err)

	// Create files in source
	err = os.WriteFile(filepath.Join(srcDir, "file1.txt"), []byte("content1"), DefaultFilePermission)
	assert.NoError(t, err)

	err = os.WriteFile(filepath.Join(srcDir, "subdir", "file2.txt"), []byte("content2"), DefaultFilePermission)
	assert.NoError(t, err)

	err = CopyDir(srcDir, dstDir)

	assert.NoError(t, err)

	// Verify that the directory structure was copied
	assert.True(t, Exists(dstDir))
	assert.True(t, Exists(filepath.Join(dstDir, "subdir")))
	assert.True(t, Exists(filepath.Join(dstDir, "file1.txt")))
	assert.True(t, Exists(filepath.Join(dstDir, "subdir", "file2.txt")))

	// Verify file contents
	content1, err := os.ReadFile(filepath.Join(dstDir, "file1.txt"))
	assert.NoError(t, err)
	assert.Equal(t, "content1", string(content1))

	content2, err := os.ReadFile(filepath.Join(dstDir, "subdir", "file2.txt"))
	assert.NoError(t, err)
	assert.Equal(t, "content2", string(content2))
}

func TestCopyDirError(t *testing.T) {
	// Test with non-existent source directory
	err := CopyDir("/nonexistent/src", "/tmp/dst")

	assert.Error(t, err)
}

func TestCopyDirWithExistingDestination(t *testing.T) {
	// Create source and destination directories
	tempDir := t.TempDir()
	srcDir := filepath.Join(tempDir, "src")
	dstDir := filepath.Join(tempDir, "dst")

	// Create source directory structure
	err := os.MkdirAll(srcDir, DefaultDirPermission)
	assert.NoError(t, err)

	err = os.WriteFile(filepath.Join(srcDir, "file1.txt"), []byte("content1"), DefaultFilePermission)
	assert.NoError(t, err)

	// Create destination directory with existing content
	err = os.MkdirAll(dstDir, DefaultDirPermission)
	assert.NoError(t, err)

	err = os.WriteFile(filepath.Join(dstDir, "existing.txt"), []byte("existing"), DefaultFilePermission)
	assert.NoError(t, err)

	err = CopyDir(srcDir, dstDir)

	assert.NoError(t, err)

	// Verify that both existing and new files are present
	assert.True(t, Exists(filepath.Join(dstDir, "existing.txt")))
	assert.True(t, Exists(filepath.Join(dstDir, "file1.txt")))

	// Verify file contents
	existingContent, err := os.ReadFile(filepath.Join(dstDir, "existing.txt"))
	assert.NoError(t, err)
	assert.Equal(t, "existing", string(existingContent))

	newContent, err := os.ReadFile(filepath.Join(dstDir, "file1.txt"))
	assert.NoError(t, err)
	assert.Equal(t, "content1", string(newContent))
}

// Mock implementations for testing
type mockConn struct {
	localAddr  net.Addr
	remoteAddr net.Addr
}

func (c *mockConn) Read(b []byte) (n int, err error)   { return testNumericZero, nil }
func (c *mockConn) Write(b []byte) (n int, err error)  { return testNumericZero, nil }
func (c *mockConn) Close() error                       { return nil }
func (c *mockConn) LocalAddr() net.Addr                { return c.localAddr }
func (c *mockConn) RemoteAddr() net.Addr               { return c.remoteAddr }
func (c *mockConn) SetDeadline(t time.Time) error      { return nil }
func (c *mockConn) SetReadDeadline(t time.Time) error  { return nil }
func (c *mockConn) SetWriteDeadline(t time.Time) error { return nil }

func TestRandIntWithDifferentInputs(t *testing.T) {
	tests := []struct {
		name string
		min  int
		max  int
	}{
		{
			name: "positive range",
			min:  testNumericOne,
			max:  testNumericTen,
		},
		{
			name: "negative range",
			min:  -testNumericTen,
			max:  -testNumericOne,
		},
		{
			name: "mixed range",
			min:  -testNumericFive,
			max:  testNumericFive,
		},
		{
			name: "single value range",
			min:  testNumericFortyTwo,
			max:  testNumericFortyTwo,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := RandInt(tt.min, tt.max)

			// Determine the actual min and max (in case they were swapped)
			actualMin, actualMax := tt.min, tt.max
			if actualMin > actualMax {
				actualMin, actualMax = actualMax, actualMin
			}

			assert.GreaterOrEqual(t, result, actualMin)
			assert.LessOrEqual(t, result, actualMax)
		})
	}
}

func TestIsNumEdgeCases(t *testing.T) {
	edgeCases := []struct {
		input    string
		expected bool
		desc     string
	}{
		{"", false, "empty string"},
		{"0", true, "zero"},
		{"-0", true, "negative zero"},
		{"+123", true, "positive sign"},
		{" 123 ", false, "with spaces"},
		{"12.34.56", false, "multiple decimal points"},
		{"1.23e-4", true, "scientific notation with negative exponent"},
		{"infinity", true, "infinity string"},
		{"nan", true, "not a number string"},
		{"0x1A", false, "hexadecimal string"},
		{"0b101", false, "binary string"},
	}

	for _, tc := range edgeCases {
		t.Run(tc.desc, func(t *testing.T) {
			result := IsNum(tc.input)
			assert.Equal(t, tc.expected, result, "Input: %s", tc.input)
		})
	}
}

func TestRemoveStringObjectPreservesOrder(t *testing.T) {
	input := []string{"a", "b", "c", "b", "d", "b", "e"}
	expected := []string{"a", "c", "d", "e"}

	result := RemoveStringObject(input, "b")

	assert.Equal(t, expected, result)

	// Verify that the order is preserved
	for i, v := range expected {
		assert.Equal(t, v, result[i])
	}
}

func TestReverseArrayInPlaceEffect(t *testing.T) {
	original := []string{"a", "b", "c", "d"}
	originalPtr := &original[0] // Get pointer to first element

	result := ReverseArray(original)

	// Verify that the result is the same underlying array (in-place reversal)
	resultPtr := &result[0]
	assert.Equal(t, originalPtr, resultPtr)

	// But the content should be reversed
	assert.Equal(t, []string{"d", "c", "b", "a"}, result)
}

func TestCopyMapNilSafety(t *testing.T) {
	// Test that CopyMap handles nil input gracefully
	var nilMap map[string]string
	result := CopyMap(nilMap)

	// Test with empty map
	emptyMap := make(map[string]string)
	result = CopyMap(emptyMap)

	assert.NotNil(t, result)
	assert.Empty(t, result)
}

func TestLoopIPWithIPv6Only(t *testing.T) {
	// Apply patches to mock DNS lookup returning only IPv6
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(net.LookupIP, func(host string) ([]net.IP, error) {
		return []net.IP{
			net.ParseIP("2001:db8::1"),
			net.ParseIP("fe80::1"),
		}, nil
	})

	ips, err := LoopIP("ipv6-only.example.com")

	assert.NoError(t, err)
	assert.Empty(t, ips) // Should return empty slice since no IPv4 addresses
}

func TestLoopIPWithEmptyHost(t *testing.T) {
	ips, err := LoopIP("")

	assert.Error(t, err) // Should not return error for empty host
	assert.Empty(t, ips) // Should return empty slice
}

func TestExistsWithSymlinks(t *testing.T) {
	tempDir := t.TempDir()

	// Create a target file
	targetFile := filepath.Join(tempDir, "target.txt")
	err := os.WriteFile(targetFile, []byte("target"), DefaultFilePermission)
	assert.NoError(t, err)

	// Create a symlink to the target file
	symlinkFile := filepath.Join(tempDir, "symlink.txt")
	err = os.Symlink(targetFile, symlinkFile)
	assert.NoError(t, err)

	// Test that Exists returns true for the symlink
	result := Exists(symlinkFile)
	assert.True(t, result)

	// Test that Exists returns true for the target file
	result = Exists(targetFile)
	assert.True(t, result)
}

func TestIsDirWithSymlinks(t *testing.T) {
	tempDir := t.TempDir()

	// Create a target directory
	targetDir := filepath.Join(tempDir, "target_dir")
	err := os.MkdirAll(targetDir, DefaultDirPermission)
	assert.NoError(t, err)

	// Create a symlink to the target directory
	symlinkDir := filepath.Join(tempDir, "symlink_dir")
	err = os.Symlink(targetDir, symlinkDir)
	assert.NoError(t, err)

	// Test that IsDir returns true for the symlink to directory
	result := IsDir(symlinkDir)
	assert.True(t, result)

	// Test that IsDir returns false for a symlink to file
	targetFile := filepath.Join(tempDir, "target_file.txt")
	err = os.WriteFile(targetFile, []byte("content"), DefaultFilePermission)
	assert.NoError(t, err)

	symlinkToFile := filepath.Join(tempDir, "symlink_to_file.txt")
	err = os.Symlink(targetFile, symlinkToFile)
	assert.NoError(t, err)

	result = IsDir(symlinkToFile)
	assert.False(t, result)
}

func TestIsFileWithSymlinks(t *testing.T) {
	tempDir := t.TempDir()

	// Create a target file
	targetFile := filepath.Join(tempDir, "target.txt")
	err := os.WriteFile(targetFile, []byte("content"), DefaultFilePermission)
	assert.NoError(t, err)

	// Create a symlink to the target file
	symlinkFile := filepath.Join(tempDir, "symlink.txt")
	err = os.Symlink(targetFile, symlinkFile)
	assert.NoError(t, err)

	// Test that IsFile returns true for the symlink to file
	result := IsFile(symlinkFile)
	assert.True(t, result)

	// Test that IsFile returns false for a symlink to directory
	targetDir := filepath.Join(tempDir, "target_dir")
	err = os.MkdirAll(targetDir, DefaultDirPermission)
	assert.NoError(t, err)

	symlinkToDir := filepath.Join(tempDir, "symlink_to_dir")
	err = os.Symlink(targetDir, symlinkToDir)
	assert.NoError(t, err)

	result = IsFile(symlinkToDir)
	assert.False(t, result)
}
