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
	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const (
	testTwoHundredValue      = 200
	testFourHundredValue     = 400
	testFourHundredFourValue = 404
	testFiveHundredValue     = 500
	testThreeHundredValue    = 300
	testOneTwentyValue       = 120
	testOneThousandValue     = 1000
	testFiftyValue           = 50
	testKiloValue            = 1024
)

func TestDownloadFile(t *testing.T) {
	tests := []struct {
		name            string
		url             string
		destinationFile string
		mockGet         func(string) (*http.Response, error)
		expectError     bool
	}{
		{
			name:            "successful download",
			url:             "http://example.com/file.txt",
			destinationFile: "/tmp/test-file.txt",
			mockGet: func(url string) (*http.Response, error) {
				assert.Equal(t, "http://example.com/file.txt", url)
				return &http.Response{
					StatusCode: testTwoHundredValue,
					Body:       io.NopCloser(strings.NewReader("test content")),
				}, nil
			},
			expectError: false,
		},
		{
			name:            "http get fails",
			url:             "http://example.com/file.txt",
			destinationFile: "/tmp/test-file.txt",
			mockGet: func(url string) (*http.Response, error) {
				return nil, fmt.Errorf("connection failed")
			},
			expectError: true,
		},
		{
			name:            "non-200 status code",
			url:             "http://example.com/file.txt",
			destinationFile: "/tmp/test-file.txt",
			mockGet: func(url string) (*http.Response, error) {
				return &http.Response{
					StatusCode: testFourHundredFourValue,
					Body:       io.NopCloser(strings.NewReader("Not Found")),
				}, nil
			},
			expectError: true,
		},
		{
			name:            "create file fails",
			url:             "http://example.com/file.txt",
			destinationFile: "/invalid/path/file.txt",
			mockGet: func(url string) (*http.Response, error) {
				return &http.Response{
					StatusCode: testTwoHundredValue,
					Body:       io.NopCloser(strings.NewReader("test content")),
				}, nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(http.Get, tt.mockGet)

			// Create a temporary file for testing
			tempDir := t.TempDir()
			testDestFile := filepath.Join(tempDir, "test-file.txt")

			err := DownloadFile(tt.url, testDestFile)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)

				// Verify that the file was downloaded correctly
				content, err := os.ReadFile(testDestFile)
				assert.NoError(t, err)
				assert.Equal(t, "test content", string(content))
			}
		})
	}
}

func TestDownloadFileWithRealHTTP(t *testing.T) {
	// Create a temporary file for download
	tempDir := t.TempDir()
	destFile := filepath.Join(tempDir, "downloaded-file.txt")

	// Apply patches to mock the HTTP request and response
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(http.Get, func(url string) (*http.Response, error) {
		return &http.Response{
			StatusCode: testTwoHundredValue,
			Body:       io.NopCloser(strings.NewReader("This is the content of the downloaded file")),
		}, nil
	})

	err := DownloadFile("http://example.com/test.txt", destFile)

	assert.NoError(t, err)

}
