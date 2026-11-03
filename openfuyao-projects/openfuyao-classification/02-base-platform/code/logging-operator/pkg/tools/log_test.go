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

package tools

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCreateLumberjackLogger(t *testing.T) {
	conf := LogConfig{
		Path:       "logs",
		FileName:   "test.log",
		MaxSize:    10, // Maximum size in megabytes before log is rotated
		MaxBackups: 5,  // Maximum number of old log files to retain
		MaxAge:     30, // Maximum number of days to retain old log files
		LocalTime:  true,
		Compress:   true, // Whether to compress/zip old log files
	}

	logger := createLumberjackLogger(&conf)

	// Build expected filename
	expectedFilename := filepath.Join("logs", "test.log")

	if logger.Filename != expectedFilename {
		t.Errorf("Expected filename %s, got %s", expectedFilename, logger.Filename)
	}
	if logger.MaxSize != 10 {
		t.Errorf("Expected MaxSize 10, got %d", logger.MaxSize)
	}
	if logger.MaxBackups != 5 {
		t.Errorf("Expected MaxBackups 5, got %d", logger.MaxBackups)
	}
	if logger.MaxAge != 30 {
		t.Errorf("Expected MaxAge 30, got %d", logger.MaxAge)
	}
	if logger.LocalTime != true {
		t.Errorf("Expected LocalTime true, got %v", logger.LocalTime)
	}
	if logger.Compress != true {
		t.Errorf("Expected Compress true, got %v", logger.Compress)
	}
}

func TestHandleError(t *testing.T) {
	// 创建一个 ResponseRecorder
	recorder := httptest.NewRecorder()

	// 模拟的错误
	testError := fmt.Errorf("test error")

	// 调用 HandleError
	HandleError(recorder, http.StatusInternalServerError, testError)

	// 检查状态码是否如预期
	if status := recorder.Code; status != http.StatusInternalServerError {
		t.Errorf("Expected status code %d, got %d", http.StatusInternalServerError, status)
	}

	// 检查响应体是否包含错误信息
	expectedMessage := "Error occurred: test error"
	responseBody := recorder.Body.String()
	if !bytes.Contains([]byte(responseBody), []byte(expectedMessage)) {
		t.Errorf("Expected response body to contain '%s', got '%s'", expectedMessage, responseBody)
	}

}

func TestAddContext(t *testing.T) {
	// 调用AddContext，添加一些参数
	newLogger := AddContext("module", "test", "id", 1)
	newLogger.Info("This is a test message")
	LogInfo("This is a test message")
	LogDebug("This is a test message")
}

func TestGetLogWriter(t *testing.T) {
	tests := []struct {
		name   string
		outMod string
	}{
		{"console mode", "console"},
		{"file mode", "file"},
		{"both mode", "both"},
		{"default mode", "invalid"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := &LogConfig{OutMod: tt.outMod}
			writer := getLogWriter(config)
			assert.NotNil(t, writer)
		})
	}
}
