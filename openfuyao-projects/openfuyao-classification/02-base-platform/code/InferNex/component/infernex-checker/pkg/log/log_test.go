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

package log

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLogInit(t *testing.T) {
	tmpDir := t.TempDir()
	logPath := filepath.Join(tmpDir, "test.log")

	err := Init(logPath)
	if err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer Close()

	if logger == nil {
		t.Error("logger should not be nil after Init")
	}
	if logFile == nil {
		t.Error("logFile should not be nil after Init")
	}

	// Verify file exists
	if _, err := os.Stat(logPath); os.IsNotExist(err) {
		t.Errorf("log file was not created at %s", logPath)
	}
}

func TestLogInitInvalidPath(t *testing.T) {
	err := Init("/invalid/path/that/does/not/exist/test.log")
	if err == nil {
		t.Error("expected error for invalid path, got nil")
		Close()
	}
}

func TestLogInfoError(t *testing.T) {
	tmpDir := t.TempDir()
	logPath := filepath.Join(tmpDir, "test.log")

	if err := Init(logPath); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	defer Close()

	Info("test info message")
	Error("test error message")

	// Read log file
	content, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}

	logStr := string(content)
	if !strings.Contains(logStr, "[INFO]") {
		t.Error("log should contain [INFO] tag")
	}
	if !strings.Contains(logStr, "test info message") {
		t.Error("log should contain info message")
	}
	if !strings.Contains(logStr, "[ERROR]") {
		t.Error("log should contain [ERROR] tag")
	}
	if !strings.Contains(logStr, "test error message") {
		t.Error("log should contain error message")
	}
}

func TestLogWithoutInit(t *testing.T) {
	// Ensure logger is nil
	Close()

	// Should not panic when logger is nil
	Info("this should not panic")
	Error("this should not panic")
}

func TestLogClose(t *testing.T) {
	tmpDir := t.TempDir()
	logPath := filepath.Join(tmpDir, "test.log")

	if err := Init(logPath); err != nil {
		t.Fatalf("Init failed: %v", err)
	}

	Close()

	if logger != nil {
		t.Error("logger should be nil after Close")
	}
	if logFile != nil {
		t.Error("logFile should be nil after Close")
	}

	// Calling Close again should not panic
	Close()
}

func TestLogReinitialize(t *testing.T) {
	tmpDir := t.TempDir()
	logPath1 := filepath.Join(tmpDir, "test1.log")
	logPath2 := filepath.Join(tmpDir, "test2.log")

	// First init
	if err := Init(logPath1); err != nil {
		t.Fatalf("first Init failed: %v", err)
	}
	Info("message in first log")

	// Second init should close first file
	if err := Init(logPath2); err != nil {
		t.Fatalf("second Init failed: %v", err)
	}
	defer Close()

	Info("message in second log")

	// Verify first file has first message
	content1, _ := os.ReadFile(logPath1)
	if !strings.Contains(string(content1), "message in first log") {
		t.Error("first log should contain first message")
	}

	// Verify second file has second message
	content2, _ := os.ReadFile(logPath2)
	if !strings.Contains(string(content2), "message in second log") {
		t.Error("second log should contain second message")
	}
}
