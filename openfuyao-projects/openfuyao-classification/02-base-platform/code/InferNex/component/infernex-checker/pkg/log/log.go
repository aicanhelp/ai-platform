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

// Package log provides a simple file-backed logger used throughout the checker
// to record SSH connections, command executions, and error stack traces.
package log

import (
	"fmt"
	"log"
	"os"
	"time"
)

const logFileMode = 0644 // owner read/write, group and others read-only

var logger *log.Logger
var logFile *os.File

// Init initializes the logger and writes to the specified file
func Init(path string) error {
	// Close any previously opened log file to avoid resource leak
	Close()

	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, logFileMode)
	if err != nil {
		return fmt.Errorf("failed to open log file: %w", err)
	}

	logFile = f
	logger = log.New(f, "", 0)
	return nil
}

// Close closes the log file
func Close() {
	if logFile != nil {
		if err := logFile.Close(); err != nil {
			_, _ = fmt.Fprintf(os.Stderr, "failed to close log file: %v\n", err)
		}
		logFile = nil
		logger = nil
	}
}

func prefix(level string) string {
	return fmt.Sprintf("[%s] [%s] ", time.Now().Format("2006-01-02 15:04:05"), level)
}

// Info logs an informational message
func Info(msg string) {
	if logger != nil {
		logger.Print(prefix("INFO") + msg)
	}
}

// Error logs an error message
func Error(msg string) {
	if logger != nil {
		logger.Print(prefix("ERROR") + msg)
	}
}
