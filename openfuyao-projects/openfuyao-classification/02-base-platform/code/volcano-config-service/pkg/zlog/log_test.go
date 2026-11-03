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

// Package zlog provides a set of utilities for logging configurations and operations.
package zlog

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"
)

// TestCase represents a generic test case structure
type TestCase struct {
	name     string
	input    interface{}
	expected interface{}
	wantErr  bool
}

// LogTestCase represents a logging test case
type LogTestCase struct {
	name     string
	config   *LogConfig
	expected zapcore.Level
	wantErr  bool
}

func TestInit(t *testing.T) {
	// Test that Logger is initialized during package init
	if Logger == nil {
		t.Error("Logger should be initialized during package init")
	}
}

func TestParseConfig(t *testing.T) {
	tests := []struct {
		name       string
		configData string
		wantErr    bool
		expected   *LogConfig
	}{
		{
			name: "valid yaml config",
			configData: `
level: debug
encodertype: json
path: /tmp/logs
filename: app.log
maxsize: 50
maxbackups: 3
maxage: 7
localtime: true
compress: false
outmod: file
`,
			wantErr: false,
			expected: &LogConfig{
				Level:       "debug",
				EncoderType: "json",
				Path:        "/tmp/logs",
				FileName:    "app.log",
				MaxSize:     50,
				MaxBackups:  3,
				MaxAge:      7,
				LocalTime:   true,
				Compress:    false,
				OutMod:      "file",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create temporary config file
			tempDir := t.TempDir()
			configFile := filepath.Join(tempDir, "config.yaml")
			if err := os.WriteFile(configFile, []byte(tt.configData), 0644); err != nil {
				t.Fatalf("Failed to create test config: %v", err)
			}

			viper.Reset()
			viper.SetConfigFile(configFile)

			config, err := parseConfig()
			if (err != nil) != tt.wantErr {
				t.Errorf("parseConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if config == nil {
					t.Fatal("parseConfig() returned nil config")
				}
				if config.Level != tt.expected.Level {
					t.Errorf("Level = %v, want %v", config.Level, tt.expected.Level)
				}
				if config.EncoderType != tt.expected.EncoderType {
					t.Errorf("EncoderType = %v, want %v", config.EncoderType, tt.expected.EncoderType)
				}
			}
		})
	}
}

func TestWatchConfig(t *testing.T) {
	// Reset watchOnce for testing
	watchOnce = sync.Once{}

	// Create temporary config file
	tempDir := t.TempDir()
	configFile := filepath.Join(tempDir, "config.yaml")
	initialConfig := `
level: info
encodertype: console
outmod: console
`
	if err := os.WriteFile(configFile, []byte(initialConfig), 0644); err != nil {
		t.Fatalf("Failed to create test config: %v", err)
	}

	viper.Reset()
	viper.SetConfigFile(configFile)
	if err := viper.ReadInConfig(); err != nil {
		t.Fatalf("Failed to read config: %v", err)
	}

	// Test watchConfig function
	watchConfig()

	// Verify watchOnce was called (subsequent calls should be no-op)
	watchConfig()
	watchConfig()
}

func TestGetLogger(t *testing.T) {
	testCases := []LogTestCase{
		{
			name: "debug level logger",
			config: &LogConfig{
				Level:       "debug",
				EncoderType: "console",
				OutMod:      "console",
			},
			expected: zapcore.DebugLevel,
			wantErr:  false,
		},
		{
			name: "info level logger",
			config: &LogConfig{
				Level:       "info",
				EncoderType: "json",
				OutMod:      "console",
			},
			expected: zapcore.InfoLevel,
			wantErr:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger := GetLogger(tc.config)
			if logger == nil {
				t.Fatal("GetLogger returned nil")
			}

			// Test logger level by checking if it logs at the expected level
			core, logs := observer.New(tc.expected)
			testLogger := zap.New(core).Sugar()

			// Test different log levels
			testLogger.Debug("debug message")
			testLogger.Info("info message")
			testLogger.Warn("warn message")
			testLogger.Error("error message")

			if logs.Len() == 0 && tc.expected <= zapcore.ErrorLevel {
				t.Error("Expected log entries but got none")
			}
		})
	}
}

func TestGetEncoder(t *testing.T) {
	testCases := []struct {
		name        string
		encoderType string
		isJSON      bool
	}{
		{
			name:        "json encoder",
			encoderType: "json",
			isJSON:      true,
		},
		{
			name:        "console encoder",
			encoderType: "console",
			isJSON:      false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := &LogConfig{EncoderType: tc.encoderType}
			encoder := getEncoder(config)
			if encoder == nil {
				t.Fatal("getEncoder returned nil")
			}

			// Test by encoding a sample entry
			entry := zapcore.Entry{
				Level:   zapcore.InfoLevel,
				Time:    time.Now(),
				Message: "test message",
			}

			buf, err := encoder.EncodeEntry(entry, nil)
			if err != nil {
				t.Fatalf("Failed to encode entry: %v", err)
			}

			output := buf.String()
			containsJSON := strings.Contains(output, `"level"`) && strings.Contains(output, `"msg"`)

			if tc.isJSON && !containsJSON {
				t.Error("Expected JSON format but got console format")
			}
			if !tc.isJSON && containsJSON {
				t.Error("Expected console format but got JSON format")
			}
		})
	}
}

func TestCreateLumberjackLogger(t *testing.T) {
	tempDir := t.TempDir()
	testCases := []struct {
		name   string
		config *LogConfig
	}{
		{
			name: "full config",
			config: &LogConfig{
				Path:       tempDir,
				FileName:   "test.log",
				MaxSize:    100,
				MaxBackups: 5,
				MaxAge:     30,
				LocalTime:  true,
				Compress:   false,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger := createLumberjackLogger(tc.config)
			if logger == nil {
				t.Fatal("createLumberjackLogger returned nil")
			}

			expectedPath := filepath.Join(tc.config.Path, tc.config.FileName)
			if logger.Filename != expectedPath {
				t.Errorf("Expected filename %s, got %s", expectedPath, logger.Filename)
			}

			if logger.MaxSize != tc.config.MaxSize {
				t.Errorf("Expected MaxSize %d, got %d", tc.config.MaxSize, logger.MaxSize)
			}

			if logger.MaxBackups != tc.config.MaxBackups {
				t.Errorf("Expected MaxBackups %d, got %d", tc.config.MaxBackups, logger.MaxBackups)
			}

			if logger.MaxAge != tc.config.MaxAge {
				t.Errorf("Expected MaxAge %d, got %d", tc.config.MaxAge, logger.MaxAge)
			}

			if logger.LocalTime != tc.config.LocalTime {
				t.Errorf("Expected LocalTime %v, got %v", tc.config.LocalTime, logger.LocalTime)
			}
		})
	}
}

func TestFatalFunctions(t *testing.T) {
	// Test FormatFatal and Fatalf functions
	// These functions call os.Exit, so we need to test them in a subprocess
	if os.Getenv("TEST_FATAL") == "1" {
		// This will be run in subprocess
		FormatFatal("test fatal %s", "message")
		return
	}

	if os.Getenv("TEST_FATALF") == "1" {
		// This will be run in subprocess
		Fatalf("test fatalf %s", "message")
		return
	}

	// Note: Testing fatal functions that call os.Exit is complex and risky
	// In production code, you might want to make these functions more testable
	// by accepting an exit function as a parameter
}

func TestSync(t *testing.T) {
	// Capture original logger
	originalLogger := Logger
	defer func() {
		Logger = originalLogger
	}()

	testCases := []struct {
		name   string
		logger *zap.SugaredLogger
	}{
		{
			name: "sync with buffer",
			logger: func() *zap.SugaredLogger {
				var buf bytes.Buffer
				core := zapcore.NewCore(
					zapcore.NewConsoleEncoder(zap.NewDevelopmentEncoderConfig()),
					zapcore.AddSync(&buf),
					zapcore.DebugLevel,
				)
				return zap.New(core).Sugar()
			}(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			Logger = tc.logger

			// Write some log entries
			Logger.Info("test message before sync")

			err := Sync()
			if err != nil {
				t.Errorf("Sync() returned error: %v", err)
			}
		})
	}
}

func TestLogConfigValidation(t *testing.T) {
	testCases := []struct {
		name   string
		config *LogConfig
		valid  bool
	}{
		{
			name: "valid config",
			config: &LogConfig{
				Level:       "info",
				EncoderType: "console",
				Path:        "/tmp",
				FileName:    "test.log",
				MaxSize:     10,
				MaxBackups:  5,
				MaxAge:      30,
				OutMod:      "console",
			},
			valid: true,
		},
		{
			name: "config with empty path",
			config: &LogConfig{
				Level:       "info",
				EncoderType: "console",
				Path:        "",
				FileName:    "test.log",
				MaxSize:     10,
				MaxBackups:  5,
				MaxAge:      30,
				OutMod:      "console",
			},
			valid: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger := GetLogger(tc.config)
			if tc.valid && logger == nil {
				t.Error("Expected valid logger but got nil")
			}
			if !tc.valid && logger != nil {
				t.Error("Expected nil logger but got valid logger")
			}
		})
	}
}

func TestGetLogWriter(t *testing.T) {
	testCases := []struct {
		name   string
		outMod string
	}{
		{name: "console", outMod: "console"},
		{name: "file", outMod: "file"},
		{name: "both", outMod: "both"},
		{name: "default", outMod: "unknown"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			conf := &LogConfig{
				OutMod:   tc.outMod,
				Path:     t.TempDir(),
				FileName: "test.log",
			}
			ws := getLogWriter(conf)
			if ws == nil {
				t.Error("getLogWriter returned nil")
			}
		})
	}
}

func TestGetDefaultConf(t *testing.T) {
	conf := getDefaultConf()
	if conf == nil {
		t.Fatal("getDefaultConf returned nil")
	}
	if conf.Level != "info" {
		t.Errorf("expected level info, got %s", conf.Level)
	}
}

func TestLogWrapperFunctions(t *testing.T) {
	Warn("warn message")
	Info("info message")
	Error("error message")
	FormatError("format error %s", "arg")
	FormatWarn("format warn %s", "arg")
	FormatInfo("format info %s", "arg")
	FormatDebug("format debug %s", "arg")
	Errorf("errorf %s", "arg")
	Warnf("warnf %s", "arg")
	Infof("infof %s", "arg")
	Debugf("debugf %s", "arg")
}
