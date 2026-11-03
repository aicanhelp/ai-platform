/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS,
 * WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for the details.
 */

package zlog

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/viper"
)

func TestGetLogger(t *testing.T) {
	// Test with console encoder
	config := &LogConfig{
		Level:       "debug",
		EncoderType: "console",
		Path:        "/tmp/test-logs",
		FileName:    "test.log",
		OutMod:      "console",
	}

	logger := GetLogger(config)
	if logger == nil {
		t.Error("GetLogger should not return nil")
	}

	// Test with JSON encoder
	configJSON := &LogConfig{
		Level:       "info",
		EncoderType: "json",
		Path:        "/tmp/test-logs",
		FileName:    "test.json",
		OutMod:      "console",
	}

	loggerJSON := GetLogger(configJSON)
	if loggerJSON == nil {
		t.Error("GetLogger with JSON encoder should not return nil")
	}

	// Test with file output
	configFile := &LogConfig{
		Level:       "warn",
		EncoderType: "console",
		Path:        "/tmp/test-logs",
		FileName:    "test-file.log",
		OutMod:      "file",
	}

	loggerFile := GetLogger(configFile)
	if loggerFile == nil {
		t.Error("GetLogger with file output should not return nil")
	}

	// Test with invalid log level (should default to info)
	configInvalidLevel := &LogConfig{
		Level:       "invalid",
		EncoderType: "console",
		Path:        "/tmp/test-logs",
		FileName:    "test-invalid.log",
		OutMod:      "console",
	}

	loggerInvalid := GetLogger(configInvalidLevel)
	if loggerInvalid == nil {
		t.Error("GetLogger with invalid level should not return nil")
	}
}

func TestGetEncoder(t *testing.T) {
	// Test console encoder
	consoleConfig := &LogConfig{
		EncoderType: "console",
	}
	consoleEncoder := getEncoder(consoleConfig)
	if consoleEncoder == nil {
		t.Error("console encoder should not be nil")
	}

	// Test JSON encoder
	jsonConfig := &LogConfig{
		EncoderType: "json",
	}
	jsonEncoder := getEncoder(jsonConfig)
	if jsonEncoder == nil {
		t.Error("JSON encoder should not be nil")
	}

	// Test invalid encoder type defaults to console
	invalidConfig := &LogConfig{
		EncoderType: "invalid",
	}
	defaultEncoder := getEncoder(invalidConfig)
	if defaultEncoder == nil {
		t.Error("default encoder should not be nil")
	}
}

func TestGetLogWriter(t *testing.T) {
	// Create test directory
	testDir := "/tmp/test-logs"
	os.MkdirAll(testDir, 0755)
	defer os.RemoveAll(testDir)

	// Test console output
	consoleConfig := &LogConfig{
		OutMod: "console",
	}
	consoleWriter := getLogWriter(consoleConfig)
	if consoleWriter == nil {
		t.Error("console writer should not be nil")
	}

	// Test file output
	fileConfig := &LogConfig{
		OutMod:     "file",
		Path:       testDir,
		FileName:   "test.log",
		MaxSize:    1,
		MaxBackups: 1,
		MaxAge:     1,
	}
	fileWriter := getLogWriter(fileConfig)
	if fileWriter == nil {
		t.Error("file writer should not be nil")
	}

	// Test both output
	bothConfig := &LogConfig{
		OutMod:     "both",
		Path:       testDir,
		FileName:   "test-both.log",
		MaxSize:    1,
		MaxBackups: 1,
		MaxAge:     1,
	}
	bothWriter := getLogWriter(bothConfig)
	if bothWriter == nil {
		t.Error("both writer should not be nil")
	}

	// Test invalid output mode (should default to console)
	invalidConfig := &LogConfig{
		OutMod:   "invalid",
		Path:     testDir,
		FileName: "test-invalid.log",
	}
	invalidWriter := getLogWriter(invalidConfig)
	if invalidWriter == nil {
		t.Error("invalid output mode writer should not be nil")
	}
}

func TestLoadConfig(t *testing.T) {
	// Create test config file
	testDir := "/tmp/test-config"
	os.MkdirAll(testDir, 0755)
	defer os.RemoveAll(testDir)

	configFile := filepath.Join(testDir, "kae-device-plugin.yaml")
	configContent := `
level: debug
encoderType: json
path: /tmp/test
fileName: test.log
maxSize: 10
maxBackups: 5
maxAge: 7
localTime: true
compress: true
outMod: file
`

	err := os.WriteFile(configFile, []byte(configContent), 0644)
	if err != nil {
		t.Fatalf("failed to write config file: %v", err)
	}

	// Reset viper instance
	viper.Reset()

	// Set config path and file
	viper.SetConfigFile(configFile)
	viper.AddConfigPath(testDir)
	viper.SetConfigName("kae-device-plugin")
	viper.SetConfigType("yaml")

	config, err := parseConfig()
	if err != nil {
		t.Errorf("parseConfig failed: %v", err)
	}

	if config == nil {
		t.Error("config should not be nil")
		return
	}

	if config.Level != "debug" {
		t.Errorf("expected level 'debug', got '%s'", config.Level)
	}

	if config.EncoderType != "json" {
		t.Errorf("expected encoderType 'json', got '%s'", config.EncoderType)
	}
}

func TestParseConfig(t *testing.T) {
	// Create a test config file
	testDir := "/tmp/test-parse-config"
	os.MkdirAll(testDir, 0755)
	defer os.RemoveAll(testDir)

	configFile := filepath.Join(testDir, "test.yaml")
	configContent := `
level: info
encoderType: console
path: /tmp/logs
fileName: app.log
`

	err := os.WriteFile(configFile, []byte(configContent), 0644)
	if err != nil {
		t.Fatalf("failed to write config file: %v", err)
	}

	// Reset and configure viper
	viper.Reset()
	viper.SetConfigFile(configFile)

	config, err := parseConfig()
	if err != nil {
		t.Errorf("parseConfig failed: %v", err)
	}

	if config == nil {
		t.Error("config should not be nil")
	}

	if config.Level != "info" {
		t.Errorf("expected level 'info', got '%s'", config.Level)
	}

	if config.EncoderType != "console" {
		t.Errorf("expected encoderType 'console', got '%s'", config.EncoderType)
	}
}

func TestDefaultConf(t *testing.T) {
	defaultConf := getDefaultConf()
	if defaultConf == nil {
		t.Error("getDefaultConf should not return nil")
	}

	if defaultConf.Level != "info" {
		t.Errorf("expected default level 'info', got '%s'", defaultConf.Level)
	}

	if defaultConf.EncoderType != "console" {
		t.Errorf("expected default encoderType 'console', got '%s'", defaultConf.EncoderType)
	}

	if defaultConf.OutMod != "console" {
		t.Errorf("expected default outMod 'console', got '%s'", defaultConf.OutMod)
	}

	if defaultConf.MaxSize != 20 {
		t.Errorf("expected default maxSize 20, got %d", defaultConf.MaxSize)
	}

	if defaultConf.MaxBackups != 0 {
		t.Errorf("expected default maxBackups 0, got %d", defaultConf.MaxBackups)
	}

	if defaultConf.MaxAge != 7 {
		t.Errorf("expected default maxAge 7, got %d", defaultConf.MaxAge)
	}

	if defaultConf.LocalTime != false {
		t.Errorf("expected default localTime false, got %v", defaultConf.LocalTime)
	}

	if defaultConf.Compress != true {
		t.Errorf("expected default compress true, got %v", defaultConf.Compress)
	}
}

func TestLogLevelMapping(t *testing.T) {
	// Test that all log levels are correctly mapped
	levels := map[string]bool{
		"debug": true,
		"info":  true,
		"warn":  true,
		"error": true,
	}

	for level := range logLevel {
		if !levels[level] {
			t.Errorf("unknown log level: %s", level)
		}
	}

	// Test case insensitivity
	config := &LogConfig{
		Level:       "DEBUG",
		EncoderType: "console",
		FileName:    "test.log",
		OutMod:      "console",
	}
	logger := GetLogger(config)
	if logger == nil {
		t.Error("GetLogger should handle uppercase log levels")
	}
}

func TestDebugLoggerFunctions(t *testing.T) {
	// Test that logger functions don't panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Logger function panicked: %v", r)
		}
	}()

	Debug("test message")
	Debugf("test format %s", "arg")
	Info("test message")
	Infof("test format %s", "arg")
	Warn("test message")
	Warnf("test format %s", "arg")
	Error("test message")
	Errorf("test format %s", "arg")

	// Test Sync - ignore errors when syncing stdout
	_ = Sync()
}

func TestGetLoggerWithNilConfig(t *testing.T) {
	// Test that GetLogger handles nil config - it should panic
	defer func() {
		if r := recover(); r != nil {
			t.Logf("GetLogger correctly panicked with nil config: %v", r)
		} else {
			t.Error("GetLogger should panic with nil config")
		}
	}()

	logger := GetLogger(nil)
	// Should not reach here
	t.Errorf("Expected panic, got logger: %v", logger)
}

func TestWatchConfig(t *testing.T) {
	// Test that watchConfig can be called multiple times without panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("watchConfig panicked: %v", r)
		}
	}()

	watchConfig()
	watchConfig()
}
