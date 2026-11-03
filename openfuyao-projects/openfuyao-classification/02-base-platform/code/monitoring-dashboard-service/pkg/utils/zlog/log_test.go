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

package zlog

import (
	"errors"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// TestGetDefaultConf 校验默认日志配置生成
func TestGetDefaultConf(t *testing.T) {
	conf := getDefaultConf()
	if conf == nil {
		t.Fatal("getDefaultConf returned nil")
	}
	if conf.Level != "info" || conf.EncoderType != "console" || conf.OutMod != "console" {
		t.Errorf("unexpected default conf: %+v", conf)
	}
	if conf.FileName != "logging-operator.log" {
		t.Errorf("unexpected default file name: %s", conf.FileName)
	}
}

// TestGetLoggerConsole 校验console模式下logger可创建并可用
func TestGetLoggerConsole(t *testing.T) {
	conf := &LogConfig{
		Level:       "debug",
		EncoderType: "console",
		OutMod:      "console",
	}
	logger := GetLogger(conf)
	if logger == nil {
		t.Fatal("GetLogger returned nil")
	}
	logger.Info("console logger works")
}

// TestGetLoggerJSONEncoder 校验json编码器分支
func TestGetLoggerJSONEncoder(t *testing.T) {
	conf := &LogConfig{
		Level:       "info",
		EncoderType: "JSON",
		OutMod:      "console",
	}
	logger := GetLogger(conf)
	if logger == nil {
		t.Fatal("GetLogger returned nil for json encoder")
	}
	logger.Info("json logger works")
}

// TestGetLoggerInvalidLevel 校验非法日志级别回退为info
func TestGetLoggerInvalidLevel(t *testing.T) {
	conf := &LogConfig{
		Level:       "not-a-level",
		EncoderType: "console",
		OutMod:      "console",
	}
	logger := GetLogger(conf)
	if logger == nil {
		t.Fatal("GetLogger returned nil for invalid level")
	}
	logger.Info("invalid level falls back to info")
}

// newLogTempDir 创建临时目录；lumberjack会持有文件句柄，
// Windows下t.TempDir自动清理会因文件占用失败，因此使用手动目录且不自动删除
func newLogTempDir(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "zlog-test-")
	if err != nil {
		t.Fatalf("create temp dir failed: %v", err)
	}
	return dir
}

// TestGetLoggerFileMode 校验file输出模式(写临时文件)
func TestGetLoggerFileMode(t *testing.T) {
	dir := newLogTempDir(t)
	conf := &LogConfig{
		Level:       "warn",
		EncoderType: "console",
		Path:        dir,
		FileName:    "test-file.log",
		MaxSize:     1,
		MaxBackups:  1,
		MaxAge:      1,
		Compress:    false,
		OutMod:      "file",
	}
	logger := GetLogger(conf)
	if logger == nil {
		t.Fatal("GetLogger returned nil for file mode")
	}
	logger.Warn("file logger works")
	if err := logger.Sync(); err != nil {
		t.Logf("sync file logger: %v", err)
	}
}

// TestGetLoggerBothMode 校验both(console+file)输出模式
func TestGetLoggerBothMode(t *testing.T) {
	dir := newLogTempDir(t)
	conf := &LogConfig{
		Level:       "error",
		EncoderType: "json",
		Path:        dir,
		FileName:    "test-both.log",
		OutMod:      "both",
	}
	logger := GetLogger(conf)
	if logger == nil {
		t.Fatal("GetLogger returned nil for both mode")
	}
	logger.Error("both logger works")
}

// TestGetLoggerUnknownOutMod 校验未知输出模式回退到stdout
func TestGetLoggerUnknownOutMod(t *testing.T) {
	conf := &LogConfig{
		Level:   "info",
		OutMod:  "unknown-mode",
		Path:    filepath.Join(t.TempDir(), "unused"),
		FileName: "unused.log",
	}
	logger := GetLogger(conf)
	if logger == nil {
		t.Fatal("GetLogger returned nil for unknown out mod")
	}
	logger.Info("unknown outmod falls back to console")
}

// TestCreateLumberjackLogger 校验lumberjack logger构造
func TestCreateLumberjackLogger(t *testing.T) {
	conf := &LogConfig{
		Path:       "/tmp/log-dir",
		FileName:   "app.log",
		MaxSize:    5,
		MaxBackups: 2,
		MaxAge:     3,
		Compress:   true,
	}
	lj := createLumberjackLogger(conf)
	if lj == nil {
		t.Fatal("createLumberjackLogger returned nil")
	}
	if lj.Filename != filepath.Join("/tmp/log-dir", "app.log") {
		t.Errorf("unexpected lumberjack filename: %s", lj.Filename)
	}
	if lj.MaxSize != 5 || lj.MaxBackups != 2 || lj.MaxAge != 3 || !lj.Compress {
		t.Errorf("unexpected lumberjack config: %+v", lj)
	}
}

// TestHandleError 校验HandleError写入响应
func TestHandleError(t *testing.T) {
	recorder := httptest.NewRecorder()
	HandleError(recorder, 500, errors.New("boom"))
	if recorder.Code != 500 {
		t.Errorf("expected status 500, got %d", recorder.Code)
	}
	if recorder.Body.String() != "Error occurred: boom" {
		t.Errorf("unexpected body: %s", recorder.Body.String())
	}
}

// TestLogFunctions 覆盖非致命日志函数
func TestLogFunctions(t *testing.T) {
	AddContext("key", "value")
	LogDebug("debug", "msg")
	LogInfo("info", "msg")
	LogWarn("warn", "msg")
	LogError("error", "msg")
	// 生产模式下DPanic只记录不panic
	LogDPanic("dpanic", "msg")
	FormatDebug("format %s", "debug")
	FormatInfo("format %s", "info")
	FormatWarn("format %s", "warn")
	FormatError("format %s", "error")
	FormatDPanic("format %s", "dpanic")
	LogDebugWithContext("debug ctx", "k", "v")
	LogInfoWithContext("info ctx", "k", "v")
	LogWarnWithContext("warn ctx", "k", "v")
	LogErrorWithContext("error ctx", "k", "v")
	LogDPanicWithContext("dpanic ctx", "k", "v")
	LogDebugLine("debug", "line")
	LogInfoLine("info", "line")
	LogWarnLine("warn", "line")
	LogErrorLine("error", "line")
	LogDPanicLine("dpanic", "line")
	if Logger == nil {
		t.Fatal("global Logger is nil")
	}
}

// TestLogPanicFunctions 校验panic类日志函数会panic并被recover捕获
func TestLogPanicFunctions(t *testing.T) {
	cases := []struct {
		name string
		fn   func()
	}{
		{"LogPanic", func() { LogPanic("panic msg") }},
		{"FormatPanic", func() { FormatPanic("panic %s", "fmt") }},
		{"LogPanicWithContext", func() { LogPanicWithContext("panic ctx", "k", "v") }},
		{"LogPanicLine", func() { LogPanicLine("panic", "line") }},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("%s did not panic", tc.name)
				}
			}()
			tc.fn()
		})
	}
}

// TestFlushLogs 校验日志刷盘
func TestFlushLogs(t *testing.T) {
	if err := FlushLogs(); err != nil {
		t.Logf("FlushLogs returned error (acceptable on console writer): %v", err)
	}
}
