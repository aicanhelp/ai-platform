/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// newBufferLogger returns a sugared logger writing to the given buffer.
func newBufferLogger(buf *bytes.Buffer, fatalPanic bool) *zap.SugaredLogger {
	encoderConfig := zap.NewProductionEncoderConfig()
	core := zapcore.NewCore(zapcore.NewConsoleEncoder(encoderConfig), zapcore.AddSync(buf), zapcore.DebugLevel)
	if fatalPanic {
		return zap.New(core, zap.AddCaller(), zap.AddCallerSkip(1),
			zap.WithFatalHook(zapcore.WriteThenPanic)).Sugar()
	}
	return zap.New(core, zap.AddCaller(), zap.AddCallerSkip(1)).Sugar()
}

func assertBufferContains(t *testing.T, buf *bytes.Buffer, substr string) {
	t.Helper()
	if !strings.Contains(buf.String(), substr) {
		t.Errorf("log output should contain %q, got %q", substr, buf.String())
	}
}

// Note: failure-path tests must run before TestLoadConfigSuccess registers a
// reachable config file path in the global viper instance.
func TestParseConfigFailure(t *testing.T) {
	if _, err := parseConfig(); err == nil {
		t.Errorf("parseConfig() should fail when no config file exists")
	}
}

func TestLoadConfigFailure(t *testing.T) {
	if _, err := loadConfig(); err == nil {
		t.Errorf("loadConfig() should fail when no config file exists")
	}
}

func TestGetDefaultConf(t *testing.T) {
	conf := getDefaultConf()
	if conf.Level != "info" {
		t.Errorf("getDefaultConf() Level = %q, want info", conf.Level)
	}
	if conf.EncoderType == "" {
		t.Errorf("getDefaultConf() EncoderType should not be empty")
	}
	if conf.MaxSize != 20 || conf.MaxBackups != 5 || conf.MaxAge != 30 {
		t.Errorf("getDefaultConf() rotation config = %d/%d/%d, want 20/5/30",
			conf.MaxSize, conf.MaxBackups, conf.MaxAge)
	}
	if !conf.Compress || conf.LocalTime {
		t.Errorf("getDefaultConf() Compress/LocalTime = %v/%v, want true/false", conf.Compress, conf.LocalTime)
	}
	if !strings.HasPrefix(filepath.ToSlash(conf.Path), defaultLogPath) {
		t.Errorf("getDefaultConf() Path = %q, want prefix %q", conf.Path, defaultLogPath)
	}
	if conf.FileName != "openfuyao-rayservice" {
		t.Errorf("getDefaultConf() FileName = %q, want openfuyao-rayservice", conf.FileName)
	}
}

func TestGetEncoder(t *testing.T) {
	jsonEncoder := getEncoder(&LogConfig{EncoderType: "json"})
	if jsonEncoder == nil {
		t.Errorf("getEncoder() should return json encoder for EncoderType json")
	}
	// any non-json type falls back to the console encoder
	consoleEncoder := getEncoder(&LogConfig{EncoderType: "text"})
	if consoleEncoder == nil {
		t.Errorf("getEncoder() should fall back to the console encoder for non json types")
	}
	upperEncoder := getEncoder(&LogConfig{EncoderType: "JSON"})
	if upperEncoder == nil {
		t.Errorf("getEncoder() should handle upper case EncoderType")
	}
}

func TestGetLogWriter(t *testing.T) {
	conf := &LogConfig{
		Path:     t.TempDir(),
		FileName: "test.log",
	}
	cases := []struct {
		name   string
		outMod string
	}{
		{name: "file-mode", outMod: "file"},
		{name: "both-mode", outMod: "both"},
		{name: "unknown-mode", outMod: "other"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			conf.OutMod = tt.outMod
			if writer := getLogWriter(conf); writer == nil {
				t.Errorf("getLogWriter() should not return nil for OutMod %q", tt.outMod)
			}
		})
	}
}

func TestCreateLumberjackLogger(t *testing.T) {
	conf := &LogConfig{
		Path:       "/var/log/openfuyao-rayservice/test",
		FileName:   "test.log",
		MaxSize:    10,
		MaxBackups: 3,
		MaxAge:     7,
		LocalTime:  true,
		Compress:   true,
	}
	logger := createLumberjackLogger(conf)
	if logger == nil {
		t.Fatalf("createLumberjackLogger() should not return nil")
	}
	if logger.Filename != filepath.Join(conf.Path, conf.FileName) {
		t.Errorf("createLumberjackLogger() Filename = %q, want %q",
			logger.Filename, filepath.Join(conf.Path, conf.FileName))
	}
	if logger.MaxSize != 10 || logger.MaxBackups != 3 || logger.MaxAge != 7 {
		t.Errorf("createLumberjackLogger() rotation config not applied: %+v", logger)
	}
	if !logger.LocalTime || !logger.Compress {
		t.Errorf("createLumberjackLogger() LocalTime/Compress not applied: %+v", logger)
	}
}

func TestGetLogger(t *testing.T) {
	conf := &LogConfig{
		Level:       "debug",
		EncoderType: "json",
		Path:        t.TempDir(),
		FileName:    "test.log",
		OutMod:      "stdout-only",
	}
	logger := GetLogger(conf)
	if logger == nil {
		t.Fatalf("GetLogger() should not return nil")
	}

	// unknown level falls back to info
	conf.Level = "not-a-level"
	logger = GetLogger(conf)
	if logger == nil {
		t.Fatalf("GetLogger() should not return nil for unknown level")
	}
}

func TestLogWrappers(t *testing.T) {
	buf := &bytes.Buffer{}
	oldLogger := Logger
	Logger = newBufferLogger(buf, false)
	defer func() { Logger = oldLogger }()

	Warn("warn-message")
	assertBufferContains(t, buf, "warn-message")

	Info("info-message")
	assertBufferContains(t, buf, "info-message")

	Error("error-message")
	assertBufferContains(t, buf, "error-message")

	FormatError("format-error-%s", "arg")
	assertBufferContains(t, buf, "format-error-arg")

	FormatWarn("format-warn-%s", "arg")
	assertBufferContains(t, buf, "format-warn-arg")

	FormatInfo("format-info-%s", "arg")
	assertBufferContains(t, buf, "format-info-arg")

	FormatDebug("format-debug-%s", "arg")
	assertBufferContains(t, buf, "format-debug-arg")

	Errorf("errorf-%s", "arg")
	assertBufferContains(t, buf, "errorf-arg")

	Warnf("warnf-%s", "arg")
	assertBufferContains(t, buf, "warnf-arg")

	Infof("infof-%s", "arg")
	assertBufferContains(t, buf, "infof-arg")

	Debugf("debugf-%s", "arg")
	assertBufferContains(t, buf, "debugf-arg")

	if err := Sync(); err != nil {
		t.Errorf("Sync() error = %v, want nil", err)
	}
}

func TestFatalWrappersPanic(t *testing.T) {
	buf := &bytes.Buffer{}
	oldLogger := Logger
	Logger = newBufferLogger(buf, true)
	defer func() { Logger = oldLogger }()

	assertPanic := func(name string, f func()) {
		t.Helper()
		defer func() {
			if r := recover(); r == nil {
				t.Errorf("%s should panic with WriteThenPanic fatal hook", name)
			}
		}()
		f()
	}

	assertPanic("FormatFatal", func() { FormatFatal("fatal-%s", "msg") })
	assertBufferContains(t, buf, "fatal-msg")

	assertPanic("Fatalf", func() { Fatalf("fatalf-%s", "msg") })
	assertBufferContains(t, buf, "fatalf-msg")
}

func TestLoadConfigSuccess(t *testing.T) {
	dir := t.TempDir()
	configContent := `level: debug
encoderType: json
path: /tmp/zlog-test
fileName: zlog-test.log
maxSize: 10
maxBackups: 3
maxAge: 7
localTime: true
compress: false
outMod: stdout-only
`
	configFile := filepath.Join(dir, defaultConfigName+"."+defaultConfigType)
	if err := os.WriteFile(configFile, []byte(configContent), 0600); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}
	viper.AddConfigPath(dir)

	conf, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig() error = %v, want nil", err)
	}
	if conf.Level != "debug" || conf.EncoderType != "json" {
		t.Errorf("loadConfig() Level/EncoderType = %q/%q, want debug/json", conf.Level, conf.EncoderType)
	}
	if conf.Path != "/tmp/zlog-test" || conf.FileName != "zlog-test.log" {
		t.Errorf("loadConfig() Path/FileName = %q/%q, want /tmp/zlog-test/zlog-test.log",
			conf.Path, conf.FileName)
	}
	if conf.MaxSize != 10 || conf.MaxBackups != 3 || conf.MaxAge != 7 {
		t.Errorf("loadConfig() rotation config = %d/%d/%d, want 10/3/7",
			conf.MaxSize, conf.MaxBackups, conf.MaxAge)
	}
	if !conf.LocalTime || conf.Compress {
		t.Errorf("loadConfig() LocalTime/Compress = %v/%v, want true/false", conf.LocalTime, conf.Compress)
	}
	if conf.OutMod != "stdout-only" {
		t.Errorf("loadConfig() OutMod = %q, want stdout-only", conf.OutMod)
	}
}
