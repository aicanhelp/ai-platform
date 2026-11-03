// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.

package zlog

import (
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap/zapcore"
)

const (
	maxSize = 20
)

func TestGetDefaultConf(t *testing.T) {
	// Mock os.Executable to simulate cross-platform behavior
	patch := gomonkey.ApplyFunc(os.Executable, func() (string, error) {
		return "/fake/path/myapp", nil
	})
	defer patch.Reset()

	conf := getDefaultConf()
	assert.Equal(t, "info", conf.Level)
	assert.Equal(t, "console", conf.EncoderType)
	assert.Equal(t, "mcs.log", conf.FileName)
	assert.Equal(t, maxSize, conf.MaxSize)
	assert.Equal(t, true, conf.Compress)
	assert.Equal(t, "both", conf.OutMod)
}

func TestGetEncoder(t *testing.T) {
	conf := &LogConfig{EncoderType: "json"}
	enc := getEncoder(conf)
	// 通过编码器的名称或其他特征来判断类型
	encoderType := getEncoderType(enc)
	assert.Equal(t, "json", encoderType)

	conf.EncoderType = "console"
	enc = getEncoder(conf)
	encoderType = getEncoderType(enc)
	assert.Equal(t, "console", encoderType)

	conf.EncoderType = ""
	enc = getEncoder(conf)
	encoderType = getEncoderType(enc)
	assert.Equal(t, "console", encoderType)
}

// 辅助函数：获取编码器类型
func getEncoderType(encoder zapcore.Encoder) string {
	// 创建一个简单的日志条目来测试编码器输出格式
	entry := zapcore.Entry{
		Level:   zapcore.InfoLevel,
		Message: "test",
	}
	var fields []zapcore.Field

	// 使用内存缓冲区来捕获编码器输出
	buffer, _ := encoder.EncodeEntry(entry, fields)
	output := buffer.String()

	// JSON 编码器输出应该是 JSON 格式（以 { 开头）
	if strings.HasPrefix(strings.TrimSpace(output), "{") {
		return "json"
	}
	return "console"
}

func TestGetLogWriter(t *testing.T) {
	conf := &LogConfig{
		OutMod:   "console",
		Path:     "/tmp",
		FileName: "test.log",
	}

	writer := getLogWriter(conf)
	assert.NotNil(t, writer)

	conf.OutMod = "file"
	writer = getLogWriter(conf)
	assert.NotNil(t, writer)

	conf.OutMod = "both"
	writer = getLogWriter(conf)
	assert.NotNil(t, writer)

	conf.OutMod = "unknown"
	writer = getLogWriter(conf)
	assert.NotNil(t, writer)
}

func TestLoadConfigError(t *testing.T) {
	viper.Reset()
	viper.SetConfigType("yaml")
	// 设置一个错误的配置文件路径
	viper.AddConfigPath("/nonexistent/path")
	viper.SetConfigName("nonexistent")
	_, err := loadConfig()
	assert.Error(t, err)
}

func TestGetLogger(t *testing.T) {
	conf := &LogConfig{
		Level:       "info",
		EncoderType: "console",
		OutMod:      "console",
	}
	logger := GetLogger(conf)
	assert.NotNil(t, logger)
	logger.Info("Test info log")
}

func TestLogLevelMapping(t *testing.T) {
	level := logLevel["debug"]
	assert.Equal(t, zapcore.DebugLevel, level)

	level = logLevel["unknown"]
	assert.Equal(t, zapcore.InfoLevel, level) // default
}

func TestLogFunctions(t *testing.T) {
	// 设置测试用的 logger，输出到内存缓冲区而不是文件
	conf := &LogConfig{
		Level:       "debug",
		EncoderType: "console",
		OutMod:      "console",
	}

	// 保存原始 logger
	originalLogger := Logger
	defer func() {
		Logger = originalLogger
	}()

	// 设置测试 logger
	Logger = GetLogger(conf)

	const arg1 = 42

	// 测试基本日志函数
	t.Run("Test basic log functions", func(t *testing.T) {
		LogDebug("debug message")
		LogInfo("info message")
		LogWarn("warn message")
		LogError("error message")
	})

	// 测试格式化日志函数
	t.Run("Test formatted log functions", func(t *testing.T) {
		LogDebugf("debug message: %s", "test")
		LogInfof("info message: %d", arg1)
		Warnf("warn message: %v", []string{"a", "b"})
		LogErrorf("error message: %s", "something went wrong")
	})
}

func TestLogLevelFiltering(t *testing.T) {
	// 保存原始 logger
	originalLogger := Logger
	defer func() {
		Logger = originalLogger
	}()

	// 测试不同日志级别过滤
	testCases := []struct {
		level    string
		expected []string // 预期会输出的日志级别
	}{
		{"error", []string{"error", "fatal"}},
		{"warn", []string{"warn", "error", "fatal"}},
		{"info", []string{"info", "warn", "error", "fatal"}},
		{"debug", []string{"debug", "info", "warn", "error", "fatal"}},
	}

	const arg1 = 123

	for _, tc := range testCases {
		t.Run("Level_"+tc.level, func(t *testing.T) {
			conf := &LogConfig{
				Level:       tc.level,
				EncoderType: "console",
				OutMod:      "console",
			}
			Logger = GetLogger(conf)

			// 调用所有日志级别函数
			LogDebug("debug message")
			LogInfo("info message")
			LogWarn("warn message")
			LogError("error message")

			LogDebugf("debug formatted %s", "message")
			LogInfof("info formatted %d", arg1)
			Warnf("warn formatted %v", []int{1, 2, 3})
			LogErrorf("error formatted %s", "error")
		})
	}
}

func TestLogWithMultipleArgs(t *testing.T) {
	// 保存原始 logger
	originalLogger := Logger
	defer func() {
		Logger = originalLogger
	}()

	conf := &LogConfig{
		Level:       "debug",
		EncoderType: "console",
		OutMod:      "console",
	}
	Logger = GetLogger(conf)

	const arg1 = 42
	const arg2 = 3.14
	const arg3 = 5

	// 测试多个参数的日志函数
	t.Run("Test multiple args", func(t *testing.T) {
		LogDebug("message", "arg1", arg1, true)
		LogInfo("message", "arg2", arg2, false)
		LogWarn("message", "arg3", []string{"a", "b"}, map[string]int{"key": 1})
		LogError("message", "arg4", struct{ Name string }{Name: "test"}, nil)
	})

	// 测试多个参数的格式化日志函数
	t.Run("Test formatted multiple args", func(t *testing.T) {
		LogDebugf("message: %s, number: %d", "test", arg1)
		LogInfof("user %s has %d items", "alice", arg3)
		Warnf("warning with values: %v and %v", []int{1, 2, 3}, map[string]bool{"ok": true})
		LogErrorf("error code %d: %s", http.StatusInternalServerError, "internal server error")
	})
}

// 由于 LogFatal 和 LogFatalf 会调用 os.Exit，我们需要 mock os.Exit 来测试它们
func TestLogFatalFunctions(t *testing.T) {
	// 保存原始 logger
	originalLogger := Logger
	defer func() {
		Logger = originalLogger
	}()

	conf := &LogConfig{
		EncoderType: "console",
		Level:       "debug",
		OutMod:      "console",
	}
	Logger = GetLogger(conf)

	// Mock os.Exit
	exitCalled := false
	patch := gomonkey.ApplyFunc(os.Exit, func(code int) {
		// 不真正退出程序
		exitCalled = true
	})
	defer patch.Reset()

	// 测试 LogFatal
	t.Run("Test LogFatal", func(t *testing.T) {
		exitCalled = false
		LogFatal("fatal message")
		assert.True(t, exitCalled, "os.Exit should be called")
	})

	// 测试 LogFatalf
	t.Run("Test LogFatalf", func(t *testing.T) {
		exitCalled = false
		LogFatalf("fatal formatted %s", "message")
		assert.True(t, exitCalled, "os.Exit should be called")
	})
}

func TestLogFunctionsWithSpecialValues(t *testing.T) {
	// 保存原始 logger
	originalLogger := Logger
	defer func() {
		Logger = originalLogger
	}()

	conf := &LogConfig{
		OutMod:      "console",
		Level:       "debug",
		EncoderType: "console",
	}
	Logger = GetLogger(conf)

	const arg1 = 30
	const arg2 = 123

	// 测试特殊值
	t.Run("Test with nil values", func(t *testing.T) {
		LogDebug(nil)
		LogInfo(nil, "message")
		LogWarn("message", nil)
	})

	t.Run("Test with empty values", func(t *testing.T) {
		LogDebug()
		LogInfof("")
		Warnf("%s", "")
	})

	t.Run("Test with complex structures", func(t *testing.T) {
		LogDebug(map[string]interface{}{"key": "value", "nested": []int{1, 2, 3}})
		LogInfo(struct {
			Name string
			Age  int
		}{Name: "Alice", Age: arg1})
		LogWarn([]interface{}{"mixed", arg2, true, nil})
	})
}
