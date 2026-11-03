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
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap/zapcore"
)

// TestCleanSpecialChar 测试特殊字符清洗逻辑
func TestCleanSpecialChar(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"Newline", "a\nb", "a\\nb"},
		{"Tab", "a\tb", "a\\tb"},
		{"Empty", "", ""},
		{"Normal", "hello", "hello"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, CleanSpecialChar(tt.input))
		})
	}
}

// TestCleanLogFields 测试日志字段清洗（防下标越界）
func TestCleanLogFields(t *testing.T) {
	const len123 = 123
	const len1 = 1
	const len2 = 2
	const len3 = 3
	// 测试用例1：正常长度数组
	args := []interface{}{"a\nb", len123, "\rtest"}
	cleaned := cleanLogFields(args)
	// 先校验长度，再访问下标
	assert.Len(t, cleaned, len(args), "清洗后数组长度与原数组不一致")
	if len(cleaned) >= len1 {
		assert.Equal(t, "a\\nb", cleaned[0])
	}
	if len(cleaned) >= len2 {
		assert.Equal(t, len123, cleaned[1])
	}
	if len(cleaned) >= len3 {
		assert.Equal(t, "\\rtest", cleaned[2])
	}

	// 测试用例3：单元素数组（边界场景）
	singleArgs := []interface{}{"\ttest"}
	singleCleaned := cleanLogFields(singleArgs)
	assert.Len(t, singleCleaned, len1)
	if len(singleCleaned) >= len1 {
		assert.Equal(t, "\\ttest", singleCleaned[0])
	}
}

// TestGetDefaultConf 测试默认配置生成
func TestGetDefaultConf(t *testing.T) {
	const len20 = 20
	conf := getDefaultConf()
	assert.Equal(t, "info", conf.Level)
	assert.Equal(t, len20, conf.MaxSize)
	assert.True(t, conf.Compress)
}

// TestGetLogWriter 测试日志输出器
func TestGetLogWriter(t *testing.T) {
	conf := &logConfig{OutMod: "console"}
	writer := getLogWriter(conf)
	assert.NotNil(t, writer)

	conf.OutMod = "file"
	conf.Path = os.TempDir()
	conf.FileName = "test.log"
	writer = getLogWriter(conf)
	assert.NotNil(t, writer)

	conf.OutMod = "both"
	writer = getLogWriter(conf)
	assert.NotNil(t, writer)
}

// TestLoggerBasic 测试基础日志方法（无panic）
func TestLoggerBasic(t *testing.T) {
	// 初始化logger后测试基础方法
	assert.NotNil(t, logger)
	Info("test info")
	Warnf("test %s", "warn")
	Debug("test debug")
	Error("test error")
	Sync()
}

// TestLogLevelMap 测试日志级别映射
func TestLogLevelMap(t *testing.T) {
	assert.Equal(t, zapcore.DebugLevel, logLevel["debug"])
	assert.Equal(t, zapcore.InfoLevel, logLevel["info"])
	assert.Equal(t, zapcore.WarnLevel, logLevel["warn"])
	assert.Equal(t, zapcore.ErrorLevel, logLevel["error"])
}

// TestParseConfig_Error 测试配置解析失败
func TestParseConfig_Error(t *testing.T) {
	// 清空viper配置，触发解析失败
	_, err := parseConfig()
	assert.Error(t, err)
}
