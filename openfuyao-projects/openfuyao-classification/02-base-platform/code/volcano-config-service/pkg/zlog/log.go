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
// It utilizes the zap logging library to configure and manage log outputs efficiently.
// This package supports different log output modes including console, file, and a combination of both.
// It includes configuration setup for logging levels, formats, and output destinations using
// the lumberjack library for log file rotation and management.
package zlog

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

const (
	defaultConfigPath = "/etc/volcano-config-service"
	defaultConfigName = "volcano-config-service"
	defaultConfigType = "yaml"
	defaultLogPath    = "/var/log/volcano-config-service"
)

// Logger base logger
var Logger *zap.SugaredLogger

var logLevel = map[string]zapcore.Level{
	"debug": zapcore.DebugLevel,
	"info":  zapcore.InfoLevel,
	"warn":  zapcore.WarnLevel,
	"error": zapcore.ErrorLevel,
}

var watchOnce = sync.Once{}

// LogConfig defines the configuration settings for the volcano config service.
type LogConfig struct {
	Level       string // - Level specifies the minimum level of logs to output.
	EncoderType string // - EncoderType determines the log output format (e.g., json or console).
	Path        string // - Path and FileName define where logs are stored on the file system.
	FileName    string // - Path and FileName define where logs are stored on the file system.
	MaxSize     int    // - MaxSize, MaxBackups, and MaxAge control log rotation behavior.
	MaxBackups  int    // - MaxSize, MaxBackups, and MaxAge control log rotation behavior.
	MaxAge      int    // - MaxSize, MaxBackups, and MaxAge control log rotation behavior.
	LocalTime   bool   // - LocalTime indicates whether to use the local time for log timestamps.
	Compress    bool   // - Compress determines whether to compress rotated log files.
	OutMod      string // - OutMod defines the output mode (e.g., file, console, or both).
}

func init() {
	var conf *LogConfig
	var err error
	if conf, err = loadConfig(); err != nil {
		fmt.Printf("loadConfig fail err is %v. use DefaultConf\n", err)
		conf = getDefaultConf()
	}
	Logger = GetLogger(conf)
}

func loadConfig() (*LogConfig, error) {
	viper.AddConfigPath(defaultConfigPath)
	viper.SetConfigName(defaultConfigName)
	viper.SetConfigType(defaultConfigType)

	config, err := parseConfig()
	if err != nil {
		return nil, err
	}
	watchConfig()
	return config, nil
}

func getDefaultConf() *LogConfig {
	var defaultConf = &LogConfig{
		Level:       "info",
		EncoderType: "console",
		Path:        defaultLogPath,
		FileName:    "volcano-config-service.log",
		MaxSize:     20,
		MaxBackups:  5,
		MaxAge:      30,
		LocalTime:   false,
		Compress:    true,
		OutMod:      "both",
	}
	exePath, err := os.Executable()
	if err != nil {
		return defaultConf
	}
	serviceName := strings.TrimSuffix(filepath.Base(exePath), filepath.Ext(filepath.Base(exePath)))
	defaultConf.Path = filepath.Join(defaultLogPath, serviceName)
	return defaultConf
}

// GetLogger initializes and returns a new instance of zap.SugaredLogger based on the provided LogConfig.
func GetLogger(conf *LogConfig) *zap.SugaredLogger {
	writeSyncer := getLogWriter(conf)
	encoder := getEncoder(conf)
	level, ok := logLevel[strings.ToLower(conf.Level)]
	if !ok {
		level = logLevel["info"]
	}
	core := zapcore.NewCore(encoder, writeSyncer, level)
	logger := zap.New(core, zap.AddCaller(), zap.AddCallerSkip(1))
	return logger.Sugar()
}

func watchConfig() {
	// 监听配置文件的变化
	watchOnce.Do(func() {
		viper.WatchConfig()
		viper.OnConfigChange(func(e fsnotify.Event) {
			Logger.Warn("ConfigClient file changed")
			// 重新加载配置
			conf, err := parseConfig()
			if err != nil {
				Logger.Warnf("Error reloading config file: %v\n", err)
			} else {
				Logger = GetLogger(conf)
			}
		})
	})
}

func parseConfig() (*LogConfig, error) {
	err := viper.ReadInConfig()
	if err != nil {
		return nil, err
	}
	var config LogConfig
	err = viper.Unmarshal(&config)
	if err != nil {
		return nil, err
	}
	return &config, nil
}

// 获取编码器,NewJSONEncoder()输出json格式，NewConsoleEncoder()输出普通文本格式
func getEncoder(conf *LogConfig) zapcore.Encoder {
	encoderConfig := zap.NewProductionEncoderConfig()
	// 指定时间格式 for example: 2021-09-11t20:05:54.852+0800
	encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	// 按级别显示不同颜色，不需要的话取值zapcore.CapitalLevelEncoder就可以了
	encoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder
	// NewJSONEncoder()输出json格式，NewConsoleEncoder()输出普通文本格式
	if strings.ToLower(conf.EncoderType) == "json" {
		return zapcore.NewJSONEncoder(encoderConfig)
	}
	return zapcore.NewConsoleEncoder(encoderConfig)
}

func getLogWriter(conf *LogConfig) zapcore.WriteSyncer {
	switch conf.OutMod {
	case "console":
		return zapcore.AddSync(os.Stdout)
	case "file":
		return zapcore.AddSync(createLumberjackLogger(conf))
	case "both":
		return zapcore.NewMultiWriteSyncer(zapcore.AddSync(createLumberjackLogger(conf)), zapcore.AddSync(os.Stdout))
	default:
		return zapcore.AddSync(os.Stdout)
	}
}

func createLumberjackLogger(conf *LogConfig) *lumberjack.Logger {
	return &lumberjack.Logger{
		Filename:   filepath.Join(conf.Path, conf.FileName),
		MaxSize:    conf.MaxSize,
		MaxBackups: conf.MaxBackups,
		MaxAge:     conf.MaxAge,
		LocalTime:  conf.LocalTime,
		Compress:   conf.Compress,
	}
}

// Warn logs the provided arguments at [WarnLevel].
// Spaces are added between arguments when neither is a string.
func Warn(args ...interface{}) {
	Logger.Warn(args...)
}

// Info logs the provided arguments at [].
// Spaces are added between arguments when neither is a string.
func Info(args ...interface{}) {
	Logger.Info(args...)
}

// Error logs the provided arguments at [].
// Spaces are added between arguments when neither is a string.
func Error(args ...interface{}) {
	Logger.Error(args...)
}

// FormatError formats the message according to the format specifier
// and logs it at [ErrorLevel].
func FormatError(template string, args ...interface{}) {
	Logger.Errorf(template, args...)
}

// FormatWarn formats the message according to the format specifier
// and logs it at [WarnLevel].
func FormatWarn(template string, args ...interface{}) {
	Logger.Warnf(template, args...)
}

// FormatInfo formats the message according to the format specifier
// and logs it at [].
func FormatInfo(template string, args ...interface{}) {
	Logger.Infof(template, args...)
}

// FormatDebug formats the message according to the format specifier
// and logs it at [DebugLevel].
func FormatDebug(template string, args ...interface{}) {
	Logger.Debugf(template, args...)
}

// FormatFatal formats the message according to the format specifier
// and calls os.Exit.
func FormatFatal(template string, args ...interface{}) {
	Logger.Fatalf(template, args...)
}

// Errorf formats the message according to the format specifier
// and logs it at [ErrorLevel].
func Errorf(template string, args ...interface{}) {
	Logger.Errorf(template, args...)
}

// Warnf formats the message according to the format specifier
// and logs it at [WarnLevel].
func Warnf(template string, args ...interface{}) {
	Logger.Warnf(template, args...)
}

// Infof formats the message according to the format specifier
// and logs it at [].
func Infof(template string, args ...interface{}) {
	Logger.Infof(template, args...)
}

// Debugf formats the message according to the format specifier
// and logs it at [DebugLevel].
func Debugf(template string, args ...interface{}) {
	Logger.Debugf(template, args...)
}

// Fatalf formats the message according to the format specifier
// and calls os.Exit.
func Fatalf(template string, args ...interface{}) {
	Logger.Fatalf(template, args...)
}

// Sync flushes any buffered log entries.
func Sync() error {
	return Logger.Sync()
}
