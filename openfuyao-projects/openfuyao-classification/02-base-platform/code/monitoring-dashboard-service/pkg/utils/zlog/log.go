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

// Package zlog provides a logging system for the logging-operator.
package zlog

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

const (
	defaultLogPath = "/var/log/logging-operator"
)

// Logger is a globally available *zap.SugaredLogger instance
var Logger *zap.SugaredLogger

var logLevel = map[string]zapcore.Level{
	"debug": zapcore.DebugLevel,
	"info":  zapcore.InfoLevel,
	"warn":  zapcore.WarnLevel,
	"error": zapcore.ErrorLevel,
}

// LogConfig defines the configuration settings for the logging system.
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
	fmt.Print("The logging backend started running.")
	conf := getDefaultConf()
	Logger = GetLogger(conf)
}

// getLogWriter returns a zapcore.WriteSyncer based on the provided LogConfig.
func getDefaultConf() *LogConfig {
	var defaultConf = &LogConfig{
		Level:       "info",
		EncoderType: "console",
		Path:        defaultLogPath,
		FileName:    "logging-operator.log",
		MaxSize:     20,
		MaxBackups:  0,
		MaxAge:      7,
		LocalTime:   false,
		Compress:    true,
		OutMod:      "console",
	}
	exePath, err := os.Executable()
	if err != nil {
		return defaultConf
	}
	// 获取运行文件名称，作为/var/log目录下的子目录
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

// getLogWriter logs an error message.
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

// createLumberjackLogger creates a new lumberjack.Logger instance based on the provided LogConfig.
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

// HandleError handles errors by logging and responding with the appropriate HTTP status code and error message.
func HandleError(response http.ResponseWriter, statusCode int, err error) {
	errorMessage := fmt.Sprintf("Error occurred: %v", err)
	// Log the error
	LogError(errorMessage)
	// Respond with error message and status code
	response.WriteHeader(statusCode)
	code, err := response.Write([]byte(errorMessage))
	if err != nil {
		errorMessage := fmt.Sprintf("Failed to write entity: %v", err)
		LogError(errorMessage)
	}
	errorMessage = fmt.Sprintf("Error code %v", code)
	LogError(errorMessage)
}

// AddContext adds a variadic number of fields to the logging context. It accepts a mix of strongly-typed Field objects
// and loosely-typed key-value pairs.
func AddContext(args ...interface{}) *zap.SugaredLogger {
	return Logger.With(args...)
}

// LogDebug writes messages at DebugLevel, inserting spaces between arguments when neither is a string.
func LogDebug(args ...interface{}) {
	Logger.Debug(args...)
}

// LogInfo logs the provided arguments at [].
// Spaces are added between arguments when neither is a string.
func LogInfo(args ...interface{}) {
	Logger.Info(args...)
}

// LogWarn logs the provided arguments at [WarnLevel].
// Spaces are added between arguments when neither is a string.
func LogWarn(args ...interface{}) {
	Logger.Warn(args...)
}

// LogError logs the provided arguments at [ErrorLevel].
// Spaces are added between arguments when neither is a string.
func LogError(args ...interface{}) {
	Logger.Error(args...)
}

// LogDPanic writes messages at DPanicLevel and panics in development environments. Spaces are added between
// arguments when neither is a string.
func LogDPanic(args ...interface{}) {
	Logger.DPanic(args...)
}

// LogPanic constructs a panic-inducing message with provided arguments, adding spaces when needed.
func LogPanic(args ...interface{}) {
	Logger.Panic(args...)
}

// LogFatal constructs a message and terminates the program using os.Exit, adding spaces when needed.
func LogFatal(args ...interface{}) {
	Logger.Fatal(args...)
}

// FormatDebug constructs a message based on a format specifier and logs it at DebugLevel.
func FormatDebug(template string, args ...interface{}) {
	Logger.Debugf(template, args...)
}

// FormatInfo constructs a message based on a format specifier and logs it at InfoLevel.
func FormatInfo(template string, args ...interface{}) {
	Logger.Infof(template, args...)
}

// FormatWarn constructs a message based on a format specifier and logs it at WarnLevel.
func FormatWarn(template string, args ...interface{}) {
	Logger.Warnf(template, args...)
}

// FormatError constructs a message based on a format specifier and logs it at ErrorLevel.
func FormatError(template string, args ...interface{}) {
	Logger.Errorf(template, args...)
}

// FormatDPanic constructs a message based on a format specifier and panics in development, as per DPanicLevel.
func FormatDPanic(template string, args ...interface{}) {
	Logger.DPanicf(template, args...)
}

// FormatPanic constructs a panic-inducing message based on a format specifier.
func FormatPanic(template string, args ...interface{}) {
	Logger.Panicf(template, args...)
}

// FormatFatal constructs a message based on a format specifier and terminates the program with os.Exit.
func FormatFatal(template string, args ...interface{}) {
	Logger.Fatalf(template, args...)
}

// LogDebugWithContext logs a debug-level message with additional context, optimizing performance when debugging is off.
func LogDebugWithContext(msg string, keysAndValues ...interface{}) {
	Logger.Debugw(msg, keysAndValues...)
}

// LogInfoWithContext logs an info-level message with additional context.
func LogInfoWithContext(msg string, keysAndValues ...interface{}) {
	Logger.Infow(msg, keysAndValues...)
}

// LogWarnWithContext logs a warn-level message with additional context.
func LogWarnWithContext(msg string, keysAndValues ...interface{}) {
	Logger.Warnw(msg, keysAndValues...)
}

// LogErrorWithContext logs an error-level message with additional context.
func LogErrorWithContext(msg string, keysAndValues ...interface{}) {
	Logger.Errorw(msg, keysAndValues...)
}

// LogDPanicWithContext logs a DPanic-level message with additional context, which panics in development.
func LogDPanicWithContext(msg string, keysAndValues ...interface{}) {
	Logger.DPanicw(msg, keysAndValues...)
}

// LogPanicWithContext logs a panic-level message with additional context.
func LogPanicWithContext(msg string, keysAndValues ...interface{}) {
	Logger.Panicw(msg, keysAndValues...)
}

// LogFatalWithContext logs a fatal-level message with additional context, then terminates the program.
func LogFatalWithContext(msg string, keysAndValues ...interface{}) {
	Logger.Fatalw(msg, keysAndValues...)
}

// LogDebugLine logs a debug-level message, always adding spaces between arguments.
func LogDebugLine(args ...interface{}) {
	Logger.Debugln(args...)
}

// LogInfoLine logs an info-level message, always adding spaces between arguments.
func LogInfoLine(args ...interface{}) {
	Logger.Infoln(args...)
}

// LogWarnLine logs a warn-level message, always adding spaces between arguments.
func LogWarnLine(args ...interface{}) {
	Logger.Warnln(args...)
}

// LogErrorLine logs an error-level message, always adding spaces between arguments.
func LogErrorLine(args ...interface{}) {
	Logger.Errorln(args...)
}

// LogDPanicLine logs a DPanic-level message and panics in development, always adding spaces between arguments.
func LogDPanicLine(args ...interface{}) {
	Logger.DPanicln(args...)
}

// LogPanicLine logs a panic-level message and panics, always adding spaces between arguments.
func LogPanicLine(args ...interface{}) {
	Logger.Panicln(args...)
}

// LogFatalLine logs a fatal-level message and terminates the program, always adding spaces between arguments.
func LogFatalLine(args ...interface{}) {
	Logger.Fatalln(args...)
}

// FlushLogs commits any pending log entries to underlying storage.
func FlushLogs() error {
	return Logger.Sync()
}
