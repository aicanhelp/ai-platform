/*
 * Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package zlog 封装日志，提供日志打印功能，支持文件和控制台输出，支持标准输出和结构化输出日志/**
package zlog

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/fsnotify/fsnotify"
	"github.com/natefinch/lumberjack"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

const (
	defaultConfigPath = "/etc/multicluster-service"
	defaultConfigName = "multicluster-service"
	defaultConfigType = "yaml"
	defaultLogPath    = "/var/log/multicluster-service"
)

// Logger 日志对象
var Logger *zap.SugaredLogger

var logLevel = map[string]zapcore.Level{
	"debug": zapcore.DebugLevel,
	"info":  zapcore.InfoLevel,
	"warn":  zapcore.WarnLevel,
	"error": zapcore.ErrorLevel,
}

var watchOnce = sync.Once{}

// LogConfig 日志配置，配置输出格式，压缩大小等
type LogConfig struct {
	Level       string
	EncoderType string
	Path        string
	FileName    string
	MaxSize     int
	MaxBackups  int
	MaxAge      int
	LocalTime   bool
	Compress    bool
	OutMod      string
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
		FileName:    "mcs.log",
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
	// 获取运行文件名称，作为/var/log目录下的子目录
	serviceName := strings.TrimSuffix(filepath.Base(exePath), filepath.Ext(filepath.Base(exePath)))
	defaultConf.Path = filepath.Join(defaultLogPath, serviceName)
	return defaultConf
}

// GetLogger 获取logger对象
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
	watchOnce.Do(
		func() {
			viper.WatchConfig()
			viper.OnConfigChange(
				func(e fsnotify.Event) {
					Logger.Warn("Config file changed")
					// 重新加载配置
					conf, err := parseConfig()
					if err != nil {
						Logger.Warnf("Error reloading config file: %v\n", err)
					} else {
						Logger = GetLogger(conf)
					}
				},
			)
		},
	)
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
	// 只输出到控制台
	if conf.OutMod == "console" {
		return zapcore.AddSync(os.Stdout)
	}
	// 日志文件配置
	lumberJackLogger := &lumberjack.Logger{
		Filename:   filepath.Join(conf.Path, conf.FileName),
		MaxSize:    conf.MaxSize,
		MaxBackups: conf.MaxBackups,
		MaxAge:     conf.MaxAge,
		LocalTime:  conf.LocalTime,
		Compress:   conf.Compress,
	}
	if conf.OutMod == "both" {
		// 控制台和文件都输出
		return zapcore.NewMultiWriteSyncer(zapcore.AddSync(lumberJackLogger), zapcore.AddSync(os.Stdout))
	}
	if conf.OutMod == "file" {
		// 只输出到文件
		return zapcore.AddSync(lumberJackLogger)
	}
	return zapcore.AddSync(os.Stdout)
}

// LogWith adds a variadic number of fields to the logging context. It accepts a
// mix of strongly-typed Field objects and loosely-typed key-value pairs. When
// processing pairs, the first element of the pair is used as the field key
// and the second as the field value.
func LogWith(args ...interface{}) *zap.SugaredLogger {
	return Logger.With(args...)
}

// LogDebug logs the provided arguments at [DebugLevel].
func LogDebug(args ...interface{}) {
	Logger.Debug(args...)
}

// LogInfo logs the provided arguments at [].
func LogInfo(args ...interface{}) {
	Logger.Info(args...)
}

// LogWarn logs the provided arguments at [WarnLevel].
func LogWarn(args ...interface{}) {
	Logger.Warn(args...)
}

// LogError logs the provided arguments at [ErrorLevel].
func LogError(args ...interface{}) {
	Logger.Error(args...)
}

// LogFatal constructs a message with the provided arguments and calls os.Exit.
func LogFatal(args ...interface{}) {
	Logger.Fatal(args...)
}

// LogDebugf formats the message according to the format specifier and logs it at [DebugLevel].
func LogDebugf(template string, args ...interface{}) {
	Logger.Debugf(template, args...)
}

// LogInfof formats the message according to the format specifier and logs it at [].
func LogInfof(template string, args ...interface{}) {
	Logger.Infof(template, args...)
}

// Warnf formats the message according to the format specifier and logs it at [WarnLevel].
func Warnf(template string, args ...interface{}) {
	Logger.Warnf(template, args...)
}

// LogErrorf formats the message according to the format specifier and logs it at [ErrorLevel].
func LogErrorf(template string, args ...interface{}) {
	Logger.Errorf(template, args...)
}

// LogFatalf formats the message according to the format specifier and calls os.Exit.
func LogFatalf(template string, args ...interface{}) {
	Logger.Fatalf(template, args...)
}

// Sync flushes any buffered log entries.
func Sync() error {
	return Logger.Sync()
}
