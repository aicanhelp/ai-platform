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

/**
* Package zlog 封装日志，提供日志打印功能，支持文件和控制台输出，支持标准输出和结构化输出日志
 */
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
	defaultConfigPath = "/etc/kae-device-plugin"
	defaultConfigName = "kae-device-plugin"
	defaultConfigType = "yaml"
	defaultLogPath    = "/var/log"
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

// LogConfig 日志配置
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
		FileName:    "root.log",
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

// GetLogger 根据配置信息，返回 SugaredLogger 对象
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
			Logger.Warn("Config file changed")
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

/* 以下封装 SugaredLogger 常用接口 */

// With 向日志上下文添加可变数量的字段。它接受强类型 Field 对象和松散类型键值对的混合。
// 处理键值对时，第一个元素用作字段键，第二个用作字段值。
func With(args ...interface{}) *zap.SugaredLogger {
	return Logger.With(args...)
}

// Debug 以 [DebugLevel] 记录提供的参数。
// 当参数都不是字符串时，会在参数之间添加空格。
func Debug(args ...interface{}) {
	Logger.Debug(args...)
}

// Info 记录提供的参数。
// 当参数都不是字符串时，会在参数之间添加空格。
func Info(args ...interface{}) {
	Logger.Info(args...)
}

// Warn 以 [WarnLevel] 记录提供的参数。
// 当参数都不是字符串时，会在参数之间添加空格。
func Warn(args ...interface{}) {
	Logger.Warn(args...)
}

// Error 以 [ErrorLevel] 记录提供的参数。
// 当参数都不是字符串时，会在参数之间添加空格。
func Error(args ...interface{}) {
	Logger.Error(args...)
}

// DPanic 以 [DPanicLevel] 记录提供的参数。
// 在开发环境中，Logger 会触发 panic。（详见 [DPanicLevel]）。
// 当参数都不是字符串时，会在参数之间添加空格。
func DPanic(args ...interface{}) {
	Logger.DPanic(args...)
}

// Panic 使用提供的参数构造消息并触发 panic。
// 当参数都不是字符串时，会在参数之间添加空格。
func Panic(args ...interface{}) {
	Logger.Panic(args...)
}

// Fatal 使用提供的参数构造消息并调用 os.Exit。
// 当参数都不是字符串时，会在参数之间添加空格。
func Fatal(args ...interface{}) {
	Logger.Fatal(args...)
}

// Debugf 根据格式说明符格式化消息，
// 并以 [DebugLevel] 记录。
func Debugf(template string, args ...interface{}) {
	Logger.Debugf(template, args...)
}

// Infof 根据格式说明符格式化消息，
// 并记录。
func Infof(template string, args ...interface{}) {
	Logger.Infof(template, args...)
}

// Warnf 根据格式说明符格式化消息，
// 并以 [WarnLevel] 记录。
func Warnf(template string, args ...interface{}) {
	Logger.Warnf(template, args...)
}

// Errorf 根据格式说明符格式化消息，
// 并以 [ErrorLevel] 记录。
func Errorf(template string, args ...interface{}) {
	Logger.Errorf(template, args...)
}

// DPanicf 根据格式说明符格式化消息，
// 并以 [DPanicLevel] 记录。
// 在开发环境中，Logger 会触发 panic。（详见 [DPanicLevel]）。
func DPanicf(template string, args ...interface{}) {
	Logger.DPanicf(template, args...)
}

// Panicf 根据格式说明符格式化消息，
// 并触发 panic。
func Panicf(template string, args ...interface{}) {
	Logger.Panicf(template, args...)
}

// Fatalf 根据格式说明符格式化消息，
// 并调用 os.Exit。
func Fatalf(template string, args ...interface{}) {
	Logger.Fatalf(template, args...)
}

// Debugw 记录带有额外上下文的消息。可变参数键值对的处理方式与 With 相同。
// 当禁用调试级别日志时，此方法速度更快。
func Debugw(msg string, keysAndValues ...interface{}) {
	Logger.Debugw(msg, keysAndValues...)
}

// Infow 记录带有额外上下文的消息。可变参数键值对的处理方式与 With 相同。
func Infow(msg string, keysAndValues ...interface{}) {
	Logger.Infow(msg, keysAndValues...)
}

// Warnw 记录带有额外上下文的消息。可变参数键值对的处理方式与 With 相同。
func Warnw(msg string, keysAndValues ...interface{}) {
	Logger.Warnw(msg, keysAndValues...)
}

// Errorw 记录带有额外上下文的消息。可变参数键值对的处理方式与 With 相同。
func Errorw(msg string, keysAndValues ...interface{}) {
	Logger.Errorw(msg, keysAndValues...)
}

// DPanicw 记录带有额外上下文的消息。在开发环境中，
// Logger 会触发 panic。（详见 DPanicLevel）。可变参数键值对的处理方式与 With 相同。
func DPanicw(msg string, keysAndValues ...interface{}) {
	Logger.DPanicw(msg, keysAndValues...)
}

// Panicw 记录带有额外上下文的消息，然后触发 panic。
// 可变参数键值对的处理方式与 With 相同。
func Panicw(msg string, keysAndValues ...interface{}) {
	Logger.Panicw(msg, keysAndValues...)
}

// Fatalw 记录带有额外上下文的消息，然后调用 os.Exit。
// 可变参数键值对的处理方式与 With 相同。
func Fatalw(msg string, keysAndValues ...interface{}) {
	Logger.Fatalw(msg, keysAndValues...)
}

// Debugln 以 [DebugLevel] 记录消息。
// 参数之间总是添加空格。
func Debugln(args ...interface{}) {
	Logger.Debugln(args...)
}

// Infoln 记录消息。
// 参数之间总是添加空格。
func Infoln(args ...interface{}) {
	Logger.Infoln(args...)
}

// Warnln 以 [WarnLevel] 记录消息。
// 参数之间总是添加空格。
func Warnln(args ...interface{}) {
	Logger.Warnln(args...)
}

// Errorln 以 [ErrorLevel] 记录消息。
// 参数之间总是添加空格。
func Errorln(args ...interface{}) {
	Logger.Errorln(args...)
}

// DPanicln 以 [DPanicLevel] 记录消息。
// 在开发环境中，Logger 会触发 panic。（详见 [DPanicLevel]）。
// 参数之间总是添加空格。
func DPanicln(args ...interface{}) {
	Logger.DPanicln(args...)
}

// Panicln 以 [PanicLevel] 记录消息并触发 panic。
// 参数之间总是添加空格。
func Panicln(args ...interface{}) {
	Logger.Panicln(args...)
}

// Fatalln 以 [FatalLevel] 记录消息并调用 os.Exit。
// 参数之间总是添加空格。
func Fatalln(args ...interface{}) {
	Logger.Fatalln(args...)
}

// Sync 刷新所有缓冲的日志条目。
func Sync() error {
	return Logger.Sync()
}
