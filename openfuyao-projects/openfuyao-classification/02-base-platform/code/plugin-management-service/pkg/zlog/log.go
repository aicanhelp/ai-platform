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

// Package zlog offers logging functions, based on zap library
package zlog

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

const (
	defaultConfigPath = "/etc/plugin-management-service/log-config"
	defaultConfigName = "plugin-management-service"
	defaultConfigType = "yaml"
	defaultLogPath    = "/var/log"
)

var logger *zap.SugaredLogger
var executable = os.Executable

var logLevel = map[string]zapcore.Level{
	"debug": zapcore.DebugLevel,
	"info":  zapcore.InfoLevel,
	"warn":  zapcore.WarnLevel,
	"error": zapcore.ErrorLevel,
}

var watchOnce = sync.Once{}

type logConfig struct {
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
	var conf *logConfig
	var err error
	if conf, err = loadConfig(); err != nil {
		fmt.Printf("loadConfig fail err is %v. use DefaultConf\n", err)
		conf = getDefaultConf()
	}
	logger = getLogger(conf)
}

func loadConfig() (*logConfig, error) {
	viper.AddConfigPath(defaultConfigPath)
	viper.SetConfigName(defaultConfigName)
	viper.SetConfigType(defaultConfigType)

	// 添加当前根目录，仅用于debug，打包构建时请勿开启
	config, err := parseConfig()
	if err != nil {
		return nil, err
	}
	watchConfig()
	return config, nil
}

func getDefaultConf() *logConfig {
	var defaultConf = &logConfig{
		Level:       "info",
		EncoderType: "console",
		Path:        defaultLogPath,
		FileName:    "root.log",
		MaxSize:     20,
		MaxBackups:  5,
		MaxAge:      30,
		LocalTime:   false,
		Compress:    true,
		OutMod:      "both",
	}
	exePath, err := executable()
	if err != nil {
		return defaultConf
	}
	// 获取运行文件名称，作为/var/log目录下的子目录
	serviceName := strings.TrimSuffix(filepath.Base(exePath), filepath.Ext(filepath.Base(exePath)))
	defaultConf.Path = filepath.Join(defaultLogPath, serviceName)
	return defaultConf
}

func getLogger(conf *logConfig) *zap.SugaredLogger {
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
			logger.Warn("ConfigClient file changed")
			// 重新加载配置
			conf, err := parseConfig()
			if err != nil {
				logger.Warnf("Error reloading config file: %v\n", err)
			} else {
				logger = getLogger(conf)
			}
		})
	})
}

func parseConfig() (*logConfig, error) {
	err := viper.ReadInConfig()
	if err != nil {
		return nil, err
	}
	var config logConfig
	err = viper.Unmarshal(&config)
	if err != nil {
		return nil, err
	}
	return &config, nil
}

// //获取编码器,NewJSONEncoder()输出json格式，NewConsoleEncoder()输出普通文本格式
func getEncoder(conf *logConfig) zapcore.Encoder {
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

func getLogWriter(conf *logConfig) zapcore.WriteSyncer {
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

// FilteredCore wrapper struct for zapcore.Core
type FilteredCore struct {
	zapcore.Core
}

// With zapcore.Core original with function
func (f *FilteredCore) With(fields []zapcore.Field) zapcore.Core {
	return &FilteredCore{
		Core: f.Core.With(fields),
	}
}

// Check zapcore.Core original check function
func (f *FilteredCore) Check(entry zapcore.Entry, ce *zapcore.CheckedEntry) *zapcore.CheckedEntry {
	entry.Message = filterLogMessage(entry.Message)

	return f.Core.Check(entry, ce)
}

// Write add custom filter before zapcore.Core original write function
func (f *FilteredCore) Write(entry zapcore.Entry, fields []zapcore.Field) error {
	return f.Core.Write(entry, fields)
}

// filterLogMessage custom sanitize function
func filterLogMessage(message string) string {
	var patterns = []string{
		`\x00`,    // 空字节
		`\x0f`,    // EBCDIC 0x0f
		`\x1b`,    // ESC
		`\x7f`,    // DEL
		`\xff`,    // 0xff
		`\"`,      // 引号
		`\t`,      // 制表符
		`<[^>]*>`, // HTML/XML 标签
		`[;&@]`,   // 重定向字符和特殊符号
	}
	re := regexp.MustCompile(strings.Join(patterns, "|"))
	message = re.ReplaceAllString(message, "")
	return message
}

// With adds a variadic number of fields to the logging context. It accepts a
// mix of strongly-typed Field objects and loosely-typed key-value pairs. When
// processing pairs, the first element of the pair is used as the field key
// and the second as the field value.
func With(args ...interface{}) *zap.SugaredLogger {
	return logger.With(args...)
}

// Error logs the provided arguments at the ErrorLevel.
// If the arguments are not strings, spaces are added between them.
func Error(args ...interface{}) {
	logger.Error(args...)
}

// Warn logs the provided arguments at the WarnLevel.
// If the arguments are not strings, spaces are added between them.
func Warn(args ...interface{}) {
	logger.Warn(args...)
}

// Info logs the provided arguments at [].
// If the arguments are not strings, spaces are added between them.
func Info(args ...interface{}) {
	logger.Info(args...)
}

// Debug logs the provided arguments at [DebugLevel].
// If the arguments are not strings, spaces are added between them.
func Debug(args ...interface{}) {
	logger.Debug(args...)
}

// Fatal constructs a message with the provided arguments and calls os.Exit.
// If the arguments are not strings, spaces are added between them.
func Fatal(args ...interface{}) {
	logger.Fatal(args...)
}

// Panic constructs a message with the provided arguments and panics.
// If the arguments are not strings, spaces are added between them.
func Panic(args ...interface{}) {
	logger.Panic(args...)
}

// DPanic logs the provided arguments at [DPanicLevel].
// In development, the logger then panics. (See [DPanicLevel] for details.)
// If the arguments are not strings, spaces are added between them.
func DPanic(args ...interface{}) {
	logger.DPanic(args...)
}

// Errorf formats the message according to the specified format string
// and logs it at the ErrorLevel.
func Errorf(template string, args ...interface{}) {
	logger.Errorf(template, args...)
}

// Warnf formats the message according to the specified format string
// and logs it at WarnLevel.
func Warnf(template string, args ...interface{}) {
	logger.Warnf(template, args...)
}

// Infof formats the message according to the specified format string
// and logs it at [].
func Infof(template string, args ...interface{}) {
	logger.Infof(template, args...)
}

// Debugf formats the message according to the specified format string
// and logs it at DebugLevel.
func Debugf(template string, args ...interface{}) {
	logger.Debugf(template, args...)
}

// Fatalf formats the message according to the specified format string
// and calls os.Exit.
func Fatalf(template string, args ...interface{}) {
	logger.Fatalf(template, args...)
}

// Sync flushes any buffered log entries.
func Sync() error {
	return logger.Sync()
}
