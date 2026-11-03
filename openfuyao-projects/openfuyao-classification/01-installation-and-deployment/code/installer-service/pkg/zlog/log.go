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
	defaultConfigPath = "/etc/installer-service/log-config"
	defaultConfigName = "installer-service"
	defaultConfigType = "yaml"
	defaultLogPath    = "/var/log"
)

var logger *zap.SugaredLogger

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
	exePath, err := os.Executable()
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
			logger.Warn("Config file changed")
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

// With 创建带固定字段的日志记录器
func With(args ...interface{}) *zap.SugaredLogger {
	return logger.With(args...)
}

// Error 等级为 ErrorLevel.
func Error(args ...interface{}) {
	logger.Error(args...)
}

// Warn 等级为 WarnLevel.
func Warn(args ...interface{}) {
	logger.Warn(args...)
}

// Info 等级为 InfoLevel
func Info(args ...interface{}) {
	logger.Info(args...)
}

// Debug 等级为 DebugLevel
func Debug(args ...interface{}) {
	logger.Debug(args...)
}

// Fatal 等级为 FatalLevel
func Fatal(args ...interface{}) {
	logger.Fatal(args...)
}

// Panic 输出 Panic 级日志
func Panic(args ...interface{}) {
	logger.Panic(args...)
}

// DPanic 输出 DPanic 级日志
func DPanic(args ...interface{}) {
	logger.DPanic(args...)
}

// Errorf 输出 Errorf 级日志
func Errorf(template string, args ...interface{}) {
	logger.Errorf(template, args...)
}

// Warnf 等级为 WarnLevel.
func Warnf(template string, args ...interface{}) {
	logger.Warnf(template, args...)
}

// Infof 等级为 InfofLevel
func Infof(template string, args ...interface{}) {
	logger.Infof(template, args...)
}

// Debugf 等级为 DebugfLevel
func Debugf(template string, args ...interface{}) {
	logger.Debugf(template, args...)
}

// Fatalf 等级为 FatalfLevel
func Fatalf(template string, args ...interface{}) {
	logger.Fatalf(template, args...)
}

// Panicf 输出 Panicf 级日志
func Panicf(template string, args ...interface{}) {
	logger.Panicf(template, args...)
}

// DPanicf 输出 DPanicf 级日志
func DPanicf(template string, args ...interface{}) {
	logger.DPanicf(template, args...)
}

// Errorw 输出 Errorw 级日志
func Errorw(msg string, keysAndValues ...interface{}) {
	logger.Errorw(msg, keysAndValues...)
}

// Warnw 输出 Warnw 级日志
func Warnw(msg string, keysAndValues ...interface{}) {
	logger.Warnw(msg, keysAndValues...)
}

// Infow 输出 Infow 级日志
func Infow(msg string, keysAndValues ...interface{}) {
	logger.Infow(msg, keysAndValues...)
}

// Debugw 输出 Debugw 级日志
func Debugw(msg string, keysAndValues ...interface{}) {
	logger.Debugw(msg, keysAndValues...)
}

// Fatalw 输出 Fatalw 级日志
func Fatalw(msg string, keysAndValues ...interface{}) {
	logger.Fatalw(msg, keysAndValues...)
}

// Panicw 输出 Panicw 级日志
func Panicw(msg string, keysAndValues ...interface{}) {
	logger.Panicw(msg, keysAndValues...)
}

// DPanicw 输出 DPanicw 级日志
func DPanicw(msg string, keysAndValues ...interface{}) {
	logger.DPanicw(msg, keysAndValues...)
}

// Errorln 输出 Errorln 级日志
func Errorln(args ...interface{}) {
	logger.Errorln(args...)
}

// Warnln 输出 Warnln 级日志
func Warnln(args ...interface{}) {
	logger.Warnln(args...)
}

// Infoln 输出 Infoln  级日志
func Infoln(args ...interface{}) {
	logger.Infoln(args...)
}

// Debugln 输出 Debugln  级日志
func Debugln(args ...interface{}) {
	logger.Debugln(args...)
}

// Fatalln logs a message at [FatalLevel] and calls os.Exit.
func Fatalln(args ...interface{}) {
	logger.Fatalln(args...)
}

// Panicln logs a message at [PanicLevel] and panics.
func Panicln(args ...interface{}) {
	logger.Panicln(args...)
}

// DPanicln logs a message at [DPanicLevel].
func DPanicln(args ...interface{}) {
	logger.DPanicln(args...)
}

// Sync flushes any buffered log entries.
func Sync() error {
	return logger.Sync()
}
