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
	defaultConfigPath = "/etc/marketplace-service/log-config"
	defaultConfigName = "marketplace-service"
	defaultConfigType = "yaml"
	defaultLogPath    = "/var/log"
)

// 特殊字符替换映射（转义换行/回车为可视字符串，避免日志换行）
var specialCharMap = map[string]string{
	"\n":   "\\n", // 换行符转义为 "\\n"
	"\r":   "\\r", // 回车符转义为 "\\r"
	"\t":   "\\t", // 制表符转义为 "\\t"
	"\000": "\\0", // 空字符转义为 "\\0"
}

// CleanSpecialChar 清洗字符串中的特殊字符，防止日志注入/格式破坏
func CleanSpecialChar(s string) string {
	if s == "" {
		return ""
	}
	// 替换所有特殊字符
	for oldChar, newChar := range specialCharMap {
		s = strings.ReplaceAll(s, oldChar, newChar)
	}
	// 可选：去除首尾空白字符（避免多余空格）
	return strings.TrimSpace(s)
}

// 清洗日志字段中的特殊字符
func cleanLogFields(args []interface{}) []interface{} {
	cleaned := make([]interface{}, 0, len(args))
	for _, arg := range args {
		if s, ok := arg.(string); ok {
			cleaned = append(cleaned, CleanSpecialChar(s))
		} else {
			cleaned = append(cleaned, arg)
		}
	}
	return cleaned
}

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

// With 提供 with级日志
func With(args ...interface{}) *zap.SugaredLogger {
	return logger.With(args...)
}

// Error 提供 Error级日志
func Error(args ...interface{}) {
	logger.Error(args...)
}

// Warn 提供 Warn级日志
func Warn(args ...interface{}) {
	logger.Warn(args...)
}

// Info 提供 Info级日志
func Info(args ...interface{}) {
	logger.Info(args...)
}

// Debug 提供 Debug级日志
func Debug(args ...interface{}) {
	logger.Debug(args...)
}

// Fatal 提供 Fatal级日志
func Fatal(args ...interface{}) {
	logger.Fatal(args...)
}

// Panic 提供 Panic级日志
func Panic(args ...interface{}) {
	logger.Panic(args...)
}

// DPanic 提供 DPanic级日志
func DPanic(args ...interface{}) {
	logger.DPanic(args...)
}

// Errorf 提供 Errorf级日志
func Errorf(template string, args ...interface{}) {
	logger.Errorf(template, args...)
}

// Warnf 提供Warnf级日志
func Warnf(template string, args ...interface{}) {
	cleanedArgs := cleanLogFields(args)
	logger.Warnf(template, cleanedArgs...)
}

// Infof 提供Infof级日志
func Infof(template string, args ...interface{}) {
	cleanedArgs := cleanLogFields(args)
	logger.Infof(template, cleanedArgs...)
}

// Debugf 提供Debugf级日志
func Debugf(template string, args ...interface{}) {
	cleanedArgs := cleanLogFields(args)
	logger.Debugf(template, cleanedArgs...)
}

// Fatalf 提供Fatalf级日志
func Fatalf(template string, args ...interface{}) {
	cleanedArgs := cleanLogFields(args)
	logger.Fatalf(template, cleanedArgs...)
}

// Panicf 提供Panicf级日志
func Panicf(template string, args ...interface{}) {
	logger.Panicf(template, args...)
}

// DPanicf 提供DPanicf级日志
func DPanicf(template string, args ...interface{}) {
	logger.DPanicf(template, args...)
}

// Errorw 提供Errorw级日志
func Errorw(msg string, keysAndValues ...interface{}) {
	logger.Errorw(msg, keysAndValues...)
}

// Warnw 提供Warnw级日志
func Warnw(msg string, keysAndValues ...interface{}) {
	logger.Warnw(msg, keysAndValues...)
}

// Infow 提供Infow级日志
func Infow(msg string, keysAndValues ...interface{}) {
	logger.Infow(msg, keysAndValues...)
}

// Debugw 提供Debugw级日志
func Debugw(msg string, keysAndValues ...interface{}) {
	logger.Debugw(msg, keysAndValues...)
}

// Fatalw 提供Fatalw级日志
func Fatalw(msg string, keysAndValues ...interface{}) {
	logger.Fatalw(msg, keysAndValues...)
}

// Panicw 提供Panicw级日志
func Panicw(msg string, keysAndValues ...interface{}) {
	logger.Panicw(msg, keysAndValues...)
}

// DPanicw 提供DPanicw级日志
func DPanicw(msg string, keysAndValues ...interface{}) {
	logger.DPanicw(msg, keysAndValues...)
}

// Errorln 提供Errorln级日志
func Errorln(args ...interface{}) {
	logger.Errorln(args...)
}

// Warnln 提供Warnln级日志
func Warnln(args ...interface{}) {
	logger.Warnln(args...)
}

// Infoln 提供Infoln级日志
func Infoln(args ...interface{}) {
	logger.Infoln(args...)
}

// Debugln 提供Debugln级日志
func Debugln(args ...interface{}) {
	logger.Debugln(args...)
}

// Fatalln 提供Fatalln级日志
func Fatalln(args ...interface{}) {
	logger.Fatalln(args...)
}

// Panicln 提供Panicln级日志
func Panicln(args ...interface{}) {
	logger.Panicln(args...)
}

// DPanicln 提供DPanicln级日志
func DPanicln(args ...interface{}) {
	logger.DPanicln(args...)
}

// Sync flushes any buffered log entries.
func Sync() error {
	return logger.Sync()
}
