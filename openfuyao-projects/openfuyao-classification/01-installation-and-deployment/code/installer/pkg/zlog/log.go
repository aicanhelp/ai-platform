/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
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

	"installer/pkg/constants"
)

const (
	defaultConfigPath = "/etc/fuyao-installer"
	defaultConfigName = "fuyao-installer"
	defaultConfigType = "yaml"
	// DefaultLogPath is lod save path
	DefaultLogPath = "/var/log/fuyao-installer"
)

// Logger is zap log
var Logger *zap.SugaredLogger

var logLevel = map[string]zapcore.Level{
	constants.LogLevelDebug: zapcore.DebugLevel,
	constants.LogLevelInfo:  zapcore.InfoLevel,
	constants.LogLevelWarn:  zapcore.WarnLevel,
	constants.LogLevelError: zapcore.ErrorLevel,
}

var watchOnce = sync.Once{}

// LogConfig is log config
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

// 初始化函数
func init() {
	var conf *LogConfig
	var err error
	if conf, err = loadConfig(); err != nil {
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
		Level:       constants.LogLevelInfo,
		EncoderType: "console",
		Path:        DefaultLogPath,
		FileName:    "installer.log",
		MaxSize:     20,
		MaxBackups:  0,
		MaxAge:      7,
		LocalTime:   false,
		Compress:    true,
		OutMod:      constants.LogOutModTwo,
	}
	exePath, err := os.Executable()
	if err != nil {
		return defaultConf
	}
	// 获取运行文件名称，作为/var/log目录下的子目录
	serviceName := strings.TrimSuffix(filepath.Base(exePath), filepath.Ext(filepath.Base(exePath)))
	defaultConf.Path = filepath.Join(DefaultLogPath, serviceName)
	return defaultConf
}

// 获取日志记录器
func GetLogger(conf *LogConfig) *zap.SugaredLogger {
	writeSyncer := getLogWriter(conf)
	encoder := getEncoder(conf)
	level, ok := logLevel[strings.ToLower(conf.Level)]
	if !ok {
		level = logLevel[constants.LogLevelInfo]
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
	//指定时间格式 for example: 2021-09-11t20:05:54.852+0800
	encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	//按级别显示不同颜色，不需要的话取值zapcore.CapitalLevelEncoder就可以了
	encoderConfig.EncodeLevel = zapcore.CapitalLevelEncoder
	//NewJSONEncoder()输出json格式，NewConsoleEncoder()输出普通文本格式
	if strings.ToLower(conf.EncoderType) == "json" {
		return zapcore.NewJSONEncoder(encoderConfig)
	}
	return zapcore.NewConsoleEncoder(encoderConfig)
}

func getLogWriter(conf *LogConfig) zapcore.WriteSyncer {
	// 只输出到控制台
	if conf.OutMod == constants.LogOutModOne {
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
	if conf.OutMod == constants.LogOutModThree {
		// 控制台和文件都输出
		return zapcore.NewMultiWriteSyncer(zapcore.AddSync(lumberJackLogger), zapcore.AddSync(os.Stdout))
	}
	if conf.OutMod == constants.LogOutModTwo {
		// 只输出到文件
		return zapcore.AddSync(lumberJackLogger)
	}
	return zapcore.AddSync(os.Stdout)
}

//以下封装 SugaredLogger 常用接口

// With函数用于添加一些字段到Logger中，返回一个新的Logger实例
func With(args ...interface{}) *zap.SugaredLogger {
	return Logger.With(args...)
}

// Debug函数用于打印调试信息
func Debug(args ...interface{}) {
	// 调用Logger的Debug方法，并传入可变参数args
	Logger.Debug(args...)
}

// Info函数用于打印日志信息，参数args为可变参数，类型为interface{}
func Info(args ...interface{}) {
	// 调用Logger的Info方法，传入可变参数args
	Logger.Info(args...)
}

// Warn函数用于打印警告日志，参数args为可变参数，类型为interface{}
func Warn(args ...interface{}) {
	// 调用Logger的Warn方法，传入可变参数args
	Logger.Warn(args...)
}

// Error函数用于记录错误日志，参数args为可变参数，类型为interface{}
func Error(args ...interface{}) {
	// 调用Logger的Error方法，传入可变参数args
	Logger.Error(args...)
}

// DPanic函数，用于记录错误日志
func DPanic(args ...interface{}) {
	// 调用Logger的DPanic方法，记录错误日志
	Logger.DPanic(args...)
}

// Panic函数用于记录错误日志并终止程序执行
func Panic(args ...interface{}) {
	// 调用Logger的Panic方法，传入参数args
	Logger.Panic(args...)
}

// Fatal函数用于输出参数并记录日志，然后退出程序
func Fatal(args ...interface{}) {
	fmt.Println(args...)
	Logger.Fatal(args...)
}

// Debugf函数用于打印调试信息，参数template为字符串模板，args为可变参数
func Debugf(template string, args ...interface{}) {
	// 调用Logger的Debugf方法，传入字符串模板和可变参数
	Logger.Debugf(template, args...)
}

// Infof函数用于打印日志信息，参数template为日志信息的格式，args为日志信息的参数
func Infof(template string, args ...interface{}) {
	// 调用Logger的Infof函数，传入template和args参数
	Logger.Infof(template, args...)
}

// Warnf函数用于打印警告日志，参数template为日志模板，args为可变参数
func Warnf(template string, args ...interface{}) {
	// 调用Logger的Warnf函数，传入日志模板和可变参数
	Logger.Warnf(template, args...)
}

// Errorf函数用于格式化输出错误信息
func Errorf(template string, args ...interface{}) {
	// 调用Logger的Errorf函数，传入格式化字符串和参数列表
	Logger.Errorf(template, args...)
}

// DPanicf函数用于记录一个DPanic级别的日志，并格式化输出日志信息
func DPanicf(template string, args ...interface{}) {
	// 调用Logger的DPanicf方法，传入模板和参数，输出日志信息
	Logger.DPanicf(template, args...)
}

// Panicf函数用于输出错误信息并终止程序执行
func Panicf(template string, args ...interface{}) {
	// 调用Logger的Panicf方法，输出错误信息并终止程序执行
	Logger.Panicf(template, args...)
}

// Fatalf函数用于打印错误信息并退出程序
func Fatalf(template string, args ...interface{}) {
	fmt.Println(getMessage(template, args))
	Logger.Fatalf(template, args...)
}

func getMessage(template string, fmtArgs []interface{}) string {
	if len(fmtArgs) == 0 {
		return template
	}

	if template != "" {
		fmt.Println(template, fmtArgs)
		return ""
	}

	if len(fmtArgs) == 1 {
		if str, ok := fmtArgs[0].(string); ok {
			return str
		}
	}
	return fmt.Sprint(fmtArgs...)
}

// Debugw函数用于记录调试信息，msg参数为要记录的信息，keysAndValues参数为键值对形式的参数
func Debugw(msg string, keysAndValues ...interface{}) {
	// 调用Logger的Debugw方法，记录调试信息
	Logger.Debugw(msg, keysAndValues...)
}

// Infow函数用于记录带有键值对的信息日志
func Infow(msg string, keysAndValues ...interface{}) {
	// 调用Logger的Infow函数，传入消息和键值对参数
	Logger.Infow(msg, keysAndValues...)
}

// Warnw函数用于记录警告级别的日志，msg参数为日志信息，keysAndValues参数为键值对，用于记录日志的额外信息
func Warnw(msg string, keysAndValues ...interface{}) {
	// 调用Logger的Warnw函数，记录警告级别的日志
	Logger.Warnw(msg, keysAndValues...)
}

// Errorw函数用于记录错误日志，msg为错误信息，keysAndValues为键值对参数
func Errorw(msg string, keysAndValues ...interface{}) {
	// 调用Logger的Errorw方法，记录错误日志
	Logger.Errorw(msg, keysAndValues...)
}

// DPanicw函数用于记录一个警告级别的日志，并输出msg和keysAndValues参数
func DPanicw(msg string, keysAndValues ...interface{}) {
	// 调用Logger的DPanicw方法，传入msg和keysAndValues参数
	Logger.DPanicw(msg, keysAndValues...)
}

// Panicw函数用于记录一条警告级别的日志，并在日志中输出指定的键值对
func Panicw(msg string, keysAndValues ...interface{}) {
	// 调用Logger的Panicw方法，传入日志消息和键值对
	Logger.Panicw(msg, keysAndValues...)
}

// Fatalw函数用于记录一条致命级别的日志，并退出程序
func Fatalw(msg string, keysAndValues ...interface{}) {
	// 调用Logger的Fatalw方法，传入日志消息和键值对参数
	Logger.Fatalw(msg, keysAndValues...)
}

// Debugln函数用于打印调试信息，参数为可变参数类型
func Debugln(args ...interface{}) {
	// 调用Logger的Debugln方法，传入可变参数
	Logger.Debugln(args...)
}

// Infoln函数用于打印日志信息，参数为可变参数类型，可以传入任意数量的参数
func Infoln(args ...interface{}) {
	// 调用Logger的Infoln方法，传入可变参数args
	Logger.Infoln(args...)
}

// Warnln函数用于输出警告信息，参数为可变参数类型
func Warnln(args ...interface{}) {
	// 调用Logger的Warnln函数，输出警告信息
	Logger.Warnln(args...)
}

// Errorln函数用于打印错误日志
func Errorln(args ...interface{}) {
	// 调用Logger的Errorln方法，传入可变参数args
	Logger.Errorln(args...)
}

// DPanicln函数用于在日志中记录一个DPanic级别的日志，并打印参数args
func DPanicln(args ...interface{}) {
	// 调用Logger的DPanicln方法，传入参数args
	Logger.DPanicln(args...)
}

// Panicln函数用于打印日志并触发panic
func Panicln(args ...interface{}) {
	// 调用Logger的Panicln方法，传入可变参数args
	Logger.Panicln(args...)
}

// Fatalln函数用于输出日志并退出程序
func Fatalln(args ...interface{}) {
	// 调用Logger的Fatalln函数，输出日志并退出程序
	Logger.Fatalln(args...)
}

// Sync函数用于同步日志，返回一个error类型的值
func Sync() error {
	// 调用Logger的Sync方法，返回一个error类型的值
	return Logger.Sync()
}
