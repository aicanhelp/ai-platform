/*
Copyright (c) 2024 Huawei Technologies Co., Ltd.
openFuyao is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/

// utils package provides utility functions for oschecktool
package utils

import (
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
)

var (
	globalLogger *log.Logger
	logWriter    io.Writer
	promptLogger *log.Logger
)

const (
	// DefaultCmdTimeoutSeconds 默认命令执行超时时间，单位：秒
	DefaultCmdTimeoutSeconds = 10
	// FileModeOnlyOwnerReadWrite 文件权限：仅文件所有者可读写
	FileModeOnlyOwnerReadWrite = 0600
	// FileModeOnlyOwnerReadExec 文件权限：仅文件所有者可执行
	FileModeOnlyOwnerReadExec = 0700
	// FileModeOwnerReadWriteExec 文件权限：文件所有者及组所有者可读写执行
	FileModeOwnerReadWriteExec = 0750
	// DirModeOwnerReadWriteExec 目录权限：文件所有者及组所有者可读写执行
	DirModeOwnerReadWriteExec = 0750
	// FileModeExecutable 文件权限：可执行
	FileModeExecutable = 0111
	// NumBase10 数字进制：10进制
	NumBase10 = 10
	// NumBites64 数字位宽：64位
	NumBites64 = 64

	logFlags = log.Ldate | log.Ltime | log.Lmicroseconds | log.Lmsgprefix
)

func init() {
	logWriter = &syncWriter{Writer: os.Stderr}
	globalLogger = GetLogger("")
	promptLogger = GetLogger("Prompt")
}

// InitLogFile 初始化日志文件，返回值为释放函数
// 如果logFilePath为空，则不进行日志文件初始化
// 释放函数用于关闭日志文件
func InitLogFile(logFilePath string) func() {
	if len(logFilePath) == 0 {
		return func() {}
	}
	baseDir := filepath.Dir(logFilePath)
	if err := os.MkdirAll(baseDir, DirModeOwnerReadWriteExec); err != nil {
		globalLogger.Printf("Failed to create log directory: %s, error: %s", baseDir, err.Error())
		return func() {}
	}
	fullLogPath, err := filepath.Abs(logFilePath)
	if err != nil {
		globalLogger.Printf("Failed to get absolute path for log file: %s, error: %s", logFilePath, err.Error())
		return func() {}
	}
	tmpWriter, err := os.OpenFile(fullLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, FileModeOnlyOwnerReadWrite)
	if err != nil {
		globalLogger.Printf("Failed to open log file: %s, error: %s", fullLogPath, err.Error())
		return func() {}
	}
	syncWriter := &syncWriter{Writer: tmpWriter}
	logWriter = syncWriter
	globalLogger.SetOutput(syncWriter)
	promptLogger = GetLogger("Prompt")
	return func() {
		CloseAll(globalLogger, tmpWriter)
	}
}

type syncWriter struct {
	logMutex sync.Mutex
	io.Writer
}

func (w *syncWriter) Write(p []byte) (int, error) {
	w.logMutex.Lock()
	defer w.logMutex.Unlock()
	return w.Writer.Write(p)
}

// ParseInt10Base 解析10进制整数
// value: 待解析的字符串
// 返回解析后的整数和可能的错误
// 错误包括：value为空、value为空格、解析失败
func ParseInt10Base(value string) (int64, error) {
	if len(value) == 0 {
		return 0, fmt.Errorf("value is empty")
	}
	value = strings.TrimSpace(value)
	if len(value) == 0 {
		return 0, fmt.Errorf("value is empty after trim")
	}
	intValue, err := strconv.ParseInt(value, NumBase10, NumBites64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse value: %s, error: %s", value, err.Error())
	}
	return intValue, nil
}

// GetDefaultLogger 获取默认的全局日志记录器
func GetDefaultLogger() *log.Logger {
	return GetLogger("")
}

// GetLogger 获取指定名称的日志记录器
// name: 日志记录器名称，如果为空，则返回默认的全局日志记录器
// 返回值为指定名称的日志记录器
// 如果name为空，则返回默认的全局日志记录器
// 如果name不为空，则返回指定名称的日志记录器
func GetLogger(name string) *log.Logger {
	if len(name) == 0 {
		if globalLogger == nil {
			globalLogger = log.New(logWriter, name, logFlags)
		}
		return globalLogger
	} else {
		return log.New(logWriter, fmt.Sprintf("[%s] ", name), logFlags)
	}
}

// PromptMsg 打印提示信息到标准输出和提示日志
// format: 格式化字符串
// args: 格式化参数
// 打印提示信息到标准输出和提示日志
// 如果format为空，则不打印任何内容
func PromptMsg(format string, args ...any) {
	if len(format) > 0 {
		promptLogger.Printf(format, args...)
		fmt.Printf(format+"\n", args...)

	}
}

// CloseAll 关闭多个io.Closer对象
// logger: 日志记录器
// closers: 多个io.Closer对象
// 如果logger不为空，则在关闭失败时记录日志
func CloseAll(logger *log.Logger, closers ...io.Closer) {
	for _, closer := range closers {
		if reflect.ValueOf(closer).IsNil() {
			continue
		}
		err := closer.Close()
		if err != nil && logger != nil {
			logger.Printf("failed to close closer: %v", err)
		}
	}
}

// CreateFile 创建文件
// path: 文件路径
// mode: 文件权限
// 返回可能的错误
// 如果文件已存在，则返回错误
// 如果文件不存在，则创建文件
func CreateFile(path string, mode os.FileMode, createParentDir bool) (*os.File, error) {
	if createParentDir {
		if err := os.MkdirAll(filepath.Dir(path), DirModeOwnerReadWriteExec); err != nil {
			return nil, fmt.Errorf("failed to create parent directory: %s, error: %s", filepath.Dir(path), err.Error())
		}
	}
	if _, err := os.Stat(path); err == nil {
		return nil, fmt.Errorf("file already exists: %s", path)
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	err = file.Chmod(mode)
	if err != nil {
		CloseAll(nil, file)
		return nil, err
	}
	return file, nil
}

// GetStartDir 获取当前执行文件的目录
func GetStartDir() string {
	var startDir = "./"
	execPath, err := os.Executable()
	if err == nil {
		startDir = filepath.Dir(execPath)
	}
	absStartDir, err := filepath.Abs(startDir)
	if err == nil {
		return absStartDir
	}
	return startDir
}
