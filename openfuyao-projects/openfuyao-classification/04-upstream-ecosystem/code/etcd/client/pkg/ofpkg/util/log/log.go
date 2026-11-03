// Package log provider logger container for OF pkg
package log

import (
	"sync"

	"go.uber.org/zap"
)

var (
	logger *zap.Logger
	mutex  sync.Mutex
)

// SetLogger 设置全局日志记录器（线程安全，仅首次设置有效）
func SetLogger(log *zap.Logger) {
	mutex.Lock()
	defer mutex.Unlock()
	if logger == nil {
		logger = log
	}
}

// GetLogger 获取全局日志记录器，若未设置则返回无操作记录器
func GetLogger() *zap.Logger {
	if logger == nil {
		return zap.NewNop()
	}
	return logger
}
