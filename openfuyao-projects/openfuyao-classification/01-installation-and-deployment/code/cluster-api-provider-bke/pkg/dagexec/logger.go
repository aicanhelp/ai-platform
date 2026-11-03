/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dagexec

import (
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// Logger is the dagexec logging facade.
// Prefer BKELogger (Event + ologger) when present; otherwise fall back to utils/log (ologger).
type Logger struct {
	bke *bkev1beta1.BKELogger
}

// NewLogger wraps an optional BKELogger.
func NewLogger(bke *bkev1beta1.BKELogger) *Logger {
	return &Logger{bke: bke}
}

func loggerFrom(execCtx *ExecutionContext) *Logger {
	if execCtx == nil {
		return &Logger{}
	}
	return &Logger{bke: execCtx.Log}
}

// Info logs at info level.
func (l *Logger) Info(format string, args ...interface{}) {
	if l != nil && l.bke != nil {
		l.bke.Info(constant.ComponentUpgradingReason, format, args...)
		return
	}
	log.Infof(format, args...)
}

// Warn logs at warn level.
func (l *Logger) Warn(format string, args ...interface{}) {
	if l != nil && l.bke != nil {
		l.bke.Warn(constant.ComponentUpgradingReason, format, args...)
		return
	}
	log.Warnf(format, args...)
}

// Error logs at error level.
func (l *Logger) Error(format string, args ...interface{}) {
	if l != nil && l.bke != nil {
		l.bke.Error(constant.ComponentUpgradingReason, format, args...)
		return
	}
	log.Errorf(format, args...)
}
