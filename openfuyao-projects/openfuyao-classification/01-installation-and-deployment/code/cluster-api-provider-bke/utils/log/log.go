/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package log provides a process-wide logging facade that routes business
// logs to either the ologger global singleton (capbke / launcher) or an
// independent instance configured in code (bkeagent).
package log

import (
	"sync"

	olog "gopkg.openfuyao.cn/common-modules/ologger/v26/log"
)

// Logger is the ologger logger type used across the repository.
type Logger = olog.Logger

// Config is the ologger configuration type.
type Config = olog.Config

var (
	mu      sync.RWMutex
	current *olog.Logger
)

func init() {
	olog.SetCallerSkip(1)
}

func getLogger() *olog.Logger {
	mu.RLock()
	defer mu.RUnlock()
	return current
}

// InitForAgent registers an independent ologger instance for bkeagent.
// Call as the first statement in cmd/bkeagent/main().
func InitForAgent(cfg olog.Config) {
	l := olog.NewLogger(cfg).WithCallerSkip(1)
	mu.Lock()
	current = l
	mu.Unlock()
}

// SetLogger injects a logger for tests or advanced scenarios.
// Passing nil restores delegation to the ologger global singleton.
func SetLogger(l *olog.Logger) {
	mu.Lock()
	defer mu.Unlock()
	if l == nil {
		current = nil
		return
	}
	current = l.WithCallerSkip(1)
}

func With(args ...any) *olog.Logger {
	if l := getLogger(); l != nil {
		return l.With(args...)
	}
	return olog.With(args...)
}

// ControllerLogger returns a logger with the controller name attached as a structured field.
func ControllerLogger(controller string) *olog.Logger {
	return With("controller", controller)
}

func Trace(msg string, args ...any) {
	if l := getLogger(); l != nil {
		l.Trace(msg, args...)
		return
	}
	olog.Trace(msg, args...)
}

func Tracef(template string, args ...any) {
	if l := getLogger(); l != nil {
		l.Tracef(template, args...)
		return
	}
	olog.Tracef(template, args...)
}

func Log(levelName string, msg string, args ...any) {
	if l := getLogger(); l != nil {
		l.Log(levelName, msg, args...)
		return
	}
	olog.Log(levelName, msg, args...)
}

func Logf(levelName string, template string, args ...any) {
	if l := getLogger(); l != nil {
		l.Logf(levelName, template, args...)
		return
	}
	olog.Logf(levelName, template, args...)
}

func Debug(msg string, args ...any) {
	if l := getLogger(); l != nil {
		l.Debug(msg, args...)
		return
	}
	olog.Debug(msg, args...)
}

func Debugf(template string, args ...any) {
	if l := getLogger(); l != nil {
		l.Debugf(template, args...)
		return
	}
	olog.Debugf(template, args...)
}

func Info(msg string, args ...any) {
	if l := getLogger(); l != nil {
		l.Info(msg, args...)
		return
	}
	olog.Info(msg, args...)
}

func Infof(template string, args ...any) {
	if l := getLogger(); l != nil {
		l.Infof(template, args...)
		return
	}
	olog.Infof(template, args...)
}

func Warn(msg string, args ...any) {
	if l := getLogger(); l != nil {
		l.Warn(msg, args...)
		return
	}
	olog.Warn(msg, args...)
}

func Warnf(template string, args ...any) {
	if l := getLogger(); l != nil {
		l.Warnf(template, args...)
		return
	}
	olog.Warnf(template, args...)
}

func Error(msg any, args ...any) {
	if l := getLogger(); l != nil {
		l.Error(msg, args...)
		return
	}
	olog.Error(msg, args...)
}

func Errorf(template any, args ...any) {
	if l := getLogger(); l != nil {
		l.Errorf(template, args...)
		return
	}
	olog.Errorf(template, args...)
}

func Critical(msg string, args ...any) {
	if l := getLogger(); l != nil {
		l.Critical(msg, args...)
		return
	}
	olog.Critical(msg, args...)
}

func Criticalf(template string, args ...any) {
	if l := getLogger(); l != nil {
		l.Criticalf(template, args...)
		return
	}
	olog.Criticalf(template, args...)
}

func SetLevel(level string) error {
	if l := getLogger(); l != nil {
		return l.SetLevel(level)
	}
	return olog.SetLevel(level)
}
