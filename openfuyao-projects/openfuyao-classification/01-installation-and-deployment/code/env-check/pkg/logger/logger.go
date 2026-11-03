/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package logger provides env-check compatible APIs backed by ologger.
package logger

import (
	"log/slog"
	"strings"
	"sync"

	olog "gopkg.openfuyao.cn/common-modules/ologger/log"
)

const (
	successLevelName  = "SUCCESS"
	successLevelValue = 2 // slog level offset above INFO for the SUCCESS level
)

// successLevelOnce registers the custom SUCCESS level once. AddLevel mutates
// ologger's global level registry (shared across all Logger instances), so the
// registration is process-wide and safe to perform a single time.
var successLevelOnce sync.Once

// Logger defines the logging capabilities used by env-check modules.
type Logger interface {
	Info(message string)
	Success(message string)
	Warning(message string)
	Error(message string)
}

// ologgerAdapter wraps a dedicated, self-contained ologger instance. Unlike the
// previous implementation that delegated to ologger's package-level globals
// (which depend on /etc/openFuyao/ologger/ologger.yaml and emit "config not
// found" warnings), this adapter owns a *olog.Logger built via NewLogger(cfg),
// giving env-check full control over the log path, level, format and rotation.
type ologgerAdapter struct {
	logger *olog.Logger
}

// NewLogger creates an env-check compatible logger backed by a dedicated
// ologger instance. Configuration is fully self-contained: it does not read
// OLOGGER_* environment variables or the ologger.yaml config file, so behavior
// is deterministic across platforms.
//
// A non-empty logFile enables file output at that path (honoring the config
// field). An empty logFile keeps logging console-only — production always
// supplies a path via config (default ./envCheck.log), while tests pass "" to
// avoid opening file handles that cannot be reclaimed on Windows.
func NewLogger(logFile string) Logger {
	successLevelOnce.Do(func() {
		olog.AddLevel(successLevelName, slog.Level(successLevelValue))
	})

	cfg := buildLoggerConfig(strings.TrimSpace(logFile))
	instance := olog.NewLogger(cfg).WithCallerSkip(1)

	return &ologgerAdapter{logger: instance}
}

// buildLoggerConfig returns a self-contained ologger configuration. It follows
// the idiomatic pattern used by bkeadm, adapted for env-check's cross-platform
// CLI nature: when a log path is provided, file logging is enabled to honor the
// configured log_file; console output is kept for operator visibility;
// sensitive-value masking and rotation compression are enabled; and the
// config-file level watcher is disabled (a CLI tool does not need runtime level
// reloading). An empty path disables file output (console-only).
func buildLoggerConfig(path string) olog.Config {
	enableConsole := true
	enableFile := path != ""
	enableSanitize := true
	compress := true
	watchLevel := false

	cfg := olog.Config{
		Level:          olog.INFO,
		Format:         "json",
		EnableFile:     &enableFile,
		EnableConsole:  &enableConsole,
		EnableSanitize: &enableSanitize,
		Compress:       &compress,
		WatchLevel:     &watchLevel,
	}
	if enableFile {
		cfg.Path = path
	}
	return cfg
}

// Info logs information.
func (l *ologgerAdapter) Info(message string) {
	l.logger.Info(message)
}

// Success logs successful operation via a custom SUCCESS level.
func (l *ologgerAdapter) Success(message string) {
	l.logger.Log(successLevelName, message)
}

// Warning logs warning information.
func (l *ologgerAdapter) Warning(message string) {
	l.logger.Warn(message)
}

// Error logs errors.
func (l *ologgerAdapter) Error(message string) {
	l.logger.Error(message)
}
