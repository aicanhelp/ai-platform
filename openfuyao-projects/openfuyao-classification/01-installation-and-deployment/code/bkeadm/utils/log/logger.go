/*
   Copyright @ 2021 bocloud <fushaosong@beyondcent.com>.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   Original file: https://gitee.com/bocloud-open-source/carina/blob/v0.9.1/utils/log/logger.go
*/

// Package log provides a logging wrapper around the ologger framework.
// It exposes the same public API as the previous Zap-based implementation
// so that callers require no changes. A dedicated Logger instance is
// created via ologger.NewLogger and writes to /var/log/openFuyao/bke.log.
package log

import (
	"fmt"
	"os"
	"strings"

	"gopkg.openfuyao.cn/common-modules/ologger/v26/log"
)

const (
	defaultLogPath = "/var/log/openFuyao/bke.log"
)

var logger *log.Logger

func init() {
	logger = log.NewLogger(bkeLoggerConfig()).WithCallerSkip(1)
}

func bkeLoggerConfig() log.Config {
	preflight := isPreflightCommand(os.Args)
	enableFile := !preflight
	// Preflight stdout is a machine/user report stream. Keep operational logs
	// in the normal file sink so JSON/text output is not polluted by logger
	// records. Existing bke commands retain their historical console logging.
	enableConsole := !preflight
	enableSanitize := true
	compress := true
	watchLevel := false

	return log.Config{
		Path:           defaultLogPath,
		Level:          log.INFO,
		Format:         "json",
		EnableFile:     &enableFile,
		EnableConsole:  &enableConsole,
		EnableSanitize: &enableSanitize,
		Compress:       &compress,
		WatchLevel:     &watchLevel,
	}
}

func isPreflightCommand(args []string) bool {
	expectFlagValue := false
	for _, arg := range args[1:] {
		if expectFlagValue {
			expectFlagValue = false
			continue
		}
		if strings.HasPrefix(arg, "--kubeconfig=") || strings.HasPrefix(arg, "--log-level=") {
			continue
		}
		switch arg {
		case "--kubeconfig", "--log-level":
			expectFlagValue = true
			continue
		case "--doc", "--help", "-h":
			continue
		default:
			return arg == "preflight"
		}
	}
	return false
}

// Debug logs a message at DEBUG level.
func Debug(args ...any) {
	logger.Debug(fmt.Sprint(args...))
}

// Debugf logs a formatted message at DEBUG level.
func Debugf(template string, args ...any) {
	logger.Debugf(template, args...)
}

// Info logs a message at INFO level.
func Info(args ...any) {
	logger.Info(fmt.Sprint(args...))
}

// Infof logs a formatted message at INFO level.
func Infof(template string, args ...any) {
	logger.Infof(template, args...)
}

// Warn logs a message at WARNING level.
func Warn(args ...any) {
	logger.Warn(fmt.Sprint(args...))
}

// Warnf logs a formatted message at WARNING level.
func Warnf(template string, args ...any) {
	logger.Warnf(template, args...)
}

// Error logs a message at ERROR level.
func Error(args ...any) {
	logger.Error(fmt.Sprint(args...))
}

// Errorf logs a formatted message at ERROR level.
func Errorf(template string, args ...any) {
	logger.Errorf(template, args...)
}

// Trace logs a message at TRACE level.
func Trace(args ...any) {
	logger.Trace(fmt.Sprint(args...))
}

// Tracef logs a formatted message at TRACE level.
func Tracef(template string, args ...any) {
	logger.Tracef(template, args...)
}

// SetLevel updates the global logger level at runtime.
func SetLevel(level string) error {
	return logger.SetLevel(level)
}

// Fatal logs a message at CRITICAL level and then calls os.Exit(1).
func Fatal(args ...any) {
	msg := fmt.Sprint(args...)
	logger.Critical(msg)
	os.Exit(1) //nolint:forbidigo // intentional: Fatal is designed to log and terminate
}

// Fatalf logs a formatted message at CRITICAL level and then calls os.Exit(1).
func Fatalf(template string, args ...any) {
	logger.Criticalf(template, args...)
	os.Exit(1) //nolint:forbidigo // intentional: Fatalf is designed to log and terminate
}

// SteppedInfo logs a message at INFO level with a step/phase identifier.
// The step name is prepended as a bracketed prefix in the log message (e.g. "[step.1] message").
func SteppedInfo(stepName string, args ...any) {
	logger.Info(fmt.Sprintf("[%s] %s", stepName, fmt.Sprint(args...)))
}

// SteppedInfof logs a formatted message at INFO level with a step/phase identifier.
func SteppedInfof(stepName, template string, args ...any) {
	logger.Infof("[%s] "+template, append([]any{stepName}, args...)...)
}
