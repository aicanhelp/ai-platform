package log

import (
	"testing"

	"go.uber.org/zap"
)

func TestSetLoggerWhenLoggerIsNil(t *testing.T) {
	originLogger := logger
	logger = nil
	defer func() {
		logger = originLogger
	}()

	SetLogger(zap.NewNop())

	if logger == nil {
		t.Errorf("expected logger exists, but logger not found")
	}
}

func TestGetLoggerWhenLoggerIsNil(t *testing.T) {
	originLogger := logger
	logger = nil
	defer func() {
		logger = originLogger
	}()

	log := GetLogger()
	if log.Core().Enabled(zap.InfoLevel) {
		t.Errorf("expected no logger, but got logger")
	}
}

func TestSetLoggerWhenLoggerNotNil(t *testing.T) {
	originLogger := logger
	newLogger := zap.NewNop()
	logger = newLogger
	defer func() {
		logger = originLogger
	}()

	SetLogger(zap.NewNop())

	if logger != newLogger {
		t.Errorf("expected logger not change, but it changed")
	}
}

func TestGetLoggerWhenLoggerNotNil(t *testing.T) {
	originLogger := logger
	logger = zap.NewNop()
	defer func() {
		logger = originLogger
	}()

	log := GetLogger()

	if log != logger {
		t.Errorf("expected get the same logger, but got other logger")
	}
}
