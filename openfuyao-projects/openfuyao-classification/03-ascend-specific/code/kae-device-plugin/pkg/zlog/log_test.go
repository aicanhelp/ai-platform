/*
 * Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
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
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/viper"
	"go.uber.org/zap"
)

func TestGetLoggerSupportsConfiguredOutputs(t *testing.T) {
	tempDir := t.TempDir()
	testCases := []LogConfig{
		{Level: "debug", EncoderType: "console", OutMod: "console"},
		{Level: "info", EncoderType: "json", OutMod: "console"},
		{Level: "unknown", EncoderType: "unknown", OutMod: "file", Path: tempDir, FileName: "kae.log"},
		{Level: "warn", EncoderType: "console", OutMod: "both", Path: tempDir, FileName: "kae.log"},
	}

	for _, config := range testCases {
		if GetLogger(&config) == nil {
			t.Fatalf("GetLogger(%+v) returned nil", config)
		}
	}
}

func TestParseConfigReadsConfiguredValues(t *testing.T) {
	viper.Reset()
	t.Cleanup(viper.Reset)

	configPath := filepath.Join(t.TempDir(), "kae-device-plugin.yaml")
	config := []byte("level: debug\nencoderType: json\noutMod: console\n")
	if err := os.WriteFile(configPath, config, 0o600); err != nil {
		t.Fatalf("write configuration: %v", err)
	}
	viper.SetConfigFile(configPath)

	parsed, err := parseConfig()
	if err != nil {
		t.Fatalf("parse configuration: %v", err)
	}
	if parsed.Level != "debug" || parsed.EncoderType != "json" {
		t.Fatalf("unexpected parsed configuration: %+v", parsed)
	}
}

func TestParseConfigReturnsErrorWithoutConfiguration(t *testing.T) {
	viper.Reset()
	t.Cleanup(viper.Reset)

	if _, err := parseConfig(); err == nil {
		t.Fatal("parseConfig succeeded without a configured file")
	}
}

func TestLoggingWrappersDelegateWithoutPanicking(t *testing.T) {
	originalLogger := Logger
	Logger = zap.NewNop().Sugar()
	t.Cleanup(func() { Logger = originalLogger })

	if With("component", "test") == nil {
		t.Fatal("With returned nil")
	}
	Debug("message")
	Info("message")
	Warn("message")
	Error("message")
	DPanic("message")
	Debugf("message %d", 1)
	Infof("message %d", 1)
	Warnf("message %d", 1)
	Errorf("message %d", 1)
	DPanicf("message %d", 1)
	Debugw("message", "key", "value")
	Infow("message", "key", "value")
	Warnw("message", "key", "value")
	Errorw("message", "key", "value")
	DPanicw("message", "key", "value")
	Debugln("message")
	Infoln("message")
	Warnln("message")
	Errorln("message")
	DPanicln("message")

	if err := Sync(); err != nil {
		t.Fatalf("sync logger: %v", err)
	}
}

func TestPanicWrappersPanic(t *testing.T) {
	originalLogger := Logger
	Logger = zap.NewNop().Sugar()
	t.Cleanup(func() { Logger = originalLogger })

	testCases := []struct {
		name string
		call func()
	}{
		{name: "Panic", call: func() { Panic("message") }},
		{name: "Panicf", call: func() { Panicf("message %d", 1) }},
		{name: "Panicw", call: func() { Panicw("message", "key", "value") }},
		{name: "Panicln", call: func() { Panicln("message") }},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("expected logger call to panic")
				}
			}()
			testCase.call()
		})
	}
}
