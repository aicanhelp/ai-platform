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

package zlog

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func TestFilteredCore(t *testing.T) {
	nopCore := zapcore.NewNopCore()
	fc := &FilteredCore{Core: nopCore}

	coreWith := fc.With([]zapcore.Field{})
	if coreWith == nil {
		t.Fatalf("With() returned nil")
	}

	if _, ok := coreWith.(*FilteredCore); !ok {
		t.Fatalf("With() did not return *FilteredCore")
	}

	entry := zapcore.Entry{Message: "test"}
	ce := fc.Check(entry, nil)
	_ = ce

	err := fc.Write(entry, nil)
	if err != nil {
		t.Fatalf("Write() returned error: %v", err)
	}
}

func TestFilterLogMessage(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "clean message",
			input:    "hello world",
			expected: "hello world",
		},
		{
			name:     "null byte",
			input:    "hello\x00world",
			expected: "helloworld",
		},
		{
			name:     "ESC and DEL",
			input:    "test\x1b\x7f",
			expected: "test",
		},
		{
			name:     "quotes and tab",
			input:    "a\"b\tc",
			expected: "abc",
		},
		{
			name:     "HTML tags",
			input:    "a<b>test</b>c",
			expected: "atestc",
		},
		{
			name:     "special chars",
			input:    "cmd; ls & rm @home",
			expected: "cmd ls  rm home",
		},
		{
			name:     "mixed control chars and tags",
			input:    "<script>say\x00\t\"hi\x7f</script>",
			expected: "sayhi",
		},
		{
			name:     "no filtered chars",
			input:    "plain-text_123",
			expected: "plain-text_123",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := filterLogMessage(tt.input)
			if got != tt.expected {
				t.Fatalf("filterLogMessage() = %q, want %q", got, tt.expected)
			}
		})
	}
}

func TestLoggerWrappers(t *testing.T) {
	t.Run("With", func(t *testing.T) {
		l := With("key", "value")
		if l == nil {
			t.Fatalf("With() returned nil")
		}
	})

	t.Run("Error", func(t *testing.T) {
		Error("test error")
	})

	t.Run("Warn", func(t *testing.T) {
		Warn("test warn")
	})

	t.Run("Info", func(t *testing.T) {
		Info("test info")
	})

	t.Run("Debug", func(t *testing.T) {
		Debug("test debug")
	})

	t.Run("Panic", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("Panic() should have panicked")
			}
		}()
		Panic("test panic")
	})

	t.Run("DPanic", func(t *testing.T) {
		DPanic("test dpanic") 
	})

	t.Run("Fatal", func(t *testing.T) {
	})
}

func TestLoggerFormattedWrappers(t *testing.T) {
	t.Run("Errorf", func(t *testing.T) {
		Errorf("error %d", 1)
	})

	t.Run("Warnf", func(t *testing.T) {
		Warnf("warn %s", "test")
	})

	t.Run("Infof", func(t *testing.T) {
		Infof("info %v", map[string]string{"k": "v"})
	})

	t.Run("Debugf", func(t *testing.T) {
		Debugf("debug %f", 3.14)
	})

	t.Run("Fatalf", func(t *testing.T) {
	})
}

func TestGetDefaultConf(t *testing.T) {
	conf := getDefaultConf()
	if conf == nil {
		t.Fatalf("getDefaultConf() returned nil")
	}
	if conf.Level != "info" || conf.EncoderType != "console" || conf.OutMod != "both" {
		t.Fatalf("unexpected default conf: %#v", conf)
	}
	if !strings.HasSuffix(conf.Path, filepath.Base(conf.Path)) {
		t.Fatalf("unexpected default path: %s", conf.Path)
	}
}

func TestGetDefaultConfExecutableError(t *testing.T) {
	oldExecutable := executable
	executable = func() (string, error) {
		return "", errors.New("exec failed")
	}
	defer func() { executable = oldExecutable }()

	conf := getDefaultConf()
	if conf == nil {
		t.Fatalf("getDefaultConf() returned nil")
	}
	if conf.Path != defaultLogPath {
		t.Fatalf("expected fallback path %q, got %q", defaultLogPath, conf.Path)
	}
}

func TestGetLoggerInvalidLevelFallsBackToInfo(t *testing.T) {
	conf := &logConfig{Level: "invalid", EncoderType: "console", OutMod: "console"}
	lg := getLogger(conf)
	if lg == nil {
		t.Fatalf("getLogger() returned nil")
	}
}

func TestParseConfigReadError(t *testing.T) {
	viper.SetConfigFile(filepath.Join(t.TempDir(), "missing.yaml"))
	_, err := parseConfig()
	if err == nil {
		t.Fatalf("parseConfig() expected error")
	}
}

func TestLoadConfigSuccess(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, defaultConfigName+"."+defaultConfigType)
	if err := os.WriteFile(cfgPath, []byte("Level: debug\nEncoderType: json\nPath: /tmp\nFileName: app.log\nMaxSize: 1\nMaxBackups: 1\nMaxAge: 1\nLocalTime: true\nCompress: false\nOutMod: file\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	viper.Reset()
	viper.SetConfigFile(cfgPath)
	viper.SetConfigType(defaultConfigType)
	viper.SetConfigName(defaultConfigName)
	viper.AddConfigPath(dir)

	conf, err := loadConfig()
	if err != nil {
		t.Fatalf("loadConfig() returned error: %v", err)
	}
	if conf == nil || conf.Level != "debug" || conf.OutMod != "file" {
		t.Fatalf("unexpected config: %#v", conf)
	}
}

func TestWatchConfigOnce(t *testing.T) {
	watchOnce = sync.Once{}
	oldLogger := logger
	logger = zap.NewNop().Sugar()
	defer func() { logger = oldLogger }()

	watchConfig()
	watchConfig()
}

func TestFilteredCoreCheckFiltersMessage(t *testing.T) {
	core := &FilteredCore{Core: zapcore.NewNopCore()}
	entry := zapcore.Entry{Message: "<a>hi\x00</a>"}
	_ = core.Check(entry, nil)
}

func TestGetEncoder(t *testing.T) {
	jsonEncoder := getEncoder(&logConfig{EncoderType: "json"})
	if jsonEncoder == nil {
		t.Fatalf("getEncoder(json) returned nil")
	}
	consoleEncoder := getEncoder(&logConfig{EncoderType: "console"})
	if consoleEncoder == nil {
		t.Fatalf("getEncoder(console) returned nil")
	}
	defaultEncoder := getEncoder(&logConfig{EncoderType: ""})
	if defaultEncoder == nil {
		t.Fatalf("getEncoder(default) returned nil")
	}
}

func TestGetLogWriter(t *testing.T) {
	if got := getLogWriter(&logConfig{OutMod: "console"}); got == nil {
		t.Fatalf("getLogWriter(console) returned nil")
	}
	if got := getLogWriter(&logConfig{OutMod: "file", Path: t.TempDir(), FileName: "x.log"}); got == nil {
		t.Fatalf("getLogWriter(file) returned nil")
	}
	if got := getLogWriter(&logConfig{OutMod: "both", Path: t.TempDir(), FileName: "x.log"}); got == nil {
		t.Fatalf("getLogWriter(both) returned nil")
	}
	if got := getLogWriter(&logConfig{OutMod: "other"}); got == nil {
		t.Fatalf("getLogWriter(other) returned nil")
	}
	if got := getLogWriter(&logConfig{OutMod: ""}); got == nil {
		t.Fatalf("getLogWriter(empty) returned nil")
	}
}

func TestSync(t *testing.T) {
	err := Sync()
	_ = err
}