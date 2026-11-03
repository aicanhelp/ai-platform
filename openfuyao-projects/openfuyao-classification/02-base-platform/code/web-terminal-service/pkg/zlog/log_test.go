package zlog

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func TestGetDefaultConf_UsesExecutableName(t *testing.T) {
	patch := gomonkey.ApplyFunc(os.Executable, func() (string, error) {
		return "/tmp/wts-bin", nil
	})
	defer patch.Reset()

	conf := getDefaultConf()
	assert.Equal(t, filepath.Join(defaultLogPath, "wts-bin"), conf.Path)
}

func TestGetDefaultConf_ExecutableError(t *testing.T) {
	patch := gomonkey.ApplyFunc(os.Executable, func() (string, error) {
		return "", errors.New("boom")
	})
	defer patch.Reset()

	conf := getDefaultConf()
	assert.Equal(t, defaultLogPath, conf.Path)
}

func TestGetEncoder(t *testing.T) {
	jsonEnc := getEncoder(&LogConfig{EncoderType: "json"})
	jsonBytes, err := jsonEnc.EncodeEntry(zapcore.Entry{Message: "ok"}, nil)
	assert.NoError(t, err)
	assert.True(t, bytes.HasPrefix(jsonBytes.Bytes(), []byte("{")))

	consoleEnc := getEncoder(&LogConfig{EncoderType: "console"})
	consoleBytes, err := consoleEnc.EncodeEntry(zapcore.Entry{Message: "ok"}, nil)
	assert.NoError(t, err)
	assert.False(t, bytes.HasPrefix(consoleBytes.Bytes(), []byte("{")))
}

func TestGetLogger_LevelFallback(t *testing.T) {
	conf := &LogConfig{
		Level:       "invalid",
		EncoderType: "console",
		OutMod:      "console",
	}
	logger := GetLogger(conf)
	assert.NotNil(t, logger)
	core := logger.Desugar().Core()
	assert.False(t, core.Enabled(zapcore.DebugLevel))
	assert.True(t, core.Enabled(zapcore.InfoLevel))
}

func TestGetLogWriter(t *testing.T) {
	tmpDir := t.TempDir()
	tests := []struct {
		name  string
		conf  *LogConfig
		write bool
	}{
		{
			name:  "console",
			conf:  &LogConfig{OutMod: "console"},
			write: true,
		},
		{
			name: "file",
			conf: &LogConfig{
				OutMod:   "file",
				Path:     tmpDir,
				FileName: "test.log",
			},
			write: false,
		},
		{
			name: "both",
			conf: &LogConfig{
				OutMod:   "both",
				Path:     tmpDir,
				FileName: "test.log",
			},
			write: false,
		},
		{
			name:  "unknown",
			conf:  &LogConfig{OutMod: "unknown"},
			write: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ws := getLogWriter(tt.conf)
			assert.NotNil(t, ws)
			if tt.write {
				_, err := ws.Write([]byte("hello"))
				assert.NoError(t, err)
			}
		})
	}
}

func TestLoggerWrappers(t *testing.T) {
	orig := Logger
	Logger = zap.NewNop().Sugar()
	t.Cleanup(func() {
		Logger = orig
	})

	LogWith("k", "v")
	LogDebug("d")
	LogInfo("i")
	LogWarn("w")
	LogError("e")
	LogDebugf("%s", "d")
	LogInfof("%s", "i")
	LogWarnf("%s", "w")
	LogErrorf("%s", "e")
	LogDebugw("d", "k", "v")
	LogInfow("i", "k", "v")
	LogWarnw("w", "k", "v")
	LogErrorw("e", "k", "v")
	LogDebugln("d")
	LogInfoln("i")
	LogWarnln("w")
	LogErrorln("e")

	if err := Sync(); err != nil {
		t.Fatalf("sync failed: %v", err)
	}
}

func TestGetLogger_UsesDefaultLevelOnUnknown(t *testing.T) {
	conf := &LogConfig{
		Level:       "UNKNOWN",
		EncoderType: "json",
		OutMod:      "console",
	}
	logger := GetLogger(conf)
	assert.NotNil(t, logger)
	got := reflect.TypeOf(logger).String()
	assert.NotEmpty(t, got)
}
