package zlog

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func writeLogConfig(t *testing.T, dir string) string {
	t.Helper()
	cfgPath := filepath.Join(dir, "marketplace-service.yaml")
	content := []byte("" +
		"level: info\n" +
		"encoderType: console\n" +
		"path: " + dir + "\n" +
		"fileName: test.log\n" +
		"maxSize: 20\n" +
		"maxBackups: 3\n" +
		"maxAge: 7\n" +
		"localTime: false\n" +
		"compress: true\n" +
		"outMod: console\n")
	assert.NoError(t, os.WriteFile(cfgPath, content, 0600))
	return cfgPath
}

func TestParseConfigSuccess(t *testing.T) {
	dir := t.TempDir()
	_ = writeLogConfig(t, dir)

	viper.Reset()
	viper.AddConfigPath(dir)
	viper.SetConfigName(defaultConfigName)
	viper.SetConfigType(defaultConfigType)

	cfg, err := parseConfig()
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.Equal(t, "info", cfg.Level)
	assert.Equal(t, "console", cfg.OutMod)
}

func TestLoadConfigSuccess(t *testing.T) {
	expected := &logConfig{
		Level:       "info",
		EncoderType: "console",
		Path:        "/tmp",
		FileName:    "x.log",
		MaxSize:     1,
		MaxBackups:  1,
		MaxAge:      1,
		LocalTime:   false,
		Compress:    true,
		OutMod:      "console",
	}

	patches := gomonkey.ApplyFunc(parseConfig, func() (*logConfig, error) {
		return expected, nil
	}).ApplyFunc(watchConfig, func() {})
	defer patches.Reset()

	cfg, err := loadConfig()
	assert.NoError(t, err)
	assert.Equal(t, expected, cfg)
}

func TestWatchConfig(t *testing.T) {
	dir := t.TempDir()
	_ = writeLogConfig(t, dir)

	viper.Reset()
	viper.AddConfigPath(dir)
	viper.SetConfigName(defaultConfigName)
	viper.SetConfigType(defaultConfigType)
	_, _ = parseConfig()

	watchOnce = sync.Once{}
	assert.NotPanics(t, func() {
		watchConfig()
		watchConfig()
	})
}

func TestPanicWrappers(t *testing.T) {
	backup := logger
	devLogger, err := zap.NewDevelopment()
	assert.NoError(t, err)
	logger = devLogger.Sugar()
	defer func() { logger = backup }()

	assert.Panics(t, func() { Panic("p") })
	assert.Panics(t, func() { Panicf("%s", "p") })
	assert.Panics(t, func() { Panicw("p", "k", "v") })
	assert.Panics(t, func() { Panicln("p") })

	assert.Panics(t, func() { DPanic("dp") })
	assert.Panics(t, func() { DPanicf("%s", "dp") })
	assert.Panics(t, func() { DPanicw("dp", "k", "v") })
	assert.Panics(t, func() { DPanicln("dp") })
}
