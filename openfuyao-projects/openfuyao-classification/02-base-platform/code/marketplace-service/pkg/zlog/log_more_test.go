package zlog

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"
)

func TestAdditionalSafeWrappers(t *testing.T) {
	backup := logger
	logger = zap.NewNop().Sugar()
	defer func() { logger = backup }()

	_ = With("k", "v")
	Warn("warn")
	Info("info")
	Debug("debug")
	Error("error")

	Infof("hello %s", "world")
	Debugf("debug %d", 1)
	Warnf("warn %s", "x")
	Errorf("error %s", "x")

	Errorw("errorw", "k", "v")
	Warnw("warnw", "k", "v")
	Infow("infow", "k", "v")
	Debugw("debugw", "k", "v")

	Errorln("errorln")
	Warnln("warnln")
	Infoln("infoln")
	Debugln("debugln")

	assert.NoError(t, Sync())
}

