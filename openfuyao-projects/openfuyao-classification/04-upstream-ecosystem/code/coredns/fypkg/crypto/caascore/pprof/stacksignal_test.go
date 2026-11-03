// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package pprof
package pprof

import (
	"os"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFYRegisterPProfSignal(t *testing.T) {
	t.Setenv("FY_ENABLE_PPROF", "true")
	t.Setenv("FY_PROFILING_PATH", os.TempDir())

	// 注册信号
	FYRegisterPProfSignal()

	pid := syscall.Getpid()

	// 注册成功后一定有pid
	assert.NotNil(t, pid)
}
