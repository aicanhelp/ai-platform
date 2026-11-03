// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

// Package pprof
package pprof

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestProfilingHandler_handle(t *testing.T) {
	pprofDir, err := os.MkdirTemp(os.TempDir(), "pprof")
	if err != nil {
		t.Fatalf("couldn't make temporary directory: %v", err)
	}

	ph := &profilingHandler{
		modules:       "coredns",
		profilingPath: pprofDir,
	}

	ph.handle()
	time.Sleep(20 * time.Second)
	count := count(pprofDir)
	assert.EqualValues(t, 5, count)

	err1 := os.RemoveAll(pprofDir)
	if err1 != nil {
		t.Fatalf("couldn't remove pprof directory: %v", err1)
	}
}

func count(pprofDir string) int {
	count := 0
	err := filepath.Walk(pprofDir, func(path string, info os.FileInfo, err error) error {
		if !info.IsDir() {
			count++
		}
		return nil
	})
	if err != nil {
		return 0
	}
	return count
}
