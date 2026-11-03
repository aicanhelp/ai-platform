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

package log

import (
	"io"
	"os"
	"sync"
	"testing"

	olog "gopkg.openfuyao.cn/common-modules/ologger/v26/log"
)

func TestDefaultDelegation(t *testing.T) {
	SetLogger(nil)
	Info("delegation smoke test")
}

func TestSetLogger(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	enableConsole := true
	enableFile := false
	cfg := olog.Config{
		Level:         olog.INFO,
		Format:        "json",
		EnableConsole: &enableConsole,
		EnableFile:    &enableFile,
	}
	l := olog.NewLogger(cfg)
	SetLogger(l)
	Info("injected logger test")
}

func TestInitForAgent(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	enableConsole := true
	enableFile := false
	cfg := DefaultAgentConfig()
	cfg.EnableConsole = &enableConsole
	cfg.EnableFile = &enableFile
	InitForAgent(cfg)
	Info("agent logger test")

	got := DefaultAgentConfig()
	if got.Path != "/var/log/openFuyao/bkeagent.log" {
		t.Fatalf("unexpected default path: %s", got.Path)
	}
}

func TestConcurrentAccess(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			Infof("concurrent log %d", i)
		}()
	}
	wg.Wait()
}

func TestSetLevel(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	enableConsole := false
	enableFile := false
	cfg := olog.Config{
		Level:         olog.INFO,
		Format:        "json",
		EnableConsole: &enableConsole,
		EnableFile:    &enableFile,
	}
	SetLogger(olog.NewLogger(cfg))

	if err := SetLevel(olog.DEBUG); err != nil {
		t.Fatalf("SetLevel failed: %v", err)
	}
}

func TestWithReturnsLogger(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	child := With("component", "test")
	if child == nil {
		t.Fatal("With returned nil")
	}
	child.Info("with fields")
}

func TestControllerLogger(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	child := ControllerLogger("bkecluster")
	if child == nil {
		t.Fatal("ControllerLogger returned nil")
	}
	child.Info("controller logger smoke test")
}

func TestInitDoesNotPanicOnDiscard(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	// Ensure InitForAgent works when file output is disabled (test env).
	enableConsole := true
	enableFile := false
	cfg := DefaultAgentConfig()
	cfg.EnableConsole = &enableConsole
	cfg.EnableFile = &enableFile
	cfg.Path = os.DevNull
	InitForAgent(cfg)

	// Redirect through discard-backed logger for isolation.
	SetLogger(olog.NewLogger(olog.Config{
		Level:  olog.INFO,
		Format: "json",
		EnableConsole: func() *bool {
			b := false
			return &b
		}(),
		EnableFile: func() *bool {
			b := false
			return &b
		}(),
	}))
	_ = io.Discard
	Info("discard path")
}

func TestAllLogLevelsWithInjectedLogger(t *testing.T) {
	t.Cleanup(func() { SetLogger(nil) })

	enableConsole := false
	enableFile := false
	cfg := olog.Config{
		Level:         olog.TRACE,
		Format:        "json",
		EnableConsole: &enableConsole,
		EnableFile:    &enableFile,
	}
	SetLogger(olog.NewLogger(cfg))

	Trace("trace")
	Tracef("trace %s", "msg")
	Log(olog.INFO, "log")
	Logf(olog.INFO, "log %s", "msg")
	Debug("debug")
	Debugf("debug %s", "msg")
	Info("info")
	Infof("info %s", "msg")
	Warn("warn")
	Warnf("warn %s", "msg")
	Error("error")
	Errorf("error %s", "msg")
	Critical("critical")
	Criticalf("critical %s", "msg")
}
