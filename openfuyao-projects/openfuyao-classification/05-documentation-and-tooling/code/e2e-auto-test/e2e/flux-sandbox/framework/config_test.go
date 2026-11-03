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

package framework

import (
	"testing"
	"time"
)

func TestLoadDefaults(t *testing.T) {
	t.Setenv("E2E_NODE_ACCESS", "")
	t.Setenv("E2E_ORPHAN_GRACE_PERIOD", "")
	t.Setenv("E2E_SCAN_INTERVAL", "")
	t.Setenv("E2E_SMALL_CAPACITY_AGENT_URL", "")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.NodeAccess != NodeAccessSSH {
		t.Errorf("NodeAccess = %q, want ssh (Linux+k8s is the primary path)", cfg.NodeAccess)
	}
	if cfg.Namespace != DefaultNamespace {
		t.Errorf("Namespace = %q, want %q", cfg.Namespace, DefaultNamespace)
	}
	if cfg.AgentPort != DefaultAgentPort {
		t.Errorf("AgentPort = %d, want %d", cfg.AgentPort, DefaultAgentPort)
	}
	if cfg.GracePeriod != 30*time.Second {
		t.Errorf("GracePeriod = %v, want 30s", cfg.GracePeriod)
	}
	if cfg.ScanInterval != 30*time.Second {
		t.Errorf("ScanInterval = %v, want 30s", cfg.ScanInterval)
	}
}

func TestLoadInvalidNodeAccess(t *testing.T) {
	t.Setenv("E2E_NODE_ACCESS", "telnet")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for invalid E2E_NODE_ACCESS")
	}
}

func TestLoadInvalidDuration(t *testing.T) {
	t.Setenv("E2E_ORPHAN_GRACE_PERIOD", "not-a-duration")
	if _, err := Load(); err == nil {
		t.Fatal("expected error for invalid E2E_ORPHAN_GRACE_PERIOD")
	}
}

func TestLoadOverrides(t *testing.T) {
	t.Setenv("E2E_NODE_ACCESS", NodeAccessDocker)
	t.Setenv("E2E_ORPHAN_GRACE_PERIOD", "1m")
	t.Setenv("E2E_SCAN_INTERVAL", "15s")
	t.Setenv("E2E_AGENT_AUTH_TOKEN", "s3cr3t")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.NodeAccess != NodeAccessDocker {
		t.Errorf("NodeAccess = %q, want docker", cfg.NodeAccess)
	}
	if cfg.GracePeriod != time.Minute {
		t.Errorf("GracePeriod = %v, want 1m", cfg.GracePeriod)
	}
	if cfg.ScanInterval != 15*time.Second {
		t.Errorf("ScanInterval = %v, want 15s", cfg.ScanInterval)
	}
	if cfg.AgentAuthToken != "s3cr3t" {
		t.Errorf("AgentAuthToken = %q, want s3cr3t", cfg.AgentAuthToken)
	}
}

func TestDerivedTimings(t *testing.T) {
	cfg := &Config{GracePeriod: 30 * time.Second, ScanInterval: 30 * time.Second}

	if got := cfg.NotCleanedWindow(); got != 25*time.Second {
		t.Errorf("NotCleanedWindow = %v, want 25s", got)
	}
	want := OrphanTimeout + 30*time.Second + 2*30*time.Second + 60*time.Second
	if got := cfg.CleanupTimeout(); got != want {
		t.Errorf("CleanupTimeout = %v, want %v", got, want)
	}

	// A tiny grace period must still leave a safe observation window.
	cfg.GracePeriod = 5 * time.Second
	if got := cfg.NotCleanedWindow(); got != 10*time.Second {
		t.Errorf("NotCleanedWindow with tiny grace = %v, want 10s floor", got)
	}
}
