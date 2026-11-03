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

package reset

import (
	"fmt"
	"sync/atomic"
	"time"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
)

const (
	// stopDelaySeconds is scheduled at Reset shutdown (not after writeback) so
	// self-manage delete still uninstalls when the API is already down.
	// Must leave enough time for Command status writeback when the API is up,
	// and stay shorter than RestartSec so cleanup runs before Restart=always sticks.
	stopDelaySeconds = 12
	// defaultRestartSec matches bkeagent.service.tmpl RestartSec.
	defaultRestartSec = 20
	agentServiceName  = "bkeagent"
)

// agentCleanupPending is set during Reset scope "shutdown". CommandReconciler
// still Flushes on exit (idempotent if already scheduled).
var agentCleanupPending atomic.Bool

// agentCleanupScheduled ensures delayed nohup stop is only started once.
var agentCleanupScheduled atomic.Bool

// MarkAgentCleanupPending records that bkeagent should be uninstalled.
func MarkAgentCleanupPending() {
	agentCleanupPending.Store(true)
}

// ClearPendingAgentCleanup drops pending/scheduled marks without stopping.
// Intended for tests.
func ClearPendingAgentCleanup() {
	agentCleanupPending.Store(false)
	agentCleanupScheduled.Store(false)
}

// scheduleAgentCleanupOnce starts delayed uninstall at most once.
func scheduleAgentCleanupOnce() error {
	if !agentCleanupScheduled.CompareAndSwap(false, true) {
		return nil
	}
	if err := scheduleAgentCleanup(); err != nil {
		agentCleanupScheduled.Store(false)
		return err
	}
	return nil
}

// FlushPendingAgentCleanup schedules delayed bkeagent uninstall if pending.
// Safe when nothing is pending or stop was already scheduled at shutdown phase.
func FlushPendingAgentCleanup() error {
	if !agentCleanupPending.Swap(false) {
		return nil
	}
	return scheduleAgentCleanupOnce()
}

// AgentCleanupPending reports whether uninstall is in progress for this Reset.
func AgentCleanupPending() bool {
	return agentCleanupPending.Load()
}

// WritebackTimeoutWhileCleanupPending bounds Command status sync while delayed
// uninstall is already scheduled. Kept just under stopDelaySeconds so management
// cluster scale-down can finish writeback before systemd stops the agent, without
// blocking self-manage delete for the default multi-minute sync window.
func WritebackTimeoutWhileCleanupPending() time.Duration {
	const margin = 2 * time.Second
	d := time.Duration(stopDelaySeconds)*time.Second - margin
	if d < time.Second {
		return time.Second
	}
	return d
}

// agentStopScript mirrors EnsureBKEAgent.executePreCommand (non-immutable) cleanup:
// stop → disable → remove binary/unit/workspace → daemon-reload.
// Also removes /usr/lib unit path (some push paths), systemd drop-ins,
// and legacy /etc/bkeagent. Agent logs under /var/log/openFuyao are kept.
// Scheduled via nohup because calling systemctl stop inside this process deadlocks
// (systemctl waits for MainPID to exit).
var agentStopScript = fmt.Sprintf(`sleep %d; `+
	`systemctl stop %s 2>&1 >/dev/null || true; `+
	`systemctl disable %s 2>&1 >/dev/null || true; `+
	`rm -rf /usr/local/bin/bkeagent* 2>&1 >/dev/null || true; `+
	`rm -f /etc/systemd/system/bkeagent.service /usr/lib/systemd/system/bkeagent.service 2>&1 >/dev/null || true; `+
	`rm -rf /etc/systemd/system/bkeagent.service.d /usr/lib/systemd/system/bkeagent.service.d 2>&1 >/dev/null || true; `+
	`rm -rf /etc/openFuyao/bkeagent /etc/bkeagent 2>&1 >/dev/null || true; `+
	`systemctl daemon-reload 2>&1 >/dev/null || true`,
	stopDelaySeconds, agentServiceName, agentServiceName,
)

var agentStopCmd = fmt.Sprintf("nohup /bin/sh -c %q >/dev/null 2>&1 &", agentStopScript)

var agentCleanupExecutor exec.Executor = &exec.CommandExecutor{}

// scheduleAgentCleanupDefault stops systemd bkeagent and removes agent files via nohup.
// Safe to call from inside the agent process (systemctl stop of MainPID deadlocks).
func scheduleAgentCleanupDefault() error {
	if err := agentCleanupExecutor.ExecuteCommand("/bin/sh", "-c", agentStopCmd); err != nil {
		return fmt.Errorf("schedule stop %s failed: %w", agentServiceName, err)
	}
	return nil
}
