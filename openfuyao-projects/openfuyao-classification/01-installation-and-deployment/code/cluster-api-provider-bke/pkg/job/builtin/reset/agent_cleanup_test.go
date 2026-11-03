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
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
)

type mockAgentCleanupExecutor struct {
	exec.Executor
	executeCommandError error
	executeCommandArgs  []string
	called              bool
}

func (m *mockAgentCleanupExecutor) WithContext(_ context.Context) exec.Executor {
	return m
}

func (m *mockAgentCleanupExecutor) ExecuteCommand(_ string, args ...string) error {
	m.called = true
	m.executeCommandArgs = append([]string(nil), args...)
	return m.executeCommandError
}

func withMockAgentCleanupExecutor(t *testing.T, mock *mockAgentCleanupExecutor) {
	t.Helper()
	original := agentCleanupExecutor
	agentCleanupExecutor = mock
	t.Cleanup(func() {
		agentCleanupExecutor = original
	})
}

func TestScheduleAgentCleanupDefaultSequence(t *testing.T) {
	mockExec := &mockAgentCleanupExecutor{}
	withMockAgentCleanupExecutor(t, mockExec)

	err := scheduleAgentCleanupDefault()

	assert.NoError(t, err)
	assert.True(t, mockExec.called)
	assert.Len(t, mockExec.executeCommandArgs, 2)
	assert.Equal(t, "-c", mockExec.executeCommandArgs[0])
	script := mockExec.executeCommandArgs[1]
	assert.Contains(t, script, "nohup")
	assert.Contains(t, script, fmt.Sprintf("sleep %d", stopDelaySeconds))
	assert.Contains(t, script, "systemctl stop bkeagent")
	assert.Contains(t, script, "systemctl disable bkeagent")
	assert.Contains(t, script, "rm -rf /usr/local/bin/bkeagent*")
	assert.Contains(t, script, "rm -f /etc/systemd/system/bkeagent.service /usr/lib/systemd/system/bkeagent.service")
	assert.Contains(t, script, "rm -rf /etc/systemd/system/bkeagent.service.d /usr/lib/systemd/system/bkeagent.service.d")
	assert.Contains(t, script, "rm -rf /etc/openFuyao/bkeagent /etc/bkeagent")
	assert.NotContains(t, script, "/var/log/openFuyao/bkeagent")
	assert.Contains(t, script, "systemctl daemon-reload")
	assert.True(t, strings.HasSuffix(strings.TrimSpace(script), "&"))
}

func TestScheduleAgentCleanupDefaultError(t *testing.T) {
	mockExec := &mockAgentCleanupExecutor{executeCommandError: fmt.Errorf("exec failed")}
	withMockAgentCleanupExecutor(t, mockExec)

	err := scheduleAgentCleanupDefault()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "schedule stop bkeagent failed")
}

func TestStopDelayShorterThanDefaultRestartSec(t *testing.T) {
	// stop is scheduled at shutdown with a writeback window; must still beat RestartSec.
	assert.Less(t, time.Duration(stopDelaySeconds)*time.Second, time.Duration(defaultRestartSec)*time.Second)
}

func TestWritebackTimeoutWhileCleanupPendingUnderStopDelay(t *testing.T) {
	budget := WritebackTimeoutWhileCleanupPending()
	assert.GreaterOrEqual(t, budget, time.Second)
	assert.Less(t, budget, time.Duration(stopDelaySeconds)*time.Second)
}

func TestScheduleAgentCleanupOnceIdempotent(t *testing.T) {
	t.Cleanup(ClearPendingAgentCleanup)

	var calls int
	orig := scheduleAgentCleanup
	scheduleAgentCleanup = func() error {
		calls++
		return nil
	}
	t.Cleanup(func() { scheduleAgentCleanup = orig })

	assert.NoError(t, scheduleAgentCleanupOnce())
	assert.NoError(t, scheduleAgentCleanupOnce())
	assert.Equal(t, 1, calls)

	assert.NoError(t, FlushPendingAgentCleanup())
	assert.Equal(t, 1, calls, "Flush with no pending must not schedule again")

	MarkAgentCleanupPending()
	assert.NoError(t, FlushPendingAgentCleanup())
	assert.Equal(t, 1, calls, "Flush after already-scheduled stop is idempotent")
}

func TestMarkFlushClearAgentCleanupPending(t *testing.T) {
	t.Cleanup(ClearPendingAgentCleanup)

	var scheduled bool
	orig := scheduleAgentCleanup
	scheduleAgentCleanup = func() error {
		scheduled = true
		return nil
	}
	t.Cleanup(func() { scheduleAgentCleanup = orig })

	assert.False(t, AgentCleanupPending())
	assert.NoError(t, FlushPendingAgentCleanup())
	assert.False(t, scheduled)

	MarkAgentCleanupPending()
	assert.True(t, AgentCleanupPending())

	ClearPendingAgentCleanup()
	assert.False(t, AgentCleanupPending())
	assert.NoError(t, FlushPendingAgentCleanup())
	assert.False(t, scheduled)

	MarkAgentCleanupPending()
	assert.NoError(t, FlushPendingAgentCleanup())
	assert.True(t, scheduled)
	assert.False(t, AgentCleanupPending())
}

func TestFlushPendingAgentCleanupBestEffort(t *testing.T) {
	t.Cleanup(ClearPendingAgentCleanup)

	orig := scheduleAgentCleanup
	scheduleAgentCleanup = func() error { return fmt.Errorf("nohup failed") }
	t.Cleanup(func() { scheduleAgentCleanup = orig })

	MarkAgentCleanupPending()
	err := FlushPendingAgentCleanup()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nohup failed")
	assert.False(t, AgentCleanupPending(), "mark cleared even when schedule fails")
}
