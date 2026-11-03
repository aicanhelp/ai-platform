/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package exec

import (
	"os/exec"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHandleInterruptProcess(t *testing.T) {
	cmd := exec.Command("sleep", "30")
	require.NoError(t, cmd.Start())
	t.Cleanup(func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	})

	timer := time.NewTimer(time.Second)
	defer timer.Stop()

	output, err, sent := handleInterruptProcess(cmd, "sleep", timer, time.Second)
	assert.True(t, sent)
	assert.NoError(t, err)
	assert.Empty(t, output)
}

func TestExecuteCommandWithOutputFileWrapper(t *testing.T) {
	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithOutputFile("echo", "", "hello")
	assert.NotNil(t, output)
	_ = err
}

func TestExecuteCommandWithOutputFileTimeoutWrapper(t *testing.T) {
	executor := &CommandExecutor{}
	timeout := 50 * time.Millisecond
	output, err := executor.ExecuteCommandWithOutputFileTimeout(timeout, "sleep", "", "1")
	assert.NotNil(t, output)
	assert.Error(t, err)
}

func TestExecuteCommandWithTimeoutSuccess(t *testing.T) {
	executor := &CommandExecutor{}
	output, err := executor.ExecuteCommandWithTimeout(2*time.Second, "echo", "done")
	assert.NoError(t, err)
	assert.Equal(t, "done", output)
}
