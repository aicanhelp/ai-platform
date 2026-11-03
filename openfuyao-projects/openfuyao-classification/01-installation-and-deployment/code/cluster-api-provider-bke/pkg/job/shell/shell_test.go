/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package shell

import (
	"context"
	"testing"
	"time"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
)

func TestTaskExecuteEmptyCommands(t *testing.T) {
	task := &Task{Exec: &exec.CommandExecutor{}}
	result, err := task.Execute(context.Background(), 0, []string{})
	if err == nil {
		t.Error("Expected error for empty commands")
	}
	if len(result) != 0 {
		t.Errorf("Expected empty result, got %v", result)
	}
}

func TestTaskExecuteSingleCommand(t *testing.T) {
	mockExec := &exec.CommandExecutor{}
	task := &Task{Exec: mockExec}
	commands := []string{"echo test"}
	result, err := task.Execute(context.Background(), 0, commands)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(result) != 1 {
		t.Errorf("Expected 1 result, got %d", len(result))
	}
}

func TestTaskExecuteWithTimeout(t *testing.T) {
	mockExec := &exec.CommandExecutor{}
	task := &Task{Exec: mockExec}
	commands := []string{"echo test"}
	result, err := task.Execute(context.Background(), 5*time.Second, commands)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(result) != 1 {
		t.Errorf("Expected 1 result, got %d", len(result))
	}
}

func TestTaskExecuteMultipleCommands(t *testing.T) {
	mockExec := &exec.CommandExecutor{}
	task := &Task{Exec: mockExec}
	commands := []string{"echo test1", "echo test2"}
	result, err := task.Execute(context.Background(), 0, commands)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(result) != 1 {
		t.Errorf("Expected 1 result, got %d", len(result))
	}
}

func TestShellInterface(t *testing.T) {
	var _ Shell = &Task{}
}
