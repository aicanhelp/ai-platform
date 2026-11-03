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

package builtin

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
)

func TestNewTask(t *testing.T) {
	client := fake.NewClientBuilder().Build()
	exec := &exec.CommandExecutor{}
	task := New(exec, client)
	assert.NotNil(t, task)
}

func TestTaskExecuteNullInstructions(t *testing.T) {
	client := fake.NewClientBuilder().Build()
	exec := &exec.CommandExecutor{}
	task := New(exec, client)
	result, err := task.Execute(context.Background(), []string{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "null")
	assert.Empty(t, result)
}

func TestTaskExecuteNotFound(t *testing.T) {
	client := fake.NewClientBuilder().Build()
	exec := &exec.CommandExecutor{}
	task := New(exec, client)
	result, err := task.Execute(context.Background(), []string{"notfound"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
	assert.Empty(t, result)
}

func TestTaskExecuteBuiltinPlugin(t *testing.T) {
	client := fake.NewClientBuilder().Build()
	exec := &exec.CommandExecutor{}
	task := New(exec, client)
	result, err := task.Execute(context.Background(), []string{"Ping"})
	assert.NoError(t, err)
	assert.Len(t, result, 2)
	assert.Equal(t, "pong", result[0])
}

func TestTaskExecuteCaseInsensitive(t *testing.T) {
	client := fake.NewClientBuilder().Build()
	exec := &exec.CommandExecutor{}
	task := New(exec, client)
	result, err := task.Execute(context.Background(), []string{"ping"})
	assert.NoError(t, err)
	assert.Len(t, result, 2)
}

func TestTaskExecuteWithArgs(t *testing.T) {
	client := fake.NewClientBuilder().Build()
	exec := &exec.CommandExecutor{}
	task := New(exec, client)
	result, err := task.Execute(context.Background(), []string{"Ping", "arg1", "arg2"})
	assert.NoError(t, err)
	assert.Len(t, result, 2)
}

func TestBuiltInInterface(t *testing.T) {
	var _ BuiltIn = &Task{}
}
