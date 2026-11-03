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
package hooks

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

// noopHookFunc returns a hook function that does nothing.
func noopHookFunc(t *testing.T) HookFunc {
	return func(info core.Info) error {
		t.Log("noop hook ran")
		return nil
	}
}

// failingHookFunc returns a hook function that always returns an error.
func failingHookFunc(t *testing.T) HookFunc {
	return func(info core.Info) error {
		t.Log("failing hook ran")
		return assert.AnError
	}
}

func TestAddHookDuplicate(t *testing.T) {
	ClearAllHooksSafe()
	const name = "dup-hook"
	_, err := AddHook(common.PreCreateContainer, name)
	assert.NoError(t, err)
	// adding the same name for the same action should fail
	_, err = AddHook(common.PreCreateContainer, name)
	assert.Error(t, err)
}

func TestAddHookActionNotFound(t *testing.T) {
	ClearAllHooksSafe()
	_, err := AddHook("DoesNotExist", "x")
	assert.Error(t, err)
}

func TestGetHookActionNotFound(t *testing.T) {
	ClearAllHooksSafe()
	hooks, err := GetHook("DoesNotExist")
	assert.Error(t, err)
	assert.Nil(t, hooks)
}

func TestGetHookFound(t *testing.T) {
	ClearAllHooksSafe()
	const name = "get-hook"
	_, err := AddHook(common.PreCreateContainer, name)
	assert.NoError(t, err)

	got, err := GetHook(common.PreCreateContainer)
	assert.NoError(t, err)
	assert.Len(t, got, 1)
	assert.Equal(t, name, got[0].name)
}

func TestRunHookFailingFuncLogsOnly(t *testing.T) {
	ClearAllHooksSafe()
	_, err := RegisterFunc("", common.PreUpdateContainerResources, "fail", failingHookFunc(t))
	assert.NoError(t, err)

	info := &core.PodInfo{
		PodRequest: core.PodRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Linux: &utils.LinuxPodSandboxConfig{CgroupParent: "x"},
			},
		},
	}
	// RunHook always returns nil; it only logs the inner hook error.
	assert.NoError(t, RunHook(common.PreUpdateContainerResources, info))
}

func TestRunHookActionNotFoundReturnsNil(t *testing.T) {
	ClearAllHooksSafe()
	info := &core.PodInfo{}
	assert.NoError(t, RunHook("DoesNotExist", info))
}

func TestRegisterFuncWithValidConfig(t *testing.T) {
	ClearAllHooksSafe()
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	assert.NoError(t, os.WriteFile(path, []byte("{}"), 0644))

	h, err := RegisterFunc(path, common.PreRunPodSandbox, "cfg-hook", noopHookFunc(t))
	assert.NoError(t, err)
	assert.NotNil(t, h)
}

func TestRegisterFuncWithInvalidJSONConfig(t *testing.T) {
	ClearAllHooksSafe()
	dir := t.TempDir()
	path := filepath.Join(dir, "cfg.json")
	assert.NoError(t, os.WriteFile(path, []byte("{bad"), 0644))

	h, err := RegisterFunc(path, common.PreRunPodSandbox, "bad-cfg-hook", noopHookFunc(t))
	assert.Error(t, err)
	assert.Nil(t, h)
}

func TestRegisterFuncWithMissingConfigFile(t *testing.T) {
	ClearAllHooksSafe()
	h, err := RegisterFunc(filepath.Join(t.TempDir(), "nope.json"),
		common.PreRunPodSandbox, "missing-cfg-hook", noopHookFunc(t))
	assert.Error(t, err)
	assert.Nil(t, h)
}

func TestRegisterFuncActionNotFound(t *testing.T) {
	ClearAllHooksSafe()
	h, err := RegisterFunc("", "UnknownAction", "x", noopHookFunc(t))
	assert.Error(t, err)
	assert.Nil(t, h)
}

func TestClearAllHooksSafeTwice(t *testing.T) {
	ClearAllHooksSafe()
	// second call should be a no-op and not panic
	ClearAllHooksSafe()
	// the registered action map should still be present (initialized by init)
	assert.NotNil(t, actionHooks)
	assert.NotEmpty(t, actionHooks)
}

func TestRunHookMultipleHooks(t *testing.T) {
	ClearAllHooksSafe()

	calls := 0
	mk := func() HookFunc {
		return func(info core.Info) error {
			calls++
			return nil
		}
	}
	_, err := RegisterFunc("", common.PreCreateContainer, "h1", mk())
	assert.NoError(t, err)
	_, err = RegisterFunc("", common.PreCreateContainer, "h2", mk())
	assert.NoError(t, err)

	info := &core.PodInfo{}
	assert.NoError(t, RunHook(common.PreCreateContainer, info))
	assert.Equal(t, 2, calls)
}

// TestRunHookReRegisters exercises re-registering the same action after
// clearing, and confirms GetHook reflects the change.
func TestHookLifecycle(t *testing.T) {
	ClearAllHooksSafe()
	_, err := RegisterFunc("", common.PreCreateContainer, "life", noopHookFunc(t))
	assert.NoError(t, err)

	got, err := GetHook(common.PreCreateContainer)
	assert.NoError(t, err)
	assert.Len(t, got, 1)

	ClearAllHooksSafe()
	got, err = GetHook(common.PreCreateContainer)
	assert.NoError(t, err)
	assert.Empty(t, got)
}
