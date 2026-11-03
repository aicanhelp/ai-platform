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

package reset

import (
	"errors"
	"os/exec"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	bkeexec "gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
)

func TestIsSystemdUnitNotLoaded(t *testing.T) {
	assert.True(t, isSystemdUnitNotLoaded("Unit kubelet.service not loaded"))
	assert.True(t, isSystemdUnitNotLoaded("could not be found"))
	assert.True(t, isSystemdUnitNotLoaded("service not found"))
	assert.False(t, isSystemdUnitNotLoaded("permission denied"))
}

func TestIsEmptyListOutput(t *testing.T) {
	assert.True(t, isEmptyListOutput(""))
	assert.True(t, isEmptyListOutput("   "))
	assert.True(t, isEmptyListOutput("."))
	assert.False(t, isEmptyListOutput("abc123"))
}

func TestListContainerIDs(t *testing.T) {
	tests := []struct {
		name    string
		mockOut func(string, ...string) (string, error)
		wantOK  bool
		wantIDs []string
	}{
		{
			name: "returns ids on success",
			mockOut: func(string, ...string) (string, error) {
				return "id1 id2", nil
			},
			wantOK:  true,
			wantIDs: []string{"id1", "id2"},
		},
		{
			name: "empty output means nothing to clean",
			mockOut: func(string, ...string) (string, error) {
				return ".", nil
			},
			wantOK: false,
		},
		{
			name: "error with empty output is ignored",
			mockOut: func(string, ...string) (string, error) {
				return "", errors.New("command failed")
			},
			wantOK: false,
		},
		{
			name: "error with output is logged and ignored",
			mockOut: func(string, ...string) (string, error) {
				return "partial", errors.New("command failed")
			},
			wantOK: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			mockOut := tt.mockOut
			patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithOutput,
				func(_ *bkeexec.CommandExecutor, _ string, _ ...string) (string, error) {
					return mockOut("", "/bin/sh", "-c", "docker ps -q")
				})

			ids, ok := listContainerIDs("docker ps -q")
			assert.Equal(t, tt.wantOK, ok)
			if tt.wantIDs != nil {
				assert.Equal(t, tt.wantIDs, ids)
			}
		})
	}
}

func TestCleanIptablesRules(t *testing.T) {
	t.Run("skip when iptables missing", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(exec.LookPath, func(string) (string, error) {
			return "", errors.New("not found")
		})
		cleanIptablesRules()
	})

	t.Run("warn on flush failure", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(exec.LookPath, func(string) (string, error) {
			return "/usr/sbin/iptables", nil
		})
		patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithCombinedOutput,
			func(_ *bkeexec.CommandExecutor, _ string, _ ...string) (string, error) {
				return "denied", errors.New("permission denied")
			})
		cleanIptablesRules()
	})
}

func TestDeleteInterfaceIfExists(t *testing.T) {
	t.Run("skip missing interface", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithCombinedOutput,
			func(_ *bkeexec.CommandExecutor, _ string, args ...string) (string, error) {
				shellCmd := ""
				if len(args) >= 2 {
					shellCmd = args[1]
				}
				if strings.Contains(shellCmd, "ip link show") {
					return "", errors.New("not found")
				}
				return "", nil
			})
		deleteInterfaceIfExists("boc0")
	})

	t.Run("warn on delete failure", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithCombinedOutput,
			func(_ *bkeexec.CommandExecutor, _ string, args ...string) (string, error) {
				shellCmd := ""
				if len(args) >= 2 {
					shellCmd = args[1]
				}
				if strings.Contains(shellCmd, "ip link show") {
					return "state UP", nil
				}
				return "", nil
			})
		patches.ApplyFunc((*bkeexec.CommandExecutor).ExecuteCommandWithOutput,
			func(_ *bkeexec.CommandExecutor, _ string, _ ...string) (string, error) {
				return "failed", errors.New("delete failed")
			})
		deleteInterfaceIfExists("boc0")
	})
}
