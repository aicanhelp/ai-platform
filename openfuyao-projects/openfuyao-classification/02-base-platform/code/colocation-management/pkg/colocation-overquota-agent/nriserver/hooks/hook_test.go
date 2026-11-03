/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

package hooks

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

func TestHooksSetUp(t *testing.T) {
	fmt.Println(actionHooks)
}

func TestRegisterFunc(t *testing.T) {

	configPath := ""
	hookName := "test"
	f := func(info core.Info) error {
		t.Log("test")
		return nil
	}
	_, err := RegisterFunc(configPath, common.PreRunPodSandbox, hookName, f)
	assert.NoError(t, err)
	for _, v := range actionHooks[common.PreRunPodSandbox] {
		t.Log(v)
	}
}

func TestRunHook(t *testing.T) {
	configPath := ""
	hookName := "test"
	f := func(info core.Info) error {
		t.Log(info.Admit())
		return nil
	}
	ClearAllHooksSafe()
	_, err := RegisterFunc(configPath, common.PreCreateContainer, hookName, f)
	assert.NoError(t, err)

	info := &core.PodInfo{
		PodRequest: core.PodRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Linux: &utils.LinuxPodSandboxConfig{
					CgroupParent: "test",
				},
			},
		},
	}

	err = RunHook(common.PreRunPodSandbox, info)
	assert.NoError(t, err)
}
