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

package gpu

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

func resetHooks(t *testing.T) {
	t.Helper()
	runner = nil
	hooks.ClearAllHooksSafe()
}

// ---------- NewGpuRunner singleton ----------

func TestNewGpuRunnerSingleton(t *testing.T) {
	resetHooks(t)
	r1 := NewGpuRunner()
	r2 := NewGpuRunner()
	assert.Same(t, r1, r2)
}

// ---------- RegisterFunc ----------

func TestRegisterFunc(t *testing.T) {
	resetHooks(t)
	NewGpuRunner().RegisterFunc()

	h, err := hooks.GetHook(common.PreCreateContainer)
	assert.NoError(t, err)
	assert.NotEmpty(t, h)

	// Re-registering exercises the "hooks already exists" internal error path.
	NewGpuRunner().RegisterFunc()
}

// ---------- SetGpu ----------

func TestSetGpu(t *testing.T) {
	info := core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.GPUInstanceAnnotation: "1,2,3",
				},
			},
		},
	}

	err := NewGpuRunner().SetGpu(&info)
	assert.NoError(t, err)
	expect := map[string]string{
		NvidiaDevices: "1,2,3",
	}
	if info.ContainerResponese.AddContainerEnvs[NvidiaDevices] != expect[NvidiaDevices] {
		t.Errorf("Expected  to be %v, got %v", "1,2,3", info.ContainerResponese.AddContainerEnvs[NvidiaDevices])

	}

}

func TestSetGpuWrongType(t *testing.T) {
	r := NewGpuRunner()
	// Pass a *core.PodInfo (wrong type) -> assertion fails.
	err := r.SetGpu(&core.PodInfo{})
	assert.Error(t, err)
}

func TestSetGpuMissingAnnotationIsNoop(t *testing.T) {
	// No GPU instance annotation -> SetGpu returns nil without populating envs.
	info := &core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.QosClassAnnotationKey: "BE",
				},
			},
		},
	}
	r := NewGpuRunner()
	assert.NoError(t, r.SetGpu(info))
	assert.Nil(t, info.ContainerResponese.AddContainerEnvs)
}

func TestSetGpuInitializesEnvMapIfNil(t *testing.T) {
	// AddContainerEnvs starts nil; SetGpu should create the map before writing.
	info := &core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.GPUInstanceAnnotation: "0",
				},
			},
		},
	}
	r := NewGpuRunner()
	assert.NoError(t, r.SetGpu(info))
	assert.NotNil(t, info.ContainerResponese.AddContainerEnvs)
	assert.Equal(t, "0", info.ContainerResponese.AddContainerEnvs[NvidiaDevices])
}
