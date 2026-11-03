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

package cpuset

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/utils/pointer"

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

// ---------- RegisterFunc ----------

func TestRegisterFunc(t *testing.T) {
	resetHooks(t)
	NewCpuSetRunner().RegisterFunc()

	// Each of the three registered actions should resolve to a non-empty hook list.
	for _, action := range []string{common.PreRunPodSandbox, common.PreCreateContainer, common.PreUpdateContainerResources} {
		h, err := hooks.GetHook(action)
		assert.NoError(t, err)
		assert.NotEmpty(t, h)
	}

	// Registering again exercises the "hooks already exists" error path internally.
	NewCpuSetRunner().RegisterFunc()
}

// ---------- NewCpuSetRunner singleton ----------

func TestNewCpuSetRunnerSingleton(t *testing.T) {
	resetHooks(t)
	r1 := NewCpuSetRunner()
	r2 := NewCpuSetRunner()
	assert.Same(t, r1, r2)
}

// ---------- SetContainerCPUSet ----------

func TestSetContainerCPUSet(t *testing.T) {

	info := core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.QosClassAnnotationKey: "BE",
				},
			},
		},
	}
	err := NewCpuSetRunner().SetContainerCPUSet(&info)
	if err != nil {
		t.Fatal(err)
	}
}

func TestSetContainerCPUSetWrongType(t *testing.T) {
	r := NewCpuSetRunner()
	// Pass a *core.PodInfo (wrong type) -> assertion fails.
	err := r.SetContainerCPUSet(&core.PodInfo{})
	assert.Error(t, err)
}

func TestSetContainerCPUSetEmptyCPUSetIsNoop(t *testing.T) {
	// For a BE pod getContainerCPUSet returns "" -> SetContainerCPUSet returns nil and
	// leaves CPUSet unset.
	info := &core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{common.QosClassAnnotationKey: "BE"},
			},
		},
	}
	r := NewCpuSetRunner()
	assert.NoError(t, r.SetContainerCPUSet(info))
	assert.Nil(t, info.ContainerResponese.Resources.CPUSet)
}

// ---------- UnsetPodCPUQuota ----------

// TestUnsetPodCPUQuota
func TestUnsetPodCPUQuota(t *testing.T) {

	info := core.PodInfo{
		PodResponese: core.PodResponse{
			Resources: utils.LinuxContainerResources{
				CPUShares:   pointer.Int64(2048),
				CFSQuota:    pointer.Int64(200000),
				MemoryLimit: pointer.Int64(2147483648),
			},
		},
	}
	err := NewCpuSetRunner().UnsetPodCPUQuota(&info)
	if err != nil {
		t.Fatal(err)
	}
}

func TestUnsetPodCPUQuotaWrongType(t *testing.T) {
	r := NewCpuSetRunner()
	// Pass a *core.ContainerInfo (wrong type) -> assertion fails.
	err := r.UnsetPodCPUQuota(&core.ContainerInfo{})
	assert.Error(t, err)
}

func TestUnsetPodCPUQuotaNoAnnotationsIsNoop(t *testing.T) {
	// No annotations -> GetPodCPUset returns "" -> UnsetPodCPUQuota returns nil and
	// leaves CFSQuota untouched.
	info := &core.PodInfo{}
	r := NewCpuSetRunner()
	assert.NoError(t, r.UnsetPodCPUQuota(info))
	assert.Nil(t, info.PodResponese.Resources.CFSQuota)
}

func TestUnsetPodCPUQuotaWithCpusetClearsQuota(t *testing.T) {
	// Pod with a NUMA-resource annotation carrying a non-empty cpuset -> the hook sets
	// CFSQuota to -1 (i.e. unsets the quota).
	info := &core.PodInfo{
		PodRequest: core.PodRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.NumaResourceAnnotation: `{"CPUset":"0-3"}`,
				},
			},
		},
	}
	r := NewCpuSetRunner()
	assert.NoError(t, r.UnsetPodCPUQuota(info))
	assert.NotNil(t, info.PodResponese.Resources.CFSQuota)
	assert.Equal(t, int64(-1), *info.PodResponese.Resources.CFSQuota)
}

func TestUnsetPodCPUQuotaInvalidAnnotationErrors(t *testing.T) {
	// Malformed JSON in the NUMA-resource annotation -> GetPodCPUset returns an error.
	info := &core.PodInfo{
		PodRequest: core.PodRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.NumaResourceAnnotation: "{not-json",
				},
			},
		},
	}
	r := NewCpuSetRunner()
	assert.Error(t, r.UnsetPodCPUQuota(info))
}

// ---------- getContainerCPUSet ----------

// TestGetcontainerCPUSet
func TestGetcontainerCPUSet(t *testing.T) {

	info := core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.QosClassAnnotationKey: "BE",
				},
			},
		},
	}
	_, err := getContainerCPUSet(&info.ContainerRequest)
	if err != nil {
		t.Fatal(err)
	}
}

func TestGetContainerCPUSetBranches(t *testing.T) {
	t.Run("BE returns empty cpuset", func(t *testing.T) {
		req := &core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{common.QosClassAnnotationKey: "BE"},
			},
		}
		set, err := getContainerCPUSet(req)
		assert.NoError(t, err)
		assert.NotNil(t, set)
		assert.Equal(t, "", *set)
	})
	t.Run("non-BE (LS) returns empty cpuset", func(t *testing.T) {
		req := &core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{common.QosClassAnnotationKey: "LS"},
			},
		}
		set, err := getContainerCPUSet(req)
		assert.NoError(t, err)
		assert.NotNil(t, set)
		assert.Equal(t, "", *set)
	})
	t.Run("no QoS annotation defaults to empty cpuset", func(t *testing.T) {
		req := &core.ContainerRequest{
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{},
			},
		}
		set, err := getContainerCPUSet(req)
		assert.NoError(t, err)
		assert.NotNil(t, set)
		assert.Equal(t, "", *set)
	})
}
