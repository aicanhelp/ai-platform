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

package beresource

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/pointer"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/core"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/nriserver/hooks"
	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

// jsonMarshal is a thin wrapper used by the cfg builders below.
func jsonMarshal(cfg core.ExtendedResourceConfiguration) ([]byte, error) {
	return json.Marshal(cfg)
}

// TestSetBEPodResources
func TestSetBEPodResources(t *testing.T) {
	_, _, marshal := core.SetupTest(t)

	info := createPodInfo(marshal)

	expect := core.PodInfo{
		PodResponese: core.PodResponse{
			Resources: utils.LinuxContainerResources{
				CPUShares:   pointer.Int64(2048),
				CFSQuota:    pointer.Int64(200000),
				MemoryLimit: pointer.Int64(2147483648),
			},
		},
	}
	err := NewBEResourceRunner().SetPodResources(&info)
	if err != nil {
		t.Errorf("SetPodResources failed,err:%v", err)
	}

	if !reflect.DeepEqual(*info.PodResponese.Resources.CPUShares, *expect.PodResponese.Resources.CPUShares) {
		t.Errorf("Expected  to be %v, got %v", *expect.PodResponese.Resources.CPUShares,
			*info.PodResponese.Resources.CPUShares)
	}
	if !reflect.DeepEqual(*info.PodResponese.Resources.CFSQuota, *expect.PodResponese.Resources.CFSQuota) {
		t.Errorf("Expected  to be %v, got %v", *expect.PodResponese.Resources.CFSQuota,
			*info.PodResponese.Resources.CFSQuota)
	}
	if !reflect.DeepEqual(*info.PodResponese.Resources.MemoryLimit, *expect.PodResponese.Resources.MemoryLimit) {
		t.Errorf("Expected  to be %v, got %v", *expect.PodResponese.Resources.MemoryLimit,
			*info.PodResponese.Resources.MemoryLimit)
	}

}

// TestSetBEPodCPUShares
func TestSetBEPodCPUShares(t *testing.T) {
	_, _, marshal := core.SetupTest(t)

	info := createPodInfo(marshal)

	expect := core.PodInfo{
		PodResponese: core.PodResponse{
			Resources: utils.LinuxContainerResources{
				CPUShares: pointer.Int64(2048),
			},
		},
	}

	err := NewBEResourceRunner().SetPodCPUShares(&info)
	if err != nil {
		t.Errorf("SetPodResources failed,err:%v", err)
	}
	if !reflect.DeepEqual(*info.PodResponese.Resources.CPUShares, *expect.PodResponese.Resources.CPUShares) {
		t.Errorf("Expected  to be %v, got %v", *expect.PodResponese.Resources.CPUShares,
			*info.PodResponese.Resources.CPUShares)
	}
}

func TestSetBEPodMemoryLimit(t *testing.T) {

	_, _, marshal := core.SetupTest(t)

	info := createPodInfo(marshal)

	expect := core.PodInfo{
		PodResponese: core.PodResponse{
			Resources: utils.LinuxContainerResources{
				MemoryLimit: pointer.Int64(2147483648),
			},
		},
	}
	err := NewBEResourceRunner().SetPodMemoryLimit(&info)
	if err != nil {
		t.Errorf("SetPodResources failed,err:%v", err)
	}

	if !reflect.DeepEqual(*info.PodResponese.Resources.MemoryLimit, *expect.PodResponese.Resources.MemoryLimit) {
		t.Errorf("Expected  to be %v, got %v", *expect.PodResponese.Resources.MemoryLimit,
			*info.PodResponese.Resources.MemoryLimit)
	}
}

// TestSetBEPodCFSQuota
func TestSetBEPodCFSQuota(t *testing.T) {
	_, _, marshal := core.SetupTest(t)

	info := createPodInfo(marshal)

	expect := core.PodInfo{
		PodResponese: core.PodResponse{
			Resources: utils.LinuxContainerResources{
				CFSQuota: pointer.Int64(200000),
			},
		},
	}

	err := NewBEResourceRunner().SetPodCFSQuota(&info)
	if err != nil {
		t.Errorf("SetPodResources failed,err:%v", err)
	}
	if !reflect.DeepEqual(*info.PodResponese.Resources.CFSQuota, *expect.PodResponese.Resources.CFSQuota) {
		t.Errorf("Expected  to be %v, got %v", *expect.PodResponese.Resources.CFSQuota,
			*info.PodResponese.Resources.CFSQuota)
	}
}

// TestSetBEContainerResources
func TestSetBEContainerResources(t *testing.T) {

	_, _, marshal := core.SetupTest(t)

	info := createContainerInfo(marshal)

	expect := core.ContainerInfo{
		ContainerResponese: core.ContainerResponse{
			Resources: utils.LinuxContainerResources{
				CPUShares:   pointer.Int64(2048),
				CFSQuota:    pointer.Int64(200000),
				MemoryLimit: pointer.Int64(2147483648),
			},
		},
	}

	err := NewBEResourceRunner().SetContainerResources(&info)
	if err != nil {
		t.Errorf("SetContainerResources failed,err:%v", err)
	}
	if !reflect.DeepEqual(*info.ContainerResponese.Resources.CPUShares, *expect.ContainerResponese.Resources.CPUShares) {
		t.Errorf("Expected  to be %v, got %v", *expect.ContainerResponese.Resources.CPUShares,
			*info.ContainerResponese.Resources.CPUShares)
	}
	if !reflect.DeepEqual(*info.ContainerResponese.Resources.CFSQuota, *expect.ContainerResponese.Resources.CFSQuota) {
		t.Errorf("Expected  to be %v, got %v", *expect.ContainerResponese.Resources.CFSQuota,
			*info.ContainerResponese.Resources.CFSQuota)
	}
	if !reflect.DeepEqual(*info.ContainerResponese.Resources.MemoryLimit,
		*expect.ContainerResponese.Resources.MemoryLimit) {
		t.Errorf("Expected  to be %v, got %v", *expect.ContainerResponese.Resources.MemoryLimit,
			*info.ContainerResponese.Resources.MemoryLimit)
	}
}

// TestSetBEContainerCPUShares
func TestSetBEContainerCPUShares(t *testing.T) {
	_, _, marshal := core.SetupTest(t)

	info := createContainerInfo(marshal)

	expect := core.ContainerInfo{
		ContainerResponese: core.ContainerResponse{
			Resources: utils.LinuxContainerResources{
				CPUShares: pointer.Int64(2048),
			},
		},
	}

	err := NewBEResourceRunner().SetContainerCPUShares(&info)
	if err != nil {
		t.Errorf("SetContainerCPUShares failed,err:%v", err)
	}
	if !reflect.DeepEqual(*info.ContainerResponese.Resources.CPUShares, *expect.ContainerResponese.Resources.CPUShares) {
		t.Errorf("Expected  to be %v, got %v", *expect.ContainerResponese.Resources.CPUShares,
			*info.ContainerResponese.Resources.CPUShares)
	}
}

// TestSetBEContainerMemoryLimit
func TestSetBEContainerMemoryLimit(t *testing.T) {
	_, _, marshal := core.SetupTest(t)

	info := createContainerInfo(marshal)

	expect := core.ContainerInfo{
		ContainerResponese: core.ContainerResponse{
			Resources: utils.LinuxContainerResources{
				MemoryLimit: pointer.Int64(2147483648),
			},
		},
	}

	err := NewBEResourceRunner().SetContainerMemoryLimit(&info)
	if err != nil {
		t.Errorf("SetContainerMemoryLimit failed,err:%v", err)
	}
	if !reflect.DeepEqual(*info.ContainerResponese.Resources.MemoryLimit,
		*expect.ContainerResponese.Resources.MemoryLimit) {
		t.Errorf("Expected  to be %v, got %v", *expect.ContainerResponese.Resources.MemoryLimit,
			*info.ContainerResponese.Resources.MemoryLimit)
	}
}

// TestSetBEContainerCFSQuota
func TestSetBEContainerCFSQuota(t *testing.T) {
	_, _, marshal := core.SetupTest(t)

	info := createContainerInfo(marshal)

	expect := core.ContainerInfo{
		ContainerResponese: core.ContainerResponse{
			Resources: utils.LinuxContainerResources{
				CFSQuota: pointer.Int64(200000),
			},
		},
	}

	err := NewBEResourceRunner().SetContainerCFSQuota(&info)
	if err != nil {
		t.Errorf("SetContainerCFSQuota failed,err:%v", err)
	}
	if !reflect.DeepEqual(*info.ContainerResponese.Resources.CFSQuota,
		*expect.ContainerResponese.Resources.CFSQuota) {
		t.Errorf("Expected  to be %v, got %v", *expect.ContainerResponese.Resources.CFSQuota,
			*info.ContainerResponese.Resources.CFSQuota)
	}

}

func createPodSandboxConfig(qosClass string, marshal []byte) utils.PodSandboxConfig {
	return utils.PodSandboxConfig{
		Annotations: map[string]string{
			common.QosClassAnnotationKey:                   qosClass,
			common.ExtenderResourceConfigurationAnnotation: string(marshal),
		},
	}
}

func createPodInfo(marshal []byte) core.PodInfo {
	info := core.PodInfo{
		PodRequest: core.PodRequest{
			PodSandboxConfig: createPodSandboxConfig("BE", marshal),
		},
	}
	return info
}

func createContainerInfo(marshal []byte) core.ContainerInfo {
	info := core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			ContainerMeta: core.ContainerMetadata{
				Name: "container1",
			},
			PodSandboxConfig: createPodSandboxConfig("BE", marshal),
		},
	}
	return info
}

// ===================== Extended coverage tests =====================

func resetHooks(t *testing.T) {
	t.Helper()
	runner = nil
	hooks.ClearAllHooksSafe()
}

// ---------- RegisterFunc ----------

func TestRegisterFunc(t *testing.T) {
	resetHooks(t)
	NewBEResourceRunner().RegisterFunc()

	for _, action := range []string{common.PreRunPodSandbox, common.PreCreateContainer, common.PreUpdateContainerResources} {
		h, err := hooks.GetHook(action)
		assert.NoError(t, err)
		assert.NotEmpty(t, h)
	}

	// Re-registering exercises the "hooks already exists" internal error path.
	NewBEResourceRunner().RegisterFunc()
}

func TestNewBEResourceRunnerSingleton(t *testing.T) {
	resetHooks(t)
	r1 := NewBEResourceRunner()
	r2 := NewBEResourceRunner()
	assert.Same(t, r1, r2)
}

// ---------- type-assertion error paths ----------

func TestSetPodResourcesWrongType(t *testing.T) {
	r := NewBEResourceRunner()
	assert.Error(t, r.SetPodResources(&core.ContainerInfo{}))
}

func TestSetContainerResourcesWrongType(t *testing.T) {
	r := NewBEResourceRunner()
	assert.Error(t, r.SetContainerResources(&core.PodInfo{}))
}

func TestSetPodCPUSharesWrongType(t *testing.T) {
	assert.Error(t, NewBEResourceRunner().SetPodCPUShares(&core.ContainerInfo{}))
}

func TestSetPodCFSQuotaWrongType(t *testing.T) {
	assert.Error(t, NewBEResourceRunner().SetPodCFSQuota(&core.ContainerInfo{}))
}

func TestSetPodMemoryLimitWrongType(t *testing.T) {
	assert.Error(t, NewBEResourceRunner().SetPodMemoryLimit(&core.ContainerInfo{}))
}

func TestSetContainerCPUSharesWrongType(t *testing.T) {
	assert.Error(t, NewBEResourceRunner().SetContainerCPUShares(&core.PodInfo{}))
}

func TestSetContainerCFSQuotaWrongType(t *testing.T) {
	assert.Error(t, NewBEResourceRunner().SetContainerCFSQuota(&core.PodInfo{}))
}

func TestSetContainerMemoryLimitWrongType(t *testing.T) {
	assert.Error(t, NewBEResourceRunner().SetContainerMemoryLimit(&core.PodInfo{}))
}

// ---------- non-BE early-return paths ----------
//
// Every Set* hook returns nil immediately when the pod is not BE, without touching the
// response. These table-driven tests confirm each hook short-circuits for HLS/LS.

func TestNonBEPodHooksAreNoop(t *testing.T) {
	_, _, marshal := core.SetupTest(t)

	cases := []struct {
		name string
		fn   func(r *BEResourceRunner, info core.Info) error
		info core.Info
	}{
		{"SetPodCPUShares", func(r *BEResourceRunner, i core.Info) error { return r.SetPodCPUShares(i) },
			&core.PodInfo{PodRequest: core.PodRequest{PodSandboxConfig: nonBEConfig("LS", marshal)}}},
		{"SetPodCFSQuota", func(r *BEResourceRunner, i core.Info) error { return r.SetPodCFSQuota(i) },
			&core.PodInfo{PodRequest: core.PodRequest{PodSandboxConfig: nonBEConfig("HLS", marshal)}}},
		{"SetPodMemoryLimit", func(r *BEResourceRunner, i core.Info) error { return r.SetPodMemoryLimit(i) },
			&core.PodInfo{PodRequest: core.PodRequest{PodSandboxConfig: nonBEConfig("LS", marshal)}}},
		{"SetContainerCPUShares", func(r *BEResourceRunner, i core.Info) error { return r.SetContainerCPUShares(i) },
			&core.ContainerInfo{ContainerRequest: core.ContainerRequest{PodSandboxConfig: nonBEConfig("HLS", marshal)}}},
		{"SetContainerCFSQuota", func(r *BEResourceRunner, i core.Info) error { return r.SetContainerCFSQuota(i) },
			&core.ContainerInfo{ContainerRequest: core.ContainerRequest{PodSandboxConfig: nonBEConfig("LS", marshal)}}},
		{"SetContainerMemoryLimit", func(r *BEResourceRunner, i core.Info) error { return r.SetContainerMemoryLimit(i) },
			&core.ContainerInfo{ContainerRequest: core.ContainerRequest{PodSandboxConfig: nonBEConfig("HLS", marshal)}}},
	}

	r := NewBEResourceRunner()
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.NoError(t, tc.fn(r, tc.info))
		})
	}
}

func nonBEConfig(qos string, marshal []byte) utils.PodSandboxConfig {
	return createPodSandboxConfig(qos, marshal)
}

// ---------- invalid extended-resource annotation ----------
//
// A malformed ExtenderResourceConfiguration annotation makes GetExtendedResourceConfiguration
// return an error; the Set* hooks must propagate it.

func TestPodHooksPropagateConfigError(t *testing.T) {
	bad := createPodSandboxConfig("BE", []byte("{not-json"))
	r := NewBEResourceRunner()

	info := &core.PodInfo{PodRequest: core.PodRequest{PodSandboxConfig: bad}}
	assert.Error(t, r.SetPodCPUShares(info))
	assert.Error(t, r.SetPodCFSQuota(info))
	assert.Error(t, r.SetPodMemoryLimit(info))

	// SetPodResources aggregates the three sub-errors into one non-nil error.
	assert.Error(t, r.SetPodResources(&core.PodInfo{PodRequest: core.PodRequest{PodSandboxConfig: bad}}))
}

func TestContainerHooksPropagateConfigError(t *testing.T) {
	bad := createPodSandboxConfig("BE", []byte("{not-json"))
	r := NewBEResourceRunner()

	ci := &core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			ContainerMeta:     core.ContainerMetadata{Name: "container1"},
			PodSandboxConfig:  bad,
		},
	}
	assert.Error(t, r.SetContainerCPUShares(ci))
	assert.Error(t, r.SetContainerCFSQuota(ci))
	assert.Error(t, r.SetContainerMemoryLimit(ci))
	assert.Error(t, r.SetContainerResources(&core.ContainerInfo{
		ContainerRequest: core.ContainerRequest{
			ContainerMeta:    core.ContainerMetadata{Name: "container1"},
			PodSandboxConfig: bad,
		},
	}))
}

// ---------- pod hooks: limit/memory set to -1 branches ----------
//
// SetPodCFSQuota sets limit=-1 (and breaks) when a container has no Limits map or no
// ExtenderResourceCPU in Limits; the resulting quota becomes MilliCPUToQuota(-1) = -1.
// SetPodMemoryLimit does the same for memory.

// configWithMissingCPULimit builds an extended-resource cfg where the container has
// Limits only for memory (no CPU), so SetPodCFSQuota hits the -1 branch.
func configWithMissingCPULimit(t *testing.T) []byte {
	t.Helper()
	mem := resource.MustParse("1Gi")
	cfg := core.ExtendedResourceConfiguration{
		Containers: map[string]core.ExtendedResourceItem{
			"container1": {
				Requests: v1.ResourceList{common.ExtenderResourceCPU: resource.MustParse("1000")},
				Limits:   v1.ResourceList{common.ExtenderResourceMemory: mem}, // no CPU limit
			},
		},
	}
	out, err := jsonMarshal(cfg)
	assert.NoError(t, err)
	return out
}

// configWithNilLimits builds an extended-resource cfg where the container has nil
// Limits, so both SetPodCFSQuota and SetPodMemoryLimit hit their -1 branches.
func configWithNilLimits(t *testing.T) []byte {
	t.Helper()
	cfg := core.ExtendedResourceConfiguration{
		Containers: map[string]core.ExtendedResourceItem{
			"container1": {
				Requests: v1.ResourceList{common.ExtenderResourceCPU: resource.MustParse("1000")},
				Limits:   nil,
			},
		},
	}
	out, err := jsonMarshal(cfg)
	assert.NoError(t, err)
	return out
}

// configWithMissingMemLimit: Limits has CPU but no memory -> SetPodMemoryLimit -1 branch.
func configWithMissingMemLimit(t *testing.T) []byte {
	t.Helper()
	cfg := core.ExtendedResourceConfiguration{
		Containers: map[string]core.ExtendedResourceItem{
			"container1": {
				Requests: v1.ResourceList{common.ExtenderResourceCPU: resource.MustParse("1000")},
				Limits:   v1.ResourceList{common.ExtenderResourceCPU: resource.MustParse("1000")},
			},
		},
	}
	out, err := jsonMarshal(cfg)
	assert.NoError(t, err)
	return out
}

func TestSetPodCFSQuotaMissingCPULimitYieldsMinusOne(t *testing.T) {
	marshal := configWithMissingCPULimit(t)
	info := createPodInfo(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetPodCFSQuota(&info))
	assert.NotNil(t, info.PodResponese.Resources.CFSQuota)
	assert.Equal(t, int64(-1), *info.PodResponese.Resources.CFSQuota)
}

func TestSetPodCFSQuotaNilLimitsYieldsMinusOne(t *testing.T) {
	marshal := configWithNilLimits(t)
	info := createPodInfo(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetPodCFSQuota(&info))
	assert.Equal(t, int64(-1), *info.PodResponese.Resources.CFSQuota)
}

func TestSetPodMemoryLimitMissingMemLimitYieldsMinusOne(t *testing.T) {
	marshal := configWithMissingMemLimit(t)
	info := createPodInfo(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetPodMemoryLimit(&info))
	assert.NotNil(t, info.PodResponese.Resources.MemoryLimit)
	assert.Equal(t, int64(-1), *info.PodResponese.Resources.MemoryLimit)
}

func TestSetPodMemoryLimitNilLimitsYieldsMinusOne(t *testing.T) {
	marshal := configWithNilLimits(t)
	info := createPodInfo(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetPodMemoryLimit(&info))
	assert.Equal(t, int64(-1), *info.PodResponese.Resources.MemoryLimit)
}

// ---------- container hooks: nil Requests/Limits paths ----------
//
// SetContainerCPUShares/CFSQuota/MemoryLimit index into configuration.Containers[name].
// When the named container has nil Requests/Limits the request/limit stays 0 (CPU) and
// the hook still produces a result without panicking.

func containerWithNilReqsLimits(marshal []byte) core.ContainerInfo {
	return createContainerInfo(marshal)
}

func TestSetContainerCPUSharesNilRequests(t *testing.T) {
	marshal := configWithNilLimits(t) // container has nil Limits, Requests non-nil w/ CPU
	info := containerWithNilReqsLimits(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetContainerCPUShares(&info))
	// CPU request is present (1000m) -> shares should be one full CPU (1024).
	assert.NotNil(t, info.ContainerResponese.Resources.CPUShares)
}

func TestSetContainerCFSQuotaNilLimits(t *testing.T) {
	marshal := configWithNilLimits(t)
	info := containerWithNilReqsLimits(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetContainerCFSQuota(&info))
	assert.NotNil(t, info.ContainerResponese.Resources.CFSQuota)
	assert.Equal(t, int64(0), *info.ContainerResponese.Resources.CFSQuota)
}

func TestSetContainerMemoryLimitNilLimits(t *testing.T) {
	marshal := configWithNilLimits(t)
	info := containerWithNilReqsLimits(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetContainerMemoryLimit(&info))
	assert.NotNil(t, info.ContainerResponese.Resources.MemoryLimit)
	assert.Equal(t, int64(0), *info.ContainerResponese.Resources.MemoryLimit)
}

// ---------- pod hooks: container with Requests but no ExtenderResourceCPU ----------
//
// SetPodCPUShares sums ExtenderResourceCPU values; when a container has Requests but no
// CPU key, the request stays 0 and shares become MinShares (2).

func configWithRequestsNoCPU(t *testing.T) []byte {
	t.Helper()
	cfg := core.ExtendedResourceConfiguration{
		Containers: map[string]core.ExtendedResourceItem{
			"container1": {
				Requests: v1.ResourceList{common.ExtenderResourceMemory: resource.MustParse("1Gi")},
				Limits:   v1.ResourceList{common.ExtenderResourceMemory: resource.MustParse("1Gi")},
			},
		},
	}
	out, err := jsonMarshal(cfg)
	assert.NoError(t, err)
	return out
}

func TestSetPodCPUSharesRequestsWithoutCPU(t *testing.T) {
	marshal := configWithRequestsNoCPU(t)
	info := createPodInfo(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetPodCPUShares(&info))
	assert.NotNil(t, info.PodResponese.Resources.CPUShares)
	assert.Equal(t, int64(MinShares), *info.PodResponese.Resources.CPUShares)
}

func TestSetContainerCPUSharesRequestsWithoutCPU(t *testing.T) {
	marshal := configWithRequestsNoCPU(t)
	info := createContainerInfo(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetContainerCPUShares(&info))
	assert.NotNil(t, info.ContainerResponese.Resources.CPUShares)
	assert.Equal(t, int64(MinShares), *info.ContainerResponese.Resources.CPUShares)
}

// ---------- container hooks: missing memory/cpu in Limits (container-level) ----------
//
// When the container's Limits map has no ExtenderResourceCPU / ExtenderResourceMemory,
// the limit/memory stays 0.

func TestSetContainerCFSQuotaMissingCPULimit(t *testing.T) {
	marshal := configWithMissingMemLimit(t) // Limits has CPU only -> CFSQuota computed
	info := createContainerInfo(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetContainerCFSQuota(&info))
	assert.NotNil(t, info.ContainerResponese.Resources.CFSQuota)
}

func TestSetContainerMemoryLimitMissingMemLimit(t *testing.T) {
	marshal := configWithMissingCPULimit(t) // Limits has memory only -> memory computed
	info := createContainerInfo(marshal)
	r := NewBEResourceRunner()
	assert.NoError(t, r.SetContainerMemoryLimit(&info))
	assert.NotNil(t, info.ContainerResponese.Resources.MemoryLimit)
}

