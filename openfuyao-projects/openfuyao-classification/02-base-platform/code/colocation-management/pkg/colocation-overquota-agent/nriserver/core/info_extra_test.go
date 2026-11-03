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
package core

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/containerd/nri/pkg/api"

	"openfuyao.com/colocation-management/pkg/utils"
)

// int64Ptr returns a pointer to the given int64 value.
func int64Ptr(v int64) *int64 { return &v }

// strPtr returns a pointer to the given string value.
func strPtr(v string) *string { return &v }

func TestContainerInfoDeny(t *testing.T) {
	c := &ContainerInfo{}
	result := c.Deny()
	assert.False(t, result.Admit)
}

func TestContainerInfoUpdate(t *testing.T) {
	// Update() calls into runnermanager.GetRunner() which performs cgroup
	// file IO. In the test environment those reads/writes fail and the errors
	// are only logged, so the function must simply not panic and exercise all
	// resource branches.
	shares := int64(1024)
	quota := int64(100000)
	memLimit := int64(1024 * 1024)
	cpuSet := "0-3"
	cpuSetMem := "0"

	tests := []struct {
		name string
		resp ContainerResponse
	}{
		{
			name: "all resources set",
			resp: ContainerResponse{
				Resources: utils.LinuxContainerResources{
					CPUShares:   &shares,
					CFSQuota:    &quota,
					MemoryLimit: &memLimit,
					CPUSet:      &cpuSet,
					CPUSetMem:   &cpuSetMem,
				},
			},
		},
		{
			name: "no resources set",
			resp: ContainerResponse{
				Resources: utils.LinuxContainerResources{},
			},
		},
		{
			name: "only cpuset",
			resp: ContainerResponse{
				Resources: utils.LinuxContainerResources{
					CPUSet: &cpuSet,
				},
			},
		},
		{
			name: "only memory",
			resp: ContainerResponse{
				Resources: utils.LinuxContainerResources{
					MemoryLimit: &memLimit,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &ContainerInfo{
				ContainerRequest: ContainerRequest{
					PodSandboxConfig: utils.PodSandboxConfig{
						Linux: &utils.LinuxPodSandboxConfig{
							CgroupParent: "kubepods/test.slice",
						},
					},
				},
				ContainerResponese: tt.resp,
			}
			// should not panic
			c.Update()
		})
	}
}

func TestContainerInfoUpdateContainer(t *testing.T) {
	shares := int64(1024)
	quota := int64(100000)
	memLimit := int64(1024 * 1024)
	cpuSet := "0-3"
	cpuSetMem := "0"

	tests := []struct {
		name     string
		resp     ContainerResponse
		wantEnvs map[string]string
	}{
		{
			name: "all resources populated",
			resp: ContainerResponse{
				Resources: utils.LinuxContainerResources{
					CPUShares:   &shares,
					CFSQuota:    &quota,
					MemoryLimit: &memLimit,
					CPUSet:      &cpuSet,
					CPUSetMem:   &cpuSetMem,
				},
				AddContainerEnvs: map[string]string{
					"FOO": "bar",
					"BAZ": "qux",
				},
			},
			wantEnvs: map[string]string{"FOO": "bar", "BAZ": "qux"},
		},
		{
			name: "no resources and no envs",
			resp: ContainerResponse{
				Resources: utils.LinuxContainerResources{},
			},
			wantEnvs: map[string]string{},
		},
		{
			name: "only cpuset mem",
			resp: ContainerResponse{
				Resources: utils.LinuxContainerResources{
					CPUSetMem: &cpuSetMem,
				},
			},
			wantEnvs: map[string]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &ContainerInfo{ContainerResponese: tt.resp}
			adjust, update, err := c.UpdateContainer()
			assert.NoError(t, err)
			assert.NotNil(t, adjust)
			assert.NotNil(t, update)
		})
	}
}

func TestContainerInfoAdmitWithAnnotations(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		wantAdmit   bool
	}{
		{
			name:        "no annotations admit",
			annotations: map[string]string{},
			wantAdmit:   true,
		},
		{
			name: "offline resource without qos",
			annotations: map[string]string{
				"openfuyao.com/extender-resource-cfg": "{}",
			},
			wantAdmit: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &ContainerInfo{
				ContainerRequest: ContainerRequest{
					PodSandboxConfig: utils.PodSandboxConfig{
						Annotations: tt.annotations,
					},
				},
			}
			result := c.Admit()
			assert.Equal(t, tt.wantAdmit, result.Admit)
		})
	}
}

func TestPodInfoDeny(t *testing.T) {
	p := &PodInfo{}
	result := p.Deny()
	assert.False(t, result.Admit)
}

func TestPodInfoUpdateContainer(t *testing.T) {
	p := &PodInfo{}
	adjust, update, err := p.UpdateContainer()
	assert.NoError(t, err)
	assert.Nil(t, adjust)
	assert.Nil(t, update)
}

func TestPodInfoUpdate(t *testing.T) {
	// PodInfo.Update mirrors ContainerInfo.Update and only logs errors from
	// the cgroup runner; the goal here is to exercise every branch.
	shares := int64(1024)
	quota := int64(100000)
	memLimit := int64(1024 * 1024)
	cpuSet := "0-1"

	tests := []struct {
		name string
		resp PodResponse
	}{
		{
			name: "all resources",
			resp: PodResponse{
				Resources: utils.LinuxContainerResources{
					CPUShares:   &shares,
					CFSQuota:    &quota,
					MemoryLimit: &memLimit,
					CPUSet:      &cpuSet,
				},
			},
		},
		{
			name: "empty resources",
			resp: PodResponse{Resources: utils.LinuxContainerResources{}},
		},
		{
			name: "only quota",
			resp: PodResponse{
				Resources: utils.LinuxContainerResources{CFSQuota: &quota},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &PodInfo{
				PodRequest: PodRequest{
					PodSandboxConfig: utils.PodSandboxConfig{
						Linux: &utils.LinuxPodSandboxConfig{
							CgroupParent: "kubepods/pod-test",
						},
					},
				},
				PodResponese: tt.resp,
			}
			p.Update()
		})
	}
}

// Verify ContainerInfo / PodInfo both satisfy the Info interface.
func TestInfoInterfaceCompliance(t *testing.T) {
	var _ Info = (*ContainerInfo)(nil)
	var _ Info = (*PodInfo)(nil)
}

// TestConvertNRIRawPod2ContainerInfoNoExtender exercises the conversion path
// where the pod has no extender-resource-cfg annotation, so the resulting
// ExtendedResourceConfiguration is left nil.
func TestConvertNRIRawPod2ContainerInfoNoExtender(t *testing.T) {
	SetCgroupDriver("systemd")
	currentContainerEngines = CONTAINERD
	pod := api.PodSandbox{
		Name:      "p",
		Uid:       "u",
		Namespace: "ns",
		Linux: &api.LinuxPodSandbox{
			CgroupParent: "kubepods.slice/kubepods-pod.slice",
		},
		Annotations: map[string]string{},
		Labels:      map[string]string{},
	}
	container := api.Container{
		Name:        "c",
		Id:          "cid",
		Annotations: map[string]string{},
		Env:         []string{},
		Labels:      map[string]string{},
		Linux: &api.LinuxContainer{
			CgroupsPath: "kubepods-poduid.slice:cri-containerd:cid",
		},
	}
	c := &ContainerInfo{}
	c.ConvertNRIRawPod2ContainerInfo(&pod, &container)
	assert.Equal(t, "c", c.ContainerRequest.ContainerMeta.Name)
	assert.Equal(t, "p", c.ContainerRequest.PodMeta.Name)
	// no extender annotation -> ExtendedResourceConfiguration stays nil
	assert.Nil(t, c.ContainerRequest.ExtendedResourceConfiguration)
}

func TestConvertNRIRawPod2PodInfoNilLinux(t *testing.T) {
	pod := api.PodSandbox{
		Name:        "p",
		Uid:         "u",
		Namespace:   "ns",
		Annotations: map[string]string{},
		Labels:      map[string]string{},
		// Linux nil -> GetLinux().GetCgroupParent() returns ""
	}
	p := &PodInfo{}
	p.ConvertNRIRawPod2PodInfo(&pod)
	assert.Equal(t, "p", p.PodRequest.PodMeta.Name)
}
