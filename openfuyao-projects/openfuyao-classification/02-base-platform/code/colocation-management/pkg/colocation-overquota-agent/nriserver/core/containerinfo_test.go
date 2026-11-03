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

package core

import (
	"reflect"
	"testing"

	"github.com/containerd/nri/pkg/api"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	"openfuyao.com/colocation-management/pkg/common"
	"openfuyao.com/colocation-management/pkg/utils"
)

func TestConvertNRIRawPod2ContainerInfo(t *testing.T) {
	request, _, marshal := SetupTest(t)
	SetCgroupDriver("systemd")

	pod := createTestPod(marshal)
	container := api.Container{
		Name:        "container1",
		Id:          "66ff3a44a254533880e6b50e8fb52e0311c9158eb66426ae244066a4f11b26e5",
		Annotations: map[string]string{},
		Env:         []string{},
		Labels:      map[string]string{},
		Linux: &api.LinuxContainer{
			CgroupsPath: "kubepods-burstable-podbb29f378_0c50_4da4_b070_be919e350db2.slice:" +
				"cri-containerd:66ff3a44a254533880e6b50e8fb52e0311c9158eb66426ae244066a4f11b26e5",
		},
	}
	info := ContainerInfo{}
	info.ConvertNRIRawPod2ContainerInfo(&pod, &container)

	expectinfo := createTestContainerInfo(request, marshal)

	if !resourceListsEqual(info.ContainerRequest.ExtendedResourceConfiguration.Requests,
		expectinfo.ContainerRequest.ExtendedResourceConfiguration.Requests) {
		t.Errorf("Expected  to be %v, got %v", expectinfo.ContainerRequest.ExtendedResourceConfiguration,
			info.ContainerRequest.ExtendedResourceConfiguration)
	}

	if !reflect.DeepEqual(info.ContainerRequest.PodMeta, expectinfo.ContainerRequest.PodMeta) {
		t.Errorf("Expected  to be %v, got %v", expectinfo.ContainerRequest.PodMeta,
			info.ContainerRequest.PodMeta)
	}

	if !reflect.DeepEqual(info.ContainerRequest.ContainerMeta, expectinfo.ContainerRequest.ContainerMeta) {
		t.Errorf("Expected  to be %v, got %v", expectinfo.ContainerRequest.ContainerMeta,
			info.ContainerRequest.ContainerMeta)
	}

	if !reflect.DeepEqual(info.ContainerRequest.ContainerConfig, expectinfo.ContainerRequest.ContainerConfig) {
		t.Errorf("Expected  to be %v, got %v", expectinfo.ContainerRequest.ContainerConfig,
			info.ContainerRequest.ContainerConfig)
	}

	if !reflect.DeepEqual(info.ContainerRequest.PodSandboxConfig.Linux.CgroupParent,
		expectinfo.ContainerRequest.PodSandboxConfig.Linux.CgroupParent) {
		t.Errorf("Expected  to be %v, got %v", expectinfo.ContainerRequest.PodSandboxConfig.Linux.CgroupParent,
			info.ContainerRequest.PodSandboxConfig.Linux.CgroupParent)
	}

	if !reflect.DeepEqual(info.ContainerRequest.PodSandboxConfig.Linux.Resources,
		expectinfo.ContainerRequest.PodSandboxConfig.Linux.Resources) {
		t.Errorf("Expected  to be %v, got %v", expectinfo.ContainerRequest.PodSandboxConfig.Linux.Resources,
			info.ContainerRequest.PodSandboxConfig.Linux.Resources)
	}
}

func TestAdmit(t *testing.T) {
	request, _, marshal := SetupTest(t)

	expectinfo := createTestContainerInfo(request, marshal)

	if !reflect.DeepEqual(expectinfo.Admit().Reason, "offline resource pod  must have BE QosLevel annotation") {
		t.Errorf("Expected  to be %v, got %v",
			"offline resource pod  must have BE QosLevel annotation",
			expectinfo.Admit().Reason,
		)
	}

	expectinfo.ContainerRequest.PodSandboxConfig.Annotations[common.QosClassAnnotationKey] = string(common.QosBe)

	if !reflect.DeepEqual(expectinfo.Admit().Reason,
		"offline resource pod  must have volcano.sh/preemptable annotation") {
		t.Errorf("Expected  to be %v, got %v",
			"offline resource pod  must have volcano.sh/preemptable annotation",
			expectinfo.Admit().Reason,
		)
	}

	expectinfo.ContainerRequest.PodSandboxConfig.Annotations[common.PriorityAnnotationKey] = "true"
	if !reflect.DeepEqual(expectinfo.Admit().Admit, true) {
		t.Errorf("Expected  to be %v, got %v", true, expectinfo.Admit().Admit)
	}

}

func createTestPod(marshal []byte) api.PodSandbox {
	pod := api.PodSandbox{
		Name:      "pod1",
		Uid:       "pod1",
		Namespace: "default",
		Linux: &api.LinuxPodSandbox{
			CgroupParent: "kubepods.slice/kubepods-podxxxxxx.slice",
		},
		Annotations: map[string]string{
			common.ExtenderResourceConfigurationAnnotation: string(marshal),
		},
		Labels: map[string]string{},
	}
	return pod
}

func createTestContainerInfo(request map[v1.ResourceName]resource.Quantity, marshal []byte) ContainerInfo {
	extenderResource := ExtendedResourceItem{
		Requests: request,
	}
	CgroupParentStr := "kubepods.slice/kubepods-burstable.slice/" +
		"kubepods-burstable-podbb29f378_0c50_4da4_b070_be919e350db2.slice/" +
		"cri-containerd-66ff3a44a254533880e6b50e8fb52e0311c9158eb66426ae244066a4f11b26e5.scope"
	expectinfo := ContainerInfo{
		ContainerRequest: ContainerRequest{
			PodMeta: PodSandboxMetadata{
				Name:      "pod1",
				UID:       "pod1",
				Namespace: "default",
			},
			ContainerMeta: ContainerMetadata{
				Name: "container1",
				ID:   "66ff3a44a254533880e6b50e8fb52e0311c9158eb66426ae244066a4f11b26e5",
			},
			PodSandboxConfig: utils.PodSandboxConfig{
				Annotations: map[string]string{
					common.ExtenderResourceConfigurationAnnotation: string(marshal),
				},
				Labels: map[string]string{},
				Linux: &utils.LinuxPodSandboxConfig{
					CgroupParent: CgroupParentStr,
				},
			},
			ContainerConfig: utils.ContainerConfig{
				map[string]string{},
				map[string]string{},
				map[string]string{},
			},
			ExtendedResourceConfiguration: &extenderResource,
		},
		ContainerResponese: ContainerResponse{},
	}
	return expectinfo
}

func resourceListsEqual(expected, actual v1.ResourceList) bool {
	if len(expected) != len(actual) {
		return false
	}

	for resourceName, expectedQuantity := range expected {
		actualQuantity, exists := actual[resourceName]
		if !exists {
			return false
		}

		if expectedQuantity.String() != actualQuantity.String() {
			return false
		}
	}

	return true
}

func TestCalculateCPUMilliRequest(t *testing.T) {
	testCases := []struct {
		name      string
		container *api.Container
		expected  int64
	}{
		{
			name:      "nil container",
			container: nil,
			expected:  DefaultCPURequestMilliCores,
		},
		{
			name:      "no linux config",
			container: &api.Container{Id: "test-container-123"},
			expected:  DefaultCPURequestMilliCores,
		},
		{
			name: "no resources config",
			container: &api.Container{
				Id:    "test-container-123",
				Linux: &api.LinuxContainer{},
			},
			expected: DefaultCPURequestMilliCores,
		},
		{
			name: "no cpu config",
			container: &api.Container{
				Id: "test-container-123",
				Linux: &api.LinuxContainer{
					Resources: &api.LinuxResources{},
				},
			},
			expected: DefaultCPURequestMilliCores,
		},
		{
			name: "valid quota and period",
			container: &api.Container{
				Id: "test-container-123",
				Linux: &api.LinuxContainer{
					Resources: &api.LinuxResources{
						Cpu: &api.LinuxCPU{
							Quota:  &api.OptionalInt64{Value: 100000},
							Period: &api.OptionalUInt64{Value: 100000},
						},
					},
				},
			},
			expected: DefaultCPURequestMilliCores, // 100000/100000 * 1000 = 1000
		},
		{
			name: "valid shares",
			container: &api.Container{
				Id: "test-container-123",
				Linux: &api.LinuxContainer{
					Resources: &api.LinuxResources{
						Cpu: &api.LinuxCPU{
							Shares: &api.OptionalUInt64{Value: 1024},
						},
					},
				},
			},
			expected: DefaultCPURequestMilliCores, // 1024/1024 * 1000 = 1000
		},
		{
			name: "quota with zero period",
			container: &api.Container{
				Id: "test-container-123",
				Linux: &api.LinuxContainer{
					Resources: &api.LinuxResources{
						Cpu: &api.LinuxCPU{
							Quota:  &api.OptionalInt64{Value: 100000},
							Period: &api.OptionalUInt64{Value: 0},
						},
					},
				},
			},
			expected: DefaultCPURequestMilliCores,
		},
		{
			name: "zero shares",
			container: &api.Container{
				Id: "test-container-123",
				Linux: &api.LinuxContainer{
					Resources: &api.LinuxResources{
						Cpu: &api.LinuxCPU{
							Shares: &api.OptionalUInt64{Value: 0},
						},
					},
				},
			},
			expected: DefaultCPURequestMilliCores,
		},
	}

	containerInfo := &ContainerInfo{}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := containerInfo.calculateCPUMilliRequest(tc.container)
			if result != tc.expected {
				t.Errorf("Expected %d, got %d", tc.expected, result)
			}
		})
	}
}

func TestHasLinuxCPUResources(t *testing.T) {
	containerInfo := &ContainerInfo{}

	tests := []struct {
		name      string
		container *api.Container
		expected  bool
	}{
		{
			name:      "nil container",
			container: nil,
			expected:  false,
		},
		{
			name:      "no linux config",
			container: &api.Container{},
			expected:  false,
		},
		{
			name: "no resources config",
			container: &api.Container{
				Linux: &api.LinuxContainer{},
			},
			expected: false,
		},
		{
			name: "no cpu config",
			container: &api.Container{
				Linux: &api.LinuxContainer{
					Resources: &api.LinuxResources{},
				},
			},
			expected: false,
		},
		{
			name: "valid cpu config",
			container: &api.Container{
				Linux: &api.LinuxContainer{
					Resources: &api.LinuxResources{
						Cpu: &api.LinuxCPU{},
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := containerInfo.hasLinuxCPUResources(tt.container)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestCalculateFromQuotaPeriod(t *testing.T) {
	containerInfo := &ContainerInfo{}

	tests := []struct {
		name         string
		cpuResources *api.LinuxCPU
		expected     int64
	}{
		{
			name:         "nil cpu resources",
			cpuResources: nil,
			expected:     0,
		},
		{
			name:         "nil quota",
			cpuResources: &api.LinuxCPU{},
			expected:     0,
		},
		{
			name: "nil period",
			cpuResources: &api.LinuxCPU{
				Quota: &api.OptionalInt64{Value: 100000},
			},
			expected: 0,
		},
		{
			name: "zero period",
			cpuResources: &api.LinuxCPU{
				Quota:  &api.OptionalInt64{Value: 100000},
				Period: &api.OptionalUInt64{Value: 0},
			},
			expected: 0,
		},
		{
			name: "valid calculation",
			cpuResources: &api.LinuxCPU{
				Quota:  &api.OptionalInt64{Value: 50000},
				Period: &api.OptionalUInt64{Value: 100000},
			},
			expected: 500, // (50000/100000) * 1000 = 500
		},
		{
			name: "negative quota",
			cpuResources: &api.LinuxCPU{
				Quota:  &api.OptionalInt64{Value: -100000},
				Period: &api.OptionalUInt64{Value: 100000},
			},
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := containerInfo.calculateFromQuotaPeriod("test-container-123", tt.cpuResources)
			if result != tt.expected {
				t.Errorf("Expected %d, got %d", tt.expected, result)
			}
		})
	}
}

func TestCalculateFromShares(t *testing.T) {
	containerInfo := &ContainerInfo{}

	tests := []struct {
		name         string
		cpuResources *api.LinuxCPU
		expected     int64
	}{
		{
			name:         "nil cpu resources",
			cpuResources: nil,
			expected:     0,
		},
		{
			name:         "nil shares",
			cpuResources: &api.LinuxCPU{},
			expected:     0,
		},
		{
			name: "zero shares",
			cpuResources: &api.LinuxCPU{
				Shares: &api.OptionalUInt64{Value: 0},
			},
			expected: 0,
		},
		{
			name: "valid calculation",
			cpuResources: &api.LinuxCPU{
				Shares: &api.OptionalUInt64{Value: 512},
			},
			expected: 500,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := containerInfo.calculateFromShares("test-container-123", tt.cpuResources)
			if result != tt.expected {
				t.Errorf("Expected %d, got %d", tt.expected, result)
			}
		})
	}
}
