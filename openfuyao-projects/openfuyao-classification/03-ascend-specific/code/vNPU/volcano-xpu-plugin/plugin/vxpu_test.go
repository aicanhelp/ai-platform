/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package plugin

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1 "k8s.io/api/core/v1"

	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

func TestIsDeviceQualified(t *testing.T) {
	tests := []struct {
		name       string
		xpuDevice  *common.XPUDevice
		val        *util.ContainerResource
		want       bool
		realMemory int
	}{
		{
			name:      "nil xpuDevices",
			xpuDevice: nil,
			val: &util.ContainerResource{
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want: false,
		},
		{
			name: "nil val",
			xpuDevice: &common.XPUDevice{
				DieID:  "device-0",
				Memory: 8192,
				Count:  4,
			},
			val:  nil,
			want: false,
		},
		{
			name: "index out of range",
			xpuDevice: &common.XPUDevice{
				DieID:    "device-0",
				Memory:   8192,
				Count:    4,
				UsedVids: 0b1111,
			},
			val: &util.ContainerResource{
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want: false,
		},
		{
			name: "device count exhausted",
			xpuDevice: &common.XPUDevice{
				DieID:    "device-0",
				Memory:   8192,
				Count:    2,
				UsedVids: 0b11,
			},
			val: &util.ContainerResource{
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want: false,
		},
		{
			name: "device count over exhausted",
			xpuDevice: &common.XPUDevice{
				DieID:    "device-0",
				Memory:   8192,
				Count:    2,
				UsedVids: 0b111,
			},
			val: &util.ContainerResource{
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want: false,
		},
		{
			name: "insufficient memory with absolute value",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Memory:     8192,
				UsedMemory: 7000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUCores: 50,
				ReqXPUMem:   2000,
			},
			want: false,
		},
		{
			name: "memory percentage calculation",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Memory:     10000,
				UsedMemory: 5000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUCores:         50,
				ReqXPUMem:           -1,
				ReqXPUMemPercentage: 60,
			},
			want: false,
		},
		{
			name: "memory percentage calculation - sufficient",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Memory:     10000,
				UsedMemory: 5000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUCores:         50,
				ReqXPUMem:           -1,
				ReqXPUMemPercentage: 40,
			},
			want:       true,
			realMemory: 4000,
		},
		{
			name: "insufficient cores",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  60,
			},
			val: &util.ContainerResource{
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want: false,
		},
		{
			name: "xpu type mismatch",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Type:       "XPU-A100",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUType:  "XPU-V100",
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want: false,
		},
		{
			name: "xpu type match",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Type:       "XPU-A100",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUType:  "XPU-A100",
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want:       true,
			realMemory: 1024,
		},
		{
			name: "dieID mismatch",
			xpuDevice: &common.XPUDevice{
				DieID:      "37E96E64-20C10A5A-8C747124-6E78030A-BF003019",
				Type:       "NPU-ASCEND-310P3",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUDieID: "37E96E64-20E1375A-4E247524-6E78030A-2F003019",
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want: false,
		},
		{
			name: "dieID match",
			xpuDevice: &common.XPUDevice{
				DieID:      "37E96E64-20C10A5A-8C747124-6E78030A-BF003019",
				Type:       "NPU-ASCEND-310P3",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUDieID: "37E96E64-20C10A5A-8C747124-6E78030A-BF003019",
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want:       true,
			realMemory: 1024,
		},
		{
			name: "dieID empty matches any",
			xpuDevice: &common.XPUDevice{
				DieID:      "37E96E64-20C10A5A-8C747124-6E78030A-BF003019",
				Type:       "NPU-ASCEND-310P3",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUDieID: "",
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want:       true,
			realMemory: 1024,
		},
		{
			name: "npu type annotation mismatch",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Type:       "NPU-ASCEND-310P3",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqUseNPUType: "NPU-ASCEND-910B",
				ReqXPUCores:   50,
				ReqXPUMem:     1024,
			},
			want: false,
		},
		{
			name: "npu type annotation match",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Type:       "NPU-ASCEND-310P3",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqUseNPUType: "NPU-ASCEND-310P3",
				ReqXPUCores:   50,
				ReqXPUMem:     1024,
			},
			want:       true,
			realMemory: 1024,
		},
		{
			name: "npu type annotation empty matches any",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Type:       "NPU-ASCEND-310P3",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqUseNPUType: "",
				ReqXPUCores:   50,
				ReqXPUMem:     1024,
			},
			want:       true,
			realMemory: 1024,
		},
		{
			name: "all resources sufficient",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Type:       "XPU-A100",
				Memory:     16384,
				UsedMemory: 2000,
				Count:      8,
				UsedVids:   0b0011,
				UsedCores:  30,
			},
			val: &util.ContainerResource{
				ReqXPUType:  "XPU-A100",
				ReqXPUCores: 50,
				ReqXPUMem:   10000,
			},
			want:       true,
			realMemory: 10000,
		},
		{
			name: "edge case - exact memory match",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Memory:     8192,
				UsedMemory: 4096,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUCores: 50,
				ReqXPUMem:   4096,
			},
			want:       true,
			realMemory: 4096,
		},
		{
			name: "edge case - exact cores match",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Memory:     8192,
				UsedMemory: 1000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  50,
			},
			val: &util.ContainerResource{
				ReqXPUCores: 50,
				ReqXPUMem:   1024,
			},
			want:       true,
			realMemory: 1024,
		},
		{
			name: "zero resource request",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Memory:     8192,
				UsedMemory: 8000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  99,
			},
			val: &util.ContainerResource{
				ReqXPUCores: 0,
				ReqXPUMem:   0,
			},
			want:       true,
			realMemory: 0,
		},
		{
			name: "multiple devices - check correct index",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-1",
				Type:       "XPU-A100",
				Memory:     16384,
				UsedMemory: 2000,
				Count:      8,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUType:  "XPU-A100",
				ReqXPUCores: 50,
				ReqXPUMem:   10000,
			},
			want:       true,
			realMemory: 10000,
		},
		{
			name: "memory percentage with zero percentage",
			xpuDevice: &common.XPUDevice{
				DieID:      "device-0",
				Memory:     10000,
				UsedMemory: 5000,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  20,
			},
			val: &util.ContainerResource{
				ReqXPUCores:         50,
				ReqXPUMem:           3000,
				ReqXPUMemPercentage: 0,
			},
			want:       true,
			realMemory: 3000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, realMem := IsDeviceQualified(tt.xpuDevice, tt.val)

			if got != tt.want {
				t.Errorf("IsDeviceQualified() = %v, want %v", got, tt.want)
			}

			if got && realMem != tt.realMemory {
				t.Errorf("realMem after calculation = %v, want %v", realMem, tt.realMemory)
			}
		})
	}
}

func TestIsDeviceQualified_MemoryPercentageCalculation(t *testing.T) {
	tests := []struct {
		name                string
		deviceMemory        uint64
		reqXPUMemPercentage int
		initialReqXPUMem    int
		expectedReqXPUMem   int
	}{
		{
			name:                "10% of 10000",
			deviceMemory:        10000,
			reqXPUMemPercentage: 10,
			initialReqXPUMem:    -1,
			expectedReqXPUMem:   1000,
		},
		{
			name:                "50% of 8192",
			deviceMemory:        8192,
			reqXPUMemPercentage: 50,
			initialReqXPUMem:    -1,
			expectedReqXPUMem:   4096,
		},
		{
			name:                "100% of 1024",
			deviceMemory:        1024,
			reqXPUMemPercentage: 100,
			initialReqXPUMem:    -1,
			expectedReqXPUMem:   1024,
		},
		{
			name:                "positive ReqXPUMem not recalculated",
			deviceMemory:        10000,
			reqXPUMemPercentage: 50,
			initialReqXPUMem:    3000,
			expectedReqXPUMem:   3000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			device := &common.XPUDevice{
				DieID:      "device-0",
				Memory:     tt.deviceMemory,
				UsedMemory: 0,
				Count:      4,
				UsedVids:   0b01,
				UsedCores:  0,
			}

			val := &util.ContainerResource{
				ReqXPUCores:         10,
				ReqXPUMem:           tt.initialReqXPUMem,
				ReqXPUMemPercentage: tt.reqXPUMemPercentage,
			}

			_, realMem := IsDeviceQualified(device, val)

			if realMem != tt.expectedReqXPUMem {
				t.Errorf("ReqXPUMem = %v, want %v", val.ReqXPUMem, tt.expectedReqXPUMem)
			}
		})
	}
}

func BenchmarkIsDeviceQualified_Pass(b *testing.B) {
	device := &common.XPUDevice{
		DieID:      "device-0",
		Type:       "XPU-A100",
		Memory:     16384,
		UsedMemory: 2000,
		Count:      8,
		UsedVids:   0b01,
		UsedCores:  20,
	}

	val := &util.ContainerResource{
		ReqXPUType:  "XPU-A100",
		ReqXPUCores: 50,
		ReqXPUMem:   10000,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		IsDeviceQualified(device, val)
	}
}

func BenchmarkIsDeviceQualified_Fail(b *testing.B) {
	device := &common.XPUDevice{
		DieID:      "device-0",
		Memory:     8192,
		UsedMemory: 7000,
		Count:      2,
		UsedVids:   0b11,
		UsedCores:  60,
	}

	val := &util.ContainerResource{
		ReqXPUCores: 50,
		ReqXPUMem:   2000,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		IsDeviceQualified(device, val)
	}
}

func vxpuTestAbsFloat(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func vxpuTestBuildPod(ns, name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:   ns,
			Name:        name,
			Annotations: map[string]string{},
		},
	}
}

func TestCalculateDeviceScore_SoftMode(t *testing.T) {
	tests := []struct {
		name          string
		device        *common.XPUDevice
		val           *util.ContainerResource
		realReqXPUMem int
		expected      float64
	}{
		{
			name: "empty device, low request",
			device: &common.XPUDevice{
				Cores: 100, UsedCores: 0, Memory: 10000, UsedMemory: 0,
			},
			val:           &util.ContainerResource{ReqXPUCores: 50},
			realReqXPUMem: 5000,
			expected:      100.0 * (50.0/100.0 + 5000.0/10000.0),
		},
		{
			name: "partially used device",
			device: &common.XPUDevice{
				Cores: 100, UsedCores: 25, Memory: 10000, UsedMemory: 2500,
			},
			val:           &util.ContainerResource{ReqXPUCores: 25},
			realReqXPUMem: 2500,
			expected:      100.0 * ((25+25.0)/100.0 + (2500+2500.0)/10000.0),
		},
		{
			name: "zero request",
			device: &common.XPUDevice{
				Cores: 100, UsedCores: 0, Memory: 10000, UsedMemory: 0,
			},
			val:           &util.ContainerResource{ReqXPUCores: 0},
			realReqXPUMem: 0,
			expected:      0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateDeviceScore(tt.device, tt.val, tt.realReqXPUMem, tt.device.Cores, false)
			if diff := vxpuTestAbsFloat(result - tt.expected); diff > 0.001 {
				t.Errorf("calculateDeviceScore(soft) = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCalculateDeviceScore_HardMode(t *testing.T) {
	tests := []struct {
		name       string
		device     *common.XPUDevice
		val        *util.ContainerResource
		totalCores int
		expected   float64
	}{
		{
			name:       "empty device, low request",
			device:     &common.XPUDevice{Cores: 100, UsedCores: 0},
			val:        &util.ContainerResource{ReqXPUCores: 25},
			totalCores: 20,
			expected:   100.0 * (25 / 20.0),
		},
		{
			name:       "partially used device",
			device:     &common.XPUDevice{Cores: 100, UsedCores: 5},
			val:        &util.ContainerResource{ReqXPUCores: 25},
			totalCores: 20,
			expected:   100.0 * ((25 + 5.0) / 20.0),
		},
		{
			name:       "full card request",
			device:     &common.XPUDevice{Cores: 100, UsedCores: 0},
			val:        &util.ContainerResource{ReqXPUCores: util.FullCore},
			totalCores: 20,
			expected:   100.0 * (util.FullCore / 20.0),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateDeviceScore(tt.device, tt.val, 0, tt.totalCores, true)
			if diff := vxpuTestAbsFloat(result - tt.expected); diff > 0.001 {
				t.Errorf("calculateDeviceScore(hard) = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestCheckSoftModeDevice_NoMutate(t *testing.T) {
	device := &common.XPUDevice{
		DieID: "dev-0", Cores: 100, Memory: 10000, Count: 4,
		UsedVids: 0b01, UsedCores: 10, UsedMemory: 1000,
	}
	val := &util.ContainerResource{ReqXPUCores: 50, ReqXPUMem: 5000}
	origCores, origMem, origVids := device.UsedCores, device.UsedMemory, device.UsedVids

	fits, realMem, score := checkSoftModeDevice(device, val)
	if !fits {
		t.Fatalf("expected fit")
	}
	if realMem != 5000 {
		t.Errorf("realMem = %v, want 5000", realMem)
	}
	if score <= 0 {
		t.Errorf("score = %v, expected > 0", score)
	}
	if device.UsedCores != origCores || device.UsedMemory != origMem || device.UsedVids != origVids {
		t.Error("checkSoftModeDevice mutated device state")
	}
}

func TestApplySoftModeDevice_Mutates(t *testing.T) {
	device := &common.XPUDevice{
		PhysicID: 0, DieID: "dev-0", Cores: 100, Memory: 10000,
		Count: 4, UsedVids: 0b01, UsedCores: 10, UsedMemory: 1000,
	}
	val := &util.ContainerResource{ReqXPUCores: 50}
	cdev := applySoftModeDevice(device, val, 5000)
	if cdev == nil {
		t.Fatal("returned nil")
	}
	if device.UsedCores != 60 || device.UsedMemory != 6000 {
		t.Errorf("device state wrong: cores=%d mem=%d", device.UsedCores, device.UsedMemory)
	}
}

func TestCheckHardModeDevice_FullCard(t *testing.T) {
	templates := TemplateInfos{
		"910B": {"vir05": {Name: "vir05", AiCore: 5, AiCPU: 1, TotalAiCore: 20, TotalAiCpu: 7}},
	}

	t.Run("empty device fits full card", func(t *testing.T) {
		device := &common.XPUDevice{
			DieID: "dev-0", Type: "NPU-ASCEND-910B", Cores: 100, UsedCores: 0,
		}
		val := &util.ContainerResource{ReqXPUCores: util.FullCore}
		res := checkHardModeDevice(device, val, templates)
		if !res.fits || !res.isFullCard || res.template.Name != util.FullCardTemplate || res.score != float64(scoreWeight) {
			t.Errorf("fits=%v isFullCard=%v template=%v score=%v", res.fits, res.isFullCard, res.template.Name, res.score)
		}
		if device.UsedCores != 0 {
			t.Error("checkHardModeDevice mutated device")
		}
	})

	t.Run("used device rejects full card", func(t *testing.T) {
		device := &common.XPUDevice{
			DieID: "dev-1", Type: "NPU-ASCEND-910B", Cores: 100, UsedCores: 5,
		}
		val := &util.ContainerResource{ReqXPUCores: util.FullCore}
		res := checkHardModeDevice(device, val, templates)
		if res.fits {
			t.Error("expected rejection")
		}
		if !res.isFullCard {
			t.Error("isFullCard should be true")
		}
	})
}

func TestCheckHardModeDevice_Template(t *testing.T) {
	templates := TemplateInfos{
		"910B": {"vir05": {Name: "vir05", AiCore: 5, AiCPU: 1, TotalAiCore: 20, TotalAiCpu: 7}},
	}
	device := &common.XPUDevice{
		DieID: "dev-0", Type: "NPU-ASCEND-910B", Cores: 100, UsedCores: 0,
	}
	val := &util.ContainerResource{ReqXPUCores: 20, ReqCpuLevel: util.AiCpuLevelLow}
	res := checkHardModeDevice(device, val, templates)
	if !res.fits {
		t.Fatal("expected fit")
	}
	if res.template == nil {
		t.Fatal("template nil")
	}
	if res.score <= 0 {
		t.Errorf("score = %v, want > 0", res.score)
	}
	if res.isFullCard {
		t.Error("isFullCard should be false")
	}
}

func TestApplyHardModeDevice_Mutates(t *testing.T) {
	device := &common.XPUDevice{PhysicID: 0, DieID: "dev-0", UsedCores: 0}
	template := &TemplateInfo{Name: "vir05", AiCore: 5, AiCPU: 1}
	cdev := applyHardModeDevice(device, template)
	if cdev == nil {
		t.Fatal("returned nil")
	}
	if device.UsedCores != 5 {
		t.Errorf("UsedCores = %d, want 5", device.UsedCores)
	}
}

func TestAllocateContainerDevices_BinpackSoft(t *testing.T) {
	devices := []*common.XPUDevice{
		{PhysicID: 0, DieID: "d0", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 10, UsedMemory: 1000},
		{PhysicID: 1, DieID: "d1", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 60, UsedMemory: 6000},
		{PhysicID: 2, DieID: "d2", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 30, UsedMemory: 3000},
	}
	req := &util.ContainerResource{ReqXPUNum: 1, ReqXPUCores: 10, ReqXPUMem: 500}
	result, _ := AllocateContainerDevices(devices, req, ScheduleConfig{Policy: util.SchedulerPolicyBinpack, PodMode: util.SoftMode})
	if len(result) != 1 || result[0].Id != "d1" {
		t.Errorf("binpack soft: want d1, got %v", result)
	}
}

func TestAllocateContainerDevices_SpreadSoft(t *testing.T) {
	devices := []*common.XPUDevice{
		{PhysicID: 0, DieID: "d0", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 10, UsedMemory: 1000},
		{PhysicID: 1, DieID: "d1", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 60, UsedMemory: 6000},
		{PhysicID: 2, DieID: "d2", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 30, UsedMemory: 3000},
	}
	req := &util.ContainerResource{ReqXPUNum: 1, ReqXPUCores: 10, ReqXPUMem: 500}
	result, _ := AllocateContainerDevices(devices, req, ScheduleConfig{Policy: util.SchedulerPolicySpread, PodMode: util.SoftMode})
	if len(result) != 1 || result[0].Id != "d0" {
		t.Errorf("spread soft: want d0, got %v", result)
	}
}

func TestAllocateContainerDevices_BinpackHard(t *testing.T) {
	templates := TemplateInfos{
		"910B": {"vir05": {Name: "vir05", AiCore: 5, AiCPU: 1, TotalAiCore: 20, TotalAiCpu: 7}},
	}
	devices := []*common.XPUDevice{
		{PhysicID: 0, DieID: "d0", Type: "NPU-ASCEND-910B", Cores: 100, Count: 4, UsedVids: 0b001, UsedCores: 0, UsedCpu: 0},
		{PhysicID: 1, DieID: "d1", Type: "NPU-ASCEND-910B", Cores: 100, Count: 4, UsedVids: 0b001, UsedCores: 5, UsedCpu: 5},
		{PhysicID: 2, DieID: "d2", Type: "NPU-ASCEND-910B", Cores: 100, Count: 4, UsedVids: 0b001, UsedCores: 2, UsedCpu: 2},
	}
	req := &util.ContainerResource{ReqXPUNum: 1, ReqXPUCores: 20, ReqCpuLevel: util.AiCpuLevelLow}
	result, _ := AllocateContainerDevices(devices, req, ScheduleConfig{Templates: templates, Policy: util.SchedulerPolicyBinpack, PodMode: util.HardMode})
	if len(result) != 1 || result[0].Id != "d1" {
		t.Errorf("binpack hard: want d1, got %v", result)
	}
}

func TestAllocateContainerDevices_SpreadHard(t *testing.T) {
	templates := TemplateInfos{
		"910B": {"vir05": {Name: "vir05", AiCore: 5, AiCPU: 1, TotalAiCore: 20, TotalAiCpu: 7}},
	}
	devices := []*common.XPUDevice{
		{PhysicID: 0, DieID: "d0", Type: "NPU-ASCEND-910B", Cores: 100, Count: 4, UsedVids: 0b001, UsedCores: 0, UsedCpu: 0},
		{PhysicID: 1, DieID: "d1", Type: "NPU-ASCEND-910B", Cores: 100, Count: 4, UsedVids: 0b001, UsedCores: 5, UsedCpu: 5},
		{PhysicID: 2, DieID: "d2", Type: "NPU-ASCEND-910B", Cores: 100, Count: 4, UsedVids: 0b001, UsedCores: 2, UsedCpu: 2},
	}
	req := &util.ContainerResource{ReqXPUNum: 1, ReqXPUCores: 20, ReqCpuLevel: util.AiCpuLevelLow}
	result, _ := AllocateContainerDevices(devices, req, ScheduleConfig{Templates: templates, Policy: util.SchedulerPolicySpread, PodMode: util.HardMode})
	if len(result) != 1 || result[0].Id != "d0" {
		t.Errorf("spread hard: want d0, got %v", result)
	}
}

func TestAllocateContainerDevices_MultipleDevices(t *testing.T) {
	devices := []*common.XPUDevice{
		{PhysicID: 0, DieID: "d0", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 10, UsedMemory: 1000},
		{PhysicID: 1, DieID: "d1", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 60, UsedMemory: 6000},
		{PhysicID: 2, DieID: "d2", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b001, UsedCores: 30, UsedMemory: 3000},
	}
	req := &util.ContainerResource{ReqXPUNum: 2, ReqXPUCores: 5, ReqXPUMem: 100}
	result, _ := AllocateContainerDevices(devices, req, ScheduleConfig{Policy: util.SchedulerPolicySpread, PodMode: util.SoftMode})
	if len(result) != 2 {
		t.Fatalf("expected 2 devices, got %d", len(result))
	}
	if req.ReqXPUNum != 0 {
		t.Errorf("ReqXPUNum = %d, want 0 after allocation", req.ReqXPUNum)
	}
}

func TestAllocateContainerDevices_NoFit(t *testing.T) {
	devices := []*common.XPUDevice{
		{PhysicID: 0, DieID: "d0", Cores: 100, Memory: 10000, Count: 4, UsedVids: 0b1111, UsedCores: 99, UsedMemory: 9999},
	}
	req := &util.ContainerResource{ReqXPUNum: 1, ReqXPUCores: 50, ReqXPUMem: 5000}
	result, _ := AllocateContainerDevices(devices, req, ScheduleConfig{Policy: util.SchedulerPolicyBinpack, PodMode: util.SoftMode})
	if len(result) != 0 {
		t.Errorf("expected 0 devices, got %d", len(result))
	}
}

func TestIsVnpuModeMatch_DefaultsToSoft(t *testing.T) {
	tests := []struct {
		name      string
		annotVal  string
		annotSet  bool
		modes     []string
		expectErr bool
	}{
		{"unannotated pod with soft devices", "", false, []string{util.SoftMode, util.SoftMode}, false},
		{"unannotated pod with hard devices (should fail)", "", false, []string{util.HardMode}, true},
		{"hard pod matches hard devices", util.HardMode, true, []string{util.HardMode}, false},
		{"hard pod with soft devices (should fail)", util.HardMode, true, []string{util.SoftMode}, true},
		{"soft pod matches soft devices", util.SoftMode, true, []string{util.SoftMode}, false},
		{"soft pod with mixed devices (should fail)", util.SoftMode, true, []string{util.SoftMode, util.HardMode}, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var devices []*common.XPUDevice
			for _, m := range tt.modes {
				devices = append(devices, &common.XPUDevice{Mode: m})
			}
			pod := vxpuTestBuildPod("default", "test-pod")
			if tt.annotSet {
				pod.Annotations = map[string]string{util.VNPUModeAnnotation: tt.annotVal}
			}
			err := isVnpuModeMatch(pod, devices)
			if (err != nil) != tt.expectErr {
				t.Errorf("isVnpuModeMatch() error = %v, wantErr %v", err, tt.expectErr)
			}
		})
	}
}
