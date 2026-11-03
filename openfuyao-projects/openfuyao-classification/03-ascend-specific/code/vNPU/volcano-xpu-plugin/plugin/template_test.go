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

	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

var tinfos = map[string]map[string]*TemplateInfo{
	"310P3": {
		"vir01":    {Name: "vir01", AiCore: 1, AiCPU: 1, TotalAiCore: 8, TotalAiCpu: 7},
		"vir02_1c": {Name: "vir02_1c", AiCore: 2, AiCPU: 1, TotalAiCore: 8, TotalAiCpu: 7},
		"vir02":    {Name: "vir02", AiCore: 2, AiCPU: 2, TotalAiCore: 8, TotalAiCpu: 7},
		"vir04_3c": {Name: "vir04_3c", AiCore: 4, AiCPU: 3, TotalAiCore: 8, TotalAiCpu: 7},
		"vir04":    {Name: "vir04", AiCore: 4, AiCPU: 4, TotalAiCore: 8, TotalAiCpu: 7},
	},
	"910B4": {
		"vir05_1c_8g":  {Name: "vir05_1c_8g", AiCore: 5, AiCPU: 1, TotalAiCore: 20, TotalAiCpu: 7},
		"vir10_3c_16g": {Name: "vir10_3c_16g", AiCore: 10, AiCPU: 3, TotalAiCore: 20, TotalAiCpu: 7},
	}}

func TestMapTemplate(t *testing.T) {
	tests := []struct {
		desc           string
		deviceType     string
		reqCore        int
		reqCpuLevel    string
		expectTemplate string
		expectMapRes   bool
	}{
		{"test get correct template of device", "NPU-ASCEND-310P3", 10, util.AiCpuLevelHigh, "vir01", true},
		{"test get correct template of device", "NPU-ASCEND-310P3", 15, util.AiCpuLevelHigh, "vir02", true},
		{"test get correct template of device", "NPU-ASCEND-310P3", 25, util.AiCpuLevelHigh, "vir02", true},
		{"test get correct template of device", "NPU-ASCEND-310P3", 25, util.AiCpuLevelLow, "vir02_1c", true},
		{"test get correct template of device", "NPU-ASCEND-310P3", 50, util.AiCpuLevelLow, "vir04_3c", true},
		{"test get correct template of device", "NPU-ASCEND-310P3", 60, util.AiCpuLevelHigh, "", false},
	}

	var templateInfos TemplateInfos
	templateInfos = tinfos
	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			res, template := templateInfos.mapTemplate(tt.deviceType, tt.reqCore, tt.reqCpuLevel)
			if res != tt.expectMapRes {
				t.Errorf("got %v, want %v", res, tt.expectMapRes)
			}
			if template != nil && template.Name != tt.expectTemplate {
				t.Errorf("got %s, want %s", template.Name, tt.expectTemplate)
			}
		})
	}
}

func TestGetTemplateUsedResources(t *testing.T) {
	tests := []struct {
		desc         string
		deviceType   string
		templateName string
		expectAiCore int
		expectAiCPU  int
	}{
		{"test get correct resource of template", "NPU-ASCEND-310P3", "vir01", 1, 1},
		{"test get correct resource of template", "NPU-ASCEND-310P3", "vir04", 4, 4},
		{"test get correct resource of template", "NPU-ASCEND-910B4", "vir10_3c_16g", 10, 3},
	}

	var templateInfos TemplateInfos
	templateInfos = tinfos
	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			core, cpu := templateInfos.getTemplateUsedResources(tt.deviceType, tt.templateName)
			if core != tt.expectAiCore {
				t.Errorf("got %d, want %d", core, tt.expectAiCore)
			}
			if cpu != tt.expectAiCPU {
				t.Errorf("got %d, want %d", cpu, tt.expectAiCPU)
			}
		})
	}
}

func TestGetDeviceTotalResources(t *testing.T) {
	tests := []struct {
		desc         string
		deviceType   string
		expectAiCore int
		expectAiCPU  int
	}{
		{"test get correct resource of device", "NPU-ASCEND-310P3", 8, 7},
		{"test get correct resource of device", "NPU-ASCEND-910B4", 20, 7},
	}

	var templateInfos TemplateInfos
	templateInfos = tinfos
	for _, tt := range tests {
		t.Run(tt.desc, func(t *testing.T) {
			core, cpu := templateInfos.getDeviceTotalResources(tt.deviceType)
			if core != tt.expectAiCore {
				t.Errorf("got %d, want %d", core, tt.expectAiCore)
			}
			if cpu != tt.expectAiCPU {
				t.Errorf("got %d, want %d", cpu, tt.expectAiCPU)
			}
		})
	}
}
