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

package config

import (
	"testing"
)

func TestGetTemplateCorePercent(t *testing.T) {
	templateTest := TemplateInfos{
		"910": {
			"vir01": &TemplateInfo{
				Name:        "vir01",
				AiCore:      16,
				AiCPU:       4,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
			"vir02": &TemplateInfo{
				Name:        "vir02",
				AiCore:      8,
				AiCPU:       2,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
			"full": &TemplateInfo{
				Name:        "full",
				AiCore:      32,
				AiCPU:       8,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
		},
		"910B": {
			"vir01": &TemplateInfo{
				Name:        "vir01",
				AiCore:      24,
				AiCPU:       6,
				TotalAiCore: 48,
				TotalAiCpu:  12,
			},
		},
	}

	testCases := []struct {
		name         string
		deviceType   string
		templateName string
		expected     float64
	}{
		{
			name:         "normal device type and template",
			deviceType:   "NPU-ASCEND-910",
			templateName: "vir01",
			expected:     0.5,
		},
		{
			name:         "another template",
			deviceType:   "NPU-ASCEND-910",
			templateName: "vir02",
			expected:     0.25,
		},
		{
			name:         "full card template",
			deviceType:   "NPU-ASCEND-910",
			templateName: "full",
			expected:     1.0,
		},
		{
			name:         "different chip type",
			deviceType:   "NPU-ASCEND-910B",
			templateName: "vir01",
			expected:     0.5,
		},
		{
			name:         "device type without prefix",
			deviceType:   "910",
			templateName: "vir01",
			expected:     0.5,
		},
		{
			name:         "device type not found",
			deviceType:   "NPU-ASCEND-999",
			templateName: "vir01",
			expected:     -1,
		},
		{
			name:         "template not found",
			deviceType:   "NPU-ASCEND-910",
			templateName: "nonexistent",
			expected:     -1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := templateTest.GetTemplateCorePercent(tc.deviceType, tc.templateName)
			if result == tc.expected {
				t.Log("test succeed")
			} else {
				t.Errorf("test failed: expected %f, got %f", tc.expected, result)
			}
		})
	}
}

func TestGetTemplateByName(t *testing.T) {
	templateTest := TemplateInfos{
		"910": {
			"vir01": &TemplateInfo{
				Name:        "vir01",
				AiCore:      16,
				AiCPU:       4,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
			"vir02": &TemplateInfo{
				Name:        "vir02",
				AiCore:      8,
				AiCPU:       2,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
		},
	}

	testCases := []struct {
		name         string
		deviceType   string
		templateName string
		expected     *TemplateInfo
	}{
		{
			name:         "normal device type and template",
			deviceType:   "NPU-ASCEND-910",
			templateName: "vir01",
			expected: &TemplateInfo{
				Name:        "vir01",
				AiCore:      16,
				AiCPU:       4,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
		},
		{
			name:         "another template",
			deviceType:   "NPU-ASCEND-910",
			templateName: "vir02",
			expected: &TemplateInfo{
				Name:        "vir02",
				AiCore:      8,
				AiCPU:       2,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
		},
		{
			name:         "device type without prefix",
			deviceType:   "910",
			templateName: "vir01",
			expected: &TemplateInfo{
				Name:        "vir01",
				AiCore:      16,
				AiCPU:       4,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
		},
		{
			name:         "empty template name returns first",
			deviceType:   "NPU-ASCEND-910",
			templateName: "",
			expected: &TemplateInfo{
				Name:        "vir01",
				AiCore:      16,
				AiCPU:       4,
				TotalAiCore: 32,
				TotalAiCpu:  8,
			},
		},
		{
			name:         "device type not found",
			deviceType:   "NPU-ASCEND-999",
			templateName: "vir01",
			expected:     nil,
		},
		{
			name:         "template not found",
			deviceType:   "NPU-ASCEND-910",
			templateName: "nonexistent",
			expected:     nil,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := templateTest.getTemplateByName(tc.deviceType, tc.templateName)
			if tc.expected == nil {
				if result == nil {
					t.Log("test succeed")
				} else {
					t.Errorf("test failed: expected nil, got %v", result)
				}
			} else {
				if result != nil && result.Name == tc.expected.Name && result.AiCore == tc.expected.AiCore {
					t.Log("test succeed")
				} else {
					t.Errorf("test failed: expected %v, got %v", tc.expected, result)
				}
			}
		})
	}
}
