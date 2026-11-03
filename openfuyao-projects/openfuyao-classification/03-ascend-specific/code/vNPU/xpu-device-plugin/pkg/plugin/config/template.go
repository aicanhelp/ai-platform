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
	"sort"
	"strings"

	"huawei.com/vxpu-device-plugin/pkg/log"
)

type TemplateInfos map[string]map[string]*TemplateInfo

type TemplateInfo struct {
	Name        string
	AiCore      int
	AiCPU       int
	TotalAiCore int
	TotalAiCpu  int
}

const deviceTypePrefix = "NPU-ASCEND-"

func (tp TemplateInfos) GetTemplateCorePercent(deviceType string, templateName string) float64 {
	template := tp.getTemplateByName(deviceType, templateName)
	if template != nil {
		return float64(template.AiCore) / float64(template.TotalAiCore)
	}
	return -1
}

func (tp TemplateInfos) getTemplateByName(deviceType string, templateName string) *TemplateInfo {
	DeviceChipName := strings.TrimPrefix(deviceType, deviceTypePrefix)
	templates, ok := tp[DeviceChipName]
	if !ok {
		log.Errorf("No template found for device type %s", deviceType)
		return nil
	}

	if templateName == "" && len(templates) > 0 {
		keys := make([]string, 0, len(templates))
		for k := range templates {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		return templates[keys[0]]
	}

	t, ok := templates[templateName]
	if !ok {
		log.Errorf("template %s not found under device type %s", templateName, deviceType)
		return nil
	}
	return t
}

type Template struct {
	Name   string `yaml:"name"`
	AiCore int    `yaml:"aiCore"`
	AiCPU  int    `yaml:"aiCPU"`
}

type Vnpu struct {
	ChipName  string     `yaml:"chipName"`
	AiCore    int        `yaml:"aiCore"`
	AiCPU     int        `yaml:"aiCPU"`
	Templates []Template `yaml:"templates"`
}

type VnpuTemplateData struct {
	Vnpus []Vnpu `yaml:"vnpus"`
}
