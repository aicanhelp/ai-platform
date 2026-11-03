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
	"sort"
	"strings"

	"k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

type TemplateInfos map[string]map[string]*TemplateInfo

type TemplateInfo struct {
	Name        string
	AiCore      int
	AiCPU       int
	TotalAiCore int
	TotalAiCpu  int
}

func (tp TemplateInfos) getTemplateUsedResources(deviceType string, templateName string) (aiCore, aiCpu int) {
	template := tp.getTemplateByName(deviceType, templateName)
	if template != nil {
		return template.AiCore, template.AiCPU
	}
	return -1, -1
}

func (tp TemplateInfos) getDeviceTotalResources(deviceType string) (aiCore, aiCpu int) {
	template := tp.getTemplateByName(deviceType, "")
	if template != nil {
		return template.TotalAiCore, template.TotalAiCpu
	}
	return -1, -1
}

func (tp TemplateInfos) getTemplateByName(deviceType string, templateName string) *TemplateInfo {
	DeviceChipName := strings.TrimPrefix(deviceType, util.DeviceType)
	templates, ok := tp[DeviceChipName]
	if !ok {
		klog.Errorf("No template found for device type %s", deviceType)
		return nil
	}

	if templateName == "" {
		sorted := tp.getSortedTemplateByDevice(deviceType)
		if len(sorted) > 0 {
			return sorted[0]
		}
		return nil
	}

	t, ok := templates[templateName]
	if !ok {
		klog.Errorf("template %s not found under device type %s", templateName, deviceType)
		return nil
	}
	return t
}

func (tp TemplateInfos) getSortedTemplateByDevice(deviceType string) []*TemplateInfo {
	DeviceChipName := strings.TrimPrefix(deviceType, util.DeviceType)
	templates, ok := tp[DeviceChipName]
	if !ok {
		klog.Errorf("No template found for device type %s", deviceType)
		return nil
	}

	var templateSlice []*TemplateInfo
	for _, t := range templates {
		templateSlice = append(templateSlice, t)
	}

	sort.Slice(templateSlice, func(i, j int) bool {
		if templateSlice[i].AiCore != templateSlice[j].AiCore {
			return templateSlice[i].AiCore < templateSlice[j].AiCore
		}
		return templateSlice[i].AiCPU < templateSlice[j].AiCPU
	})

	return templateSlice
}

func (tp TemplateInfos) mapTemplate(deviceType string, reqCore int, reqCpuLevel string) (bool, *TemplateInfo) {
	var minCore int
	var matchTemplates []*TemplateInfo
	templateSlice := tp.getSortedTemplateByDevice(deviceType)
	for _, t := range templateSlice {
		if minCore > 0 {
			if t.AiCore <= minCore {
				matchTemplates = append(matchTemplates, t)
				continue
			} else {
				break
			}
		}
		if t.AiCore*util.Base100 >= t.TotalAiCore*reqCore {
			minCore = t.AiCore
			matchTemplates = append(matchTemplates, t)
		}
	}

	if len(matchTemplates) == 0 {
		return false, nil
	}

	if reqCpuLevel == util.AiCpuLevelHigh {
		return true, matchTemplates[len(matchTemplates)-1]
	}

	return true, matchTemplates[0]
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
