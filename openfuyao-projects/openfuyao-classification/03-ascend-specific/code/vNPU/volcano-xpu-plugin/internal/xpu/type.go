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

// Package xpu implements xpu plugin
package xpu

import (
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/plugin"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

var npuPlugin = &plugin.SchedulerPlugin{
	PluginName:                 util.NPUPluginName,
	VxpuName:                   util.VNPUName,
	VxpuType:                   util.VNPUType,
	VxpuCore:                   util.VNPUCore,
	VxpuMemory:                 util.VNPUMemory,
	NodeXPURegisterAnno:        util.NodeNPURegisterAnnotation,
	AssignedXPUsToAllocateAnno: util.AssignedNPUsToAllocateAnnotations,
	AssignedXPUsToNodeAnno:     util.AssignedNPUsToNodeAnnotations,
	AssignedXPUsToPodAnno:      util.AssignedNPUsToPodAnnotations,
	NodeXPUTopologyAnno:        util.NodeNPUTopologyAnnotation,
	NodeXPUHandshakeAnno:       util.NodeNPUHandshakeAnnotation,
}
