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

// Package plugin implements xpu scheduler plugin
package plugin

import (
	"fmt"

	klog "k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

// InitNodeXPUDevices init xpu devices of node
func (sh *ScheduleHandler) InitNodeXPUDevices(sJob *SchedulerJob, nodeInfo *api.NodeInfo) {
	xpuDevices := sJob.handler.GetXPUDevicesFromNode(nodeInfo, sh.Templates)
	sh.Lock()
	sh.XPUDevices[nodeInfo.Name] = xpuDevices
	sh.Unlock()
}

// NodePredicate Predicate nodes.
func (sh *ScheduleHandler) NodePredicate(task *api.TaskInfo, node *api.NodeInfo) error {
	if sh == nil || task == nil || node == nil {
		klog.V(util.LogErrorLevel).Infof("NodePredicate got null parameter(s), which is invalid.")
		return fmt.Errorf("got null parameter(s)")
	}
	klog.V(util.LogDebugLevel).Infof("enter node(%s) predicate", node.Name)
	defer klog.V(util.LogDebugLevel).Infof("leave node(%s) predicate", node.Name)

	sJob, ok := sh.Jobs[task.Job]
	if !ok {
		klog.V(util.LogDebugLevel).Infof("NodePredicate not support job:%s.", util.SafePrint(task.Job))
		return nil
	}
	if !util.IsXPUName(sJob.ReqXPUName) || !IsXPUTask(sJob, task) {
		klog.V(util.LogDebugLevel).Infof("NodePredicate job:%#v is not xpu job or task %s is not xpu task.",
			sJob, task.Name)
		return nil
	}
	if err := sJob.preCheckNodePredicate(task, node); err != nil {
		return fmt.Errorf("%s", err.Error())
	}
	code, err := sJob.handler.NodePredicateForTask(sJob, task, node, sh)
	if err != nil {
		_ = code
		klog.V(util.LogDebugLevel).Infof("CheckNodeXPUByTask %s:%s ,cannot be selected.", node.Name, util.SafePrint(err))
		return err
	}
	klog.V(util.LogDebugLevel).Infof("%s NodePredicate %s select success.", PluginName, node.Name)
	return nil
}

func computeNodeScore(nodeName string, devices map[int]*common.XPUDevice, podMode string) float64 {
	if len(devices) == 0 {
		return 0
	}
	sumUsedCores := 0
	sumTotalCores := 0
	sumUsedMem := uint64(0)
	sumTotalMem := uint64(0)
	for _, dev := range devices {
		sumUsedCores += dev.UsedCores
		sumTotalCores += dev.Cores
		sumUsedMem += dev.UsedMemory
		sumTotalMem += dev.Memory
	}
	if sumTotalCores == 0 {
		return 0
	}
	coresScore := float64(sumUsedCores) / float64(sumTotalCores)
	if podMode == util.HardMode {
		return float64(scoreWeight) * coresScore
	}
	if sumTotalMem == 0 {
		return 0
	}
	memScore := float64(sumUsedMem) / float64(sumTotalMem)
	return float64(scoreWeight) * (coresScore + memScore)
}
