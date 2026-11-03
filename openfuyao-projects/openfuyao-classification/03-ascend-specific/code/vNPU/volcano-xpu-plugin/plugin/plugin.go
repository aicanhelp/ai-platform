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
	"errors"
	"fmt"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/nodelock"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

// XPUBuilder PluginBuilder plugin management
type XPUBuilder = func() XPUSchedulerPlugin

// XPUSchedulerPlugin for xpu plugin has function.
type XPUSchedulerPlugin interface {
	ValidXPUJob() *api.ValidateResult
	GetXPUDevicesFromNode(*api.NodeInfo, TemplateInfos) map[int]*common.XPUDevice
	NodePredicateForTask(*SchedulerJob, *api.TaskInfo, *api.NodeInfo, *ScheduleHandler) (int, error)
	Allocate(*SchedulerJob, *api.TaskInfo, *api.NodeInfo, map[int]*common.XPUDevice, TemplateInfos) error
	Deallocate(*api.TaskInfo, *api.NodeInfo) error
}

// SchedulerPlugin for all volcano-npu plugin.
type SchedulerPlugin struct {
	PluginName                 string
	VxpuName                   string
	VxpuType                   string
	VxpuCore                   string
	VxpuMemory                 string
	NodeXPURegisterAnno        string
	AssignedXPUsToAllocateAnno string
	AssignedXPUsToPodAnno      string
	AssignedXPUsToNodeAnno     string
	NodeXPUTopologyAnno        string
	NodeXPUHandshakeAnno       string
}

// ValidXPUJob check job req xpu num
func (sp *SchedulerPlugin) ValidXPUJob() *api.ValidateResult {
	if sp == nil {
		err := errors.New(util.ArgumentError)
		return &api.ValidateResult{Pass: false, Reason: err.Error(), Message: err.Error()}
	}
	return nil
}

func getInUseXPUDevices(inUseDeviceMap map[string][]common.ContainerDevice, annoName string, pod *v1.Pod) {
	if inUseDeviceMap == nil {
		err := errors.New(util.ArgumentError)
		klog.V(util.LogErrorLevel).Infof("getInUseXPUDevices err: %s", err.Error())
		return
	}
	annoValue, exists := pod.Annotations[annoName]
	if !exists || annoValue == "" {
		return
	}
	podDevices := DecodePodDevices(annoValue)
	for _, containerDevices := range podDevices {
		for _, device := range containerDevices {
			if _, ok := inUseDeviceMap[device.Id]; !ok {
				inUseDeviceMap[device.Id] = []common.ContainerDevice{}
			}
			policy, policyErr := getSchedulerPolicy(pod)
			if policyErr != nil {
				klog.V(util.LogErrorLevel).Infof("getInUseXPUDevices invalid scheduler policy for pod %s/%s: %v",
					pod.Namespace, pod.Name, policyErr)
			}
			device.Policy = policy
			inUseDeviceMap[device.Id] = append(inUseDeviceMap[device.Id], device)
		}
	}
}

// GetXPUDevicesFromNode get xpu infos from node
func (sp *SchedulerPlugin) GetXPUDevicesFromNode(node *api.NodeInfo, templates TemplateInfos) map[int]*common.XPUDevice {
	infos, ok := node.Node.Annotations[sp.NodeXPURegisterAnno]
	if !ok {
		klog.V(util.LogWarningLevel).Infof("Get XPU Devices failed, annotation %s not exist on node %s",
			sp.NodeXPURegisterAnno, node.Name)
		return nil
	}
	xpuDevices := DecodeNodeDevices(infos, node.Name)
	inUseDeviceMap := make(map[string][]common.ContainerDevice)
	for _, pod := range node.Pods() {
		getInUseXPUDevices(inUseDeviceMap, sp.AssignedXPUsToPodAnno, pod)
	}
	for _, v := range xpuDevices {
		if devices, ok := inUseDeviceMap[v.DieID]; ok {
			updateXPUDeviceUsage(v, devices, templates, node.Name)
		}
	}
	klog.V(util.LogDebugLevel).Infof("Get XPU device for node: %s, device num: %d", node.Name, len(xpuDevices))
	return xpuDevices
}

func updateXPUDeviceUsage(v *common.XPUDevice, devices []common.ContainerDevice, templates TemplateInfos, nodeName string) {
	var firstPolicy string
	for i, x := range devices {
		if x.Template != "" {
			usedCore, usedCpu := templates.getTemplateUsedResources(v.Type, x.Template)
			v.UsedCores += usedCore
			v.UsedCpu += usedCpu
		} else {
			v.UsedMemory += x.UsedMemory
			v.UsedCores += x.UsedCores
		}
		v.MarkVidUsed(x.Vid)
		if i == 0 {
			firstPolicy = x.Policy
		} else if x.Policy != firstPolicy {
			klog.V(util.LogErrorLevel).Infof("node %s device %s policy inconsistent: first policy %s, current policy %s",
				nodeName, v.DieID, firstPolicy, x.Policy)
		} else {
			// policy matches firstPolicy; no action needed
		}
	}
	v.Policy = firstPolicy
	v.InUse = true
}

// NodePredicateForTask evaluate whether node meet the task requirement
func (sp *SchedulerPlugin) NodePredicateForTask(sJob *SchedulerJob, task *api.TaskInfo,
	node *api.NodeInfo, sh *ScheduleHandler) (int, error) {
	if sp == nil || sJob == nil || task == nil || len(node.Node.Annotations) == 0 {
		err := errors.New(util.ArgumentError)
		klog.V(util.LogErrorLevel).Infof("NodePredicateForTask err: %s", err.Error())
		return api.Unschedulable, err
	}

	xpuTask, ok := sJob.Tasks[task.UID]
	if !ok {
		return api.Unschedulable, fmt.Errorf("node predicate failed, task: %s is not exist in job %s",
			task.Name, sJob.Id)
	}

	nodeMemLock := nodelock.GetNodeMemoryLock(node.Name)
	nodeMemLock.Lock()
	defer nodeMemLock.Unlock()

	sh.GetAllocatableXPUDevicesByNode(sJob)
	var score float64
	fit, _, err := sp.EvaluateXPUDeviceAllocation(task.Pod, sh.getNodeXPUDevices(node.Name), sh.Templates, &score)
	if err != nil || !fit {
		return api.Unschedulable, fmt.Errorf("%s predicate failed: no suitable devices, err: %v",
			sp.PluginName, err)
	}
	podMode := task.Pod.Annotations[util.VNPUModeAnnotation]
	if podMode == "" {
		podMode = util.SoftMode
	}
	nodeScore := computeNodeScore(node.Name, sh.getNodeXPUDevices(node.Name), podMode)
	nodePolicy := util.GetNodeSchedulerPolicy(task.Pod)
	if nodePolicy == util.SchedulerPolicySpread {
		var maxScore int
		if podMode == util.HardMode {
			maxScore = maxHardNodeScore
		} else {
			maxScore = maxSoftNodeScore
		}
		nodeScore = float64(maxScore) - nodeScore
	}
	xpuTask.Lock()
	xpuTask.ScoreMap[node.Name] = nodeScore
	xpuTask.Unlock()
	klog.V(util.LogDebugLevel).Infof("task[%v] node[%v] nodeScore[%.4f] policy[%s] mode[%s]",
		task.Name, node.Name, nodeScore, nodePolicy, podMode)
	return 0, nil
}

// AnnotatePodWithDevices set the selected xpu devices to pod annotation
func (sp *SchedulerPlugin) AnnotatePodWithDevices(task *api.TaskInfo, nodeName string, podDevices string) {
	if task == nil || task.Pod == nil || task.Pod.Annotations == nil {
		klog.V(util.LogErrorLevel).Infof("AnnotatePodWithDevices err: %s", util.ObjectNilError)
		return
	}

	task.Pod.Annotations[sp.AssignedXPUsToNodeAnno] = nodeName
	task.Pod.Annotations[sp.AssignedXPUsToAllocateAnno] = podDevices
	task.Pod.Annotations[sp.AssignedXPUsToPodAnno] = podDevices
	tmp := strconv.FormatInt(time.Now().UnixNano(), util.Base10)
	task.Pod.Annotations[util.BindTimeAnnotations] = tmp
	task.Pod.Annotations[util.DeviceBindPhase] = util.DeviceBindAllocating
	klog.V(util.LogDebugLevel).Infof("AnnotatePodWithDevices %s==%v : %s.", task.Name, tmp, podDevices)
	return
}

// getXPUReqFromContainer get xpu request number from container
func (sp *SchedulerPlugin) getXPUReqFromContainer(container *v1.Container) int {
	var number int = 0
	xpuNum, ok := container.Resources.Limits[v1.ResourceName(sp.VxpuName)]
	if ok {
		number = int(xpuNum.Value())
	}
	return number
}

// getPodDeviceFromAllocateXPUs select xpu devices from node
func (sp *SchedulerPlugin) getPodDeviceFromAllocateXPUs(
	pod *v1.Pod, allocateXPUs []int, xpuDevices map[int]*common.XPUDevice) string {
	selectDevices := PodDevices{}
	start := 0
	length := len(allocateXPUs)
	for _, v := range pod.Spec.Containers {
		xpuNum := sp.getXPUReqFromContainer(&v)
		if xpuNum == 0 {
			continue
		}
		if start+xpuNum > length {
			klog.V(util.LogErrorLevel).Infof(`getPodDeviceFromAllocateXPUs failed, insufficient number of xpu devices,
				request xpu number: %d, allocate: %v`, start+xpuNum, allocateXPUs)
			return ""
		}
		cds, err := getContainerDevices(allocateXPUs[start:start+xpuNum], xpuDevices)
		if err != nil {
			klog.V(util.LogErrorLevel).Infof(
				"getPodDeviceFromAllocateXPUs failed, err: %v", err)
			return ""
		}
		selectDevices = append(selectDevices, cds)
		start += xpuNum
	}
	return EncodePodDevices(selectDevices)
}

func (sp *SchedulerPlugin) getSelectXPUs(task *api.TaskInfo, xpuDevices map[int]*common.XPUDevice, templateInfos TemplateInfos) string {
	fit, device, err := sp.EvaluateXPUDeviceAllocation(task.Pod, xpuDevices, templateInfos, nil)
	if err != nil || !fit {
		klog.V(util.LogErrorLevel).Infof("%s Allocate failed: no suitable devices was selected.",
			sp.PluginName)
		return ""
	}
	return EncodePodDevices(device)
}

// Allocate select xpu for task from node
func (sp *SchedulerPlugin) Allocate(sJob *SchedulerJob, task *api.TaskInfo,
	node *api.NodeInfo, xpuDevices map[int]*common.XPUDevice, templateInfos TemplateInfos) error {
	if sJob == nil || sp == nil || task == nil || node == nil {
		err := errors.New(util.ArgumentError)
		klog.V(util.LogErrorLevel).Infof("%s Allocate err: %s.", sp.PluginName, err.Error())
		return err
	}
	if _, ok := sJob.Tasks[task.UID]; !ok {
		klog.V(util.LogErrorLevel).Infof("Allocate task %s is not exist in job %s.",
			task.Name, sJob.Id)
		return fmt.Errorf("task %s not found in job %s", task.Name, sJob.Id)
	}

	selectedXPUs := sp.getSelectXPUs(task, xpuDevices, templateInfos)
	if selectedXPUs == "" {
		klog.V(util.LogErrorLevel).Infof("%s Allocate failed: no suitable node was selected.",
			sp.PluginName)
		return fmt.Errorf("no suitable XPU devices on node %s for task %s", node.Name, task.Name)
	}
	klog.V(util.LogDebugLevel).Infof("%s Allocate task<%s> select xpu <%v>.",
		sp.PluginName, task.Name, selectedXPUs)
	sp.AnnotatePodWithDevices(task, node.Name, selectedXPUs)
	return nil
}

// Deallocate remove xpu allocation from node
func (sp *SchedulerPlugin) Deallocate(_ *api.TaskInfo, _ *api.NodeInfo) error {
	return nil
}
