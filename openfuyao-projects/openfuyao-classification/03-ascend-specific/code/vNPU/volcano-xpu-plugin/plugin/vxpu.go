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
	"errors"
	"fmt"
	"sort"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

type candidateDevice struct {
	device        *common.XPUDevice
	realReqXPUMem int
	template      *TemplateInfo
	isFullCard    bool
	deviceScore   float64
}

type hardModeResult struct {
	fits       bool
	template   *TemplateInfo
	score      float64
	isFullCard bool
}

type ScheduleConfig struct {
	Templates TemplateInfos
	Policy    string
	PodMode   string
}

// IsDeviceQualified verify request xpu
func IsDeviceQualified(device *common.XPUDevice, val *util.ContainerResource) (bool, int) {
	if val == nil || device == nil {
		return false, 0
	}

	if device.Policy != "" && device.Policy != val.ReqPolicy {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, policy not the same, 
		deviceId: %s, request policy: %s, device policy: %s`,
			val, device.DieID, val.ReqPolicy, device.Policy)
		return false, 0
	}
	if len(val.ReqXPUDieID) != 0 && device.DieID != val.ReqXPUDieID {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, dieID not the same, 
		deviceId: %s, request dieID: %s`, val, device.DieID, val.ReqXPUDieID)
		return false, 0
	}
	if len(val.ReqUseNPUType) != 0 && device.Type != val.ReqUseNPUType {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, npu type not the same, 
		deviceId: %s, request npu type: %s, device type: %s`, val, device.DieID, val.ReqUseNPUType, device.Type)
		return false, 0
	}

	realReqXPUMem := val.ReqXPUMem
	if device.Count <= int(device.GetUsedVidCount()) {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, count is not enough, 
		deviceId: %s, max count: %d, used vids: %x`,
			val, device.DieID, device.Count, device.UsedVids)
		return false, 0
	}
	if val.ReqXPUMemPercentage != 0 && val.ReqXPUMem < 0 {
		realReqXPUMem = int(device.Memory) * val.ReqXPUMemPercentage / util.Base100
	}
	if device.Memory-device.UsedMemory < uint64(realReqXPUMem) {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, memory is not enough, 
		deviceId: %s, request memory: %d, exist memory: %d`, val, device.DieID, realReqXPUMem,
			device.Memory-device.UsedMemory)
		return false, 0
	}
	if util.Base100-device.UsedCores < val.ReqXPUCores {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, cores is not enough, 
		deviceId: %s, request cores: %d, exist cores: %d`, val, device.DieID, val.ReqXPUCores,
			util.Base100-device.UsedCores)
		return false, 0
	}
	if len(val.ReqXPUType) != 0 && device.Type != val.ReqXPUType {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, xpu type not the same, 
		deviceId: %s, request xpu: %s, device xpu: %s`, val, device.DieID, val.ReqXPUType, device.Type)
		return false, 0
	}
	return true, realReqXPUMem
}

func checkSoftModeDevice(device *common.XPUDevice, val *util.ContainerResource) (bool, int, float64) {
	if val.IsPrivileged {
		klog.V(util.LogDebugLevel).Infof("Container is privileged, not qualified for soft mode device %s", device.DieID)
		return false, 0, 0
	}
	qualified, realReqXPUMem := IsDeviceQualified(device, val)
	if !qualified {
		return false, 0, 0
	}
	deviceScore := calculateDeviceScore(device, val, realReqXPUMem, device.Cores, false)
	return true, realReqXPUMem, deviceScore
}

func applySoftModeDevice(device *common.XPUDevice, val *util.ContainerResource, realReqXPUMem int) *common.ContainerDevice {
	vid := device.AllocateVid()
	device.UsedMemory += uint64(realReqXPUMem)
	device.UsedCores += val.ReqXPUCores
	return &common.ContainerDevice{
		Index:      device.PhysicID,
		Id:         device.DieID,
		Type:       device.Type,
		UsedMemory: uint64(realReqXPUMem),
		UsedCores:  val.ReqXPUCores,
		Vid:        vid,
	}
}

func checkHardModeDevice(device *common.XPUDevice, val *util.ContainerResource,
	templates TemplateInfos) hardModeResult {
	noneFit := hardModeResult{}
	if val == nil {
		return noneFit
	}

	totalCores, totalCpu := templates.getDeviceTotalResources(device.Type)

	if val.ReqXPUCores == util.FullCore {
		if device.UsedCores > 0 {
			return hardModeResult{isFullCard: true}
		}
		return hardModeResult{fits: true, isFullCard: true, score: float64(scoreWeight),
			template: &TemplateInfo{Name: util.FullCardTemplate, AiCore: totalCores, AiCPU: totalCpu,
				TotalAiCore: totalCores, TotalAiCpu: totalCpu}}
	}

	if totalCores <= 0 {
		return noneFit
	}
	unUsedCores := totalCores - device.UsedCores
	unUsedCpu := totalCpu - device.UsedCpu
	if unUsedCores <= 0 || unUsedCpu <= 0 {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, available resource < 0, 
		deviceId: %s, totalCores %d, unUsedCores %d, unUsedCpu %d `, val, device.DieID, totalCores, unUsedCores, unUsedCpu)
		return noneFit
	}
	if unUsedCores*util.Base100 < totalCores*val.ReqXPUCores {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, cores is not enough,
		deviceId: %s, totalCores %d, unUsedCores %d`, val, device.DieID, totalCores, unUsedCores)
		return noneFit
	}

	hasMatch, template := templates.mapTemplate(device.Type, val.ReqXPUCores, val.ReqCpuLevel)
	if !hasMatch {
		return noneFit
	}
	if template.AiCore > unUsedCores || template.AiCPU > unUsedCpu {
		klog.V(util.LogDebugLevel).Infof(`Calculate device for container request %v, resource is not enough,
		deviceId: %s, template.AiCore %d, unUsedCores %d template.AiCPU %d, unUsedCpu %d`,
			val, device.DieID, template.AiCore, unUsedCores, template.AiCPU, unUsedCpu)
		return noneFit
	}

	deviceScore := calculateDeviceScore(device, val, 0, totalCores, true)
	return hardModeResult{fits: true, template: template, score: deviceScore}
}

func applyHardModeDevice(device *common.XPUDevice, template *TemplateInfo) *common.ContainerDevice {
	device.UsedCores += template.AiCore
	device.UsedCpu += template.AiCPU
	vid := device.AllocateVid()
	return &common.ContainerDevice{
		Index:    device.PhysicID,
		Id:       device.DieID,
		Type:     device.Type,
		Vid:      vid,
		Template: template.Name,
	}
}

func calculateDeviceScore(device *common.XPUDevice, val *util.ContainerResource,
	realReqXPUMem int, totalCores int, isHardMode bool) float64 {
	if isHardMode {
		return float64(scoreWeight) * float64(val.ReqXPUCores+device.UsedCores) / float64(totalCores)
	}
	coresScore := float64(val.ReqXPUCores+device.UsedCores) / float64(device.Cores)
	memScore := float64(realReqXPUMem+int(device.UsedMemory)) / float64(device.Memory)
	return float64(scoreWeight) * (coresScore + memScore)
}

// AllocateContainerDevices allocate for container xpu device request with scheduling policy
func AllocateContainerDevices(xpuDevices []*common.XPUDevice, containerReq *util.ContainerResource,
	config ScheduleConfig) ([]common.ContainerDevice, float64) {
	var candidates []candidateDevice
	var totalScore float64

	for _, device := range xpuDevices {
		if config.PodMode == util.HardMode {
			res := checkHardModeDevice(device, containerReq, config.Templates)
			if !res.fits {
				continue
			}
			candidates = append(candidates, candidateDevice{
				device:      device,
				template:    res.template,
				isFullCard:  res.isFullCard,
				deviceScore: res.score,
			})
		} else {
			fits, realReqXPUMem, deviceScore := checkSoftModeDevice(device, containerReq)
			if !fits {
				continue
			}
			candidates = append(candidates, candidateDevice{
				device:        device,
				realReqXPUMem: realReqXPUMem,
				deviceScore:   deviceScore,
			})
		}
	}

	if config.Policy == util.SchedulerPolicySpread {
		sort.SliceStable(candidates, func(i, j int) bool {
			return candidates[i].deviceScore < candidates[j].deviceScore
		})
	} else {
		sort.SliceStable(candidates, func(i, j int) bool {
			return candidates[i].deviceScore > candidates[j].deviceScore
		})
	}

	var result []common.ContainerDevice
	for _, cand := range candidates {
		if containerReq.ReqXPUNum <= 0 {
			break
		}
		var cdev *common.ContainerDevice
		if config.PodMode == util.HardMode {
			cdev = applyHardModeDevice(cand.device, cand.template)
		} else {
			cdev = applySoftModeDevice(cand.device, containerReq, cand.realReqXPUMem)
		}
		if cdev != nil {
			result = append(result, *cdev)
			containerReq.ReqXPUNum--
			totalScore += cand.deviceScore
		}
	}
	return result, totalScore
}

func buildXPUDevices(devs map[int]*common.XPUDevice) ([]*common.XPUDevice, error) {
	xpuDevices := make([]*common.XPUDevice, len(devs))
	for index, dev := range devs {
		if index >= len(xpuDevices) {
			return nil, errors.New("xpu device index error")
		}
		xpuDevices[index] = dev
	}
	return xpuDevices, nil
}

func (sp *SchedulerPlugin) parseContainerRequests(pod *v1.Pod,
	schedulerPolicy string, aiCpuLevel string) ([]*util.ContainerResource, error) {
	var resourceRequests []*util.ContainerResource
	useDieID := pod.Annotations[util.VNPUUseDieIDAnnotation]
	useNPUType := pod.Annotations[util.VNPUUseNPUTypeAnnotation]
	for _, c := range pod.Spec.Containers {
		cr := GetXPUResourceFromContainer(&c, sp.VxpuName, sp.VxpuCore, sp.VxpuMemory, sp.VxpuType)
		if cr.ReqXPUNum < 0 {
			continue
		}
		if err := sp.validateContainerResource(&c, &cr, schedulerPolicy); err != nil {
			return nil, err
		}
		cr.ReqCpuLevel = aiCpuLevel
		cr.ReqPolicy = schedulerPolicy
		cr.ReqXPUDieID = useDieID
		cr.ReqUseNPUType = useNPUType
		cr.IsPrivileged = c.SecurityContext != nil && c.SecurityContext.Privileged != nil && *c.SecurityContext.Privileged
		resourceRequests = append(resourceRequests, &cr)
	}
	return resourceRequests, nil
}

func (sp *SchedulerPlugin) validateContainerResource(c *v1.Container,
	cr *util.ContainerResource, policy string) error {
	if cr.ReqXPUNum == 0 || (cr.ReqXPUNum > 1 && cr.ReqXPUCores > 0 && cr.ReqXPUCores < util.FullCore) {
		errMsg := fmt.Sprintf("Container %s invalid %s limit: %d", c.Name, sp.VxpuName, cr.ReqXPUNum)
		klog.V(util.LogErrorLevel).Infof("%s", errMsg)
		return fmt.Errorf("%s", errMsg)
	}
	if cr.ReqXPUCores > util.FullCore {
		errMsg := fmt.Sprintf("Container %s invalid %s limit: %d", c.Name, sp.VxpuCore, cr.ReqXPUCores)
		klog.V(util.LogErrorLevel).Infof("%s", errMsg)
		return fmt.Errorf("%s", errMsg)
	}
	if (policy == util.PolicyElastic || policy == util.PolicyFixedShare) && cr.ReqXPUCores == 0 {
		errMsg := fmt.Sprintf("Container %s invalid %s limit: %d, elastic or fixed-share must have xpu cores",
			c.Name, sp.VxpuCore, cr.ReqXPUCores)
		klog.V(util.LogErrorLevel).Infof("%s", errMsg)
		return fmt.Errorf("%s", errMsg)
	}
	if cr.ReqXPUMem == 0 {
		errMsg := fmt.Sprintf("Container %s invalid %s limit: %d", c.Name, sp.VxpuMemory, cr.ReqXPUMem)
		klog.V(util.LogErrorLevel).Infof("%s", errMsg)
		return fmt.Errorf("%s", errMsg)
	}
	return nil
}

func allocateResources(xpuDevices []*common.XPUDevice, resourceRequests []*util.ContainerResource,
	config ScheduleConfig, score *float64) (PodDevices, error) {
	podDevices := PodDevices{}
	for _, val := range resourceRequests {
		if val.ReqXPUNum > len(xpuDevices) {
			return nil, fmt.Errorf("no enough xpu cards on node, request: %d, have: %d",
				val.ReqXPUNum, len(xpuDevices))
		}
		klog.V(util.LogDebugLevel).Infof("Allocating device for container request %v", val)
		cdevs, containerScore := AllocateContainerDevices(xpuDevices, val, config)
		if val.ReqXPUNum > 0 {
			return nil, fmt.Errorf("not enough xpu fitted on this node")
		}
		podDevices = append(podDevices, cdevs)
		if score != nil {
			*score += containerScore
		}
	}
	return podDevices, nil
}

// EvaluateXPUDeviceAllocation allocate pod xpu devices
func (sp *SchedulerPlugin) EvaluateXPUDeviceAllocation(pod *v1.Pod,
	devs map[int]*common.XPUDevice, templates TemplateInfos, score *float64) (bool, PodDevices, error) {
	if pod == nil || devs == nil {
		return false, PodDevices{}, fmt.Errorf("pod or devices nil")
	}

	xpuDevices, err := buildXPUDevices(devs)
	if err != nil {
		return false, PodDevices{}, err
	}
	if err := isVnpuModeMatch(pod, xpuDevices); err != nil {
		return false, PodDevices{}, err
	}

	klog.V(util.LogDebugLevel).Infof("Allocate devices for pod %s/%s", pod.Namespace, pod.Name)

	aiCpuLevel := getAiCpuLevel(pod)
	schedulerPolicy, policyErr := getSchedulerPolicy(pod)
	if policyErr != nil {
		return false, PodDevices{}, policyErr
	}
	podMode := pod.Annotations[util.VNPUModeAnnotation]
	if podMode == "" {
		podMode = util.SoftMode
	}

	resourceRequests, parseErr := sp.parseContainerRequests(pod, schedulerPolicy, aiCpuLevel)
	if parseErr != nil {
		return false, PodDevices{}, parseErr
	}

	config := ScheduleConfig{
		Templates: templates,
		Policy:    util.GetDeviceSchedulerPolicy(pod),
		PodMode:   podMode,
	}
	podDevices, allocErr := allocateResources(xpuDevices, resourceRequests, config, score)
	if allocErr != nil {
		return false, PodDevices{}, allocErr
	}
	return true, podDevices, nil
}

func getAiCpuLevel(pod *v1.Pod) string {
	level, ok := pod.Annotations[util.HardAiCpuLevelAnnotation]
	if !ok {
		return util.AiCpuLevelLow
	}
	return level
}

func getSchedulerPolicy(pod *v1.Pod) (string, error) {
	policy, ok := pod.Annotations[util.SoftSchedulerPolicyAnnotation]
	if !ok {
		return util.PolicyFixedShare, nil
	}
	if policy != util.PolicyFixedShare && policy != util.PolicyElastic && policy != util.PolicyBestEffort {
		return "", fmt.Errorf("invalid scheduler policy: %s, must be one of [fixed-share, elastic, best-effort]", policy)
	}
	return policy, nil
}

func isVnpuModeMatch(pod *v1.Pod, devices []*common.XPUDevice) error {
	podVnpuMode, ok := pod.Annotations[util.VNPUModeAnnotation]
	if !ok {
		podVnpuMode = util.SoftMode
	}
	for _, device := range devices {
		if device.Mode != podVnpuMode {
			return fmt.Errorf("vnpu mode not match, pod mode is %s, device mode is %s", podVnpuMode, device.Mode)
		}
	}
	return nil
}
