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
package kubelet

import (
	"encoding/json"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

const (
	// ResizeLimitOnStarting Enable resize configuration parameters
	ResizeLimitOnStarting = "resize.on.starting"
)

type resizeResource struct {
	Name string            `json:"name"`
	CPU  resource.Quantity `json:"cpu"`
}

func disableResize(pod *v1.Pod) bool {
	return pod.Annotations[ResizeLimitOnStarting] == ""
}

func fyResizeLimit(pod *v1.Pod, kl *Kubelet, pcm cm.PodContainerManager) (*v1.Pod, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) || pod.Annotations == nil ||
		disableResize(pod) {
		return pod, nil
	}

	if qos.GetPodQOS(pod) == v1.PodQOSGuaranteed {
		klog.V(2).Infof("Resize cancel: Pod QOS is immutable %s", pod.Name)
		return pod, nil
	}

	var resizeResources []resizeResource
	resizeResourceStr := pod.Annotations[ResizeLimitOnStarting]
	err := json.Unmarshal([]byte(resizeResourceStr), &resizeResources)
	if err != nil {
		klog.V(2).Infof("Failed to unmarshal resize resource: %v", err)
		return pod, nil
	}

	if len(resizeResources) == 0 {
		klog.V(2).Infof("No resize resource found in pod annotations")
		return pod, nil
	}

	if isPodSteady(pod) {
		oldPod, ok := kl.podManager.GetPodByUID(pod.UID)
		if !ok {
			klog.V(2).InfoS("Cancel hwResize", "reason", "pod not found", "pod", pod.Name)
			return pod, nil
		}
		return oldPod, nil
	}

	resizeMap := parseToResizeMap(resizeResources)
	cpuLimitMap := make(map[string]resource.Quantity)

	for _, rContainer := range pod.Spec.Containers {
		if rContainer.Resources.Limits == nil || rContainer.ReadinessProbe == nil {
			continue
		}

		resizeCpuLimit, ok := resizeMap[rContainer.Name]
		if !ok {
			continue
		}

		if rContainer.Resources.Limits.Cpu().Cmp(resizeCpuLimit) > 0 {
			klog.V(2).Infof("Resize CPU limit exceeded for container %s", rContainer.Name)
			continue
		}
		cpuLimitMap[rContainer.Name] = rContainer.Resources.Limits[v1.ResourceCPU]
		rContainer.Resources.Limits[v1.ResourceCPU] = resizeCpuLimit

		if pcm.Exists(pod) {
			if err := kl.containerRuntime.FySetPodCpuCgroup(pod); err != nil {
				klog.ErrorS(err, "Cancel hwResize: Errors set the pod`s cgroup", "pod", pod.Name)
				recoverCpuLimit(pod, cpuLimitMap)
			}
		}
	}
	return pod, nil
}

func isPodSteady(pod *v1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == v1.PodReady && condition.Status == v1.ConditionTrue {
			return true
		}
	}
	return false
}

func parseToResizeMap(resizeResources []resizeResource) map[string]resource.Quantity {
	resizeMap := make(map[string]resource.Quantity)
	for _, rContainer := range resizeResources {
		cpuLimit := rContainer.CPU
		resizeMap[rContainer.Name] = cpuLimit
	}
	return resizeMap
}

func recoverCpuLimit(pod *v1.Pod, limitMap map[string]resource.Quantity) {
	for _, rContainer := range pod.Spec.Containers {
		cpuLimit, ok := limitMap[rContainer.Name]
		if ok {
			rContainer.Resources.Limits[v1.ResourceCPU] = cpuLimit
		}
	}
}
