/*
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 * 此外，本项目部分代码引用了 Apache License 2.0 许可的 Kubernetes qos-helper的代码
 * This project includes code from Kubernetes qos-helper, which is licensed under the Apache License 2.0.
 * 修改内容:GetKubePodQosClass方法拆分降低代码复杂度
 *
 * 此文件部分引用了 Apache License 2.0 许可的代码，原始版权声明如下：
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package utils

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"

	"openfuyao.com/colocation-management/pkg/common"
)

// OfflinePod 判断pod是否是离线pod
func OfflinePod(pod *corev1.Pod) bool {
	if pod == nil {
		return false
	}
	var anno string
	var label string

	if pod.Annotations != nil {
		anno = pod.Annotations[common.PriorityAnnotationKey]
	}

	if pod.Labels != nil {
		label = pod.Labels[common.PriorityAnnotationKey]
	}

	// Annotations have a higher priority than labels
	return anno == "true" || label == "true"
}

// OfflinePodMeta 判断pod是否是离线pod
func OfflinePodMeta(podMeta *metav1.PartialObjectMetadata) bool {
	if podMeta == nil {
		return false
	}

	var anno string
	var label string

	if podMeta.Annotations != nil {
		anno = podMeta.Annotations[common.PriorityAnnotationKey]
	}

	if podMeta.Labels != nil {
		label = podMeta.Labels[common.PriorityAnnotationKey]
	}

	// Annotations have a higher priority than labels
	return anno == "true" || label == "true"
}

// ContributivePod 判断pod是否是ContributivePod
func ContributivePod(podPhase corev1.PodPhase) bool {
	return podPhase != corev1.PodSucceeded && podPhase != corev1.PodFailed
}

// GetQosLevelByPod 获取pod的QosLevel
func GetQosLevelByPod(pod *corev1.Pod) common.QosLevel {
	qosLevel := GetPodQosLevelByName(pod.Annotations)
	if qosLevel != common.QosNone {
		return qosLevel
	}
	return common.QosNone

}

// GetPodQosLevelByName 根据annotation获取pod的QosLevel
func GetPodQosLevelByName(annotations map[string]string) common.QosLevel {
	if annotations == nil {
		return common.QosNone
	}

	if qos, exist := annotations[common.QosClassAnnotationKey]; exist {
		switch common.QosLevel(qos) {
		case common.QosBe, common.QosLs, common.QosHLS:
			return common.QosLevel(qos)
		default:
			return common.QosNone
		}
	}
	return common.QosNone
}

// GetPodQosLevelByKubeQos 目前保留 仅仅用annotation作为分类依据
func GetPodQosLevelByKubeQos(KubeQos corev1.PodQOSClass) common.QosLevel {
	switch KubeQos {
	case corev1.PodQOSGuaranteed:
		return common.QosNone
	case corev1.PodQOSBurstable:
		return common.QosLs
	case corev1.PodQOSBestEffort:
		return common.QosBe
	default:
		return common.QosNone
	}
}

// GetKubePodQosClass 获取pod的QosLevel
func GetKubePodQosClass(pod *corev1.Pod) corev1.PodQOSClass {
	if qos := pod.Status.QOSClass; len(qos) > 0 {
		return qos
	}

	allContainers := getAllContainers(pod)
	if usesUnsupportedQoSResources(allContainers) {
		return corev1.PodQOSBestEffort
	}

	requests := calculateTotalRequests(allContainers)
	limits := calculateTotalLimits(allContainers)

	if hasNoResourceRequirements(requests, limits) {
		return corev1.PodQOSBestEffort
	}

	if isGuaranteedQoS(requests, limits) {
		return corev1.PodQOSGuaranteed
	}

	return corev1.PodQOSBurstable
}

// getAllContainers 获取pod的所有容器
func getAllContainers(pod *corev1.Pod) []corev1.Container {
	containers := make([]corev1.Container, 0, len(pod.Spec.Containers)+len(pod.Spec.InitContainers))
	containers = append(containers, pod.Spec.Containers...)
	containers = append(containers, pod.Spec.InitContainers...)
	return containers
}

// usesUnsupportedQoSResources 判断pod是否使用了不支持的Qos资源
func usesUnsupportedQoSResources(containers []corev1.Container) bool {
	for _, container := range containers {
		for name := range container.Resources.Requests {
			if !isSupportedQoSComputeResource(name) {
				return true
			}
		}
		for name := range container.Resources.Limits {
			if !isSupportedQoSComputeResource(name) {
				return true
			}
		}
	}
	return false
}

// calculateTotalLimits 获取pod的所有容器的limits总和
func calculateTotalLimits(containers []corev1.Container) corev1.ResourceList {
	totalLimits := corev1.ResourceList{}
	zeroQuantity := resource.MustParse("0")

	for _, container := range containers {
		containerLimits := getValidContainerResources(container.Resources.Limits, zeroQuantity)
		totalLimits = mergeResourceLists(totalLimits, containerLimits)
	}
	return totalLimits
}

// calculateTotalRequests 获取pod的所有容器的requests总和
func calculateTotalRequests(containers []corev1.Container) corev1.ResourceList {
	totalRequests := corev1.ResourceList{}
	zeroQuantity := resource.MustParse("0")

	for _, container := range containers {
		containerRequests := getValidContainerResources(container.Resources.Requests, zeroQuantity)
		totalRequests = mergeResourceLists(totalRequests, containerRequests)
	}
	return totalRequests
}

// getValidContainerResources 从资源列表中获取有效的资源（大于0的）
func getValidContainerResources(resources corev1.ResourceList, zeroQuantity resource.Quantity) corev1.ResourceList {
	validResources := corev1.ResourceList{}

	for name, quantity := range resources {
		if quantity.Cmp(zeroQuantity) == 1 {
			validResources[name] = quantity.DeepCopy()
		}
	}
	return validResources
}

// mergeResourceLists 合并两个资源列表，相同资源名的数量相加
func mergeResourceLists(list1, list2 corev1.ResourceList) corev1.ResourceList {
	result := list1.DeepCopy()
	if result == nil {
		result = corev1.ResourceList{}
	}

	for name, quantity := range list2 {
		if existing, exists := result[name]; exists {
			newQuantity := quantity.DeepCopy()
			newQuantity.Add(existing)
			result[name] = newQuantity
		} else {
			result[name] = quantity.DeepCopy()
		}
	}
	return result
}

// hasNoResourceRequirements 获取pod的QosLevel
func hasNoResourceRequirements(requests, limits corev1.ResourceList) bool {
	return len(requests) == 0 && len(limits) == 0
}

// isGuaranteedQoS 获取pod的QosLevel
func isGuaranteedQoS(requests, limits corev1.ResourceList) bool {
	// Check if all limits are set and match requests exactly
	if len(requests) != len(limits) {
		return false
	}
	for name, req := range requests {
		lim, exists := limits[name]
		if !exists || lim.Cmp(req) != 0 {
			return false
		}
	}
	return true
}
func isSupportedQoSComputeResource(name corev1.ResourceName) bool {
	return supportedQoSComputeResources.Has(string(name))
}

var supportedQoSComputeResources = sets.NewString(string(corev1.ResourceCPU), string(corev1.ResourceMemory))

// HLSPodMeta 判断是否为HLS pod
func HLSPodMeta(podMeta *metav1.PartialObjectMetadata) bool {
	if podMeta.Annotations == nil {
		return false
	}
	if anno, exist := podMeta.Annotations[common.QosClassAnnotationKey]; exist {
		return anno == string(common.QosHLS)
	} else {
		return false
	}
}

// HLSPod 判断是否为HLS pod
func HLSPod(pod *corev1.Pod) bool {
	if pod == nil || pod.Annotations == nil {
		return false
	}

	if anno, exist := pod.Annotations[common.QosClassAnnotationKey]; exist {
		return anno == string(common.QosHLS)
	} else {
		return false
	}
}
