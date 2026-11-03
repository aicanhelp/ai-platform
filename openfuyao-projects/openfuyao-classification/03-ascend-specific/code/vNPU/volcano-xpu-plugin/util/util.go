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

// Package util defines data structure and provide util function for xpu scheduler plugin implementation
package util

import (
	"fmt"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

// IsXPUName determine whether it is the xpu name
func IsXPUName(s string) bool {
	return s == VNPUName
}

// GetXPUType get the xpu device type if configured
func GetXPUType(container *v1.Container, xpuType string) string {
	for k := range container.Resources.Limits {
		if strings.HasPrefix(string(k), xpuType) {
			device := strings.TrimPrefix(string(k), xpuType)
			return device
		}
	}
	return ""
}

// SafePrint safe print error
func SafePrint(args ...interface{}) string {
	msg := fmt.Sprint(args...)
	trimMsg := strings.Replace(msg, "\r", " ", -1)
	trimMsg = strings.Replace(trimMsg, "\n", " ", -1)
	return trimMsg
}

// GetVXPUResource for get vxpu resource from container Limits
func GetVXPUResource(container *v1.Container, resourceName string) int {
	resourceNum, ok := container.Resources.Limits[v1.ResourceName(resourceName)]
	if ok {
		return int(resourceNum.Value())
	}
	return -1
}

func GetNodeSchedulerPolicy(pod *v1.Pod) string {
	return getPlacementStrategy(pod, NodeSchedulerPolicyAnnotation)
}

func GetDeviceSchedulerPolicy(pod *v1.Pod) string {
	return getPlacementStrategy(pod, DeviceSchedulerPolicyAnnotation)
}

func getPlacementStrategy(pod *v1.Pod, annotationKey string) string {
	if pod == nil || pod.Annotations == nil {
		return SchedulerPolicyBinpack
	}
	policy, ok := pod.Annotations[annotationKey]
	if !ok || policy == "" {
		return SchedulerPolicyBinpack
	}
	if policy != SchedulerPolicyBinpack && policy != SchedulerPolicySpread {
		klog.V(LogWarningLevel).Infof("invalid scheduler policy %q for pod %s/%s (annotation: %s), fallback to %s",
			policy, pod.Namespace, pod.Name, annotationKey, SchedulerPolicyBinpack)
		return SchedulerPolicyBinpack
	}
	return policy
}
