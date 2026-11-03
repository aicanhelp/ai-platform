/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package pod

import (
	v1 "k8s.io/api/core/v1"
)

// pod status phase check
func VerifyPodStatus(pod v1.Pod) bool {
	if pod.Status.Phase == "Running" {
		for _, containerStatus := range pod.Status.ContainerStatuses {
			if !containerStatus.Ready {
				return false
			}
		}
		return true
	}
	return false
}

// pod image check
func ImageCheck(pod v1.Pod, imageList []string) bool {
	var imageMap map[string]int
	imageMap = make(map[string]int)
	for _, eachImage := range imageList {
		imageMap[eachImage] = 0
	}
	for _, eachConatainerStatus := range pod.Status.ContainerStatuses {
		if _, exists := imageMap[eachConatainerStatus.Image]; exists {
			imageMap[eachConatainerStatus.Image] += 1
		}
	}
	for _, v := range imageMap {
		if v == 0 {
			return false
		}
	}
	return true
}
