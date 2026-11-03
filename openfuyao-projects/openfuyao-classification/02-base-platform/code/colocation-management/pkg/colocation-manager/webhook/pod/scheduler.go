/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: hongchi hu
* Date: 2025-5-25
 */

package pod

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

// SchedulerMutator implements PodMutator to set scheduler name to volcano
type SchedulerMutator struct{}

// NewSchedulerMutator inits SchedulerMutator
func NewSchedulerMutator() PodMutator {
	return &SchedulerMutator{}
}

// MutatePod forcibly sets the scheduler name to "volcano" for this pod
func (m *SchedulerMutator) MutatePod(pod *v1.Pod) (bool, error) {
	if pod == nil {
		return false, fmt.Errorf("pod cannot be nil")
	}

	const volcanoScheduler = "volcano"

	// 记录原始调度器名称（用于日志）
	oldScheduler := pod.Spec.SchedulerName

	// 强制设置为volcano
	pod.Spec.SchedulerName = volcanoScheduler

	// 根据是否修改过返回不同日志
	if oldScheduler == volcanoScheduler {
		klog.Infof("Pod %s/%s already uses volcano scheduler (no change needed)",
			pod.Namespace, pod.Name)
		return false, nil
	}

	klog.Infof("Forcibly set scheduler from %q to %q for pod %s/%s",
		oldScheduler, volcanoScheduler,
		pod.Namespace, pod.Name)
	return true, nil
}
