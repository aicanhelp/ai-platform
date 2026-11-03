/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY, FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package plugin

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/nodelock"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

const (
	maxLockRetries    = 30
	lockRetryInterval = 500 * time.Millisecond
)

func IsXPUTask(sJob *SchedulerJob, task *api.TaskInfo) bool {
	if _, ok := sJob.Tasks[task.UID]; !ok {
		return false
	}
	if util.IsXPUName(sJob.Tasks[task.UID].ReqXPUName) {
		return true
	}
	return false
}

func (sh *ScheduleHandler) AllocateXPUForTask(task *api.TaskInfo, ssn *framework.Session) error {
	if task == nil || ssn == nil {
		klog.V(util.LogErrorLevel).Infof("AllocateXPUForTask %s.", util.ArgumentError)
		return fmt.Errorf("%s", util.ArgumentError)
	}
	sJob, ok := sh.Jobs[task.Job]
	if !ok {
		klog.V(util.LogDebugLevel).Infof("AllocateXPUForTask %s not req npu.", task.Name)
		return nil
	}
	if !IsXPUTask(sJob, task) {
		klog.V(util.LogDebugLevel).Infof("AllocateXPUForTask %s no need to set pod annotation.", task.Name)
		return nil
	}
	if !sJob.JobReadyTag {
		klog.V(util.LogDebugLevel).Infof("AllocateXPUForTask %s not allow allocate npu.", task.Name)
		return nil
	}
	nodeName := task.NodeName
	node, found := ssn.Nodes[nodeName]
	if !found {
		klog.V(util.LogWarningLevel).Infof("%s AllocateXPUForTask %s not exist.", PluginName, nodeName)
		return fmt.Errorf("node %s not exist", nodeName)
	}

	nodelock.UseClient(ssn.KubeClient())
	if err := sh.acquireNodeLock(nodeName, task.Pod); err != nil {
		return err
	}
	return sh.allocateDevices(sJob, task, node, nodeName)
}

func (sh *ScheduleHandler) acquireNodeLock(nodeName string, pod *v1.Pod) error {
	if pod == nil {
		return nil
	}
	for attempt := 0; attempt < maxLockRetries; attempt++ {
		lockErr := nodelock.LockNodeWithPod(nodeName, util.VXPULockName, pod)
		if lockErr == nil {
			return nil
		}
		sh.logLockRetry(nodeName, attempt, lockErr)
		time.Sleep(lockRetryInterval)
	}
	klog.V(util.LogErrorLevel).Infof("AllocateXPUForTask lock node %s failed after %d attempts", nodeName, maxLockRetries)
	return fmt.Errorf("lock node %s failed after %d attempts", nodeName, maxLockRetries)
}

func (sh *ScheduleHandler) logLockRetry(nodeName string, attempt int, lockErr error) {
	lockState, _ := nodelock.GetLockState(nodeName, util.VXPULockName)
	if lockState.IsHeld() {
		klog.V(util.LogWarningLevel).Infof(
			"AllocateXPUForTask node %s is locked by pod %s/%s since %v (age: %v), retry %d/%d: %v",
			nodeName, lockState.Namespace, lockState.PodName, lockState.Time,
			time.Since(lockState.Time), attempt+1, maxLockRetries, lockErr)
	} else {
		klog.V(util.LogWarningLevel).Infof(
			"AllocateXPUForTask node %s lock acquisition failed, retry %d/%d: %v",
			nodeName, attempt+1, maxLockRetries, lockErr)
	}
}

func (sh *ScheduleHandler) allocateDevices(sJob *SchedulerJob, task *api.TaskInfo, node *api.NodeInfo, nodeName string) error {
	sh.GetAllocatableXPUDevicesByNode(sJob)
	err := sJob.handler.Allocate(sJob, task, node, sh.getNodeXPUDevices(nodeName), sh.Templates)
	if err != nil {
		klog.V(util.LogErrorLevel).Infof("AllocateXPUForTask allocate failed: %v.", err)
		if task.Pod != nil {
			if releaseErr := nodelock.ReleaseNodeLockWithPod(nodeName, util.VXPULockName, task.Pod); releaseErr != nil {
				klog.V(util.LogErrorLevel).Infof("AllocateXPUForTask release lock failed: %v", releaseErr)
			}
		}
		return fmt.Errorf("allocate failed on node %s: %v", nodeName, err)
	}
	if task.Pod != nil {
		nodelock.ReleaseNodeMemoryLockOnly(nodeName)
	}
	return nil
}

func (sh *ScheduleHandler) DeallocateXPUForTask(task *api.TaskInfo) {
	if sh == nil || task == nil {
		klog.V(util.LogErrorLevel).Infof("DeallocateXPUForTask failed: %s.", util.ArgumentError)
		return
	}
	if task.Pod != nil && task.NodeName != "" {
		if err := nodelock.ReleaseNodeLockAnnotationOnly(task.NodeName, util.VXPULockName, task.Pod); err != nil {
			klog.V(util.LogErrorLevel).Infof("DeallocateXPUForTask release lock failed: %v, node: %s, pod: %s/%s",
				err, task.NodeName, task.Pod.Namespace, task.Pod.Name)
		}
	}
}
