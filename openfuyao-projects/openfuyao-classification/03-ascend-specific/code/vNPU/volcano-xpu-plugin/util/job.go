/*
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Package plugin is using for HuaWei Ascend pin affinity schedule frame.
*/
package util

import (
	"strings"

	"k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/api"
)

// XPUJob only xpu vcJob have.
type XPUJob struct {
	// the mapKey is taskID, not Name.
	Tasks             map[api.TaskID]*XPUTask
	SelectServers     string
	XPUTaskNum        int
	SchedulingTaskNum int
	ReqXPUName        string
	ReqXPUNum         int
}

// GetXPUTaskNumInJob get the XPU task number in one job. for some task has no XPU.
func (xJob *XPUJob) GetXPUTaskNumInJob() int {
	if xJob == nil || !IsXPUName(xJob.ReqXPUName) {
		return 0
	}
	taskNum := 0
	for _, task := range xJob.Tasks {
		if IsXPUName(task.ReqXPUName) {
			taskNum++
		}
	}
	return taskNum
}

// GetSchedulingTaskNum get the num of scheduling task
func (xJob *XPUJob) GetSchedulingTaskNum() int {
	if xJob == nil || !IsXPUName(xJob.ReqXPUName) {
		return 0
	}
	schedulingTaskNum := 0
	for _, task := range xJob.Tasks {
		if task.NodeName == "" {
			schedulingTaskNum++
		}
	}
	return schedulingTaskNum
}

// GetJobReferenceName get job reference name
func GetJobReferenceName(job *api.JobInfo) string {
	if job == nil || job.PodGroup == nil {
		return ""
	}
	ownerReferences := job.PodGroup.OwnerReferences
	if len(ownerReferences) == 0 {
		return ""
	}
	return ownerReferences[0].Name
}

// SelectorMatches check the selectors
func SelectorMatches(jobSelectors, conf map[string]string) bool {
	for jobKey, jobValue := range jobSelectors {
		value, ok := conf[jobKey]
		if !ok {
			klog.V(LogErrorLevel).Infof("conf has no job selector key:%s.", jobKey)
			return false
		}
		if !strings.Contains(value, jobValue) {
			klog.V(LogErrorLevel).Infof("conf has no job selector value:%s.", jobValue)
			return false
		}
	}
	return true
}
