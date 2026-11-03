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
package main

import (
	"sync"

	"k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/internal/xpu"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/plugin"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

const (
	PluginName = "volcano-vxpu"
)

var scheduleHandler *plugin.ScheduleHandler

func init() {
	scheduleHandler = NewXpuScheduleHandler()
}

type volcanoXPUPlugin struct {
	Scheduler *plugin.ScheduleHandler
	Arguments framework.Arguments
}

// NewXpuScheduleHandler XPU scheduler plugin start by frame.
func NewXpuScheduleHandler() *plugin.ScheduleHandler {
	sh := &plugin.ScheduleHandler{
		XPUPlugins:     map[string]plugin.XPUBuilder{},
		XPUDevices:     map[string]map[int]*common.XPUDevice{},
		Jobs:           map[api.JobID]*plugin.SchedulerJob{},
		DeleteJobInfos: map[api.JobID]*api.JobInfo{},
		SessionID:      "",
		Templates:      map[string]map[string]*plugin.TemplateInfo{},
		Mutex:          &sync.Mutex{},
	}
	sh.RegisterXPUScheduler(util.NPUPluginName, xpu.GetNPUPlugin)
	klog.V(util.LogDebugLevel).Infof("NewXpuScheduleHandler %#v.", sh.XPUPlugins)
	return sh
}

// New return huawei xpu plugin
func New(arguments framework.Arguments) framework.Plugin {
	return &volcanoXPUPlugin{Scheduler: scheduleHandler, Arguments: arguments}
}

func (xp *volcanoXPUPlugin) Name() string {
	return PluginName
}

func addJobValidFn(ssn *framework.Session, xp *volcanoXPUPlugin) {
	ssn.AddJobValidFn(xp.Name(), func(obj interface{}) *api.ValidateResult {
		return xp.Scheduler.JobValid(obj)
	})
}

func addPredicateFn(ssn *framework.Session, xp *volcanoXPUPlugin) {
	ssn.AddPredicateFn(xp.Name(), func(taskInfo *api.TaskInfo, nodeInfo *api.NodeInfo) error {
		err := xp.Scheduler.NodePredicate(taskInfo, nodeInfo)
		if err != nil {
			xp.Scheduler.Jobs[taskInfo.Job].Lock()
			xp.Scheduler.Jobs[taskInfo.Job].Reason[err.Error()] += nodeInfo.Name + " "
			xp.Scheduler.Jobs[taskInfo.Job].Unlock()
		}
		return err
	})
}

func addBatchNodeOrderFn(ssn *framework.Session, xp *volcanoXPUPlugin) {
	ssn.AddBatchNodeOrderFn(xp.Name(), func(task *api.TaskInfo, nodes []*api.NodeInfo) (map[string]float64, error) {
		score, err := xp.Scheduler.BatchNodeOrderFn(task, nodes)
		if err != nil {
			if setErr := xp.Scheduler.MarkJobPendingWithReason(ssn.Jobs[task.Job], err.Error()); setErr != nil {
				klog.V(util.LogErrorLevel).Infof("%s setJobFailed err:%s.", PluginName, util.SafePrint(setErr))
			}
		}
		if vcJob, ok := xp.Scheduler.Jobs[task.Job]; ok && vcJob.JobReadyTag == false {
			if _, exist := xp.Scheduler.DeleteJobInfos[task.Job]; !exist {
				xp.Scheduler.DeleteJobInfos[task.Job] = ssn.Jobs[task.Job]
				delete(ssn.Jobs, task.Job)
			}
		}
		return score, nil
	})
}

func addEventHandler(ssn *framework.Session, xp *volcanoXPUPlugin) {
	ssn.AddEventHandler(&framework.EventHandler{
		AllocateFunc: func(event *framework.Event) {
			if event == nil {
				klog.V(util.LogErrorLevel).Infof("AllocateFunc event nil.")
				return
			}
			if err := xp.Scheduler.AllocateXPUForTask(event.Task, ssn); err != nil {
				klog.V(util.LogErrorLevel).Infof("AllocateFunc failed for task %s: %v", event.Task.Name, err)
				event.Err = err
			}
		},
		DeallocateFunc: func(event *framework.Event) {
			if event == nil {
				klog.V(util.LogErrorLevel).Infof("DeallocateFunc event nil.")
				return
			}
			xp.Scheduler.DeallocateXPUForTask(event.Task)
		},
	})
}

func addJobReadyFn(ssn *framework.Session, xp *volcanoXPUPlugin) {
	ssn.AddJobReadyFn(xp.Name(), func(obj interface{}) bool {
		ji, ok := obj.(*api.JobInfo)
		if !ok {
			klog.V(util.LogErrorLevel).Info("obj assertion failed.")
			return false
		}
		job, ok := xp.Scheduler.Jobs[ji.UID]
		if !ok {
			return true
		}
		return job.JobReadyTag
	})
}

func (xp *volcanoXPUPlugin) OnSessionOpen(ssn *framework.Session) {
	if xp == nil || ssn == nil {
		klog.V(util.LogErrorLevel).Infof("OnSessionOpen : %s.", util.ArgumentError)
		return
	}
	klog.V(util.LogDebugLevel).Infof("enter %s OnSessionOpen.", PluginName)
	defer klog.V(util.LogDebugLevel).Infof("leave %s OnSessionOpen.", PluginName)

	if err := xp.Scheduler.InitXPUSession(ssn); err != nil {
		klog.V(util.LogErrorLevel).Infof("InitXPUSession : %s, xpu plugin will not be initialized.", err)
		return
	}

	addJobValidFn(ssn, xp)
	addPredicateFn(ssn, xp)
	addBatchNodeOrderFn(ssn, xp)
	addEventHandler(ssn, xp)
	addJobReadyFn(ssn, xp)
}

func (xp *volcanoXPUPlugin) OnSessionClose(ssn *framework.Session) {
	if xp == nil || ssn == nil {
		klog.V(util.LogErrorLevel).Infof("OnSessionClose failed: %s.", util.ArgumentError)
		return
	}
	klog.V(util.LogDebugLevel).Infof("enter %s OnSessionClose.", PluginName)
	defer klog.V(util.LogDebugLevel).Infof("leave %s OnSessionClose.", PluginName)
	if ssn.Jobs == nil && len(xp.Scheduler.DeleteJobInfos) != 0 {
		ssn.Jobs = make(map[api.JobID]*api.JobInfo)
	}
	for _, job := range ssn.Jobs {
		if job.PodGroup.Status.Phase == util.PodGroupInqueue || job.PodGroup.Status.Phase == util.PodGroupPending {
			xp.Scheduler.MarkJobPendingDueToNodeFailure(job)
		}
		if len(job.PodGroup.Annotations) != 0 && job.PodGroup.Annotations[util.PodDeleteTimes] == util.TagOfPodPending {
			xp.Scheduler.UpdatePodGroupPendingReason(job, util.JobRestartReason)
		}
	}
	for jobId, jobInfo := range xp.Scheduler.DeleteJobInfos {
		ssn.Jobs[jobId] = jobInfo
	}
}
