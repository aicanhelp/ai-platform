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
package plugin

import (
	"errors"
	"fmt"
	"strings"
	"sync"

	"k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

// Init the SchedulerJob's init.
func (sJob *SchedulerJob) Init(vcJob *api.JobInfo, sh *ScheduleHandler) error {
	if sJob == nil || vcJob == nil {
		klog.V(util.LogErrorLevel).Infof("SchedulerJob_Init: parameter is nil.")
		return errors.New("parameter is nil")
	}
	if initErr := sJob.initJobInfo(vcJob); initErr != nil {
		klog.V(util.LogErrorLevel).Infof("%s initJobInfo %s", vcJob.UID, initErr)
		return initErr
	}

	if !sJob.isJobSupportByPlugin(sh) {
		klog.V(util.LogErrorLevel).Infof("%s IsJobSupportByPlugin not has suitable plugin.", sJob.Id)
		return fmt.Errorf("%s's plugin not registe", sJob.Id)
	}

	sJob.initPluginByJobInfo(sh)
	return nil
}

func (sJob *SchedulerJob) initJobInfo(vcJob *api.JobInfo) error {
	name, num, tasks := GetJobXPUTasks(vcJob)
	// job do not have xpu task
	if len(tasks) == 0 {
		return errors.New("initJobInfo failed: job's task is nil")
	}

	sJob.JobReadyTag = true
	sJob.UnschedulableReason = UnschedulableReason{
		Reason: map[string]string{},
		Mutex:  &sync.Mutex{},
	}
	sJob.Id = vcJob.UID
	sJob.NameSpace = vcJob.Namespace
	sJob.ReferenceName = util.GetJobReferenceName(vcJob)
	sJob.Selector = GetSelectorFromVcJob(vcJob)
	sJob.Label = GetLabelFromVcJob(vcJob)
	sJob.Annotation = vcJob.PodGroup.Annotations
	sJob.handler = nil
	sJob.XPUJob = &util.XPUJob{
		ReqXPUName: name,
		ReqXPUNum:  num,
		Tasks:      tasks,
	}
	if name == "" {
		return errors.New("initJobInfo failed: nil XPU")
	}
	sJob.XPUTaskNum = sJob.GetXPUTaskNumInJob()
	if vcJob.MinAvailable != int32(len(vcJob.Tasks)) {
		sJob.SchedulingTaskNum = defaultSchedulingTaskNum
		return nil
	}
	sJob.SchedulingTaskNum = sJob.GetSchedulingTaskNum()
	return nil
}

func (sJob *SchedulerJob) isJobSupportByPlugin(sh *ScheduleHandler) bool {
	name := sJob.getPluginNameByReq()
	if name == "" {
		return false
	}
	return sh.IsPluginRegistered(name)
}

func isSelectorContains(value, jobValue string) bool {
	for _, v := range strings.Split(value, "|") {
		if strings.EqualFold(v, jobValue) {
			return true
		}
	}
	return false
}

// GetTaskSelectors get task's selector.
func GetTaskSelectors(task *api.TaskInfo) map[string]string {
	if task == nil {
		klog.V(util.LogErrorLevel).Infof("GetTaskSelectors task nil.")
		return nil
	}
	return task.Pod.Spec.NodeSelector
}

// GetTaskLabels get task's Labels.
func GetTaskLabels(task *api.TaskInfo) map[string]string {
	if task == nil {
		klog.V(util.LogErrorLevel).Infof("GetTaskLabels task nil.")
		return nil
	}
	return task.Pod.Labels
}

func getLabel(res map[string]string, taskSelector map[string]string) {
	if res == nil {
		klog.V(util.LogErrorLevel).Infof("getLabel nil.")
		return
	}
	for k, v := range taskSelector {
		label, ok := res[k]
		if !ok {
			res[k] = v
			continue
		}
		if isSelectorContains(label, v) {
			continue
		}
		res[k] = label + "|" + v
	}
}

// GetLabelFromVcJob get job's label, not task's.
func GetLabelFromVcJob(job *api.JobInfo) map[string]string {
	if job == nil {
		klog.V(util.LogErrorLevel).Infof("GetLabelFromVcJob job nil.")
		return nil
	}
	res := make(map[string]string, util.MapInitNum)
	for labelKey, labelValue := range job.PodGroup.Labels {
		res[labelKey] = labelValue
	}
	for _, task := range job.Tasks {
		taskSelector := GetTaskLabels(task)
		getLabel(res, taskSelector)
	}
	return res
}

// GetSelectorFromVcJob get job selector.
func GetSelectorFromVcJob(job *api.JobInfo) map[string]string {
	var res = make(map[string]string, util.MapInitNum)
	for _, task := range job.Tasks {
		taskSelector := task.Pod.Spec.NodeSelector
		getLabel(res, taskSelector)
	}
	return res
}

// GetJobXPUTasks get XPUTask from jobInfo.
func GetJobXPUTasks(vcJob *api.JobInfo) (string, int, map[api.TaskID]*util.XPUTask) {
	if vcJob == nil {
		return "", 0, nil
	}
	if len(vcJob.Tasks) == 0 {
		klog.V(util.LogWarningLevel).Infof("GetJobXPUTasks %s not init has no task.", vcJob.Name)
		return "", 0, nil
	}

	name := ""
	num := 0
	resultMap := make(map[api.TaskID]*util.XPUTask, util.MapInitNum)
	for taskID, taskInfo := range vcJob.Tasks {
		tr := GetXPUResourceFromTaskInfo(taskInfo, util.VNPUName)

		// task is not xpu task, continue
		if tr.ReqXPUNum <= 0 {
			continue
		}

		resultMap[taskID] = &util.XPUTask{
			Name:         taskInfo.Name,
			NameSpace:    taskInfo.Namespace,
			TaskResource: tr,
			Selector:     GetTaskSelectors(taskInfo),
			Label:        GetTaskLabels(taskInfo),
			NodeName:     taskInfo.NodeName,
			Annotation:   taskInfo.Pod.Annotations,
			PodStatus:    taskInfo.Pod.Status.Phase,
			ScoreMap:     make(map[string]float64),
		}
		if name == "" {
			name = tr.ReqXPUName
		} else if tr.ReqXPUName != "" && tr.ReqXPUName != name {
			klog.V(util.LogErrorLevel).Infof(
				"GetJobXPUTasks %s error, a job can only apply for one type of XPU resource.", vcJob.Name)
			return "", 0, nil
		} else {
			// name matches ReqXPUName; no action needed
		}
		num += tr.ReqXPUNum
	}
	return name, num, resultMap
}

func (sJob *SchedulerJob) initPluginByJobInfo(sh *ScheduleHandler) {
	if sJob == nil {
		return
	}
	pluginName := sJob.getPluginNameByReq()
	if pluginName == "" {
		return
	}
	plugin, ok := sh.XPUPlugins[pluginName]
	if !ok {
		return
	}
	sJob.handler = plugin()
}

func (sJob *SchedulerJob) getPluginNameByReq() string {
	name := sJob.ReqXPUName
	if strings.Contains(name, util.VNPUName) {
		return util.NPUPluginName
	}
	return ""
}

// ValidJobFn valid job.
func (sJob SchedulerJob) ValidJobFn() *api.ValidateResult {
	if result := sJob.handler.ValidXPUJob(); result != nil {
		klog.V(util.LogErrorLevel).Infof("%s ValidXPUJob failed:%s.", PluginName, result.Message)
		return result
	}
	klog.V(util.LogDebugLevel).Infof("%s valid ok.", sJob.Id)
	return nil
}

// preCheckNodePredicate PreCheck Predicate nodes.
func (sJob SchedulerJob) preCheckNodePredicate(taskInfo *api.TaskInfo, nodeInfo *api.NodeInfo) error {
	if !util.SelectorMatches(sJob.Selector, nodeInfo.Node.Labels) {
		err := fmt.Errorf("job selector not match node label")
		klog.V(util.LogWarningLevel).Infof("job(%s) selector not match node<%s> label", sJob.Id, nodeInfo.Name)
		return err
	}
	return nil
}
