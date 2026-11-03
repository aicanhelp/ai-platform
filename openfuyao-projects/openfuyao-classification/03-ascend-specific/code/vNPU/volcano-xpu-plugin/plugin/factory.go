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
	"context"
	"errors"
	"fmt"
	"strings"

	"gopkg.in/yaml.v2"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/framework"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/nodelock"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

// RegisterXPUScheduler register the plugin,like factory.
func (sh *ScheduleHandler) RegisterXPUScheduler(name string, xb XPUBuilder) {
	if sh == nil || xb == nil {
		klog.V(util.LogErrorLevel).Infof("RegisterXPUScheduler : %s.", util.ObjectNilError)
		return
	}
	if _, ok := sh.XPUPlugins[name]; ok {
		klog.V(util.LogErrorLevel).Infof("XPU Scheduler[%s] has been registered before.", name)
		return
	}

	sh.XPUPlugins[name] = xb
	klog.V(util.LogDebugLevel).Infof("XPU Scheduler[%s] registered.", name)

	sh.getTemplateConfigFromCm()
}

func (sh *ScheduleHandler) getTemplateConfigFromCm() {
	clientSet, err := getK8sClient()
	if err != nil {
		klog.Errorf("getTemplateConfigFromCm err: %v", err)
		return
	}

	cms, err := clientSet.CoreV1().ConfigMaps("").List(context.Background(), metav1.ListOptions{})
	if err != nil {
		klog.Errorf("getTemplateConfigFromCm err: %v", err)
		return
	}

	var templateData VnpuTemplateData
	for _, cm := range cms.Items {
		if cm.Name == util.TemplateConfigCmName {
			yamlData, exists := cm.Data[util.TemplateYamlName]
			if !exists {
				klog.Errorf("getTemplateConfigFromCm err: ConfigMap not exist key: %s", util.TemplateYamlName)
				return
			}
			if err := yaml.Unmarshal([]byte(yamlData), &templateData); err != nil {
				klog.Errorf("getTemplateConfigFromCm err: parse template yaml failed %v", err)
				return
			}
			break
		}
	}

	for _, v := range templateData.Vnpus {
		templateMap := map[string]*TemplateInfo{}
		for _, t := range v.Templates {
			templateMap[t.Name] = &TemplateInfo{
				Name:        t.Name,
				AiCore:      t.AiCore,
				AiCPU:       t.AiCPU,
				TotalAiCore: v.AiCore,
				TotalAiCpu:  v.AiCPU,
			}
		}
		templateMap[util.FullCardTemplate] = &TemplateInfo{
			Name:        util.FullCardTemplate,
			AiCore:      v.AiCore,
			AiCPU:       v.AiCPU,
			TotalAiCore: v.AiCore,
			TotalAiCpu:  v.AiCPU,
		}
		sh.Templates[v.ChipName] = templateMap
	}
}

func getK8sClient() (kubernetes.Interface, error) {
	config, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get k8s config. Error: %v", err)
	}
	clientSet, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create clientset. Error: %v", err)
	}
	return clientSet, nil
}

// UnRegisterXPUScheduler unRegister the plugin
func (sh *ScheduleHandler) UnRegisterXPUScheduler(name string) error {
	if sh == nil {
		return errors.New(util.ArgumentError)
	}
	if _, ok := sh.XPUPlugins[name]; ok {
		sh.XPUPlugins[name] = nil
		delete(sh.XPUPlugins, name)
		klog.V(util.LogDebugLevel).Infof("XPU Scheduler[%s] delete.", name)
	}
	klog.V(util.LogDebugLevel).Infof("XPU Scheduler[%s] unRegistered.", name)
	return nil
}

// IsPluginRegistered Determine if the plugin is registered.
func (sh *ScheduleHandler) IsPluginRegistered(name string) bool {
	if sh == nil {
		klog.V(util.LogErrorLevel).Infof("IsPluginRegistered %s", util.ObjectNilError)
		return false
	}

	if _, ok := sh.XPUPlugins[name]; ok {
		return true
	}

	klog.V(util.LogErrorLevel).Infof("IsPluginRegistered %s not in XPUPlugins %+v", name, sh.XPUPlugins)
	return false
}

// InitXPUSession init xpu plugin and nodes.
func (sh *ScheduleHandler) InitXPUSession(ssn *framework.Session) error {
	if sh == nil || ssn == nil {
		klog.V(util.LogErrorLevel).Infof("InitXPUSession failed: %s.", util.ArgumentError)
		return errors.New(util.ArgumentError)
	}
	klog.V(util.LogDebugLevel).Infof("enter %s InitXPUSession.", PluginName)
	defer klog.V(util.LogDebugLevel).Infof("leave %s InitXPUSession.", PluginName)

	sh.InitJobsInSession(ssn)
	sh.InitDeleteJobInfos()
	sh.SessionID = ssn.UID
	sh.Nodes = ssn.NodeList
	nodelock.UseClient(ssn.KubeClient())
	return nil
}

// InitDeleteJobInfos init empty deleted jobinfos.
func (sh *ScheduleHandler) InitDeleteJobInfos() {
	if sh == nil {
		klog.V(util.LogErrorLevel).Infof("InitDeleteJobInfos failed: %s.", util.ArgumentError)
		return
	}
	sh.DeleteJobInfos = map[api.JobID]*api.JobInfo{}
}

func (sh *ScheduleHandler) getNodeXPUDevices(nodeName string) map[int]*common.XPUDevice {
	sh.Lock()
	xpuDevices, ok := sh.XPUDevices[nodeName]
	sh.Unlock()
	if !ok {
		xpuDevices = map[int]*common.XPUDevice{}
		klog.V(util.LogDebugLevel).Infof("getNodeXPUDevices xpu devices of node %s is empty", nodeName)
	}
	return xpuDevices
}

// BatchNodeOrderFn Score the selected nodes.
func (sh *ScheduleHandler) BatchNodeOrderFn(task *api.TaskInfo, nodes []*api.NodeInfo) (map[string]float64, error) {
	if sh == nil || task == nil || len(nodes) == 0 {
		klog.V(util.LogErrorLevel).Infof("BatchNodeOrderFn failed: %s.", util.ArgumentError)
		return nil, errors.New(util.ArgumentError)
	}
	klog.V(util.LogDebugLevel).Infof("Enter batchNodeOrderFn")
	defer klog.V(util.LogDebugLevel).Infof("leaving batchNodeOrderFn")

	scores := make(map[string]float64, len(nodes))
	for _, node := range nodes {
		if node == nil {
			continue
		}
		scores[node.Name] = 0.0
	}

	sJob, ok := sh.Jobs[task.Job]
	if !ok {
		klog.V(util.LogWarningLevel).Infof("BatchNodeOrderFn %s not req xpu.", task.Name)
		return scores, nil
	}

	xpuTask, ok := sJob.Tasks[task.UID]
	if !ok {
		klog.V(util.LogErrorLevel).Infof("task[%v] not exist", task.Name)
		return make(map[string]float64), nil
	}
	return xpuTask.ScoreMap, nil
}

// GetAllocatableXPUDevicesByNode get allocatable xpu devices by node.
func (sh *ScheduleHandler) GetAllocatableXPUDevicesByNode(sJob *SchedulerJob) map[string][]*common.XPUDevice {
	unUseXPUDevicesOfNodes := make(map[string][]*common.XPUDevice)
	for _, v := range sh.Nodes {
		sh.InitNodeXPUDevices(sJob, v)
		xpuDevices := sh.getNodeXPUDevices(v.Name)
		var unUseXPUDevices []*common.XPUDevice
		for _, v := range xpuDevices {
			if !v.InUse && v.Health {
				unUseXPUDevices = append(unUseXPUDevices, v)
			}
		}
		unUseXPUDevicesOfNodes[v.Name] = unUseXPUDevices
	}
	return unUseXPUDevicesOfNodes
}

// InitJobsInSession init jobs in session
func (sh *ScheduleHandler) InitJobsInSession(ssn *framework.Session) {
	if sh == nil || ssn == nil {
		klog.V(util.LogErrorLevel).Infof("InitJobsInSession failed: %s.", util.ArgumentError)
		return
	}
	sh.Jobs = make(map[api.JobID]*SchedulerJob, util.MapInitNum)
	for jobID, jobInfo := range ssn.Jobs {
		sJob := SchedulerJob{}
		if err := sJob.Init(jobInfo, sh); err != nil {
			klog.V(util.LogErrorLevel).Infof("%s InitJobsInSession failed: %s.", jobInfo.Name, util.SafePrint(err))
			continue
		}
		sh.Jobs[jobID] = &sJob
	}
	return
}

// JobValid the job valid, used by volcano frame.
func (sh *ScheduleHandler) JobValid(obj interface{}) *api.ValidateResult {
	if sh == nil {
		return &api.ValidateResult{Pass: false, Reason: util.ObjectNilError,
			Message: fmt.Sprintf("validJobFn [%#v] failed:%s", obj, util.ObjectNilError)}
	}
	klog.V(util.LogDebugLevel).Infof("enter job valid")
	defer klog.V(util.LogDebugLevel).Infof("leave job valid")
	job, ok := obj.(*api.JobInfo)
	if !ok {
		reason := "job convert failed"
		klog.V(util.LogErrorLevel).Infof("%s :%#v.", reason, obj)
		return &api.ValidateResult{Pass: false, Reason: reason,
			Message: fmt.Sprintf("validJobFn [%#v] failed:%s", obj, reason)}
	}
	if !IsJobReady(job) {
		reason := "job is not ready"
		klog.V(util.LogErrorLevel).Infof("%s job(%s) not ready:%s.", PluginName, job.Name,
			job.PodGroup.Status.Phase)
		return &api.ValidateResult{Pass: false, Reason: reason,
			Message: fmt.Sprintf("validJobFn [%#v] failed:%s", obj, reason)}
	}
	vcJob, ok := sh.Jobs[job.UID]
	if !ok {
		klog.V(util.LogErrorLevel).Infof("%s %s not support or init", PluginName, job.Name)
		return nil
	}

	result := vcJob.ValidJobFn()
	if result != nil {
		if setErr := sh.MarkJobPendingWithReason(job, result.Message); setErr != nil {
			klog.V(util.LogErrorLevel).Infof("%s setJobFailed err: %s.", PluginName, util.SafePrint(setErr))
		}
		return result
	}
	return nil
}

// MarkJobPendingWithReason mark the pod and podGroup pending reason.
func (sh *ScheduleHandler) MarkJobPendingWithReason(vcJob *api.JobInfo, reason interface{}) error {
	if sh == nil || vcJob == nil {
		klog.V(util.LogErrorLevel).Infof("MarkJobPendingWithReason job null")
		return errors.New(util.ArgumentError)
	}

	var tmpReason string
	switch value := reason.(type) {
	case string:
		vcJob.JobFitErrors = value
		tmpReason = value
	case map[api.TaskID]*api.FitErrors:
		vcJob.NodesFitErrors = value
		for _, nodeErrors := range value {
			tmpReason += nodeErrors.Error()
		}
	default:
		return fmt.Errorf("assert reason(%T) failed", reason)
	}
	sh.UpdatePodGroupPendingReason(vcJob, tmpReason)
	return nil
}

// UpdatePodGroupPendingReason update pod group pending reason
func (sh *ScheduleHandler) UpdatePodGroupPendingReason(job *api.JobInfo, reason string) {
	job.JobFitErrors = reason
	if len(job.PodGroup.Status.Conditions) == 0 {
		return
	}

	jobCondition := job.PodGroup.Status.Conditions[0].DeepCopy()
	jobCondition.Type = util.PodGroupUnschedulableType
	jobCondition.Status = v1.ConditionTrue
	jobCondition.LastTransitionTime = metav1.Now()
	jobCondition.TransitionID = string(sh.SessionID)
	jobCondition.Reason = reason
	jobCondition.Message = reason

	for k, value := range job.PodGroup.Status.Conditions {
		if strings.Contains(value.Message, reason) {
			job.PodGroup.Status.Conditions[k].LastTransitionTime = jobCondition.LastTransitionTime
			job.PodGroup.Status.Conditions[k].TransitionID = jobCondition.TransitionID
			return
		}
	}
	job.PodGroup.Status.Conditions = append(job.PodGroup.Status.Conditions, *jobCondition)
}

// MarkJobPendingDueToNodeFailure Marks job as not blocked if enough tasks remain after excluding node fit failures.
func (sh ScheduleHandler) MarkJobPendingDueToNodeFailure(job *api.JobInfo) {
	if int32(len(job.Tasks)-len(job.NodesFitErrors)) >= job.MinAvailable {
		klog.V(util.LogDebugLevel).Infof(
			"Job %s is not blocked by node constraints: total tasks=%d, min required=%d, node fit failures=%d",
			job.Name, len(job.Tasks), job.MinAvailable, len(job.NodesFitErrors),
		)
		return
	}
	if err := sh.MarkJobPendingWithReason(job, job.NodesFitErrors); err != nil {
		klog.V(util.LogErrorLevel).Infof("%s mark job pending err:%s.", PluginName, err)
	}
}

// jobTerminatingPodCount counts the number of Pods in a Job that are in a terminating state.
func jobTerminatingPodCount(job *api.JobInfo) int {
	count := 0
	for _, task := range job.Tasks {
		if task.Pod != nil && task.Pod.DeletionTimestamp != nil {
			count++
		}
	}
	return count
}

// IsJobReady Determine if the task is ready.
func IsJobReady(job *api.JobInfo) bool {
	if job.ValidTaskNum() < job.MinAvailable {
		klog.V(util.LogErrorLevel).Infof("Job %s/%s not ready: valid %d < min %d",
			job.Namespace, job.Name, job.ValidTaskNum(), job.MinAvailable)
		return false
	}
	if jobTerminatingPodCount(job) > 0 {
		klog.V(util.LogErrorLevel).Infof("Job %s/%s not ready: %d terminating pods",
			job.Namespace, job.Name, jobTerminatingPodCount(job))
		return false
	}
	return true
}
