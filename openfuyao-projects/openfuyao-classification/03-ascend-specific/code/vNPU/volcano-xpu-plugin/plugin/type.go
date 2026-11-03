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
	"sync"

	"k8s.io/apimachinery/pkg/types"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

const (
	PluginName = "volcano-vxpu"
)

const (
	scoreWeight              = 100
	defaultSchedulingTaskNum = -1
	maxSoftNodeScore         = 200
	maxHardNodeScore         = 100
)

// ScheduleHandler information for the current plugin
type ScheduleHandler struct {
	XPUPlugins     map[string]XPUBuilder
	XPUDevices     map[string]map[int]*common.XPUDevice
	Jobs           map[api.JobID]*SchedulerJob
	DeleteJobInfos map[api.JobID]*api.JobInfo
	SessionID      types.UID
	Nodes          []*api.NodeInfo
	Templates      TemplateInfos
	*sync.Mutex
}

// SchedulerJob the plugin define job info
type SchedulerJob struct {
	Id            api.JobID
	ReferenceName string
	NameSpace     string
	Annotation    map[string]string
	Selector      map[string]string
	Label         map[string]string
	UnschedulableReason
	handler     XPUSchedulerPlugin
	JobReadyTag bool
	*util.XPUJob
}

// UnschedulableReason the message of pod pending
type UnschedulableReason struct {
	Reason map[string]string
	*sync.Mutex
}

// ContainerDevices description of all xpus in the container
type ContainerDevices []common.ContainerDevice

// PodDevices description of all xpus in the pod
type PodDevices []ContainerDevices
