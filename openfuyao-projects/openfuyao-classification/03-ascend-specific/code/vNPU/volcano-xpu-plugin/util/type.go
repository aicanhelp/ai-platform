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

const (
	// LogErrorLevel for log error.
	LogErrorLevel = 1
	// LogWarningLevel for log warning.
	LogWarningLevel = 2
	// LogInfoLevel for log information.
	LogInfoLevel = 3
	// LogDebugLevel for log debug.
	LogDebugLevel = 4
	// ArgumentError argument nil error.
	ArgumentError = "invalid argument"
	// ObjectNilError object or argument nil error.
	ObjectNilError = "object or argument is nil"
	// Comma for ,
	Comma = ","
	// Semicolon for ;
	Semicolon = ";"
	// XPUHexKilo for const 1000,volcano frame used.
	XPUHexKilo = 1000
	// MapInitNum for map init length.
	MapInitNum = 3
	// XPUDeviceLen for xpu device param number.
	XPUDeviceLen = 9
	// ContainerLength for container device param number.
	ContainerLength = 7
	// Base2 for const 2.
	Base2 = 2
	// Base10 for const 10.
	Base10 = 10
	// Base100 for const 100.
	Base100 = 100
	// Base1024 for const 1024.
	Base1024 = 1024
	// XpuMultiplier for calculate score without topology
	XpuMultiplier = 100
)

const (
	// PodGroupInqueue the pg Inqueue status
	PodGroupInqueue = "Inqueue"
	// PodGroupPending the pg Pending status
	PodGroupPending = "Pending"
	// PodGroupRunning the pg Running status
	PodGroupRunning = "Running"
	// PodGroupUnknown the pg Unknown status
	PodGroupUnknown = "Unknown"
	// PodGroupUnschedulableType the pg Unschedulable Condition
	PodGroupUnschedulableType = "Unschedulable"
	// PodDeleteTimes the tag of single pod has been deleted
	PodDeleteTimes = "pod-delete-times"
	// TagOfPodPending the limitation on pod pending times
	TagOfPodPending = "ready"
	// JobRestartReason the reason of job restart
	JobRestartReason = "task reach max pending session"
)

const (
	// NPUPluginName for npu
	NPUPluginName = "npu"

	// VNPUName for NPU card
	VNPUName = "huawei.com/vnpu-number"
	// VNPUType for NPU card, for example: huawei.com/vnpu-type.310P: 1
	VNPUType = "huawei.com/vnpu-type."
	// VNPUCore for vnpu coreP
	VNPUCore = "huawei.com/vnpu-cores"
	// VNPUMemory for vnpu memory
	VNPUMemory = "huawei.com/vnpu-memory.1Gi"
	// AscendNPUDevice device type supported by the device plugin
	AscendNPUDevice = "NPU"
	// NodeNPURegisterAnnotation for npu register annotation
	NodeNPURegisterAnnotation = "huawei.com/node-vnpu-register"
	// AssignedNPUsToAllocateAnnotations pod's npu devices to allocate to container
	AssignedNPUsToAllocateAnnotations = "huawei.com/vnpu-devices-to-allocate"
	// AssignedNPUsToPodAnnotations pod's devices to allocate to container by scheduler
	AssignedNPUsToPodAnnotations = "huawei.com/vnpu-ids-new"
	// AssignedNPUsToNodeAnnotations assigned node name
	AssignedNPUsToNodeAnnotations = "huawei.com/vnpu-node"
	// NodeNPUTopologyAnnotation for node npu topology
	NodeNPUTopologyAnnotation = "huawei.com/node-npu-topology"
	// NodeNPUHandshakeAnnotation handshake timestamp for vnpu register
	NodeNPUHandshakeAnnotation = "huawei.com/node-vnpu-handshake"

	// BindTimeAnnotations set pod BindTimeAnnotations for using by device-plugin.
	BindTimeAnnotations = "huawei.com/bind-time"
	// DeviceBindPhase allocate bind phase
	DeviceBindPhase = "huawei.com/bind-phase"
	// DeviceBindAllocating bind phase allocating
	DeviceBindAllocating = "allocating"

	// VXPULockName is the annotation key for node-level VXPU allocation lock
	VXPULockName = "vnpu/mutex.lock"
)

const (
	ConfigMapNamespace   = "xpu"
	TemplateConfigCmName = "vnpu.virtual-template.config"
	TemplateYamlName     = "virtual-template-config.yaml"
	DeviceType           = "NPU-ASCEND-"

	AiCpuLevelHigh           = "high"
	AiCpuLevelLow            = "low"
	HardAiCpuLevelAnnotation = "huawei.com/vnpu-hard-aicpu-level"

	VNPUModeAnnotation = "huawei.com/vnpu-mode"
	SoftMode           = "soft"
	HardMode           = "hard"
	FullCore           = 100
	FullCardTemplate   = "full"

	VNPUUseDieIDAnnotation   = "huawei.com/vnpu-use-npu-dieID"
	VNPUUseNPUTypeAnnotation = "huawei.com/vnpu-use-npu-type"

	SoftSchedulerPolicyAnnotation = "huawei.com/vnpu-soft-scheduler-policy"
	PolicyFixedShare              = "fixed-share"
	PolicyElastic                 = "elastic"
	PolicyBestEffort              = "best-effort"

	NodeSchedulerPolicyAnnotation   = "huawei.com/vnpu-pod-node-scheduler-policy"
	DeviceSchedulerPolicyAnnotation = "huawei.com/vnpu-pod-device-scheduler-policy"
	SchedulerPolicyBinpack          = "binpack"
	SchedulerPolicySpread           = "spread"

	CoreDeviationRatio = 0.05
)
