/*
 *
 *  * Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

// common 常量
package common

import (
	"os"
	"time"
)

const (

	// FreshContainerMinMetricsResolution is the minimum resolution of metrics for fresh container
	FreshContainerMinMetricsResolution = 10 * time.Second

	// DefaultInsecureKubeletMetricsPort is the default port for kubelet metrics
	DefaultInsecureKubeletMetricsPort = 10255

	// DefaultSecureKubeletMetricsPort is the default port for kubelet metrics
	DefaultSecureKubeletMetricsPort = 10250

	// DefaultMetricsScrapePeriod is the default period for scraping metrics
	DefaultMetricsScrapePeriod = 60 * time.Second

	// DefaultReportsUploadPeriod is the default period for uploading reports to master
	DefaultReportsUploadPeriod = 300 * time.Second

	// DefaultReportsBufCapacity is the default capacity of reports buffer
	DefaultReportsBufCapacity = 5

	// DefaultExporterPort is the default port for include exporter
	DefaultExporterPort = 8900

	// DefaultExporterPeriod is the default period for exporting metrics
	DefaultExporterPeriod = 30 * time.Second

	// CushionTime is the cushion time for metrics
	CushionTime = 5 * time.Second

	// DefaultHttpClientTimeout for http client
	DefaultHttpClientTimeout = 10 * time.Second

	// NanoCoresPerCore per core
	NanoCoresPerCore = 1e9
)

const (
	// DefaultLog Always logged
	DefaultLog = iota
	// BasicDebugLog Important but non-critical logs
	BasicDebugLog
	// GeneralDebugLog More detailed operational logs
	GeneralDebugLog
	// VerboseDebugLog Granular internal state
	VerboseDebugLog
	// AdvanceDebugLog Low-level details
	AdvanceDebugLog
	// TraceDebugLog Extremely verbose
	TraceDebugLog
	// MicroscopicDebugLog Overwhelming detail
	MicroscopicDebugLog
)

// OverheadFactor is the factor of overhead
const OverheadFactor = 0.1

// SpecNodeNameField is the field of spec.nodeName
const SpecNodeNameField = "spec.nodeName"

const (
	// ColocationNodeLabel 混部节点标签
	ColocationNodeLabel string = "node.openfuyao.com/colocation"
	// OversubNodeLabel 超卖节点标签
	OversubNodeLabel string = "node.openfuyao.com/oversold"
	// DefaultCgroupRoot specifies the default cgroup root directory.
	DefaultCgroupRoot = "/sys/fs/cgroup"

	// PriorityAnnotationKey is annotation key to mark offline pod in rubik
	PriorityAnnotationKey = "volcano.sh/preemptable"
	// ExtenderResourceCPU is the annotation key to mark be cpu resource
	ExtenderResourceCPU = "kubernetes.io/be-cpu"
	// ExtenderResourceMemory is the annotation key to mark be memory resource
	ExtenderResourceMemory = "kubernetes.io/be-memory"
	// QosClassAnnotationKey is annotation key to mark overquota qos level
	QosClassAnnotationKey = "openfuyao.com/qos-level"
	// QosBe is the qos level for BE pod
	QosBe QosLevel = "BE"
	// QosLs is the qos level for LS pod
	QosLs QosLevel = "LS"
	// QosHLS is the qos level for HLS pod
	QosHLS QosLevel = "HLS"
	// QosNone is the qos level for None pod
	QosNone QosLevel = "None"

	// OversoldAnnotation is the annotation for oversold status.
	OversoldAnnotation = "openfuyao.com/oversold"
	// ExtenderResourceConfigurationAnnotation is the annotation for extender resource configuration.
	ExtenderResourceConfigurationAnnotation = "openfuyao.com/extender-resource-cfg"
	// NumaResourceAnnotation is the annotation for NUMA resource allocation.
	NumaResourceAnnotation = "openfuyao.com/numa-resource"
	// ContainerEngineAnnotation is the annotation for container engine type.
	ContainerEngineAnnotation = "openfuyao.com/container-engine"
	// GPUInstanceAnnotation is the annotation for GPU instance.
	GPUInstanceAnnotation = "openfuyao.com/gpu-instance"
	// RubikQuotaTurboAnnotation is the annotation for Rubik quota turbo.
	RubikQuotaTurboAnnotation = "volcano.sh/quota-turbo"
)

// global config for cgroup driver choice
const (
	// CgroupDriverSystemd specifies the systemd cgroup driver.
	CgroupDriverSystemd = "systemd"
	// CgroupDriverCgroupfs specifies the cgroupfs cgroup driver.
	CgroupDriverCgroupfs = "cgroupfs"
)

// container engine
const (
	// ContainerEngineCrio specifies the CRI-O container engine.
	ContainerEngineCrio = "crio"
	// ContainerEngineContainerd specifies the containerd container engine.
	ContainerEngineContainerd = "cri-containerd"
	// ContainerEngineIsula specifies the iSula container engine.
	ContainerEngineIsula = "isula"
	// ContainerEngineDocker specifies the Docker container engine.
	ContainerEngineDocker = "docker"
)

const (
	// PreRunPodSandbox is a hook point before running a pod sandbox.
	PreRunPodSandbox = "PreRunPodSandbox"
	// PreCreateContainer is a hook point before creating a container.
	PreCreateContainer = "PreCreateContainer"
	// PreUpdateContainerResources is a hook point before updating container resources.
	PreUpdateContainerResources = "PreUpdateContainerResources"
	// PreRemoveRunPodSandbox is a hook point before removing a pod sandbox.
	PreRemoveRunPodSandbox = "PreRemoveRunPodSandbox"
)

const (
	// KubepodsCgroup is the cgroup name for Kubernetes pods.
	KubepodsCgroup = "kubepods"

	// CPUSharesPath is the path to the cpu.shares cgroup file.
	CPUSharesPath = "cpu.shares"
	// CPUQuotaPath is the path to the cpu.cfs_quota_us cgroup file.
	CPUQuotaPath = "cpu.cfs_quota_us"
	// CPUSetPath is the path to the cpuset.cpus cgroup file.
	CPUSetPath = "cpuset.cpus"
	// CPUSetMemPath is the path to the cpuset.mems cgroup file.
	CPUSetMemPath = "cpuset.mems"
	// MemoryLimitPath is the path to the memory.limit_in_bytes cgroup file.
	MemoryLimitPath = "memory.limit_in_bytes"
	// CgroupCPUDir is the cgroup directory for CPU controller.
	CgroupCPUDir = "cpu/"
	// CgroupCPUSetDir is the cgroup directory for CPUset controller.
	CgroupCPUSetDir = "cpuset/"
	// CgroupCPUSetMemDir is the cgroup directory for CPUset memory controller.
	CgroupCPUSetMemDir = "cpuset/"
	// CgroupMemDir is the cgroup directory for memory controller.
	CgroupMemDir = "memory/"
)

// QosLevel is the overquota qos level of pod
type QosLevel string

// File permission
const (
	// DefaultUmask specifies the default umask for file creation.
	DefaultUmask os.FileMode = 0077
	// DefaultFileMode specifies the default file permission mode.
	DefaultFileMode os.FileMode = 0600
	// DefaultDirMode specifies the default directory permission mode.
	DefaultDirMode os.FileMode = 0700
	// DefaultDumpLogFileMode specifies the default permission mode for dump log files.
	DefaultDumpLogFileMode os.FileMode = 0400
	// TmpTestDir specifies the temporary directory for test files.
	TmpTestDir = "/tmp/colocation-management-test-file"
)

const (
	// KlogFlushInterval is the interval of klog flush.
	KlogFlushInterval = 5 * time.Second
)
