// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package v1

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// NPUClusterPolicySpec defines the desired state of NPUClusterPolicy
type NPUClusterPolicySpec struct {
	// Operator configuration
	Operator OperatorSpec `json:"operator"`
	// Daemonset configuration
	Daemonsets DaemonsetsSpec `json:"daemonsets"`
	// Ascend Driver componet spec
	Driver DriverSpec `json:"driver"`
	// Ascend Docker Runtime component spec
	OCIRuntime OCIRuntimeSpec `json:"ociRuntime"`
	// Volcano component spec
	Volcano      VolcanoSpec    `json:"volcano,omitempty"`
	VCScheduler  SchedulerSpec  `json:"vcscheduler"`
	VCController ControllerSpec `json:"vccontroller"`
	// Ascend Operator component spec
	Trainer TrainerSpec `json:"trainer"`
	// Ascend Device Plugin component spec
	DevicePlugin DevicePluginSpec `json:"devicePlugin"`
	// Ascend NodeD component spec
	NodeD NodeDSpec `json:"nodeD"`
	// MindCluster inference controller. Disabled unless explicitly managed.
	// +optional
	InferOperator InferOperatorSpec `json:"inferOperator,omitempty"`
	// MindCluster RDMA shared device plugin. Disabled unless explicitly managed.
	// +optional
	RDMADevicePlugin RDMADevicePluginSpec `json:"rdmaDevicePlugin,omitempty"`
	// Ascend NPU Exporter component spec
	Exporter ExporterSpec `json:"exporter"`
	// Ascend ClusterD component spec
	ClusterD ClusterDSpec `json:"clusterd"`
	// Ascend Resilience Controller component spec
	RSController RSControllerSpec `json:"rscontroller"`
	// MindIO component spec
	MindIOTFT MindIOTFTSpec `json:"mindiotft"`
	// MindIO component spec
	MindIOACP MindIOACPSpec `json:"mindioacp"`
	// vNPU component spec
	VNPU VNPUSpec `json:"vnpu,omitempty"`
	// Dynamic Resource Allocation component spec
	DRA DRASpec `json:"dra,omitempty"`
}

// CommandSpec defines the properties for container
type CommandSpec struct {
	Command []string `json:"command,omitempty"`
	Args    []string `json:"args,omitempty"`
}

// ImageSpec defines the properties for an image
type ImageSpec struct {
	// Image registry
	Registry string `json:"registry,omitempty"`

	// Image repository
	Repository string `json:"repository,omitempty"`

	// Image tag
	Tag string `json:"tag,omitempty"`

	// Image pull policy
	// +default="IfNotPresent"
	ImagePullPolicy string `json:"imagePullPolicy,omitempty"`

	// Image pull secrets
	// +default=[]
	ImagePullSecrets []string `json:"imagePullSecrets,omitempty"`
}

// Name returns the complete image name
func (s ImageSpec) Name() string {
	registry := s.Registry
	repo := s.Repository
	tag := s.Tag
	var image string
	if len(repo) == 0 {
		return ""
	}
	if len(registry) == 0 && len(tag) == 0 {
		if len(repo) != 0 {
			image = repo
		}
	} else {
		if len(registry) == 0 {
			image = fmt.Sprintf("%s:%s", repo, tag)
		} else {
			image = fmt.Sprintf("%s/%s:%s", registry, repo, tag)
		}
	}
	return image
}

// OperatorSpec defines the configuration for the operator
type OperatorSpec struct {
	// Name of CRI Runtime Handler
	// +kubebuilder:default=ascend
	RuntimeClass string `json:"runtimeClass,omitempty"`

	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
}

// PlacementSpec defines scheduling placement for operands.
type PlacementSpec struct {
	// Node selector for operand pods
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// Tolerations for operand pods
	// +optional
	// +listType=atomic
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`
}

// DaemonsetsSpec defines common configuration for all Daemonsets
type DaemonsetsSpec struct {
	// Common labels
	// +optional
	Labels map[string]string `json:"labels,omitempty"`
	// Common annotations
	// +optional
	Annotations map[string]string `json:"annotations,omitempty"`
	// Common node selector
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// Common tolerations
	// +optional
	// +listType=atomic
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`

	// Common image
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
}

// OCIRuntimeSpec defines the properties for Ascend Docker Runtime deployment
type OCIRuntimeSpec struct {
	// Whether to manage Docker Runtime with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Interval for health checking
	// +default=300
	// +kubebuilder:validation:Minimum=60
	// +kubebuilder:validation:Maximum=1800
	Interval int32 `json:"interval,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Init runtime image configuration
	// +default={}
	InitRuntimeImage ImageSpec `json:"initRuntimeImageSpec,omitempty"`
	// Init config image configuration
	// +default={}
	InitConfigImage ImageSpec `json:"initConfigImageSpec,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// VolcanoFlavor defines the managed Volcano asset flavor.
// +enum
// +kubebuilder:validation:Enum=mindcluster;vnpu;external
type VolcanoFlavor string

const (
	// VolcanoFlavorMindCluster uses the default MindCluster Volcano assets.
	VolcanoFlavorMindCluster VolcanoFlavor = "mindcluster"
	// VolcanoFlavorVNPU uses vNPU-compatible Volcano assets.
	VolcanoFlavorVNPU VolcanoFlavor = "vnpu"
	// VolcanoFlavorExternal indicates Volcano is managed outside NPU Operator.
	VolcanoFlavorExternal VolcanoFlavor = "external"
)

// VolcanoLogSpec defines Volcano log command parameters.
type VolcanoLogSpec struct {
	// Log directory
	Dir string `json:"dir,omitempty"`
	// Log file path
	File string `json:"file,omitempty"`
	// Verbosity level
	Level *int32 `json:"level,omitempty"`
}

// VolcanoAdmissionSpec defines Volcano admission webhook management configuration.
type VolcanoAdmissionSpec struct {
	// Whether to manage Volcano admission webhook
	Enabled *bool `json:"enabled,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Init image configuration
	// +default={}
	InitImage ImageSpec `json:"initImageSpec,omitempty"`
	// Log configuration
	Log VolcanoLogSpec `json:"log,omitempty"`
}

// VolcanoSpec defines shared Volcano management configuration.
type VolcanoSpec struct {
	// Volcano asset flavor
	// +kubebuilder:validation:Enum=mindcluster;vnpu;external
	Flavor VolcanoFlavor `json:"flavor,omitempty"`
	// Pod and container runAsUser override for Volcano workloads
	RunAsUser *int64 `json:"runAsUser,omitempty"`
	// Pod and container runAsGroup override for Volcano workloads
	RunAsGroup *int64 `json:"runAsGroup,omitempty"`
	// Volcano admission webhook configuration
	Admission VolcanoAdmissionSpec `json:"admission,omitempty"`
}

// SchedulerSpec defines the properties for Volcano Scheduler deployment
type SchedulerSpec struct {
	// Whether to manage Volcano with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Init image path
	InitImage ImageSpec `json:"initImageSpec,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// VolumeSpec add volumes and volumeMounts
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Resources required by scheduler container
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Configmap controls the scheduler plugin
	SchedulerConfigMap string `json:"schedulerConfigMap,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// ControllerSpec defines the properties for Volcano Controller deployment
type ControllerSpec struct {
	// Whether to manage Volcano controller with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Init image path
	InitImage ImageSpec `json:"initImageSpec,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// VolumeSpec add volumes and volumeMounts
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Resources required by scheduler container
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// MindIOTFTSpec defines the properties for MindIO Training Fault Tolerance daemonset
type MindIOTFTSpec struct {
	// whether to manage MindIO-TFT with NPU Operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Image configuration
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to TFT
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// MindIOACPSpec defines the properties for MindIO Async Checkpoint Persistence daemonset
type MindIOACPSpec struct {
	// whether to manage MindIO-ACP with NPU Operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Version of mindIO
	Version string `json:"version,omitempty"`
	// Image configuration
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to ACP
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// ServiceMonitorSpec defines whether an operand ServiceMonitor should be managed.
type ServiceMonitorSpec struct {
	// Whether to create ServiceMonitor for metrics scraping
	// +default=false
	Enabled bool `json:"enabled,omitempty"`
}

// VNPUComponentSpec defines common vNPU daemonset configuration.
type VNPUComponentSpec struct {
	// Whether to manage this vNPU component
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to component pod
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Container command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// Add volumes and volume mounts
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Container resources
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// VNPUDevicePluginSpec defines vNPU device plugin configuration.
type VNPUDevicePluginSpec struct {
	// Whether to manage vNPU device plugin
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to component pod
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Container command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// Add volumes and volume mounts
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Container resources
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
	// Number of vNPU devices exposed per physical NPU
	// +default=20
	DeviceSplitCount int32 `json:"deviceSplitCount,omitempty"`
	// Whether to log to console
	// +default=true
	LoggingConsole bool `json:"loggingConsole,omitempty"`
}

// VNPUExporterSpec defines vNPU exporter configuration.
type VNPUExporterSpec struct {
	// Whether to manage vNPU exporter
	// +default=false
	Managed bool `json:"managed,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to component pod
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Container command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// Add volumes and volume mounts
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Container resources
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
	// Metrics service port
	// +default=8082
	Port int32 `json:"port,omitempty"`
	// Whether HTTPS is enabled for exporter
	// +default=false
	HTTPS bool `json:"https,omitempty"`
	// ServiceMonitor configuration
	ServiceMonitor ServiceMonitorSpec `json:"serviceMonitor,omitempty"`
}

// VNPUSpec defines vNPU component configuration.
type VNPUSpec struct {
	// Whether to manage vNPU components
	// +default=false
	Managed bool `json:"managed,omitempty"`
	// Shared placement for vNPU components
	PlacementSpec `json:",inline"`
	// DaemonSet update strategy name, such as RollingUpdate
	UpdateStrategy string `json:"updateStrategy,omitempty"`
	// Priority class for vNPU pods
	PriorityClassName string `json:"priorityClassName,omitempty"`
	// Node mode config used by vnpu.node-mode.config
	NodeModeConfig string `json:"nodeModeConfig,omitempty"`
	// Virtual template config used by vnpu.virtual-template.config
	VirtualTemplateConfig string `json:"virtualTemplateConfig,omitempty"`
	// vNPU client update daemonset
	ClientUpdate VNPUComponentSpec `json:"clientUpdate,omitempty"`
	// vNPU device plugin daemonset
	DevicePlugin VNPUDevicePluginSpec `json:"devicePlugin,omitempty"`
	// vNPU exporter daemonset
	Exporter VNPUExporterSpec `json:"exporter,omitempty"`
}

// DRAComponentSpec defines common DRA daemonset configuration.
type DRAComponentSpec struct {
	// Whether to manage this DRA component
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to component pod
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Container command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// Add volumes and volume mounts
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Container resources
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// DRADeviceClassTemplate defines a managed DRA DeviceClass template.
type DRADeviceClassTemplate struct {
	// DeviceClass name
	Name string `json:"name,omitempty"`
	// CEL selector expression
	SelectorCEL string `json:"selectorCEL,omitempty"`
}

// DRADeviceClassesSpec defines DRA DeviceClass management configuration.
type DRADeviceClassesSpec struct {
	// Whether to manage DRA DeviceClasses
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// DeviceClass templates
	// +optional
	// +listType=atomic
	Templates []DRADeviceClassTemplate `json:"templates,omitempty"`
}

// DRASoftVNPUSpec defines DRA soft-vNPU configuration.
type DRASoftVNPUSpec struct {
	// Whether to manage DRA soft-vNPU support
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Shared placement for DRA soft-vNPU support components
	PlacementSpec `json:",inline"`
	// Number of soft-share vNPU instances per physical NPU
	// +default=16
	ShareCount int32 `json:"shareCount,omitempty"`
	// Base directory for soft-share shm files
	SoftShareShmDir string `json:"softShareShmDir,omitempty"`
	// Host directory for soft-share device configs
	SoftShareDevConfigDir string `json:"softShareDevConfigDir,omitempty"`
	// Path to NPU profile config in the container
	NPUProfileConfigPath string `json:"npuProfileConfigPath,omitempty"`
	// ConfigMap content for dra.config
	ProfileConfig string `json:"profileConfig,omitempty"`
	// Name of the chip capabilities ConfigMap consumed by the DRA driver
	ChipCapabilitiesConfigMap string `json:"chipCapabilitiesConfigMap,omitempty"`
	// Namespace of the chip capabilities ConfigMap consumed by the DRA driver
	ChipCapabilitiesNamespace string `json:"chipCapabilitiesNamespace,omitempty"`
	// ConfigMap content for capabilities.yaml
	ChipCapabilities string `json:"chipCapabilities,omitempty"`
	// vCANN-RT installer daemonset used by DRA soft-vNPU
	VCANNRTInstaller DRAComponentSpec `json:"vcannrtInstaller,omitempty"`
}

// DRAWebhookSpec defines DRA validation webhook configuration.
type DRAWebhookSpec struct {
	DRAComponentSpec `json:",inline"`
	// Secure port that the webhook listens on
	// +default=443
	Port int32 `json:"port,omitempty"`
	// Certificate file path passed to the webhook process
	TLSCertFile string `json:"tlsCertFile,omitempty"`
	// Private key file path passed to the webhook process
	TLSPrivateKeyFile string `json:"tlsPrivateKeyFile,omitempty"`
	// CA bundle injected into ValidatingWebhookConfiguration
	CABundle string `json:"caBundle,omitempty"`
	// Failure policy for ValidatingWebhookConfiguration
	// +kubebuilder:validation:Enum=Ignore;Fail
	FailurePolicy string `json:"failurePolicy,omitempty"`
}

// DRASpec defines Dynamic Resource Allocation component configuration.
type DRASpec struct {
	// Whether to manage DRA components
	// +default=false
	Managed bool `json:"managed,omitempty"`
	// DRA driver name, must match DeviceClass selectors
	DriverName string `json:"driverName,omitempty"`
	// Device profile used by DRA driver
	DeviceProfile string `json:"deviceProfile,omitempty"`
	// Root directory for CDI specs
	CDIRoot string `json:"cdiRoot,omitempty"`
	// DRA healthcheck port, <0 disables the server
	HealthcheckPort int32 `json:"healthcheckPort,omitempty"`
	// Number of devices generated for gpu profile
	NumDevices int32 `json:"numDevices,omitempty"`
	// Out-of-cluster kubeconfig path used by DRA driver
	Kubeconfig string `json:"kubeconfig,omitempty"`
	// Kubernetes API QPS used by DRA driver
	KubeAPIQPS string `json:"kubeAPIQPS,omitempty"`
	// Kubernetes API burst used by DRA driver
	KubeAPIBurst int32 `json:"kubeAPIBurst,omitempty"`
	// Feature gates passed to DRA driver logging configuration
	FeatureGates string `json:"featureGates,omitempty"`
	// Deprecated: hard-vNPU mode is selected by DRA profile config or node labels.
	EnableHardVNPU bool `json:"enableHardVNPU,omitempty"`
	// Kubelet plugins directory
	KubeletPluginsDirectoryPath string `json:"kubeletPluginsDirectoryPath,omitempty"`
	// Kubelet plugin registrar directory
	KubeletRegistrarDirectoryPath string `json:"kubeletRegistrarDirectoryPath,omitempty"`
	// Shared placement for DRA components
	PlacementSpec `json:",inline"`
	// DRA kubelet plugin daemonset
	KubeletPlugin DRAComponentSpec `json:"kubeletPlugin,omitempty"`
	// Managed DRA DeviceClasses
	DeviceClasses DRADeviceClassesSpec `json:"deviceClasses,omitempty"`
	// DRA soft-vNPU support components
	SoftVNPU DRASoftVNPUSpec `json:"softVNPU,omitempty"`
	// DRA validation webhook
	Webhook DRAWebhookSpec `json:"webhook,omitempty"`
}

// ResourceRequirements defines the compute resource requirements
type ResourceRequirements struct {
	// Same as corev1.ResourceRequirements.Limits
	Limits corev1.ResourceList `json:"limits,omitempty"`
	// Same as corev1.ResourceRequirements.Requests
	Requests corev1.ResourceList `json:"requests,omitempty"`
}

// VolumesSpec defines the volume path of container
type VolumesSpec struct {
	Volumes      []corev1.Volume      `json:"volumes,omitempty"`
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`
}

// TrainerSpec defines the properties for Ascend Operator deployment
type TrainerSpec struct {
	// Whether to manage Ascend Operator with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Init image configuration
	// +default={}
	InitImage ImageSpec `json:"initImageSpec,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to pod
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// Add volume and volumeMount
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Container resources
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Log rotation
	// +default={"logFile":"/var/log/mindx-dl/ascend-operator/ascend-operator.log"}
	LogRotate LogRotate `json:"logRotate,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// LogRotate defines the configuration for log rotation
type LogRotate struct {
	// Log level
	// +default="info"
	LogLevel LogLevel `json:"logLevel,omitempty"` // NOTE: -1 ~ 3, debug ~ critical
	// Log file
	LogFile string `json:"logFile,omitempty"`
	// Number of rotations before log file is removed
	// +default=30
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=30
	Rotate int32 `json:"rotate,omitempty"` // NOTE: -maxBackups
	// Maximum days to keep before log file is removed
	// +default=7
	// +kubebuilder:validation:Minimum=7
	// +kubebuilder:validation:Maximum=700
	MaxAge int32 `json:"maxAge,omitempty"`
	// Enable/disable compression of rotated log files
	// +default=false
	Compress bool `json:"compress,omitempty"` // NOTE: -isCompress
}

// +enum
// +kubebuilder:validation:Enum=debug;info;warning;error;critical

// LogLevel defines the log level of NPU operator components
type LogLevel string

// Value returns the log level value
func (l LogLevel) Value() int {
	switch l {
	case LogDebug:
		return LogDebugValue
	case LogInfo:
		return LogInfoValue
	case LogWarning:
		return LogWarningValue
	case LogError:
		return LogErrorValue
	case LogCritical:
		return LogCriticalValue
	default:
		panic(fmt.Sprintf("unsupported LogLevel: %v", l))
	}
}

const (
	// LogDebug indicates debug log level
	LogDebug LogLevel = "debug"
	// LogInfo indicates info log level
	LogInfo LogLevel = "info"
	// LogWarning indicates warning log level
	LogWarning LogLevel = "warning"
	// LogError indicates error log level
	LogError LogLevel = "error"
	// LogCritical indicates critical log level
	LogCritical LogLevel = "critical"

	// LogDebugValue indicates value of debug log level
	LogDebugValue = -1
	// LogInfoValue indicates value of info log level
	LogInfoValue = 0
	// LogWarningValue indicates value of warning log level
	LogWarningValue = 1
	// LogErrorValue indicates value of error log level
	LogErrorValue = 2
	// LogCriticalValue indicates value of critical log level
	LogCriticalValue = 3
)

// DevicePluginSpec defines the properties for Ascend Device Plugin deployment
type DevicePluginSpec struct {
	// Device Plugin lifecycle management switch
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Scheduling placement for Device Plugin
	PlacementSpec `json:",inline"`
	// Image used by the Device Plugin init container
	// +default={}
	InitImage ImageSpec `json:"initImageSpec,omitempty"`
	// Main Device Plugin container image
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Extra environment variables for the Device Plugin pod
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Device Plugin process command and arguments
	// +default={}
	Command CommandSpec `json:"commandSpec,omitempty"`
	// Additional volumes and mounts for Device Plugin
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Resource requirements for Device Plugin containers
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Device Plugin log rotation settings
	// +default={"logFile":"/var/log/mindx-dl/devicePlugin/devicePlugin.log"}
	// +kubebuilder:validation:XValidation:rule="!has(self.compress) || !self.compress",message="Compress is unsupported"
	LogRotate LogRotate `json:"logRotate,omitempty"`
}

// NodeDSpec defines the properties for Ascend NodeD deployment
type NodeDSpec struct {
	// Configure NodeD for container checkpoint and restore.
	// +optional
	ContainerSnapshot ContainerSnapshotSpec `json:"containerSnapshot,omitempty"`
	// Whether to manage NodeD with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Interval for sending heartbeat messages
	// +default=5
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=300
	HeartbeatInterval int32 `json:"heartbeatInterval,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to Device Plugin
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Change resources from container
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Interval for scanning hardware failure on nodes
	// +default=60
	// +kubebuilder:validation:Minimum=60
	// +kubebuilder:validation:Maximum=600
	PollInterval int32 `json:"pollInterval,omitempty"` // NOTE: monitorPeriod
	// Log rotation
	// +default={"logFile":"/var/log/mindx-dl/noded/noded.log"}
	// +kubebuilder:validation:XValidation:rule="!has(self.compress) || !self.compress",message="Compress is unsupported"
	LogRotate LogRotate `json:"logRotate,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// ExporterSpec defines the properties for Ascend NPU Exporter deployment
type ExporterSpec struct {
	// Optional Prometheus Operator integration; disabled by default.
	// +optional
	ServiceMonitor ServiceMonitorSpec `json:"serviceMonitor,omitempty"`
	// Whether to manage NPU Exporter with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Log rotation
	// +default={"logFile":"/var/log/mindx-dl/npu-exporter/npu-exporter.log"}
	// +kubebuilder:validation:XValidation:rule="!has(self.compress) || !self.compress",message="Compress is unsupported"
	LogRotate LogRotate `json:"logRotate,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to Device Plugin
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Change resources from container
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// DriverSpec defines the properties for Ascend NPU Driver daemonset
type DriverSpec struct {
	// Whether to manage NPU Driver with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Log rotation
	// +default={"logFile":"/var/log/mindx-dl/driver/driver.log"}
	// +kubebuilder:validation:XValidation:rule="!has(self.compress) || !self.compress",message="Compress is unsupported"
	LogRotate LogRotate `json:"logRotate,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Init image configuration
	// +default={}
	InitImage ImageSpec `json:"initImageSpec,omitempty"`
	// Add env to Driver
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// ClusterDSpec defines the properties for ClusterD deployment
type ClusterDSpec struct {
	// Whether to manage ClusterD with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Log rotation
	// +default={"logFile":"/var/log/mindx-dl/clusterd/clusterd.log"}
	// +kubebuilder:validation:XValidation:rule="!has(self.compress) || !self.compress",message="Compress is unsupported"
	LogRotate LogRotate `json:"logRotate,omitempty"`
	// Image configuration
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Add env to Device Plugin
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Command configuration
	Command CommandSpec `json:"commandSpec,omitempty"`
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Change resources from container
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// RSControllerSpec defines the properties for Resilience Controller deployment
type RSControllerSpec struct {
	// Whether to manage Resilience Controller with NPU operator
	// +default=true
	Managed bool `json:"managed,omitempty"`
	// Log rotation
	// +default={"logFile":"/var/log/mindx-dl/resilience-controller/run.log"}
	// +kubebuilder:validation:XValidation:rule="!has(self.compress) || !self.compress",message="Compress is unsupported"
	LogRotate LogRotate `json:"logRotate,omitempty"`
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Image configuration
	// +default={}
	Image   ImageSpec   `json:"imageSpec,omitempty"`
	Command CommandSpec `json:"commandSpec,omitempty"`
	// Placement configuration
	PlacementSpec `json:",inline"`
}

// NPUClusterPolicyStatus defines the observed state of NPUClusterPolicy
type NPUClusterPolicyStatus struct {
	// State of NPUClusterPolicy
	Phase Phase `json:"phase"`
	// Namespace where the operator is installed
	Namespace string `json:"namespace,omitempty"`
	// Observations of NPUClusterPolicy's current state.
	// Known .status.conditions.type are: "Ready" and "Error"
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
	// Statuses for all NPU operator components, one entry per component
	// +listType=atomic
	ComponentStatuses []ComponentStatus `json:"componentStatuses,omitempty"`
}

// UpdateConditions defines the function that update conditions of NPUClusterPolicyStatus.
func (s *NPUClusterPolicyStatus) UpdateConditions(oldType string, newType string, reason string, message string) bool {
	changed := meta.SetStatusCondition(&s.Conditions, metav1.Condition{
		Type:    newType,
		Status:  metav1.ConditionTrue,
		Reason:  reason,
		Message: message,
	})
	changed = meta.SetStatusCondition(&s.Conditions, metav1.Condition{
		Type:   oldType,
		Status: metav1.ConditionFalse,
		Reason: newType,
	}) || changed
	return changed
}

// UpdateComponentStatus add or update component status of NPUClusterPolicyStatus.
func (s *NPUClusterPolicyStatus) UpdateComponentStatus(name string, state ComponentState) bool {
	if cs := s.GetComponentStatus(name); cs != nil {
		if cs.State == state {
			return false
		}
		cs.PrevState = cs.State
		cs.State = state
	} else {
		s.ComponentStatuses = append(s.ComponentStatuses, ComponentStatus{Name: name, State: state})
	}
	return true
}

// GetComponentStatus returns the component status of NPUClusterPolicyStatus by component name.
func (s *NPUClusterPolicyStatus) GetComponentStatus(name string) *ComponentStatus {
	for i := range s.ComponentStatuses {
		cs := &s.ComponentStatuses[i]
		if cs.Name == name {
			return cs
		}
	}
	return nil
}

// +enum
// +kubebuilder:validation:Enum=Ready;NotReady

// Phase defines the current state of NPUClusterPolicy
type Phase string

const (
	// PolicyReady indicates all components of NPUClusterPolicy are ready
	PolicyReady Phase = "Ready"
	// PolicyNotReady indicates some/all components of NPUClusterPolicy are not ready
	PolicyNotReady Phase = "NotReady"
)

// ComponentStatus defines the current status for an NPU operator component
type ComponentStatus struct {
	// Name of the component
	Name string `json:"name"`
	// State of the component
	State ComponentState `json:"state,omitempty"`
	// Previous state of the component for debugging purpose
	PrevState ComponentState `json:"prevState,omitempty"`
}

// ComponentState defines the state of an NPU operator component
type ComponentState struct {
	// Phase of the component state
	// +default="pending"
	Phase ComponentPhase `json:"type,omitempty"`
	// Unique, one-word, CamelCase reason for the component state
	Reason string `json:"reason,omitempty"`
	// Human-readable message indicating details about the component state
	Message string `json:"message,omitempty"`
}

// +enum
// +kubebuilder:validation:Enum=pending;deploying;running;terminated;unmanaged

// ComponentPhase defines the current state of an NPU operator component
type ComponentPhase string

const (
	// ComponentPending indicates the component is pending deployment, or is failed to deploy and pending redeployment
	ComponentPending ComponentPhase = "pending"
	// ComponentDeploying indicates the component is being deployed
	ComponentDeploying ComponentPhase = "deploying"
	// ComponentRunning indicates the component is successfully deployed and running
	ComponentRunning ComponentPhase = "running"
	// ComponentTerminated indicates the component is terminated and pending restart
	ComponentTerminated ComponentPhase = "terminated"
	// ComponentUnmanaged indicates the component is not managed by the operator
	ComponentUnmanaged ComponentPhase = "unmanaged"
)

// +genclient
// +genclient:nonNamespaced
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Cluster
// +kubebuilder:printcolumn:name="Status",type=string,JSONPath=`.status.phase`,priority=0
// +kubebuilder:printcolumn:name="Age",type=string,JSONPath=`.metadata.creationTimestamp`,priority=0
// +kubebuilder:validation:XValidation:rule="self.metadata.name == 'cluster'",message="Only one NPUClusterPolicy with name 'cluster' can be created in a cluster"

// NPUClusterPolicy is the Schema for the npuclusterpolicies API
type NPUClusterPolicy struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   NPUClusterPolicySpec   `json:"spec,omitempty"`
	Status NPUClusterPolicyStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// NPUClusterPolicyList contains a list of NPUClusterPolicy
type NPUClusterPolicyList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []NPUClusterPolicy `json:"items"`
}

func init() {
	SchemeBuilder.Register(&NPUClusterPolicy{}, &NPUClusterPolicyList{})
}
