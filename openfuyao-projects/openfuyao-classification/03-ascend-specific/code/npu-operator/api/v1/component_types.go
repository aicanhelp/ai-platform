// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package v1

import corev1 "k8s.io/api/core/v1"

// ComponentWorkloadSpec contains reusable deployment settings, without
// component-specific configuration or validation. It is inlined in each spec.
type ComponentWorkloadSpec struct {
	// Whether to deploy and reconcile this component.
	// +default=false
	Managed bool `json:"managed,omitempty"`
	// Main and init container image configuration.
	// +default={}
	Image ImageSpec `json:"imageSpec,omitempty"`
	// Environment overrides, merged by name.
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// Override the main process command and arguments independently.
	// +optional
	Command CommandSpec `json:"commandSpec,omitempty"`
	// Additional volumes and main container mounts, merged by name.
	// +optional
	VolumeSpec VolumesSpec `json:"volumeSpec,omitempty"`
	// Main container resource requirements.
	// +optional
	Resources ResourceRequirements `json:"resources,omitempty"`
	// Scheduling placement for the component.
	PlacementSpec `json:",inline"`
}

// InferOperatorSpec configures the MindCluster inference controller.
type InferOperatorSpec struct {
	ComponentWorkloadSpec `json:",inline"`
	// Logging parameters for the default command.
	// +default={}
	LogRotate LogRotate `json:"logRotate,omitempty"`
	// Overrides for infer-operator-config data, including fault_handling.
	// +optional
	Config map[string]string `json:"config,omitempty"`
}

// RDMADevicePluginSpec configures the MindCluster RDMA shared device plugin.
// +kubebuilder:validation:XValidation:rule="!has(self.logRotate) || !has(self.logRotate.compress) || !self.logRotate.compress",message="RDMA log compression is unsupported"
type RDMADevicePluginSpec struct {
	ComponentWorkloadSpec `json:",inline"`
	// Logging parameters for the default command. Compression is unsupported.
	// +default={}
	LogRotate LogRotate `json:"logRotate,omitempty"`
	// Overrides for rdma-devices data. config.json contains the device configuration.
	// +optional
	Config map[string]string `json:"config,omitempty"`
}

// ContainerSnapshotSpec enables the NodeD container snapshot prerequisites.
type ContainerSnapshotSpec struct {
	// Add host PID access, runtime mounts and PATH required by snapshot support.
	// Requires a snapshot-capable NodeD image and the corresponding host tools.
	// +default=false
	Enabled bool `json:"enabled,omitempty"`
}
