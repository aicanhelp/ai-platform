/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1beta1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// KubeletConfigSpec defines the desired state of KubeletConfig
type KubeletConfigSpec struct {
	// KubeletConfig defines the kubelet configuration
	// +optional
	KubeletConfig map[string]runtime.RawExtension `json:"kubeletConfig,omitempty"`

	// KubeletService defines the kubelet systemd service configuration
	// +optional
	KubeletService *KubeletServiceSpec `json:"kubeletService,omitempty"`

	// Files defines additional files to be created on the node
	// +optional
	Files []FileSpec `json:"files,omitempty"`

	// Commands defines additional commands to be executed
	// +optional
	Commands []CommandSpec `json:"commands,omitempty"`
}

// KubeletServiceSpec defines the kubelet systemd service configuration
type KubeletServiceSpec struct {
	// Enabled indicates whether to create the kubelet service
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// ServiceName is the name of the systemd service
	// +optional
	ServiceName string `json:"serviceName,omitempty"`

	// Unit defines the [Unit] section of the systemd service
	// +optional
	Unit KubeletUnit `json:"unit,omitempty"`

	// Service defines the [Service] section of the systemd service
	// +optional
	Service KubeletService `json:"service,omitempty"`

	// Install defines the [Install] section of the systemd service
	// +optional
	Install KubeletInstall `json:"install,omitempty"`

	// Variables defines template variables for the service file
	// +optional
	Variables map[string]string `json:"variables,omitempty"`
}

// KubeletUnit defines the [Unit] section
type KubeletUnit struct {
	// Description of the service
	// +optional
	Description string `json:"description,omitempty"`

	// Documentation URL
	// +optional
	Documentation string `json:"documentation,omitempty"`

	// After defines services that this service should start after
	// +optional
	After []string `json:"after,omitempty"`

	// Wants defines services that this service wants
	// +optional
	Wants []string `json:"wants,omitempty"`

	// Requires defines services that this service requires
	// +optional
	Requires []string `json:"requires,omitempty"`
}

// KubeletService defines the [Service] section
type KubeletService struct {
	// ExecStart defines the command to start the service
	ExecStart string `json:"execStart"`

	// Restart policy
	// +optional
	Restart string `json:"restart,omitempty"`

	// StartLimitInterval defines the interval for start limit
	// +optional
	StartLimitInterval int `json:"startLimitInterval,omitempty"`

	// RestartSec defines the restart delay
	// +optional
	RestartSec int `json:"restartSec,omitempty"`

	// Environment variables
	// +optional
	Environment []string `json:"environment,omitempty"`

	// EnvironmentFile variables
	// +optional
	EnvironmentFile []string `json:"environmentFile,omitempty"`

	// +optional
	ExecStartPre []string `json:"execStartPre,omitempty"`

	// +optional
	StartLimitBurst int `json:"startLimitBurst,omitempty"`

	// +optional
	KillMode string `json:"killMode,omitempty"`

	// +optional
	StandardOutput string `json:"standardOutput,omitempty"`

	// +optional
	StandardError string `json:"standardError,omitempty"`

	// +optional
	SyslogIdentifier string `json:"syslogIdentifier,omitempty"`

	// WorkingDirectory
	// +optional
	WorkingDirectory string `json:"workingDirectory,omitempty"`

	// User to run the service as
	// +optional
	User string `json:"user,omitempty"`

	// Group to run the service as
	// +optional
	Group string `json:"group,omitempty"`

	// CustomExtra defines user custom variables for the [Service] section
	// +optional
	CustomExtra map[string]string `json:"customExtra,omitempty"`
}

// KubeletInstall defines the [Install] section
type KubeletInstall struct {
	// WantedBy defines the target that wants this service
	// +optional
	WantedBy []string `json:"wantedBy,omitempty"`
	// RequiredBy defines the target that requires this service
	// +optional
	RequiredBy []string `json:"requiredBy,omitempty"`
}

// FileSpec defines a file to be created on the node
type FileSpec struct {
	// Path is the file path on the node
	Path string `json:"path"`

	// Content is the file content
	Content string `json:"content"`

	// Permissions defines the file permissions
	// +optional
	Permissions string `json:"permissions,omitempty"`

	// Owner defines the file owner
	// +optional
	Owner string `json:"owner,omitempty"`
}

// CommandSpec defines a command to be executed on the node
type CommandSpec struct {
	// Command is the command to execute
	Command string `json:"command"`

	// Args are the command arguments
	// +optional
	Args []string `json:"args,omitempty"`

	// WorkingDir is the working directory
	// +optional
	WorkingDir string `json:"workingDir,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=kct
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="AGE",type="date",JSONPath=".metadata.creationTimestamp"
// KubeletConfig is the Schema for the kubeletconfigtemplates API
type KubeletConfig struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec KubeletConfigSpec `json:"spec,omitempty"`
}

// +kubebuilder:object:root=true
type KubeletConfigList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []KubeletConfig `json:"items"`
}
