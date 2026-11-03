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

// ContainerdConfigSpec defines the desired state of ContainerdConfig
type ContainerdConfigSpec struct {
	// ConfigType indicates the type of configuration
	// +kubebuilder:validation:Enum=service;main;registry;combined
	// +kubebuilder:default=combined
	ConfigType string `json:"configType,omitempty"`

	// Description provides human-readable description of this configuration
	// +optional
	Description string `json:"description,omitempty"`

	// Service contains systemd service drop-in configuration
	// +optional
	Service *ServiceConfig `json:"service,omitempty"`

	// Main contains main config.toml configuration
	// +optional
	Main *MainConfig `json:"main,omitempty"`

	// Registry contains containerd v2.1+ registry configuration
	// +optional
	Registry *RegistryConfig `json:"registry,omitempty"`

	// Script defines shell script execution configuration
	// +optional
	Script *ScriptConfig `json:"script,omitempty"`
}

// ServiceConfig defines systemd service configuration
type ServiceConfig struct {
	// ExecStart defines the complete ExecStart command
	// Example: "/usr/bin/containerd --config /etc/containerd/config.toml"
	// +optional
	ExecStart string `json:"execStart,omitempty"`

	// Slice specifies the systemd slice for resource control
	// +optional
	// +kubebuilder:default="system.slice"
	Slice string `json:"slice,omitempty"`

	// KillMode specifies how processes of this service shall be killed
	// One of: "control-group" (default), "process", "mixed", "none"
	// - control-group: All processes in the control group will be killed
	// - process: Only the main process itself is killed
	// - mixed: The main process is killed with SIGTERM, other processes with SIGKILL
	// - none: No processes are killed
	// +optional
	// +kubebuilder:default="process"
	// +kubebuilder:validation:Enum=control-group;process;mixed;none
	KillMode string `json:"killMode,omitempty"`

	// Restart specifies when the service shall be restarted
	// One of: "no", "on-success", "on-failure", "on-abnormal", "on-watchdog", "on-abort", "always"
	// - no: Never restart
	// - on-success: Restart only when the service process exits cleanly
	// - on-failure: Restart only when the service process exits with a non-zero exit code
	// - on-abnormal: Restart when the process is terminated by a signal
	// - on-abort: Restart only when the service process exits due to an uncaught signal
	// - always: Always restart
	// +optional
	// +kubebuilder:default="always"
	// +kubebuilder:validation:Enum=no;on-success;on-failure;on-abnormal;on-watchdog;on-abort;always
	Restart string `json:"restart,omitempty"`

	// RestartSec configures the time to sleep before restarting a service
	// Specified as a time span value (e.g., "5s", "1min 30s", "300ms")
	// +optional
	// +kubebuilder:default="5s"
	RestartSec string `json:"restartSec,omitempty"`

	// StartLimitInterval specifies the interval for the start rate limiting
	// +optional
	// +kubebuilder:default="10s"
	StartLimitInterval string `json:"startLimitInterval,omitempty"`

	// StartLimitBurst specifies the burst limit for start attempts
	// +optional
	// +kubebuilder:default=5
	StartLimitBurst int `json:"startLimitBurst,omitempty"`

	// TimeoutStopSec configures the time to wait for stop before timing out
	// +optional
	// +kubebuilder:default="90s"
	TimeoutStopSec string `json:"timeoutStopSec,omitempty"`

	// Logging configuration for systemd service
	// +optional
	Logging *ServiceLogging `json:"logging,omitempty"`

	// CustomExtra defines user custom variables for the service
	// +optional
	CustomExtra map[string]string `json:"customExtra,omitempty"`
}

// ServiceLogging 服务日志配置
type ServiceLogging struct {
	// StandardOutput specifies stdout destination
	// One of: "inherit", "null", "tty", "journal", "syslog", "kmsg", "journal+console", "syslog+console", "kmsg+console"
	// +optional
	// +kubebuilder:default="journal"
	// +kubebuilder:validation:Enum=inherit;null;tty;journal;syslog;kmsg;journal+console;syslog+console;kmsg+console
	StandardOutput string `json:"standardOutput,omitempty"`

	// StandardError specifies stderr destination
	// One of: "inherit", "null", "tty", "journal", "syslog", "kmsg", "journal+console", "syslog+console", "kmsg+console"
	// +optional
	// +kubebuilder:default="journal"
	// +kubebuilder:validation:Enum=inherit;null;tty;journal;syslog;kmsg;journal+console;syslog+console;kmsg+console
	StandardError string `json:"standardError,omitempty"`

	// SyslogIdentifier specifies the syslog identifier
	// +optional
	SyslogIdentifier string `json:"syslogIdentifier,omitempty"`

	// LogLevelMax specifies the maximum log level
	// +optional
	// +kubebuilder:validation:Enum=emerg;alert;crit;err;warning;notice;info;debug
	LogLevelMax string `json:"logLevelMax,omitempty"`
}

// MainConfig defines the main config.toml configuration for containerd v2.1+
type MainConfig struct {
	// MetricsAddress specifies the address for metrics exposure
	// If set, enables Prometheus metrics endpoint
	// Example: "0.0.0.0:1338", "127.0.0.1:1338"
	// +optional
	MetricsAddress string `json:"metricsAddress,omitempty"`

	// Root directory for containerd state
	// +optional
	// +kubebuilder:default="/var/lib/containerd"
	Root string `json:"root,omitempty"`

	// State directory for containerd
	// +optional
	// +kubebuilder:default="/run/containerd"
	State string `json:"state,omitempty"`

	// SandboxImage specifies the pause container image
	// +optional
	// +kubebuilder:default="registry.k8s.io/pause:3.9"
	SandboxImage string `json:"sandboxImage,omitempty"`

	// ConfigPath specifies the registry config directory
	// +optional
	// +kubebuilder:default="/etc/containerd/certs.d"
	ConfigPath string `json:"configPath,omitempty"`

	// RawTOML allows raw TOML configuration for advanced use cases
	// This will be used as-is if provided, ignoring other fields
	// +optional
	RawTOML string `json:"rawTOML,omitempty"`
}

// RegistryConfig defines containerd v2.1+ registry configuration
type RegistryConfig struct {
	// ConfigPath specifies the registry config directory
	// +optional
	// +kubebuilder:default="/etc/containerd/certs.d"
	ConfigPath string `json:"configPath,omitempty"`

	// Configs defines registry-specific configurations
	// +optional
	Configs map[string]RegistryHostConfig `json:"configs,omitempty"`
}

// RegistryHostConfig defines containerd v2.1+ registry host configuration
type RegistryHostConfig struct {
	// Host defines the registry host URL
	// +optional
	Host string `json:"host,omitempty"`

	// Capabilities defines allowed operations
	// +optional
	// +kubebuilder:default={"pull","resolve"}
	Capabilities []string `json:"capabilities,omitempty"`

	// SkipVerify skips TLS certificate verification
	// +optional
	SkipVerify bool `json:"skipVerify,omitempty"`

	// PlainHTTP uses HTTP instead of HTTPS
	// +optional
	PlainHTTP bool `json:"plainHTTP,omitempty"`

	// Insecure uses insecure connection
	// +optional
	Insecure bool `json:"insecure,omitempty"`

	// TLS configuration
	// +optional
	TLS *TLSConfig `json:"tls,omitempty"`

	// Auth configuration
	// +optional
	Auth *RegistryAuthConfig `json:"auth,omitempty"`

	// Header contains additional headers
	// +optional
	Header map[string][]string `json:"header,omitempty"`

	// OverridePath enables path override for mirrors
	// +optional
	OverridePath bool `json:"overridePath,omitempty"`
}

// RegistryAuthConfig defines containerd v2.1+ registry authentication
type RegistryAuthConfig struct {
	// Username for authentication
	// +optional
	Username string `json:"username,omitempty"`

	// Password for authentication
	// +optional
	Password string `json:"password,omitempty"`

	// Auth base64 encoded auth string
	// +optional
	Auth string `json:"auth,omitempty"`

	// IdentityToken for token authentication
	// +optional
	IdentityToken string `json:"identityToken,omitempty"`

	// RegistryToken for registry-specific tokens
	// +optional
	RegistryToken string `json:"registryToken,omitempty"`
}

// TLSConfig defines TLS configuration for containerd v2.1+
type TLSConfig struct {
	// InsecureSkipVerify skips TLS certificate verification
	// +optional
	InsecureSkipVerify bool `json:"insecureSkipVerify,omitempty"`

	// CAFile path to CA certificate
	// +optional
	CAFile string `json:"caFile,omitempty"`

	// CertFile path to client certificate
	// +optional
	CertFile string `json:"certFile,omitempty"`

	// KeyFile path to client private key
	// +optional
	KeyFile string `json:"keyFile,omitempty"`
}

// ScriptConfig defines shell script execution configuration
type ScriptConfig struct {
	// Content contains the shell script content to execute
	// +optional
	Content string `json:"content,omitempty"`

	// Path specifies the path to a shell script file
	// If both Content and Path are provided, Content takes precedence
	// +optional
	Path string `json:"path,omitempty"`

	// Args specifies arguments to pass to the script
	// +optional
	Args []string `json:"args,omitempty"`

	// Interpreter specifies the shell interpreter to use
	// +optional
	// +kubebuilder:default="/bin/bash"
	Interpreter string `json:"interpreter,omitempty"`
}
