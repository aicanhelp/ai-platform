/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package agentapi defines the request and response types exchanged between
// the flux-sandbox control plane and the agent running on each node.
package agentapi

// VolumeMount describes a host path bind-mounted into a sandbox container.
type VolumeMount struct {
	Source      string `json:"source"`             // host path to mount from.
	Destination string `json:"destination"`        // container path to mount to.
	ReadOnly    bool   `json:"readOnly,omitempty"` // whether the mount is read-only.
}

// SandboxSpec describes the desired state of a sandbox.
type SandboxSpec struct {
	SandboxID     string            `json:"sandboxId"`               // business identifier assigned by the control plane.
	SandboxName   string            `json:"sandboxName"`             // human-readable sandbox name.
	Image         string            `json:"image"`                   // container image reference.
	CPU           string            `json:"cpu,omitempty"`           // CPU request, e.g. "500m" or "2".
	Memory        string            `json:"memory,omitempty"`        // memory request, e.g. "512Mi".
	Command       []string          `json:"command,omitempty"`       // entrypoint command.
	Args          []string          `json:"args,omitempty"`          // command arguments.
	Env           map[string]string `json:"env,omitempty"`           // environment variables.
	WorkingDir    string            `json:"workingDir,omitempty"`    // working directory inside the sandbox.
	RuntimeConfig map[string]string `json:"runtimeConfig,omitempty"` // runtime-specific opaque configuration.
	VolumeMounts  []VolumeMount     `json:"volumeMounts,omitempty"`  // host volumes to bind-mount.
	Labels        map[string]string `json:"labels,omitempty"`        // labels applied to the sandbox.
	Annotations   map[string]string `json:"annotations,omitempty"`   // annotations applied to the sandbox.
}

// EndpointType identifies the kind of service exposed by a sandbox endpoint.
type EndpointType string

const (
	// EndpointTypeExecd is the execd service port.
	EndpointTypeExecd EndpointType = "execd"
	// EndpointTypeEnvd is the E2B envd service port inside the microVM.
	EndpointTypeEnvd EndpointType = "envd"
)

// EndpointSource indicates where the sandbox endpoint address comes from.
type EndpointSource string

const (
	// EndpointSourceAgentPod means the endpoint uses the agent pod IP.
	EndpointSourceAgentPod EndpointSource = "agent_pod"
	// EndpointSourceRuntime means the endpoint uses an address provided by the runtime.
	EndpointSourceRuntime EndpointSource = "runtime"
)

// Endpoint describes a network access point exposed by a sandbox.
// The IP may be an IP address (the containerd runtime uses the agent pod IP)
// or a DNS name (the E2B runtime uses the Orchestrator Service domain name).
type Endpoint struct {
	IP   string       `json:"ip"`   // endpoint address (IP or DNS name).
	Port int32        `json:"port"` // TCP port of the endpoint.
	Type EndpointType `json:"type"` // endpoint type.
}

// SandboxStatus reports the current state of a sandbox.
type SandboxStatus struct {
	SandboxID string `json:"sandboxId"`          // business identifier of the sandbox.
	Phase     string `json:"phase"`              // lifecycle phase, e.g. running or delete_failed.
	Message   string `json:"message,omitempty"`  // human-readable status detail.
	ExitCode  int    `json:"exitCode,omitempty"` // process exit code, if terminated.
	CreatedAt int64  `json:"createdAt"`          // creation timestamp in unix seconds.
}

// CreateSandboxRequest is the request body for creating a sandbox.
type CreateSandboxRequest struct {
	Sandbox SandboxSpec `json:"sandbox"` // sandbox specification.
}

// CreateSandboxResponse is the response returned after creating a sandbox.
type CreateSandboxResponse struct {
	Success        bool           `json:"success"`                  // whether the creation succeeded.
	Message        string         `json:"message,omitempty"`        // error message when success is false.
	SandboxID      string         `json:"sandboxId"`                // identifier of the created sandbox.
	CreatedAt      int64          `json:"createdAt"`                // creation timestamp in unix seconds.
	Endpoints      []Endpoint     `json:"endpoints,omitempty"`      // access endpoints for the sandbox.
	EndpointSource EndpointSource `json:"endpointSource,omitempty"` // origin of the endpoint addresses.
}

// DeleteSandboxRequest is the request body for deleting a sandbox.
type DeleteSandboxRequest struct {
	SandboxID string `json:"sandboxId"` // identifier of the sandbox to delete.
}

// DeleteSandboxResponse is the response returned after deleting a sandbox.
type DeleteSandboxResponse struct {
	Success bool   `json:"success"`           // whether the deletion succeeded.
	Message string `json:"message,omitempty"` // error message when success is false.
}

// FluxSandboxAgentStatus reports the aggregate state of an agent and its sandboxes.
type FluxSandboxAgentStatus struct {
	FluxSandboxAgentID string          `json:"fluxsandboxagentId"` // agent identifier (pod name).
	NodeName           string          `json:"nodeName"`           // node hosting the agent.
	RuntimeType        string          `json:"runtimeType"`        // container runtime type.
	Capacity           int             `json:"capacity"`           // maximum sandboxes the agent can host.
	Allocated          int             `json:"allocated"`          // currently active sandboxes.
	Images             []string        `json:"images"`             // images cached locally.
	SandboxStatuses    []SandboxStatus `json:"sandboxStatuses"`    // status of each sandbox.
}
