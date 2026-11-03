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

// Package v1alpha1 defines the v1alpha1 API types for the Flux-Sandbox control
// plane: SandboxGroup, SubSandboxGroup, Sandbox and their supporting specs,
// statuses and shared labels/phases.
package v1alpha1

// CapacitySpec is the capacity declaration for a SandboxGroup.
// Field names align with Design doc §5.0.3:
// agentPodMin/agentPodMax/sandboxBufferMin/sandboxBufferMax.
type CapacitySpec struct {
	AgentPodMin        int `json:"agentPodMin"`                // Minimum number of agent pods to maintain.
	AgentPodMax        int `json:"agentPodMax"`                // Maximum number of agent pods allowed.
	SandboxBufferMin   int `json:"sandboxBufferMin,omitempty"` // Minimum idle sandboxes kept as a warm buffer.
	SandboxBufferMax   int `json:"sandboxBufferMax,omitempty"` // Maximum idle sandboxes kept as a warm buffer.
	MaxSandboxesPerPod int `json:"maxSandboxesPerPod"`         // Maximum sandboxes co-located on a single agent pod.
}

const (
	// LabelSandboxID is the label key holding the sandbox ID.
	LabelSandboxID = "sandbox.flux.io/sandbox-id"
	// LabelOpenSandboxID is the label key holding the opensandbox.io sandbox ID.
	LabelOpenSandboxID = "opensandbox.io/sandbox-id"
	// LabelSandboxGroup is the label key holding the owning SandboxGroup name.
	LabelSandboxGroup = "sandbox.flux.io/sandbox-group"
	// LabelScheduler is the label key holding the scheduler name.
	LabelScheduler = "sandbox.flux.io/scheduler"
	// LabelSubSandboxGroup is the label key holding the SubSandboxGroup name.
	LabelSubSandboxGroup = "sandbox.flux.io/sub-sandbox-group"
	// LabelControllerOwned marks objects owned by the flux-sandbox controller.
	LabelControllerOwned = "sandbox.flux.io/controller-owned"
	// PhasePending marks a resource waiting to become active.
	PhasePending = "Pending"
	// PhaseRunning marks a resource that is actively running.
	PhaseRunning = "Running"
	// PhaseTerminating marks a resource being torn down.
	PhaseTerminating = "Terminating"
	// PhaseDeleted marks a resource that has been fully removed.
	PhaseDeleted = "Deleted"
	// PhaseFailed marks a resource that encountered a fatal error.
	PhaseFailed = "Failed"
	// PhaseLost marks a resource whose state is unknown (heartbeat lost).
	PhaseLost = "Lost"
	// PhaseReady marks a resource that is ready to serve traffic.
	PhaseReady = "Ready"
	// PhaseNotReady marks a resource that is not yet ready to serve traffic.
	PhaseNotReady = "NotReady"
	// PhasePausing marks a sandbox that is being paused (releasing CPU/memory).
	PhasePausing = "Pausing"
	// PhasePaused marks a sandbox that has been paused; its slot is released.
	PhasePaused = "Paused"
	// PhasePauseFailed marks a sandbox whose pause operation failed.
	PhasePauseFailed = "PauseFailed"
	// PhaseResuming marks a paused sandbox that is being resumed.
	PhaseResuming = "Resuming"
	// PhaseResumeFailed marks a sandbox whose resume operation failed.
	PhaseResumeFailed = "ResumeFailed"
	// PhaseSnapshotting marks a sandbox that is being checkpointed into a snapshot.
	PhaseSnapshotting = "Snapshotting"
)
