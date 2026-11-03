/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// +k8s:deepcopy-gen=package

package v1beta1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

const (
	ConditionTrue    ConditionStatus = "True"
	ConditionFalse   ConditionStatus = "False"
	ConditionUnknown ConditionStatus = "Unknown"
)

type BKEClusterPhase string
type BKEClusterPhases []BKEClusterPhase

func (in BKEClusterPhase) String() string { return string(in) }

func (in BKEClusterPhase) In(phases BKEClusterPhases) bool {
	for _, phase := range phases {
		if phase == in {
			return true
		}
	}
	return false
}

func (in BKEClusterPhase) NotIn(phases BKEClusterPhases) bool {
	return !in.In(phases)
}

func (in *BKEClusterPhases) Add(phases ...BKEClusterPhase) {
	for _, phase := range phases {
		*in = append(*in, phase)
	}

}

func (in *BKEClusterPhases) Remove(phases ...BKEClusterPhase) {
	// 创建一个新的切片，用于存储结果
	result := make([]BKEClusterPhase, 0, len(*in))

	for _, p := range *in {
		found := false

		// 检查 p 是否在 phases 中
		for _, phase := range phases {
			if p == phase {
				found = true
				break
			}
		}

		// 如果 p 不在 phases 中，将其添加到结果切片中
		if !found {
			result = append(result, p)
		}
	}

	// 将结果切片赋值回原始切片
	*in = result
}

type ClusterStatus string

type ClusterHealthState string

type BKEClusterPhaseStatus string

type ClusterConditionType string

type ConditionStatus string

// DeclarativeUpgradeStatus persists declarative upgrade progress across controller restarts.
type DeclarativeUpgradeStatus struct {
	// TargetVersion is the desired ClusterVersion for this upgrade execution plan.
	// When it changes, Completed should be reset.
	// +optional
	TargetVersion string `json:"targetVersion,omitempty"`

	// StartedAt marks the first time we initialized progress for TargetVersion.
	// +optional
	StartedAt *metav1.Time `json:"startedAt,omitempty"`

	// FinishedAt marks when the upgrade plan completed successfully.
	// +optional
	FinishedAt *metav1.Time `json:"finishedAt,omitempty"`

	// LastError records the last observed error during DAG execution for this TargetVersion.
	// +optional
	LastError string `json:"lastError,omitempty"`

	// LastFailure records the last failed component for this TargetVersion, for easier debugging.
	// It must NOT be used for skip decisions.
	// +optional
	LastFailure *DeclarativeUpgradeFailureRecord `json:"lastFailure,omitempty"`

	// Completed holds component completion records for this TargetVersion.
	// +optional
	Completed []DeclarativeUpgradeComponentRecord `json:"completed,omitempty"`
}

type DeclarativeUpgradeComponentRecord struct {
	// Name is the declarative upgrade component name (DAG node name).
	// +required
	Name string `json:"name"`
	// Version is the component version key used in the DAG node.
	// +optional
	Version string `json:"version,omitempty"`
	// CompletedAt is the time the component finished successfully.
	// +required
	CompletedAt metav1.Time `json:"completedAt"`
}

// DeclarativeUpgradeFailureRecord records last failure info for a declarative upgrade component.
type DeclarativeUpgradeFailureRecord struct {
	// Name is the declarative upgrade component name (DAG node name).
	// +required
	Name string `json:"name"`
	// Version is the component version key used in the DAG node.
	// +optional
	Version string `json:"version,omitempty"`
	// FailedAt is the time the component execution failed.
	// +required
	FailedAt metav1.Time `json:"failedAt"`
	// Error is a short error message for the failure.
	// +optional
	Error string `json:"error,omitempty"`
	// Attempt is a best-effort counter for consecutive failures of the same component+version.
	// +optional
	Attempt int32 `json:"attempt,omitempty"`
}

// ResetForTarget resets upgrade progress when the target version changes.
// It also clears FinishedAt and LastError.
func (s *DeclarativeUpgradeStatus) ResetForTarget(targetVersion string, now metav1.Time) {
	if s == nil {
		return
	}
	s.TargetVersion = targetVersion
	s.StartedAt = &now
	s.FinishedAt = nil
	s.LastError = ""
	s.LastFailure = nil
	s.Completed = nil
}

// EnsureInitialized ensures the declarative upgrade status is present and aligned to targetVersion.
// It returns true when a reset was performed (i.e. target changed or status missing).
func (s *DeclarativeUpgradeStatus) EnsureInitialized(targetVersion string, now metav1.Time) bool {
	if s == nil {
		return true
	}
	if s.TargetVersion != targetVersion {
		s.ResetForTarget(targetVersion, now)
		return true
	}
	if s.StartedAt == nil {
		s.StartedAt = &now
	}
	// New execution of same target should clear FinishedAt.
	if s.FinishedAt != nil {
		s.FinishedAt = nil
	}
	return false
}

func normalizeUpgradeComponentVersion(version string) string {
	if version == "" {
		return defaultComponentVersion
	}
	return version
}

// NOTE: Keep this constant in sync with DAG executor default.
const defaultComponentVersion = "v1.0.0"

// IsCompleted returns true when a component+version has already completed for current TargetVersion.
func (s *DeclarativeUpgradeStatus) IsCompleted(name, version string) bool {
	if s == nil {
		return false
	}
	version = normalizeUpgradeComponentVersion(version)
	for i := range s.Completed {
		rec := s.Completed[i]
		if rec.Name != name {
			continue
		}
		if normalizeUpgradeComponentVersion(rec.Version) == version {
			return true
		}
	}
	return false
}

// MarkCompleted records a completion for the component+version if not already present.
func (s *DeclarativeUpgradeStatus) MarkCompleted(name, version string, now metav1.Time) {
	if s == nil {
		return
	}
	version = normalizeUpgradeComponentVersion(version)
	if s.IsCompleted(name, version) {
		return
	}
	s.Completed = append(s.Completed, DeclarativeUpgradeComponentRecord{
		Name:        name,
		Version:     version,
		CompletedAt: now,
	})
}

// MarkFailure updates LastFailure and LastError for debugging.
// Attempt is increased only when the same component+version fails consecutively.
func (s *DeclarativeUpgradeStatus) MarkFailure(name, version, errMsg string, now metav1.Time) {
	if s == nil {
		return
	}
	version = normalizeUpgradeComponentVersion(version)
	var attempt int32 = 1
	if s.LastFailure != nil &&
		s.LastFailure.Name == name &&
		normalizeUpgradeComponentVersion(s.LastFailure.Version) == version {
		if s.LastFailure.Attempt > 0 {
			attempt = s.LastFailure.Attempt + 1
		}
	}
	s.LastFailure = &DeclarativeUpgradeFailureRecord{
		Name:     name,
		Version:  version,
		FailedAt: now,
		Error:    errMsg,
		Attempt:  attempt,
	}
	s.LastError = errMsg
}

func (s *DeclarativeUpgradeStatus) ClearFailure() {
	if s == nil {
		return
	}
	s.LastFailure = nil
}

// LifecyclePhase is the lifecycle stage of a declarative upgrade component.
// +kubebuilder:validation:Enum=Pending;Installing;Installed;Upgrading;RollingBack;Uninstalling;Removed;Failed
type LifecyclePhase string

const (
	LifecyclePhasePending      LifecyclePhase = "Pending"
	LifecyclePhaseInstalling   LifecyclePhase = "Installing"
	LifecyclePhaseInstalled    LifecyclePhase = "Installed"
	LifecyclePhaseUpgrading    LifecyclePhase = "Upgrading"
	LifecyclePhaseRollingBack  LifecyclePhase = "RollingBack"
	LifecyclePhaseUninstalling LifecyclePhase = "Uninstalling"
	LifecyclePhaseRemoved      LifecyclePhase = "Removed"
	LifecyclePhaseFailed       LifecyclePhase = "Failed"
)

// LifecycleComponentType is the status-side component scope (node vs cluster).
// Distinct from executor install types (inline / yaml / helm).
// +kubebuilder:validation:Enum=node;cluster
type LifecycleComponentType string

const (
	LifecycleComponentTypeNode    LifecycleComponentType = "node"
	LifecycleComponentTypeCluster LifecycleComponentType = "cluster"
)

// ComponentLifecycleStatus records one component's lifecycle on BKECluster status.
type ComponentLifecycleStatus struct {
	// Name is the component name; usually matches the map key in ClusterComponentStatuses.
	// +required
	Name string `json:"name"`

	// NodeIP is set for node-scoped components; empty for cluster-scoped components.
	// +optional
	NodeIP string `json:"nodeIP,omitempty"`

	// ComponentType is the status-side scope: node or cluster.
	// +required
	ComponentType LifecycleComponentType `json:"componentType"`

	// Phase is the current lifecycle phase.
	// +required
	Phase LifecyclePhase `json:"phase"`

	// CurrentVersion is the last successfully installed/aligned version.
	// +optional
	CurrentVersion string `json:"currentVersion,omitempty"`

	// TargetVersion is the desired version for the in-progress upgrade.
	// +optional
	TargetVersion string `json:"targetVersion,omitempty"`

	// LastTransitionTime is updated when Phase changes.
	// +optional
	LastTransitionTime *metav1.Time `json:"lastTransitionTime,omitempty"`

	// Message holds failure details or supplemental notes.
	// +optional
	Message string `json:"message,omitempty"`
}

// ComponentConfigHash 定义了 Kubernetes 控制面三大组件的独立配置 Hash
type ComponentConfigHash struct {
	APIServer         string `json:"apiServer,omitempty"`
	ControllerManager string `json:"controllerManager,omitempty"`
	Scheduler         string `json:"scheduler,omitempty"`
}

// BKEClusterStatus defines the observed state of BKECluster
type BKEClusterStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	// +optional
	Ready bool `json:"ready"`

	// +optional
	OpenFuyaoVersion string `json:"openFuyaoVersion,omitempty"`

	// +optional
	KubernetesVersion string `json:"kubernetesVersion,omitempty"`

	// +optional
	EtcdVersion string `json:"etcdVersion,omitempty"`

	// +optional
	ContainerdVersion string `json:"containerdVersion,omitempty"`

	// NginxProxyVersion tracks the deployed nginx-proxy image tag (nginx load balance mode).
	// +optional
	NginxProxyVersion string `json:"nginxProxyVersion,omitempty"`

	// +optional
	AgentStatus BKEAgentStatus `json:"agentStatus"`

	// Phase is the current phase of the cluster.
	// Deprecated: Use ClusterStatus as the single source of truth.
	// Phase is auto-synced from ClusterStatus via SyncStatusFields.
	// +optional
	Phase BKEClusterPhase `json:"phase,omitempty"`

	// ControlPlaneConfigHash records the control plane static-pod config fingerprint
	// used by EnsureMasterConfigUpdate. It stays at the last success baseline while
	// reconcile failures are below ALLOWED_FAILED_COUNT; after that limit it is set
	// to the attempted desired hash. On success it is always updated to the applied hash.
	// A later update runs only when the Spec hash differs from this value.
	// +optional
	ControlPlaneConfigHash ComponentConfigHash `json:"controlPlaneConfigHash,omitempty"`

	// EtcdConfigHash records the etcd static-pod config fingerprint used by
	// EnsureEtcdConfigUpdate, with the same success / retry-limit / thorough-failure
	// semantics as ControlPlaneConfigHash. A later update runs only when the Spec
	// hash differs from this value.
	// +optional
	EtcdConfigHash string `json:"etcdConfigHash,omitempty"`

	// ClusterStatus is the current operate status of the cluster.
	// This is the single source of truth for cluster state.
	// +optional
	ClusterStatus ClusterStatus `json:"clusterStatus,omitempty"`

	// ClusterHealthState
	// Deprecated: Use ClusterStatus as the single source of truth.
	// ClusterHealthState is auto-synced from ClusterStatus via SyncStatusFields.
	// +optional
	ClusterHealthState ClusterHealthState `json:"clusterHealthState,omitempty"`

	// AddonStatus is the current status of the addons.
	AddonStatus []Product `json:"addonStatus,omitempty"`

	// +kubebuilder:object:generate:=true
	// +optional
	PhaseStatus PhaseStatus `json:"phaseStatus,omitempty"`

	// +kubebuilder:object:generate:=true
	// +optional
	Conditions ClusterConditions `json:"conditions,omitempty"`

	// DeclarativeUpgrade holds progress for declarative DAG upgrades.
	// +optional
	DeclarativeUpgrade *DeclarativeUpgradeStatus `json:"declarativeUpgrade,omitempty"`

	// ClusterComponentStatuses records lifecycle status per component name (DAG/Bundle key).
	// +optional
	ClusterComponentStatuses map[string]ComponentLifecycleStatus `json:"clusterComponentStatuses,omitempty"`
}

// +kubebuilder:object:generate=true

type ClusterConditions []ClusterCondition

type ClusterCondition struct {
	Type ClusterConditionType `json:"type"`

	// AddonName is the name of the current reconcile addon
	// +optional
	AddonName string `json:"addonName,omitempty"`

	// Status of the condition, one of True, False, Unknown.
	Status ConditionStatus `json:"status"`

	// Last time the condition transitioned from one status to another.
	// This should be when the underlying condition changed. If that is not known, then using the time when
	// the API field changed is acceptable.
	LastTransitionTime *metav1.Time `json:"lastTransitionTime,omitempty"`

	// The reason for the condition's last transition in CamelCase.
	// The specific API may choose whether or not this field is considered a guaranteed API.
	// This field may not be empty.
	// +optional
	Reason string `json:"reason,omitempty"`

	// A human readable message indicating details about the transition.
	// This field may be empty.
	// +optional
	Message string `json:"message,omitempty"`
}

type BKEAgentStatus struct {
	// +optional
	// +kubebuilder:default:=0
	Replies int32 `json:"replies,omitempty"`
	// +optional
	// +kubebuilder:default:=0
	UnavailableReplies int32 `json:"unavailableReplies,omitempty"`
	// +optional
	// +kubebuilder:default:="0/0"
	Status string `json:"status,omitempty"`
}

func (agentStatus *BKEAgentStatus) Reset() {
	agentStatus.Replies = 0
	agentStatus.UnavailableReplies = 0
	agentStatus.Status = "0/0"
}

func (agentStatus *BKEAgentStatus) Ready() bool {
	return agentStatus.UnavailableReplies == 0
}

func (agentStatus *BKEAgentStatus) Equal(other *BKEAgentStatus) bool {
	return agentStatus.Replies == other.Replies &&
		agentStatus.UnavailableReplies == other.UnavailableReplies &&
		agentStatus.Status == other.Status
}

type PhaseStatus []PhaseState

type PhaseState struct {
	// Name is the name of the phase name
	// +required
	Name BKEClusterPhase `json:"name,omitempty"`
	// StartTime is the start time of the phase
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`
	// EndTime is the end time of the phase
	// +optional
	EndTime *metav1.Time `json:"endTime,omitempty"`
	// Status is the status of the phase
	// +required
	Status BKEClusterPhaseStatus `json:"status,omitempty"`
	// Message is the message of the phase
	// +optional
	Message string `json:"message,omitempty"`
}
