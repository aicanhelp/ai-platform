/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
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
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestBKEClusterPhaseString(t *testing.T) {
	tests := []struct {
		name     string
		phase    BKEClusterPhase
		expected string
	}{
		{
			name:     "empty phase",
			phase:    BKEClusterPhase(""),
			expected: "",
		},
		{
			name:     "provisioning phase",
			phase:    BKEClusterPhase("Provisioning"),
			expected: "Provisioning",
		},
		{
			name:     "provisioned phase",
			phase:    BKEClusterPhase("Provisioned"),
			expected: "Provisioned",
		},
		{
			name:     "running phase",
			phase:    BKEClusterPhase("Running"),
			expected: "Running",
		},
		{
			name:     "failed phase",
			phase:    BKEClusterPhase("Failed"),
			expected: "Failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.phase.String()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBKEClusterPhaseIn(t *testing.T) {
	allPhases := BKEClusterPhases{
		BKEClusterPhase("Provisioning"),
		BKEClusterPhase("Provisioned"),
		BKEClusterPhase("Running"),
		BKEClusterPhase("Failed"),
	}

	tests := []struct {
		name     string
		phase    BKEClusterPhase
		phases   BKEClusterPhases
		expected bool
	}{
		{
			name:     "phase in phases",
			phase:    BKEClusterPhase("Running"),
			phases:   allPhases,
			expected: true,
		},
		{
			name:     "phase not in phases",
			phase:    BKEClusterPhase("Unknown"),
			phases:   allPhases,
			expected: false,
		},
		{
			name:     "empty phase in empty phases",
			phase:    BKEClusterPhase(""),
			phases:   BKEClusterPhases{},
			expected: false,
		},
		{
			name:     "phase in single element phases",
			phase:    BKEClusterPhase("Provisioning"),
			phases:   BKEClusterPhases{BKEClusterPhase("Provisioning")},
			expected: true,
		},
		{
			name:     "phase not in single element phases",
			phase:    BKEClusterPhase("Running"),
			phases:   BKEClusterPhases{BKEClusterPhase("Provisioning")},
			expected: false,
		},
		{
			name:     "empty phase in phases with empty",
			phase:    BKEClusterPhase(""),
			phases:   BKEClusterPhases{BKEClusterPhase("")},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.phase.In(tt.phases)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBKEClusterPhaseNotIn(t *testing.T) {
	allPhases := BKEClusterPhases{
		BKEClusterPhase("Provisioning"),
		BKEClusterPhase("Provisioned"),
		BKEClusterPhase("Running"),
	}

	tests := []struct {
		name     string
		phase    BKEClusterPhase
		phases   BKEClusterPhases
		expected bool
	}{
		{
			name:     "phase not in phases",
			phase:    BKEClusterPhase("Failed"),
			phases:   allPhases,
			expected: true,
		},
		{
			name:     "phase in phases",
			phase:    BKEClusterPhase("Running"),
			phases:   allPhases,
			expected: false,
		},
		{
			name:     "empty phase not in empty phases",
			phase:    BKEClusterPhase(""),
			phases:   BKEClusterPhases{},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.phase.NotIn(tt.phases)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBKEClusterPhasesAdd(t *testing.T) {
	tests := []struct {
		name           string
		initial        BKEClusterPhases
		toAdd          []BKEClusterPhase
		expectedLen    int
		expectedPhases BKEClusterPhases
	}{
		{
			name:           "add single phase to empty",
			initial:        BKEClusterPhases{},
			toAdd:          []BKEClusterPhase{BKEClusterPhase("Provisioning")},
			expectedLen:    numOne,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Provisioning")},
		},
		{
			name:           "add multiple phases to empty",
			initial:        BKEClusterPhases{},
			toAdd:          []BKEClusterPhase{BKEClusterPhase("Provisioning"), BKEClusterPhase("Provisioned")},
			expectedLen:    numTwo,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Provisioned")},
		},
		{
			name:           "add phases to existing",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning")},
			toAdd:          []BKEClusterPhase{BKEClusterPhase("Provisioned"), BKEClusterPhase("Running")},
			expectedLen:    numThree,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Provisioned"), BKEClusterPhase("Running")},
		},
		{
			name:           "add duplicate phase",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning")},
			toAdd:          []BKEClusterPhase{BKEClusterPhase("Provisioning")},
			expectedLen:    numTwo,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Provisioning")},
		},
		{
			name:           "add no phases",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning")},
			toAdd:          []BKEClusterPhase{},
			expectedLen:    numOne,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Provisioning")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			phases := tt.initial
			phases.Add(tt.toAdd...)
			assert.Len(t, phases, tt.expectedLen)
			for i, p := range tt.expectedPhases {
				assert.Equal(t, p, phases[i])
			}
		})
	}
}

func TestBKEClusterPhasesRemove(t *testing.T) {
	tests := []struct {
		name           string
		initial        BKEClusterPhases
		toRemove       []BKEClusterPhase
		expectedLen    int
		expectedPhases BKEClusterPhases
	}{
		{
			name:           "remove single phase",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Running")},
			toRemove:       []BKEClusterPhase{BKEClusterPhase("Provisioning")},
			expectedLen:    numOne,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Running")},
		},
		{
			name:           "remove multiple phases",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Provisioned"), BKEClusterPhase("Running")},
			toRemove:       []BKEClusterPhase{BKEClusterPhase("Provisioning"), BKEClusterPhase("Running")},
			expectedLen:    numOne,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Provisioned")},
		},
		{
			name:           "remove phase not in list",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning")},
			toRemove:       []BKEClusterPhase{BKEClusterPhase("Running")},
			expectedLen:    numOne,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Provisioning")},
		},
		{
			name:           "remove all phases",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Running")},
			toRemove:       []BKEClusterPhase{BKEClusterPhase("Provisioning"), BKEClusterPhase("Running")},
			expectedLen:    numZero,
			expectedPhases: BKEClusterPhases{},
		},
		{
			name:           "remove from empty list",
			initial:        BKEClusterPhases{},
			toRemove:       []BKEClusterPhase{BKEClusterPhase("Provisioning")},
			expectedLen:    numZero,
			expectedPhases: BKEClusterPhases{},
		},
		{
			name:           "remove no phases",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Running")},
			toRemove:       []BKEClusterPhase{},
			expectedLen:    numTwo,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Running")},
		},
		{
			name:           "remove duplicate phases",
			initial:        BKEClusterPhases{BKEClusterPhase("Provisioning"), BKEClusterPhase("Provisioning"), BKEClusterPhase("Running")},
			toRemove:       []BKEClusterPhase{BKEClusterPhase("Provisioning")},
			expectedLen:    numOne,
			expectedPhases: BKEClusterPhases{BKEClusterPhase("Running")},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			phases := tt.initial
			phases.Remove(tt.toRemove...)
			assert.Len(t, phases, tt.expectedLen)
			for i, p := range tt.expectedPhases {
				assert.Equal(t, p, phases[i])
			}
		})
	}
}

func TestBKEAgentStatusReset(t *testing.T) {
	tests := []struct {
		name          string
		initialStatus BKEAgentStatus
		expected      BKEAgentStatus
	}{
		{
			name: "reset with non-zero values",
			initialStatus: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 2,
				Status:             "5/10",
			},
			expected: BKEAgentStatus{
				Replies:            0,
				UnavailableReplies: 0,
				Status:             "0/0",
			},
		},
		{
			name: "reset with zero values",
			initialStatus: BKEAgentStatus{
				Replies:            0,
				UnavailableReplies: 0,
				Status:             "0/0",
			},
			expected: BKEAgentStatus{
				Replies:            0,
				UnavailableReplies: 0,
				Status:             "0/0",
			},
		},
		{
			name: "reset with custom status",
			initialStatus: BKEAgentStatus{
				Replies:            100,
				UnavailableReplies: 50,
				Status:             "50/100",
			},
			expected: BKEAgentStatus{
				Replies:            0,
				UnavailableReplies: 0,
				Status:             "0/0",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.initialStatus.Reset()
			assert.Equal(t, tt.expected.Replies, tt.initialStatus.Replies)
			assert.Equal(t, tt.expected.UnavailableReplies, tt.initialStatus.UnavailableReplies)
			assert.Equal(t, tt.expected.Status, tt.initialStatus.Status)
		})
	}
}

func TestBKEAgentStatusReady(t *testing.T) {
	tests := []struct {
		name     string
		status   BKEAgentStatus
		expected bool
	}{
		{
			name: "ready with zero unavailable replies",
			status: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 0,
				Status:             "10/10",
			},
			expected: true,
		},
		{
			name: "not ready with unavailable replies",
			status: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 2,
				Status:             "8/10",
			},
			expected: false,
		},
		{
			name: "ready with zero replies",
			status: BKEAgentStatus{
				Replies:            0,
				UnavailableReplies: 0,
				Status:             "0/0",
			},
			expected: true,
		},
		{
			name: "not ready with all unavailable",
			status: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 10,
				Status:             "0/10",
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.status.Ready()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBKEAgentStatusEqual(t *testing.T) {
	tests := []struct {
		name     string
		status   BKEAgentStatus
		other    BKEAgentStatus
		expected bool
	}{
		{
			name: "equal status",
			status: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 2,
				Status:             "8/10",
			},
			other: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 2,
				Status:             "8/10",
			},
			expected: true,
		},
		{
			name: "different replies",
			status: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 2,
				Status:             "8/10",
			},
			other: BKEAgentStatus{
				Replies:            20,
				UnavailableReplies: 2,
				Status:             "8/10",
			},
			expected: false,
		},
		{
			name: "different unavailable replies",
			status: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 2,
				Status:             "8/10",
			},
			other: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 0,
				Status:             "8/10",
			},
			expected: false,
		},
		{
			name: "different status string",
			status: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 2,
				Status:             "8/10",
			},
			other: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 2,
				Status:             "10/10",
			},
			expected: false,
		},
		{
			name: "both zero",
			status: BKEAgentStatus{
				Replies:            0,
				UnavailableReplies: 0,
				Status:             "0/0",
			},
			other: BKEAgentStatus{
				Replies:            0,
				UnavailableReplies: 0,
				Status:             "0/0",
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.status.Equal(&tt.other)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestBKEClusterStatusFields(t *testing.T) {
	t.Run("full status", func(t *testing.T) {
		status := BKEClusterStatus{
			Ready:             true,
			OpenFuyaoVersion:  "v1.0.0",
			KubernetesVersion: "v1.25.6",
			EtcdVersion:       "3.5.9",
			ContainerdVersion: "1.6.20",
			AgentStatus: BKEAgentStatus{
				Replies:            10,
				UnavailableReplies: 0,
				Status:             "10/10",
			},
			Phase:              BKEClusterPhase("Running"),
			ClusterStatus:      ClusterStatus("Operational"),
			ClusterHealthState: ClusterHealthState("Healthy"),
			AddonStatus: []Product{
				{Name: "calico", Version: "v3.24.5"},
			},
			Conditions: ClusterConditions{
				{
					Type:               ClusterConditionType("Ready"),
					Status:             ConditionTrue,
					LastTransitionTime: &metav1.Time{},
					Reason:             "ClusterReady",
					Message:            "Cluster is ready",
				},
			},
		}

		assert.True(t, status.Ready)
		assert.Equal(t, "v1.0.0", status.OpenFuyaoVersion)
		assert.Equal(t, "v1.25.6", status.KubernetesVersion)
		assert.Equal(t, "3.5.9", status.EtcdVersion)
		assert.Equal(t, "1.6.20", status.ContainerdVersion)
		assert.True(t, status.AgentStatus.Ready())
		assert.Equal(t, BKEClusterPhase("Running"), status.Phase)
		assert.Equal(t, ClusterStatus("Operational"), status.ClusterStatus)
		assert.Len(t, status.AddonStatus, numOne)
		assert.Len(t, status.Conditions, numOne)
	})

	t.Run("minimal status", func(t *testing.T) {
		status := BKEClusterStatus{
			Ready: false,
		}

		assert.False(t, status.Ready)
		assert.Empty(t, status.OpenFuyaoVersion)
		assert.Empty(t, status.KubernetesVersion)
		assert.Empty(t, status.EtcdVersion)
		assert.Empty(t, status.ContainerdVersion)
		assert.Empty(t, status.Phase)
		assert.Empty(t, status.ClusterStatus)
		assert.Empty(t, status.AddonStatus)
		assert.Empty(t, status.Conditions)
	})
}

func TestClusterConditionFields(t *testing.T) {
	t.Run("full condition", func(t *testing.T) {
		now := metav1.Now()
		condition := ClusterCondition{
			Type:               ClusterConditionType("Ready"),
			AddonName:          "calico",
			Status:             ConditionTrue,
			LastTransitionTime: &now,
			Reason:             "ClusterReady",
			Message:            "Cluster is ready",
		}

		assert.Equal(t, ClusterConditionType("Ready"), condition.Type)
		assert.Equal(t, "calico", condition.AddonName)
		assert.Equal(t, ConditionTrue, condition.Status)
		assert.NotNil(t, condition.LastTransitionTime)
		assert.Equal(t, "ClusterReady", condition.Reason)
		assert.Equal(t, "Cluster is ready", condition.Message)
	})

	t.Run("minimal condition", func(t *testing.T) {
		condition := ClusterCondition{
			Type:   ClusterConditionType("Ready"),
			Status: ConditionFalse,
		}

		assert.Equal(t, ClusterConditionType("Ready"), condition.Type)
		assert.Equal(t, ConditionFalse, condition.Status)
		assert.Empty(t, condition.AddonName)
		assert.Nil(t, condition.LastTransitionTime)
		assert.Empty(t, condition.Reason)
		assert.Empty(t, condition.Message)
	})
}

func TestClusterConditionsFields(t *testing.T) {
	t.Run("multiple conditions", func(t *testing.T) {
		conditions := ClusterConditions{
			{Type: ClusterConditionType("Ready"), Status: ConditionTrue},
			{Type: ClusterConditionType("Health"), Status: ConditionTrue},
			{Type: ClusterConditionType("AddonReady"), Status: ConditionFalse},
		}

		assert.Len(t, conditions, numThree)
		assert.Equal(t, ConditionTrue, conditions[numZero].Status)
		assert.Equal(t, ConditionFalse, conditions[numTwo].Status)
	})

	t.Run("empty conditions", func(t *testing.T) {
		conditions := ClusterConditions{}

		assert.Empty(t, conditions)
	})
}

func TestPhaseStateFields(t *testing.T) {
	t.Run("full phase state", func(t *testing.T) {
		now := metav1.Now()
		endTime := metav1.Now()
		phaseState := PhaseState{
			Name:      BKEClusterPhase("Provisioned"),
			StartTime: &now,
			EndTime:   &endTime,
			Status:    BKEClusterPhaseStatus("Success"),
			Message:   "Phase completed successfully",
		}

		assert.Equal(t, BKEClusterPhase("Provisioned"), phaseState.Name)
		assert.NotNil(t, phaseState.StartTime)
		assert.NotNil(t, phaseState.EndTime)
		assert.Equal(t, BKEClusterPhaseStatus("Success"), phaseState.Status)
		assert.Equal(t, "Phase completed successfully", phaseState.Message)
	})

	t.Run("minimal phase state", func(t *testing.T) {
		phaseState := PhaseState{
			Name:   BKEClusterPhase("Provisioning"),
			Status: BKEClusterPhaseStatus("InProgress"),
		}

		assert.Equal(t, BKEClusterPhase("Provisioning"), phaseState.Name)
		assert.Equal(t, BKEClusterPhaseStatus("InProgress"), phaseState.Status)
		assert.Nil(t, phaseState.StartTime)
		assert.Nil(t, phaseState.EndTime)
		assert.Empty(t, phaseState.Message)
	})
}

func TestPhaseStatusFields(t *testing.T) {
	t.Run("multiple phase states", func(t *testing.T) {
		phaseStatus := PhaseStatus{
			{Name: BKEClusterPhase("Provisioning"), Status: BKEClusterPhaseStatus("InProgress")},
			{Name: BKEClusterPhase("Provisioned"), Status: BKEClusterPhaseStatus("Success")},
			{Name: BKEClusterPhase("Running"), Status: BKEClusterPhaseStatus("InProgress")},
		}

		assert.Len(t, phaseStatus, numThree)
		assert.Equal(t, BKEClusterPhaseStatus("InProgress"), phaseStatus[numZero].Status)
		assert.Equal(t, BKEClusterPhaseStatus("Success"), phaseStatus[numOne].Status)
	})

	t.Run("empty phase status", func(t *testing.T) {
		phaseStatus := PhaseStatus{}

		assert.Empty(t, phaseStatus)
	})
}

func TestConditionStatusConstants(t *testing.T) {
	assert.Equal(t, ConditionStatus("True"), ConditionTrue)
	assert.Equal(t, ConditionStatus("False"), ConditionFalse)
	assert.Equal(t, ConditionStatus("Unknown"), ConditionUnknown)
}

func TestDeclarativeUpgradeStatusLifecycle(t *testing.T) {
	now := metav1.Now()
	status := &DeclarativeUpgradeStatus{
		TargetVersion: "old",
		FinishedAt:    &now,
		LastError:     "previous",
		LastFailure:   &DeclarativeUpgradeFailureRecord{Name: "kubelet"},
		Completed: []DeclarativeUpgradeComponentRecord{{
			Name:    "kube-apiserver",
			Version: "v1",
		}},
	}

	status.ResetForTarget("v26.05", now)
	assert.Equal(t, "v26.05", status.TargetVersion)
	assert.Equal(t, &now, status.StartedAt)
	assert.Nil(t, status.FinishedAt)
	assert.Empty(t, status.LastError)
	assert.Nil(t, status.LastFailure)
	assert.Nil(t, status.Completed)

	later := metav1.Now()
	assert.False(t, status.EnsureInitialized("v26.05", later))
	assert.Equal(t, &now, status.StartedAt)
	status.FinishedAt = &later
	assert.False(t, status.EnsureInitialized("v26.05", later))
	assert.Nil(t, status.FinishedAt)
	assert.True(t, status.EnsureInitialized("v26.06", later))
	assert.Equal(t, "v26.06", status.TargetVersion)
}

func TestDeclarativeUpgradeStatusCompletedAndFailure(t *testing.T) {
	now := metav1.Now()
	status := &DeclarativeUpgradeStatus{}
	assert.False(t, status.IsCompleted("kubelet", ""))

	status.MarkCompleted("kubelet", "", now)
	assert.True(t, status.IsCompleted("kubelet", defaultComponentVersion))
	status.MarkCompleted("kubelet", defaultComponentVersion, now)
	assert.Len(t, status.Completed, 1)

	status.MarkFailure("kubelet", "", "boom", now)
	assert.NotNil(t, status.LastFailure)
	assert.Equal(t, defaultComponentVersion, status.LastFailure.Version)
	assert.Equal(t, int32(1), status.LastFailure.Attempt)
	assert.Equal(t, "boom", status.LastError)

	status.MarkFailure("kubelet", defaultComponentVersion, "boom again", now)
	assert.Equal(t, int32(2), status.LastFailure.Attempt)
	assert.Equal(t, "boom again", status.LastError)

	status.MarkFailure("etcd", "v2", "other", now)
	assert.Equal(t, int32(1), status.LastFailure.Attempt)
	assert.Equal(t, "v2", status.LastFailure.Version)

	status.ClearFailure()
	assert.Nil(t, status.LastFailure)
}

func TestDeclarativeUpgradeStatusNilReceivers(t *testing.T) {
	now := metav1.Now()
	var status *DeclarativeUpgradeStatus
	status.ResetForTarget("v1", now)
	assert.True(t, status.EnsureInitialized("v1", now))
	assert.False(t, status.IsCompleted("kubelet", "v1"))
	status.MarkCompleted("kubelet", "v1", now)
	status.MarkFailure("kubelet", "v1", "boom", now)
	status.ClearFailure()
}
