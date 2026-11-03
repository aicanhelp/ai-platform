/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"errors"
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestMapPhaseToClusterStatus(t *testing.T) {
	tests := []struct {
		name     string
		phase    confv1beta1.BKEClusterPhase
		err      error
		expected confv1beta1.ClusterStatus
	}{
		// Init phases — success
		{"InitControlPlane success", bkev1beta1.InitControlPlane, nil, bkev1beta1.ClusterInitializing},
		{"JoinControlPlane success", bkev1beta1.JoinControlPlane, nil, bkev1beta1.ClusterInitializing},
		{"JoinWorker success", bkev1beta1.JoinWorker, nil, bkev1beta1.ClusterInitializing},
		{"FakeInitControlPlane success", bkev1beta1.FakeInitControlPlane, nil, bkev1beta1.ClusterInitializing},
		{"FakeJoinControlPlane success", bkev1beta1.FakeJoinControlPlane, nil, bkev1beta1.ClusterInitializing},
		{"FakeJoinWorker success", bkev1beta1.FakeJoinWorker, nil, bkev1beta1.ClusterInitializing},
		// Init phases — failure
		{"InitControlPlane failed", bkev1beta1.InitControlPlane, errors.New("err"), bkev1beta1.ClusterInitializationFailed},
		{"JoinControlPlane failed", bkev1beta1.JoinControlPlane, errors.New("err"), bkev1beta1.ClusterInitializationFailed},
		{"JoinWorker failed", bkev1beta1.JoinWorker, errors.New("err"), bkev1beta1.ClusterInitializationFailed},
		{"FakeInitControlPlane failed", bkev1beta1.FakeInitControlPlane, errors.New("err"), bkev1beta1.ClusterInitializationFailed},
		{"FakeJoinControlPlane failed", bkev1beta1.FakeJoinControlPlane, errors.New("err"), bkev1beta1.ClusterInitializationFailed},
		{"FakeJoinWorker failed", bkev1beta1.FakeJoinWorker, errors.New("err"), bkev1beta1.ClusterInitializationFailed},
		// Upgrade phases — success
		{"UpgradeControlPlane success", bkev1beta1.UpgradeControlPlane, nil, bkev1beta1.ClusterUpgrading},
		{"UpgradeWorker success", bkev1beta1.UpgradeWorker, nil, bkev1beta1.ClusterUpgrading},
		{"UpgradeEtcd success", bkev1beta1.UpgradeEtcd, nil, bkev1beta1.ClusterUpgrading},
		// Upgrade phases — failure
		{"UpgradeControlPlane failed", bkev1beta1.UpgradeControlPlane, errors.New("err"), bkev1beta1.ClusterUpgradeFailed},
		{"UpgradeWorker failed", bkev1beta1.UpgradeWorker, errors.New("err"), bkev1beta1.ClusterUpgradeFailed},
		{"UpgradeEtcd failed", bkev1beta1.UpgradeEtcd, errors.New("err"), bkev1beta1.ClusterUpgradeFailed},
		// Scale phases — success
		{"MasterScalingUp success", bkev1beta1.MasterScalingUp, nil, bkev1beta1.ClusterMasterScalingUp},
		{"MasterScalingDown success", bkev1beta1.MasterScalingDown, nil, bkev1beta1.ClusterMasterScalingDown},
		{"WorkerScalingUp success", bkev1beta1.WorkerScalingUp, nil, bkev1beta1.ClusterWorkerScalingUp},
		{"WorkerScalingDown success", bkev1beta1.WorkerScalingDown, nil, bkev1beta1.ClusterWorkerScalingDown},
		// Scale phases — failure
		{"MasterScalingUp failed", bkev1beta1.MasterScalingUp, errors.New("err"), bkev1beta1.ClusterMasterScalingUpFailed},
		{"MasterScalingDown failed", bkev1beta1.MasterScalingDown, errors.New("err"), bkev1beta1.ClusterMasterScalingDownFailed},
		{"WorkerScalingUp failed", bkev1beta1.WorkerScalingUp, errors.New("err"), bkev1beta1.ClusterWorkerScalingUpFailed},
		{"WorkerScalingDown failed", bkev1beta1.WorkerScalingDown, errors.New("err"), bkev1beta1.ClusterWorkerScalingDownFailed},
		// Special phases
		{"FailedBootstrapNode", bkev1beta1.FailedBootstrapNode, nil, bkev1beta1.ClusterInitializationFailed},
		{"FailedBootstrapNode with err", bkev1beta1.FailedBootstrapNode, errors.New("err"), bkev1beta1.ClusterInitializationFailed},
		{"ClusterReady phase", bkev1beta1.ClusterReady, nil, bkev1beta1.ClusterStatusReady},
		{"ClusterReady phase with err", bkev1beta1.ClusterReady, errors.New("err"), bkev1beta1.ClusterStatusReady},
		// Unknown / default
		{"unknown phase", confv1beta1.BKEClusterPhase("UnknownPhase"), nil, bkev1beta1.ClusterUnknown},
		{"empty phase", confv1beta1.BKEClusterPhase(""), nil, bkev1beta1.ClusterUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MapPhaseToClusterStatus(tt.phase, tt.err)
			if got != tt.expected {
				t.Errorf("MapPhaseToClusterStatus(%q, %v) = %q, want %q", tt.phase, tt.err, got, tt.expected)
			}
		})
	}
}

func TestMapClusterHealthStateToClusterStatus(t *testing.T) {
	tests := []struct {
		name     string
		state    confv1beta1.ClusterHealthState
		expected confv1beta1.ClusterStatus
	}{
		{"Deploying", bkev1beta1.Deploying, bkev1beta1.ClusterInitializing},
		{"DeployFailed", bkev1beta1.DeployFailed, bkev1beta1.ClusterInitializationFailed},
		{"Upgrading", bkev1beta1.Upgrading, bkev1beta1.ClusterUpgrading},
		{"UpgradeFailed", bkev1beta1.UpgradeFailed, bkev1beta1.ClusterUpgradeFailed},
		{"Managing", bkev1beta1.Managing, bkev1beta1.ClusterManaging},
		{"ManageFailed", bkev1beta1.ManageFailed, bkev1beta1.ClusterManageFailed},
		{"Unhealthy", bkev1beta1.Unhealthy, bkev1beta1.ClusterUnhealthy},
		{"Healthy", bkev1beta1.Healthy, bkev1beta1.ClusterStatusReady},
		{"Deleting", bkev1beta1.Deleting, bkev1beta1.ClusterDeleting},
		{"unknown", confv1beta1.ClusterHealthState("Unknown"), bkev1beta1.ClusterUnknown},
		{"empty", confv1beta1.ClusterHealthState(""), bkev1beta1.ClusterUnknown},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MapClusterHealthStateToClusterStatus(tt.state)
			if got != tt.expected {
				t.Errorf("MapClusterHealthStateToClusterStatus(%q) = %q, want %q", tt.state, got, tt.expected)
			}
		})
	}
}

func TestMapClusterStatusToClusterHealthState(t *testing.T) {
	toHealthState := func(s confv1beta1.ClusterStatus) confv1beta1.ClusterHealthState {
		return confv1beta1.ClusterHealthState(s)
	}
	tests := []struct {
		name     string
		status   confv1beta1.ClusterStatus
		expected confv1beta1.ClusterHealthState
	}{
		// Explicitly mapped — original 9-value enum semantics preserved
		{"ClusterReady", bkev1beta1.ClusterStatusReady, bkev1beta1.Healthy},
		{"ClusterUnhealthy", bkev1beta1.ClusterUnhealthy, bkev1beta1.Unhealthy},
		{"ClusterInitializing", bkev1beta1.ClusterInitializing, bkev1beta1.Deploying},
		{"ClusterInitializationFailed", bkev1beta1.ClusterInitializationFailed, bkev1beta1.DeployFailed},
		{"ClusterUpgrading", bkev1beta1.ClusterUpgrading, bkev1beta1.Upgrading},
		{"ClusterUpgradeFailed", bkev1beta1.ClusterUpgradeFailed, bkev1beta1.UpgradeFailed},
		{"ClusterManaging", bkev1beta1.ClusterManaging, bkev1beta1.Managing},
		{"ClusterManageFailed", bkev1beta1.ClusterManageFailed, bkev1beta1.ManageFailed},
		{"ClusterDeployingAddon", bkev1beta1.ClusterDeployingAddon, bkev1beta1.Deploying},
		{"ClusterDeployAddonFailed", bkev1beta1.ClusterDeployAddonFailed, bkev1beta1.DeployFailed},
		{"ClusterDeleting", bkev1beta1.ClusterDeleting, bkev1beta1.Deleting},
		// Unmapped — falls back to ClusterStatus string value
		{"ClusterUnknown", bkev1beta1.ClusterUnknown, toHealthState(bkev1beta1.ClusterUnknown)},
		{"ClusterChecking", bkev1beta1.ClusterChecking, toHealthState(bkev1beta1.ClusterChecking)},
		{"ClusterPaused", bkev1beta1.ClusterPaused, toHealthState(bkev1beta1.ClusterPaused)},
		{"ClusterPauseFailed", bkev1beta1.ClusterPauseFailed, toHealthState(bkev1beta1.ClusterPauseFailed)},
		{"ClusterDryRun", bkev1beta1.ClusterDryRun, toHealthState(bkev1beta1.ClusterDryRun)},
		{"ClusterDryRunFailed", bkev1beta1.ClusterDryRunFailed, toHealthState(bkev1beta1.ClusterDryRunFailed)},
		{"ClusterMasterScalingUp", bkev1beta1.ClusterMasterScalingUp, toHealthState(bkev1beta1.ClusterMasterScalingUp)},
		{"ClusterMasterScalingDown", bkev1beta1.ClusterMasterScalingDown, toHealthState(bkev1beta1.ClusterMasterScalingDown)},
		{"ClusterWorkerScalingUp", bkev1beta1.ClusterWorkerScalingUp, toHealthState(bkev1beta1.ClusterWorkerScalingUp)},
		{"ClusterWorkerScalingDown", bkev1beta1.ClusterWorkerScalingDown, toHealthState(bkev1beta1.ClusterWorkerScalingDown)},
		{"ClusterMasterScalingUpFailed", bkev1beta1.ClusterMasterScalingUpFailed, toHealthState(bkev1beta1.ClusterMasterScalingUpFailed)},
		{"ClusterMasterScalingDownFailed", bkev1beta1.ClusterMasterScalingDownFailed, toHealthState(bkev1beta1.ClusterMasterScalingDownFailed)},
		{"ClusterWorkerScalingUpFailed", bkev1beta1.ClusterWorkerScalingUpFailed, toHealthState(bkev1beta1.ClusterWorkerScalingUpFailed)},
		{"ClusterWorkerScalingDownFailed", bkev1beta1.ClusterWorkerScalingDownFailed, toHealthState(bkev1beta1.ClusterWorkerScalingDownFailed)},
		{"ClusterDeleteFailed", bkev1beta1.ClusterDeleteFailed, toHealthState(bkev1beta1.ClusterDeleteFailed)},
		{"empty", confv1beta1.ClusterStatus(""), ""},
		{"unknown", confv1beta1.ClusterStatus("Unknown"), toHealthState(confv1beta1.ClusterStatus("Unknown"))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MapClusterStatusToClusterHealthState(tt.status)
			if got != tt.expected {
				t.Errorf("MapClusterStatusToClusterHealthState(%q) = %q, want %q", tt.status, got, tt.expected)
			}
		})
	}
}

func TestMapClusterStatusToPhase(t *testing.T) {
	toPhase := func(s confv1beta1.ClusterStatus) confv1beta1.BKEClusterPhase {
		return confv1beta1.BKEClusterPhase(s)
	}
	tests := []struct {
		name     string
		status   confv1beta1.ClusterStatus
		expected confv1beta1.BKEClusterPhase
	}{
		// Explicitly mapped — original Phase enum semantics preserved
		{"ClusterInitializing", bkev1beta1.ClusterInitializing, bkev1beta1.InitControlPlane},
		{"ClusterInitializationFailed", bkev1beta1.ClusterInitializationFailed, bkev1beta1.InitControlPlane},
		{"ClusterUpgrading", bkev1beta1.ClusterUpgrading, bkev1beta1.UpgradeControlPlane},
		{"ClusterUpgradeFailed", bkev1beta1.ClusterUpgradeFailed, bkev1beta1.UpgradeControlPlane},
		{"ClusterMasterScalingUp", bkev1beta1.ClusterMasterScalingUp, bkev1beta1.MasterScalingUp},
		{"ClusterMasterScalingUpFailed", bkev1beta1.ClusterMasterScalingUpFailed, bkev1beta1.MasterScalingUp},
		{"ClusterMasterScalingDown", bkev1beta1.ClusterMasterScalingDown, bkev1beta1.MasterScalingDown},
		{"ClusterMasterScalingDownFailed", bkev1beta1.ClusterMasterScalingDownFailed, bkev1beta1.MasterScalingDown},
		{"ClusterWorkerScalingUp", bkev1beta1.ClusterWorkerScalingUp, bkev1beta1.WorkerScalingUp},
		{"ClusterWorkerScalingUpFailed", bkev1beta1.ClusterWorkerScalingUpFailed, bkev1beta1.WorkerScalingUp},
		{"ClusterWorkerScalingDown", bkev1beta1.ClusterWorkerScalingDown, bkev1beta1.WorkerScalingDown},
		{"ClusterWorkerScalingDownFailed", bkev1beta1.ClusterWorkerScalingDownFailed, bkev1beta1.WorkerScalingDown},
		{"ClusterReady", bkev1beta1.ClusterStatusReady, bkev1beta1.ClusterReady},
		// Unmapped — falls back to ClusterStatus string value
		{"ClusterUnknown", bkev1beta1.ClusterUnknown, toPhase(bkev1beta1.ClusterUnknown)},
		{"ClusterChecking", bkev1beta1.ClusterChecking, toPhase(bkev1beta1.ClusterChecking)},
		{"ClusterPaused", bkev1beta1.ClusterPaused, toPhase(bkev1beta1.ClusterPaused)},
		{"ClusterPauseFailed", bkev1beta1.ClusterPauseFailed, toPhase(bkev1beta1.ClusterPauseFailed)},
		{"ClusterDryRun", bkev1beta1.ClusterDryRun, toPhase(bkev1beta1.ClusterDryRun)},
		{"ClusterDryRunFailed", bkev1beta1.ClusterDryRunFailed, toPhase(bkev1beta1.ClusterDryRunFailed)},
		{"ClusterDeployingAddon", bkev1beta1.ClusterDeployingAddon, toPhase(bkev1beta1.ClusterDeployingAddon)},
		{"ClusterDeployAddonFailed", bkev1beta1.ClusterDeployAddonFailed, toPhase(bkev1beta1.ClusterDeployAddonFailed)},
		{"ClusterManaging", bkev1beta1.ClusterManaging, toPhase(bkev1beta1.ClusterManaging)},
		{"ClusterManageFailed", bkev1beta1.ClusterManageFailed, toPhase(bkev1beta1.ClusterManageFailed)},
		{"ClusterDeleting", bkev1beta1.ClusterDeleting, toPhase(bkev1beta1.ClusterDeleting)},
		{"ClusterDeleteFailed", bkev1beta1.ClusterDeleteFailed, toPhase(bkev1beta1.ClusterDeleteFailed)},
		{"ClusterUnhealthy", bkev1beta1.ClusterUnhealthy, toPhase(bkev1beta1.ClusterUnhealthy)},
		{"empty", confv1beta1.ClusterStatus(""), ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MapClusterStatusToPhase(tt.status)
			if got != tt.expected {
				t.Errorf("MapClusterStatusToPhase(%q) = %q, want %q", tt.status, got, tt.expected)
			}
		})
	}
}

func TestSyncStatusFields(t *testing.T) {
	tests := []struct {
		name           string
		status         confv1beta1.ClusterStatus
		expectedPhase  confv1beta1.BKEClusterPhase
		expectedHealth confv1beta1.ClusterHealthState
	}{
		{
			"ClusterInitializing",
			bkev1beta1.ClusterInitializing,
			bkev1beta1.InitControlPlane,
			bkev1beta1.Deploying,
		},
		{
			"ClusterUpgrading",
			bkev1beta1.ClusterUpgrading,
			bkev1beta1.UpgradeControlPlane,
			bkev1beta1.Upgrading,
		},
		{
			"ClusterMasterScalingUpFailed",
			bkev1beta1.ClusterMasterScalingUpFailed,
			bkev1beta1.MasterScalingUp,
			confv1beta1.ClusterHealthState(bkev1beta1.ClusterMasterScalingUpFailed),
		},
		{
			"ClusterReady",
			bkev1beta1.ClusterStatusReady,
			bkev1beta1.ClusterReady,
			bkev1beta1.Healthy,
		},
		{
			"ClusterPaused (unmapped, falls back to status for both)",
			bkev1beta1.ClusterPaused,
			confv1beta1.BKEClusterPhase(bkev1beta1.ClusterPaused),
			confv1beta1.ClusterHealthState(bkev1beta1.ClusterPaused),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cluster := &bkev1beta1.BKECluster{}
			cluster.Status.ClusterStatus = tt.status
			SyncStatusFields(cluster)
			if cluster.Status.Phase != tt.expectedPhase {
				t.Errorf("Phase = %q, want %q", cluster.Status.Phase, tt.expectedPhase)
			}
			if cluster.Status.ClusterHealthState != tt.expectedHealth {
				t.Errorf("ClusterHealthState = %q, want %q", cluster.Status.ClusterHealthState, tt.expectedHealth)
			}
		})
	}

	t.Run("nil cluster does not panic", func(t *testing.T) {
		SyncStatusFields(nil)
	})
}

func TestSetClusterStatus(t *testing.T) {
	t.Run("sets status and syncs fields", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		SetClusterStatus(cluster, bkev1beta1.ClusterInitializing)
		if cluster.Status.ClusterStatus != bkev1beta1.ClusterInitializing {
			t.Errorf("ClusterStatus = %q, want %q", cluster.Status.ClusterStatus, bkev1beta1.ClusterInitializing)
		}
		if cluster.Status.Phase != bkev1beta1.InitControlPlane {
			t.Errorf("Phase = %q, want %q", cluster.Status.Phase, bkev1beta1.InitControlPlane)
		}
		if cluster.Status.ClusterHealthState != bkev1beta1.Deploying {
			t.Errorf("ClusterHealthState = %q, want %q", cluster.Status.ClusterHealthState, bkev1beta1.Deploying)
		}
	})

	t.Run("overwrites previous status and resyncs", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		SetClusterStatus(cluster, bkev1beta1.ClusterStatusReady)
		if cluster.Status.Phase != bkev1beta1.ClusterReady {
			t.Fatalf("Phase = %q after ClusterReady, want %q", cluster.Status.Phase, bkev1beta1.ClusterReady)
		}
		SetClusterStatus(cluster, bkev1beta1.ClusterUpgrading)
		if cluster.Status.ClusterStatus != bkev1beta1.ClusterUpgrading {
			t.Errorf("ClusterStatus = %q, want %q", cluster.Status.ClusterStatus, bkev1beta1.ClusterUpgrading)
		}
		if cluster.Status.Phase != bkev1beta1.UpgradeControlPlane {
			t.Errorf("Phase = %q after upgrade, want %q", cluster.Status.Phase, bkev1beta1.UpgradeControlPlane)
		}
		if cluster.Status.ClusterHealthState != bkev1beta1.Upgrading {
			t.Errorf("ClusterHealthState = %q after upgrade, want %q", cluster.Status.ClusterHealthState, bkev1beta1.Upgrading)
		}
	})

	t.Run("nil cluster does not panic", func(t *testing.T) {
		SetClusterStatus(nil, bkev1beta1.ClusterStatusReady)
	})

	t.Run("scale failed status carries status as health state", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		SetClusterStatus(cluster, bkev1beta1.ClusterMasterScalingUpFailed)
		if cluster.Status.ClusterStatus != bkev1beta1.ClusterMasterScalingUpFailed {
			t.Errorf("ClusterStatus = %q, want %q", cluster.Status.ClusterStatus, bkev1beta1.ClusterMasterScalingUpFailed)
		}
		if cluster.Status.Phase != bkev1beta1.MasterScalingUp {
			t.Errorf("Phase = %q, want %q", cluster.Status.Phase, bkev1beta1.MasterScalingUp)
		}
		expected := confv1beta1.ClusterHealthState(bkev1beta1.ClusterMasterScalingUpFailed)
		if cluster.Status.ClusterHealthState != expected {
			t.Errorf("ClusterHealthState = %q, want %q", cluster.Status.ClusterHealthState, expected)
		}
	})
}

func TestValidateStatusConsistency(t *testing.T) {
	t.Run("consistent cluster returns nil", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		SetClusterStatus(cluster, bkev1beta1.ClusterInitializing)
		if err := ValidateStatusConsistency(cluster); err != nil {
			t.Errorf("expected nil, got %v", err)
		}
	})

	t.Run("consistent ClusterReady returns nil", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		SetClusterStatus(cluster, bkev1beta1.ClusterStatusReady)
		if err := ValidateStatusConsistency(cluster); err != nil {
			t.Errorf("expected nil, got %v", err)
		}
	})

	t.Run("empty ClusterStatus returns error", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		err := ValidateStatusConsistency(cluster)
		if err == nil {
			t.Error("expected error for empty ClusterStatus, got nil")
		}
	})

	t.Run("inconsistent Phase returns error", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		cluster.Status.ClusterStatus = bkev1beta1.ClusterInitializing
		cluster.Status.Phase = bkev1beta1.UpgradeControlPlane
		cluster.Status.ClusterHealthState = bkev1beta1.Deploying
		err := ValidateStatusConsistency(cluster)
		if err == nil {
			t.Error("expected error for inconsistent Phase, got nil")
		}
	})

	t.Run("inconsistent ClusterHealthState returns error", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		cluster.Status.ClusterStatus = bkev1beta1.ClusterStatusReady
		cluster.Status.Phase = bkev1beta1.ClusterReady
		cluster.Status.ClusterHealthState = bkev1beta1.Unhealthy
		err := ValidateStatusConsistency(cluster)
		if err == nil {
			t.Error("expected error for inconsistent ClusterHealthState, got nil")
		}
	})

	t.Run("nil cluster returns error", func(t *testing.T) {
		err := ValidateStatusConsistency(nil)
		if err == nil {
			t.Error("expected error for nil cluster, got nil")
		}
	})

	t.Run("scale failed status consistent with fallback health", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{}
		SetClusterStatus(cluster, bkev1beta1.ClusterWorkerScalingDownFailed)
		if err := ValidateStatusConsistency(cluster); err != nil {
			t.Errorf("expected nil for scale-failed consistency, got %v", err)
		}
	})
}

func TestRoundTripMapping(t *testing.T) {
	statuses := []confv1beta1.ClusterStatus{
		bkev1beta1.ClusterInitializing,
		bkev1beta1.ClusterInitializationFailed,
		bkev1beta1.ClusterUpgrading,
		bkev1beta1.ClusterUpgradeFailed,
		bkev1beta1.ClusterMasterScalingUp,
		bkev1beta1.ClusterMasterScalingUpFailed,
		bkev1beta1.ClusterMasterScalingDown,
		bkev1beta1.ClusterMasterScalingDownFailed,
		bkev1beta1.ClusterWorkerScalingUp,
		bkev1beta1.ClusterWorkerScalingUpFailed,
		bkev1beta1.ClusterWorkerScalingDown,
		bkev1beta1.ClusterWorkerScalingDownFailed,
		bkev1beta1.ClusterStatusReady,
	}

	for _, status := range statuses {
		t.Run(string(status), func(t *testing.T) {
			cluster := &bkev1beta1.BKECluster{}
			SetClusterStatus(cluster, status)
			if err := ValidateStatusConsistency(cluster); err != nil {
				t.Errorf("round-trip validation failed for %s: %v", status, err)
			}
		})
	}
}
