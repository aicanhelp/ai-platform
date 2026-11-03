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
	"fmt"
	"strings"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

var mapperLogger = log.With("name", "mapper")

// SetClusterStatus sets ClusterStatus on the BKECluster and automatically
// synchronises the deprecated Phase and ClusterHealthState fields.
// All new code must use this function instead of directly assigning
// Status.ClusterStatus, Status.Phase or Status.ClusterHealthState.
func SetClusterStatus(cluster *bkev1beta1.BKECluster, status confv1beta1.ClusterStatus) {
	if cluster == nil {
		return
	}
	cluster.Status.ClusterStatus = status
	SyncStatusFields(cluster)
}

// SyncStatusFields synchronises the deprecated Phase and ClusterHealthState
// fields from the single source of truth ClusterStatus.
// Call this after any direct ClusterStatus assignment that did not go
// through SetClusterStatus.
func SyncStatusFields(cluster *bkev1beta1.BKECluster) {
	if cluster == nil {
		return
	}
	newPhase := MapClusterStatusToPhase(cluster.Status.ClusterStatus)
	if newPhase == "" {
		mapperLogger.Warnf("MapClusterStatusToPhase returned empty, status=%s", cluster.Status.ClusterStatus)
	}
	cluster.Status.Phase = newPhase

	newHealth := MapClusterStatusToClusterHealthState(cluster.Status.ClusterStatus)
	if newHealth == "" && cluster.Status.ClusterStatus != "" {
		mapperLogger.Warnf("MapClusterStatusToClusterHealthState returned empty, status=%s", cluster.Status.ClusterStatus)
	}
	cluster.Status.ClusterHealthState = newHealth
}

// ValidateStatusConsistency verifies that Phase, ClusterStatus and
// ClusterHealthState are consistent with each other according to the
// mapping functions. Returns nil when consistent, an error otherwise.
func ValidateStatusConsistency(cluster *bkev1beta1.BKECluster) error {
	if cluster == nil {
		return fmt.Errorf("cluster is nil")
	}
	var errs []string

	if cluster.Status.ClusterStatus == "" {
		errs = append(errs, "ClusterStatus is empty")
	}

	expectedPhase := MapClusterStatusToPhase(cluster.Status.ClusterStatus)
	if cluster.Status.Phase != expectedPhase {
		errs = append(errs, fmt.Sprintf("Phase mismatch: expected %s, got %s",
			expectedPhase, cluster.Status.Phase))
	}

	expectedHealth := MapClusterStatusToClusterHealthState(cluster.Status.ClusterStatus)
	if cluster.Status.ClusterHealthState != expectedHealth {
		errs = append(errs, fmt.Sprintf("ClusterHealthState mismatch: expected %s, got %s",
			expectedHealth, cluster.Status.ClusterHealthState))
	}

	if len(errs) > 0 {
		return fmt.Errorf("status inconsistency: %s", strings.Join(errs, "; "))
	}
	return nil
}

// MapPhaseToClusterStatus maps a BKEClusterPhase (plus an optional error)
// to the canonical ClusterStatus value.
// When err is non-nil the corresponding *Failed status is returned.
func MapPhaseToClusterStatus(phase confv1beta1.BKEClusterPhase, err error) confv1beta1.ClusterStatus {
	switch phase {
	case bkev1beta1.InitControlPlane,
		bkev1beta1.JoinControlPlane,
		bkev1beta1.JoinWorker,
		bkev1beta1.FakeInitControlPlane,
		bkev1beta1.FakeJoinControlPlane,
		bkev1beta1.FakeJoinWorker:
		if err != nil {
			return bkev1beta1.ClusterInitializationFailed
		}
		return bkev1beta1.ClusterInitializing
	case bkev1beta1.UpgradeControlPlane,
		bkev1beta1.UpgradeWorker,
		bkev1beta1.UpgradeEtcd:
		if err != nil {
			return bkev1beta1.ClusterUpgradeFailed
		}
		return bkev1beta1.ClusterUpgrading
	case bkev1beta1.MasterScalingUp:
		if err != nil {
			return bkev1beta1.ClusterMasterScalingUpFailed
		}
		return bkev1beta1.ClusterMasterScalingUp
	case bkev1beta1.MasterScalingDown:
		if err != nil {
			return bkev1beta1.ClusterMasterScalingDownFailed
		}
		return bkev1beta1.ClusterMasterScalingDown
	case bkev1beta1.WorkerScalingUp:
		if err != nil {
			return bkev1beta1.ClusterWorkerScalingUpFailed
		}
		return bkev1beta1.ClusterWorkerScalingUp
	case bkev1beta1.WorkerScalingDown:
		if err != nil {
			return bkev1beta1.ClusterWorkerScalingDownFailed
		}
		return bkev1beta1.ClusterWorkerScalingDown
	case bkev1beta1.FailedBootstrapNode:
		return bkev1beta1.ClusterInitializationFailed
	case bkev1beta1.ClusterReady:
		return bkev1beta1.ClusterStatusReady
	default:
		return bkev1beta1.ClusterUnknown
	}
}

// MapClusterHealthStateToClusterStatus maps a ClusterHealthState to the
// canonical ClusterStatus value. Used by legacy code that previously set
// ClusterHealthState directly (e.g. markBKEClusterHealthyStatus).
func MapClusterHealthStateToClusterStatus(state confv1beta1.ClusterHealthState) confv1beta1.ClusterStatus {
	switch state {
	case bkev1beta1.Deploying:
		return bkev1beta1.ClusterInitializing
	case bkev1beta1.DeployFailed:
		return bkev1beta1.ClusterInitializationFailed
	case bkev1beta1.Upgrading:
		return bkev1beta1.ClusterUpgrading
	case bkev1beta1.UpgradeFailed:
		return bkev1beta1.ClusterUpgradeFailed
	case bkev1beta1.Managing:
		return bkev1beta1.ClusterManaging
	case bkev1beta1.ManageFailed:
		return bkev1beta1.ClusterManageFailed
	case bkev1beta1.Unhealthy:
		return bkev1beta1.ClusterUnhealthy
	case bkev1beta1.Healthy:
		return bkev1beta1.ClusterStatusReady
	case bkev1beta1.Deleting:
		return bkev1beta1.ClusterDeleting
	default:
		return bkev1beta1.ClusterUnknown
	}
}

// MapClusterStatusToClusterHealthState maps ClusterStatus back to the
// deprecated ClusterHealthState field. Explicitly mapped values preserve
// the original 9-value enum semantics for backward compatibility.
// Unmapped values fall back to the ClusterStatus string itself, so the
// deprecated field always carries a meaningful value during the transition.
func MapClusterStatusToClusterHealthState(status confv1beta1.ClusterStatus) confv1beta1.ClusterHealthState {
	switch status {
	case bkev1beta1.ClusterInitializing:
		return bkev1beta1.Deploying
	case bkev1beta1.ClusterInitializationFailed:
		return bkev1beta1.DeployFailed
	case bkev1beta1.ClusterDeployingAddon:
		return bkev1beta1.Deploying
	case bkev1beta1.ClusterDeployAddonFailed:
		return bkev1beta1.DeployFailed
	case bkev1beta1.ClusterUpgrading:
		return bkev1beta1.Upgrading
	case bkev1beta1.ClusterUpgradeFailed:
		return bkev1beta1.UpgradeFailed
	case bkev1beta1.ClusterManaging:
		return bkev1beta1.Managing
	case bkev1beta1.ClusterManageFailed:
		return bkev1beta1.ManageFailed
	case bkev1beta1.ClusterUnhealthy:
		return bkev1beta1.Unhealthy
	case bkev1beta1.ClusterStatusReady:
		return bkev1beta1.Healthy
	case bkev1beta1.ClusterDeleting:
		return bkev1beta1.Deleting
	default:
		// Transition fallback: carry ClusterStatus string value for unmapped statuses.
		// Callers should migrate away from ClusterHealthState.
		mapperLogger.Warnf("ClusterStatus %q has no explicit ClusterHealthState mapping, falling back to string value", status)
		return confv1beta1.ClusterHealthState(status)
	}
}

// MapClusterStatusToPhase maps ClusterStatus back to the deprecated Phase
// field. Explicitly mapped values preserve the original Phase enum semantics.
// Unmapped values fall back to the ClusterStatus string itself, so the
// deprecated field always carries a meaningful value during the transition.
func MapClusterStatusToPhase(status confv1beta1.ClusterStatus) confv1beta1.BKEClusterPhase {
	switch status {
	case bkev1beta1.ClusterInitializing,
		bkev1beta1.ClusterInitializationFailed:
		return bkev1beta1.InitControlPlane
	case bkev1beta1.ClusterUpgrading,
		bkev1beta1.ClusterUpgradeFailed:
		return bkev1beta1.UpgradeControlPlane
	case bkev1beta1.ClusterMasterScalingUp,
		bkev1beta1.ClusterMasterScalingUpFailed:
		return bkev1beta1.MasterScalingUp
	case bkev1beta1.ClusterMasterScalingDown,
		bkev1beta1.ClusterMasterScalingDownFailed:
		return bkev1beta1.MasterScalingDown
	case bkev1beta1.ClusterWorkerScalingUp,
		bkev1beta1.ClusterWorkerScalingUpFailed:
		return bkev1beta1.WorkerScalingUp
	case bkev1beta1.ClusterWorkerScalingDown,
		bkev1beta1.ClusterWorkerScalingDownFailed:
		return bkev1beta1.WorkerScalingDown
	case bkev1beta1.ClusterStatusReady:
		return bkev1beta1.ClusterReady
	default:
		// Transition fallback: carry ClusterStatus string value for unmapped statuses.
		// Callers should migrate away from Phase.
		mapperLogger.Warnf("ClusterStatus %q has no explicit Phase mapping, falling back to string value", status)
		return confv1beta1.BKEClusterPhase(status)
	}
}
