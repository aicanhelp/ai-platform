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

// Package clustertracker provides functionality for tracking and monitoring
// BKE cluster states and determining when remote cluster tracking is allowed.
// It includes utilities for checking cluster conditions and operational states
// to ensure proper cluster management and health monitoring.
package clustertracker

import (
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
)

// AllowTrackerRemoteCluster determines whether remote cluster tracking is allowed.
// Tracking is not allowed when the cluster is paused, being deleted, reset,
// agent switching is in progress, or during node addition.
func AllowTrackerRemoteCluster(bkeCluster *bkev1beta1.BKECluster) bool {
	if bkeCluster == nil {
		return false
	}

	// Check cluster basic state
	if isClusterInInvalidState(bkeCluster) {
		return false
	}

	// Check cluster operation state
	if isClusterInOperationState(bkeCluster) {
		return false
	}

	return condition.HasConditionStatus(bkev1beta1.TargetClusterReadyCondition, bkeCluster, confv1beta1.ConditionTrue)
}

// isClusterInInvalidState checks if the cluster is in an invalid state (paused, being deleted, resetting).
func isClusterInInvalidState(bkeCluster *bkev1beta1.BKECluster) bool {
	return bkeCluster.Spec.Pause ||
		!bkeCluster.DeletionTimestamp.IsZero() ||
		bkeCluster.Spec.Reset ||
		condition.HasConditionStatus(bkev1beta1.SwitchBKEAgentCondition, bkeCluster, confv1beta1.ConditionTrue)
}

// isClusterInOperationState checks if the cluster is in an operational state (scaling, initializing, paused, upgrading).
func isClusterInOperationState(bkeCluster *bkev1beta1.BKECluster) bool {
	// Define set of cluster states where tracking is not allowed
	invalidStates := map[confv1beta1.ClusterStatus]bool{
		bkev1beta1.ClusterMasterScalingUp:   true,
		bkev1beta1.ClusterMasterScalingDown: true,
		bkev1beta1.ClusterWorkerScalingUp:   true,
		bkev1beta1.ClusterWorkerScalingDown: true,
		bkev1beta1.ClusterInitializing:      true,
		bkev1beta1.ClusterPaused:            true,
		bkev1beta1.ClusterUpgrading:         true,
	}

	return invalidStates[bkeCluster.Status.ClusterStatus]
}
