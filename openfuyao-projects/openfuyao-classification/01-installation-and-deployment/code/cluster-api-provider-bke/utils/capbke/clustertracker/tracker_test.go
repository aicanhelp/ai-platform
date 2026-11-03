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

package clustertracker

import (
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestAllowTrackerRemoteCluster_NilCluster(t *testing.T) {
	result := AllowTrackerRemoteCluster(nil)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_PausedCluster(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			Pause: true,
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_DeletingCluster(t *testing.T) {
	now := metav1.Now()
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			DeletionTimestamp: &now,
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_ResetCluster(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			Reset: true,
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_SwitchingAgent(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			Conditions: []confv1beta1.ClusterCondition{
				{
					Type:   bkev1beta1.SwitchBKEAgentCondition,
					Status: confv1beta1.ConditionTrue,
				},
			},
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_MasterScalingUp(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus: bkev1beta1.ClusterMasterScalingUp,
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_WorkerScalingDown(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus: bkev1beta1.ClusterWorkerScalingDown,
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_Initializing(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus: bkev1beta1.ClusterInitializing,
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_Upgrading(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus: bkev1beta1.ClusterUpgrading,
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.False(t, result)
}

func TestAllowTrackerRemoteCluster_Success(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			Conditions: []confv1beta1.ClusterCondition{
				{
					Type:   bkev1beta1.TargetClusterReadyCondition,
					Status: confv1beta1.ConditionTrue,
				},
			},
		},
	}
	result := AllowTrackerRemoteCluster(cluster)
	assert.True(t, result)
}

func TestIsClusterInInvalidState(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			Pause: true,
		},
	}
	result := isClusterInInvalidState(cluster)
	assert.True(t, result)
}

func TestIsClusterInOperationState(t *testing.T) {
	cluster := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			ClusterStatus: bkev1beta1.ClusterPaused,
		},
	}
	result := isClusterInOperationState(cluster)
	assert.True(t, result)
}
