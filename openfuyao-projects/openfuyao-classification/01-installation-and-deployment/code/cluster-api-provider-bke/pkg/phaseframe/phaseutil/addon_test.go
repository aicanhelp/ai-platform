/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
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
	"testing"

	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func TestAllowDeployAddon(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{}
	cluster := &clusterv1.Cluster{
		Status: clusterv1.ClusterStatus{
			Conditions: []clusterv1.Condition{
				{Type: clusterv1.ControlPlaneInitializedCondition, Status: "False"},
			},
		},
	}
	result := AllowDeployAddon(bkeCluster, cluster)
	if result {
		t.Error("expected false when control plane not initialized")
	}
}

func TestAllowDeployAddonWithBKENodes(t *testing.T) {
	nodes := bkev1beta1.BKENodes{}
	cluster := &clusterv1.Cluster{
		Status: clusterv1.ClusterStatus{
			Conditions: []clusterv1.Condition{
				{Type: clusterv1.ControlPlaneInitializedCondition, Status: "False"},
			},
		},
	}
	result := AllowDeployAddonWithBKENodes(nodes, cluster)
	if result {
		t.Error("expected false")
	}
}
