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
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func AllowDeployAddon(bkeCluster *bkev1beta1.BKECluster, cluster *clusterv1.Cluster) bool {
	// 如果master节点未被初始化，不允许部署addon
	if !conditions.IsTrue(cluster, clusterv1.ControlPlaneInitializedCondition) {
		return false
	}

	return IsNodeBootFlagSet(bkeCluster)
}

// AllowDeployAddonWithBKENodes checks if addon deployment is allowed using pre-fetched BKENodes.
// Use this function in controller context where BKENodes are fetched via NodeFetcher.
func AllowDeployAddonWithBKENodes(bkenodes bkev1beta1.BKENodes, cluster *clusterv1.Cluster) bool {
	// 如果master节点未被初始化，不允许部署addon
	if !conditions.IsTrue(cluster, clusterv1.ControlPlaneInitializedCondition) {
		return false
	}

	return IsNodeBootFlagSetWithBKENodes(bkenodes)
}
