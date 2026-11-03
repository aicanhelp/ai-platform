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
package phases

import (
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
)

func fetchBKENodesIfCPInitialized(ctx *phaseframe.PhaseContext, bkeCluster *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, bool) {
	if err := ctx.RefreshCtxCluster(); err == nil {
		if !conditions.IsTrue(ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
			return nil, false
		}
	}
	bkeNodes, err := ctx.NodeFetcher().GetBKENodesWrapperForCluster(ctx, bkeCluster)
	if err != nil {
		return nil, false
	}
	return bkeNodes, true
}

func getDeleteTargetNodesIfDeployed(ctx *phaseframe.PhaseContext, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, bool) {
	bkeNodes, _ := ctx.NodeFetcher().GetBKENodesWrapperForCluster(ctx.Context, bkeCluster)
	if !phaseutil.ClusterEndDeployedWithContext(ctx.Context, ctx.Client, ctx.Cluster, bkeCluster, bkeNodes) {
		return nil, false
	}
	targetNodes, err := GetTargetClusterNodes(ctx.Context, ctx.Client, bkeCluster)
	if err != nil {
		ctx.Log.Debug("scale-in check: Failed to get target cluster nodes: %v", err)
		return nil, false
	}
	return targetNodes, true
}
