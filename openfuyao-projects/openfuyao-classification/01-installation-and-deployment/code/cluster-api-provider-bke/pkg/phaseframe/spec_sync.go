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

package phaseframe

import (
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

// SyncUpgradeTargetsToClusterSpec persists etcd/containerd/kubernetes targets from VersionContext into
// BKECluster.Spec.ClusterConfig before dispatching node upgrade commands.
func (pc *PhaseContext) SyncUpgradeTargetsToClusterSpec() error {
	if pc == nil || pc.BKECluster == nil {
		return nil
	}
	if pc.VersionContext == nil {
		pc.BuildAndSetVersionContext()
	}
	vc := pc.VersionContext
	if vc == nil || pc.BKECluster.Spec.ClusterConfig == nil {
		return nil
	}
	if !upgrade.ClusterSpecHasUpgradeTargets(pc.BKECluster.Spec.ClusterConfig.Cluster, vc) {
		upgrade.ApplyVersionContextTargetsToClusterSpec(pc.BKECluster, vc)
		return nil
	}
	if pc.Client == nil {
		upgrade.ApplyVersionContextTargetsToClusterSpec(pc.BKECluster, vc)
		return nil
	}
	return mergecluster.SyncStatusUntilComplete(pc.Client, pc.BKECluster, func(bc *bkev1beta1.BKECluster) {
		upgrade.ApplyVersionContextTargetsToClusterSpec(bc, vc)
	})
}
