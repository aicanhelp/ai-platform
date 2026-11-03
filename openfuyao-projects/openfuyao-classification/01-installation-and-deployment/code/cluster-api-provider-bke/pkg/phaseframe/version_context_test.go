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
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func TestNeedExecuteWithVersionContext_UsesContext(t *testing.T) {
	pc := NewReconcilePhaseCtx(t.Context())
	pc.SetBKECluster(&bkev1beta1.BKECluster{})
	vc := upgrade.NewVersionContext()
	vc.SetCurrent(upgrade.ComponentEtcd, "3.5.10")
	vc.SetTarget(upgrade.ComponentEtcd, "3.5.12")
	pc.SetVersionContext(vc)

	bp := NewBasePhase(pc, "EnsureEtcdUpgrade")
	legacyCalled := false
	got := bp.NeedExecuteWithVersionContext(upgrade.ComponentEtcd, nil, pc.BKECluster, func(_, _ *bkev1beta1.BKECluster) bool {
		legacyCalled = true
		return true
	})
	if !got || legacyCalled {
		t.Fatalf("expected context path need=true legacyCalled=false, got=%v legacy=%v", got, legacyCalled)
	}
}

func TestNeedExecuteWithVersionContext_FallsBackToLegacy(t *testing.T) {
	pc := NewReconcilePhaseCtx(t.Context())
	cluster := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{EtcdVersion: "3.5.12"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{EtcdVersion: "3.5.10"},
	}
	pc.SetBKECluster(cluster)

	bp := NewBasePhase(pc, "EnsureEtcdUpgrade")
	got := bp.NeedExecuteWithVersionContext(upgrade.ComponentEtcd, nil, cluster, func(_, n *bkev1beta1.BKECluster) bool {
		return n.Spec.ClusterConfig.Cluster.EtcdVersion != n.Status.EtcdVersion
	})
	if !got {
		t.Fatal("expected legacy path to require upgrade")
	}
}

func TestBuildAndSetVersionContext(t *testing.T) {
	pc := NewReconcilePhaseCtx(t.Context())
	pc.SetBKECluster(&bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{EtcdVersion: "3.5.12"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{EtcdVersion: "3.5.10"},
	})
	pc.BuildAndSetVersionContext()
	if pc.VersionContext == nil || !pc.VersionContext.NeedsUpgrade(upgrade.ComponentEtcd) {
		t.Fatal("expected version context built from cluster")
	}
}
