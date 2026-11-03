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

package phases

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func masterUpgradeTestCluster(specK8s, statusK8s string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test-master", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: specK8s},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			KubernetesVersion:  statusK8s,
			ClusterHealthState: bkev1beta1.Healthy,
		},
	}
}

func newMasterUpgradePhase(t *testing.T, cluster *bkev1beta1.BKECluster, vc *upgrade.VersionContext) *EnsureMasterUpgrade {
	t.Helper()
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(cluster).Build()
	pc := &phaseframe.PhaseContext{
		Context:        context.Background(),
		BKECluster:     cluster,
		Client:         c,
		Scheme:         scheme,
		VersionContext: vc,
		Log:            bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, cluster),
	}
	return NewEnsureMasterUpgrade(pc).(*EnsureMasterUpgrade)
}

func TestEnsureMasterUpgrade_isKubernetesMasterNeedUpgrade(t *testing.T) {
	tests := []struct {
		name    string
		cluster *bkev1beta1.BKECluster
		want    bool
	}{
		{name: "nil cluster", cluster: nil, want: false},
		{
			name: "nil cluster config",
			cluster: &bkev1beta1.BKECluster{
				Status: confv1beta1.BKEClusterStatus{KubernetesVersion: "v1.28.0"},
			},
			want: false,
		},
		{name: "empty target", cluster: masterUpgradeTestCluster("", "v1.28.0"), want: false},
		{name: "empty current", cluster: masterUpgradeTestCluster("v1.29.0", ""), want: false},
		{name: "spec differs from status", cluster: masterUpgradeTestCluster("v1.29.0", "v1.28.0"), want: true},
		{name: "spec matches status", cluster: masterUpgradeTestCluster("v1.28.0", "v1.28.0"), want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			phase := &EnsureMasterUpgrade{}
			if tt.cluster != nil {
				phase = newMasterUpgradePhase(t, tt.cluster, nil)
			}
			assert.Equal(t, tt.want, phase.isKubernetesMasterNeedUpgrade(nil, tt.cluster))
		})
	}
}

func TestEnsureMasterUpgrade_NeedExecute_VersionContext(t *testing.T) {
	cluster := masterUpgradeTestCluster("v1.29.0", "v1.28.0")

	t.Run("version context needs upgrade", func(t *testing.T) {
		vc := upgrade.NewVersionContext()
		vc.SetCurrent(upgrade.ComponentKubernetesMaster, "v1.28.0")
		vc.SetTarget(upgrade.ComponentKubernetesMaster, "v1.29.0")

		phase := newMasterUpgradePhase(t, cluster, vc)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(fetchBKENodesIfCPInitialized, func(_ *phaseframe.PhaseContext, _ *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, bool) {
			return bkev1beta1.BKENodes{}, true
		})
		patches.ApplyFunc(phaseutil.GetNeedUpgradeMasterNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1"}}
		})
		require.True(t, phase.NeedExecute(nil, cluster))
	})

	t.Run("version context already at target", func(t *testing.T) {
		vc := upgrade.NewVersionContext()
		vc.SetCurrent(upgrade.ComponentKubernetesMaster, "v1.28.0")
		vc.SetTarget(upgrade.ComponentKubernetesMaster, "v1.28.0")

		phase := newMasterUpgradePhase(t, cluster, vc)
		assert.False(t, phase.NeedExecute(nil, cluster))
	})

	t.Run("legacy path when version context undecided", func(t *testing.T) {
		phase := newMasterUpgradePhase(t, cluster, nil)
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(fetchBKENodesIfCPInitialized, func(_ *phaseframe.PhaseContext, _ *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, bool) {
			return bkev1beta1.BKENodes{}, true
		})
		patches.ApplyFunc(phaseutil.GetNeedUpgradeMasterNodesWithBKENodes, func(_ *bkev1beta1.BKECluster, _ bkev1beta1.BKENodes) bkenode.Nodes {
			return bkenode.Nodes{{IP: "10.0.0.1"}}
		})
		require.True(t, phase.NeedExecute(nil, cluster))
	})

	t.Run("legacy path no upgrade when versions match", func(t *testing.T) {
		matched := masterUpgradeTestCluster("v1.28.0", "v1.28.0")
		phase := newMasterUpgradePhase(t, matched, nil)
		assert.False(t, phase.NeedExecute(nil, matched))
	})
}
