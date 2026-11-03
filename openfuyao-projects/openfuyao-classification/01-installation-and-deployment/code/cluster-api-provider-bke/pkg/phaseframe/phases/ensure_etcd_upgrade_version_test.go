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
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

func etcdTestCluster(specVer, statusVer string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test-etcd", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{EtcdVersion: specVer},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			EtcdVersion:        statusVer,
			ClusterHealthState: bkev1beta1.Healthy,
		},
	}
}

func newEtcdUpgradePhase(t *testing.T, cluster *bkev1beta1.BKECluster, vc *upgrade.VersionContext) *EnsureEtcdUpgrade {
	t.Helper()
	pc := phaseframe.NewReconcilePhaseCtx(t.Context())
	pc.SetBKECluster(cluster)
	if vc != nil {
		pc.SetVersionContext(vc)
	}
	return &EnsureEtcdUpgrade{BasePhase: phaseframe.NewBasePhase(pc, EnsureEtcdUpgradeName)}
}

func TestEnsureEtcdUpgrade_Version(t *testing.T) {
	t.Run("returns status etcd version", func(t *testing.T) {
		cluster := etcdTestCluster("3.5.12", "3.5.10")
		phase := newEtcdUpgradePhase(t, cluster, nil)
		assert.Equal(t, "3.5.10", phase.Version())
	})

	t.Run("empty when context missing", func(t *testing.T) {
		phase := &EnsureEtcdUpgrade{}
		assert.Empty(t, phase.Version())
	})
}

func TestEnsureEtcdUpgrade_resolveEtcdUpgradeVersion(t *testing.T) {
	t.Run("prefers version context target over spec", func(t *testing.T) {
		cluster := etcdTestCluster("3.5.12", "3.5.10")
		vc := upgrade.NewVersionContext()
		vc.SetTarget(upgrade.ComponentEtcd, "v3.5.21-of.1")
		phase := newEtcdUpgradePhase(t, cluster, vc)
		assert.Equal(t, "v3.5.21-of.1", phase.resolveEtcdUpgradeVersion())
	})

	t.Run("falls back to spec when version context missing", func(t *testing.T) {
		cluster := etcdTestCluster("v3.5.21-of.1", "3.5.10")
		phase := newEtcdUpgradePhase(t, cluster, nil)
		assert.Equal(t, "v3.5.21-of.1", phase.resolveEtcdUpgradeVersion())
	})
}

func TestEnsureEtcdUpgrade_isEtcdNeedUpgrade(t *testing.T) {
	tests := []struct {
		name    string
		cluster *bkev1beta1.BKECluster
		want    bool
	}{
		{
			name:    "nil cluster",
			cluster: nil,
			want:    false,
		},
		{
			name: "nil cluster config",
			cluster: &bkev1beta1.BKECluster{
				Status: confv1beta1.BKEClusterStatus{EtcdVersion: "3.5.10"},
			},
			want: false,
		},
		{
			name:    "empty spec version",
			cluster: etcdTestCluster("", "3.5.10"),
			want:    false,
		},
		{
			name:    "empty status version",
			cluster: etcdTestCluster("3.5.12", ""),
			want:    false,
		},
		{
			name:    "spec differs from status",
			cluster: etcdTestCluster("3.5.12", "3.5.10"),
			want:    true,
		},
		{
			name:    "spec matches status",
			cluster: etcdTestCluster("3.5.12", "3.5.12"),
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			phase := &EnsureEtcdUpgrade{}
			if tt.cluster != nil {
				phase = newEtcdUpgradePhase(t, tt.cluster, nil)
			}
			assert.Equal(t, tt.want, phase.isEtcdNeedUpgrade(nil, tt.cluster))
		})
	}
}

func TestEnsureEtcdUpgrade_NeedExecute(t *testing.T) {
	cluster := etcdTestCluster("3.5.12", "3.5.10")

	t.Run("version context needs upgrade", func(t *testing.T) {
		vc := upgrade.NewVersionContext()
		vc.SetCurrent(upgrade.ComponentEtcd, "3.5.10")
		vc.SetTarget(upgrade.ComponentEtcd, "3.5.12")

		phase := newEtcdUpgradePhase(t, cluster, vc)
		require.True(t, phase.NeedExecute(nil, cluster))
	})

	t.Run("version context already at target", func(t *testing.T) {
		vc := upgrade.NewVersionContext()
		vc.SetCurrent(upgrade.ComponentEtcd, "3.5.12")
		vc.SetTarget(upgrade.ComponentEtcd, "3.5.12")

		phase := newEtcdUpgradePhase(t, cluster, vc)
		assert.False(t, phase.NeedExecute(nil, cluster))
	})

	t.Run("legacy path when version context undecided", func(t *testing.T) {
		phase := newEtcdUpgradePhase(t, cluster, nil)
		require.True(t, phase.NeedExecute(nil, cluster))
	})

	t.Run("legacy path no upgrade when versions match", func(t *testing.T) {
		matched := etcdTestCluster("3.5.12", "3.5.12")
		phase := newEtcdUpgradePhase(t, matched, nil)
		assert.False(t, phase.NeedExecute(nil, matched))
	})

	t.Run("default need execute false", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyMethod(&phaseframe.BasePhase{}, "DefaultNeedExecute",
			func(_ *phaseframe.BasePhase, _, _ *bkev1beta1.BKECluster) bool {
				return false
			})

		vc := upgrade.NewVersionContext()
		vc.SetCurrent(upgrade.ComponentEtcd, "3.5.10")
		vc.SetTarget(upgrade.ComponentEtcd, "3.5.12")
		phase := newEtcdUpgradePhase(t, cluster, vc)
		assert.False(t, phase.NeedExecute(nil, cluster))
	})
}
