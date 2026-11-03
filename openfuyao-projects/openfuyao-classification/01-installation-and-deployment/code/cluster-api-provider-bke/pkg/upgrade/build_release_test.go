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

package upgrade

import (
	"testing"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	apiv1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
)

func testReleaseBundle() *releasemanifest.Bundle {
	return &releasemanifest.Bundle{
		Release: apiv1.ReleaseImage{
			Spec: apiv1.ReleaseImageSpec{
				Version: "v2.6.0",
				Install: &apiv1.ReleaseImageInstallSpec{
					Components: []apiv1.ReleaseImageInstallComponent{
						{Name: ComponentKubernetesMaster, Version: "v1.29.1-of.1"},
					},
				},
				Upgrade: &apiv1.ReleaseImageUpgradeSpec{
					Components: []apiv1.ReleaseImageUpgradeComponent{
						{Name: ComponentEtcd, Version: "v3.5.21-of.1"},
						{Name: ComponentProvider, Version: "v1.0.0"},
					},
				},
			},
		},
	}
}

func TestFillTargetFromBundle(t *testing.T) {
	vc := NewVersionContext()
	FillTargetFromBundle(vc, testReleaseBundle())

	if vc.GetTarget(ComponentEtcd) != "v3.5.21-of.1" {
		t.Fatalf("etcd target: %q", vc.GetTarget(ComponentEtcd))
	}
	if vc.GetTarget(ComponentProvider) != "v1.0.0" {
		t.Fatalf("provider target: %q", vc.GetTarget(ComponentProvider))
	}
	if vc.GetTarget(ComponentKubernetesMaster) != "v1.29.1-of.1" {
		t.Fatalf("kubernetes-master target: %q", vc.GetTarget(ComponentKubernetesMaster))
	}
}

func TestBuildVersionContextForUpgrade_FromBundleAndCluster(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{EtcdVersion: "v3.5.21-of.1"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{EtcdVersion: "v3.5.12-of.1"},
	}

	vc := BuildVersionContextForUpgrade(testReleaseBundle(), nil, bc)
	if !vc.NeedsUpgrade(ComponentEtcd) {
		t.Fatal("expected etcd upgrade from cluster current vs bundle target")
	}
	if !vc.NeedsUpgrade(ComponentProvider) {
		t.Fatal("expected manifest component upgrade when current is empty")
	}
}

func TestBuildVersionContextForUpgrade_ProviderFromClusterComponentStatus(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			ClusterComponentStatuses: map[string]confv1beta1.ComponentLifecycleStatus{
				ComponentProvider: {Name: ComponentProvider, CurrentVersion: "v1.0.0"},
			},
		},
	}
	vc := BuildVersionContextForUpgrade(testReleaseBundle(), nil, bc)
	if vc.NeedsUpgrade(ComponentProvider) {
		t.Fatal("expected provider skip when clusterComponentStatuses matches target")
	}
}

func TestBuildVersionContextForUpgrade_BKEAgentFromAddonStatus(t *testing.T) {
	bundle := testReleaseBundle()
	bundle.Release.Spec.Upgrade.Components = append(
		bundle.Release.Spec.Upgrade.Components,
		apiv1.ReleaseImageUpgradeComponent{Name: ComponentBKEAgent, Version: "v2.0.0"},
	)
	bc := &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			AddonStatus: []confv1beta1.Product{{Name: ComponentBKEAgent, Version: "v2.0.0"}},
		},
	}
	vc := BuildVersionContextForUpgrade(bundle, nil, bc)
	if vc.NeedsUpgrade(ComponentBKEAgent) {
		t.Fatal("expected bkeagent skip when addonStatus matches target")
	}
}

func TestBuildVersionContextForUpgrade_CurrentBundle(t *testing.T) {
	target := testReleaseBundle()
	current := testReleaseBundle()

	vc := BuildVersionContextForUpgrade(target, current, nil)
	if vc.NeedsUpgrade(ComponentEtcd) {
		t.Fatal("expected no etcd upgrade when current bundle matches target")
	}
}

func TestAnyTargetNeedsUpgrade(t *testing.T) {
	vc := NewVersionContext()
	vc.SetTarget(ComponentProvider, "v1.0.0")
	if !vc.AnyTargetNeedsUpgrade() {
		t.Fatal("empty current should need upgrade")
	}
	vc.SetCurrent(ComponentProvider, "v1.0.0")
	if vc.AnyTargetNeedsUpgrade() {
		t.Fatal("matched versions should not need upgrade")
	}
}
