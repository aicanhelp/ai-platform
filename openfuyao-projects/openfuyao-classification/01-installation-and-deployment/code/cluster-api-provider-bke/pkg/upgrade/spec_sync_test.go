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
)

func TestApplyVersionContextTargetsToClusterSpec(t *testing.T) {
	vc := NewVersionContext()
	vc.SetTarget(ComponentEtcd, "v3.6.7-of.8")
	vc.SetTarget(ComponentContainerd, "1.7.0")

	bc := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					EtcdVersion:       "v3.6.7-of.1",
					ContainerdVersion: "1.6.0",
				},
			},
		},
	}

	ApplyVersionContextTargetsToClusterSpec(bc, vc)
	if bc.Spec.ClusterConfig.Cluster.EtcdVersion != "v3.6.7-of.8" {
		t.Fatalf("etcd version = %q, want v3.6.7-of.8", bc.Spec.ClusterConfig.Cluster.EtcdVersion)
	}
	if bc.Spec.ClusterConfig.Cluster.ContainerdVersion != "1.7.0" {
		t.Fatalf("containerd version = %q, want 1.7.0", bc.Spec.ClusterConfig.Cluster.ContainerdVersion)
	}
}

func TestApplyVersionContextTargetsToClusterSpec_Kubernetes(t *testing.T) {
	vc := NewVersionContext()
	vc.SetTarget(releaseComponentKubernetes, "v1.33.1-of.2")

	bc := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{KubernetesVersion: "v1.33.1-of.1"},
			},
		},
	}

	ApplyVersionContextTargetsToClusterSpec(bc, vc)
	if bc.Spec.ClusterConfig.Cluster.KubernetesVersion != "v1.33.1-of.2" {
		t.Fatalf("kubernetes version = %q, want v1.33.1-of.2", bc.Spec.ClusterConfig.Cluster.KubernetesVersion)
	}
}

func TestKubernetesTargetFromVersionContext_PrefersMaster(t *testing.T) {
	vc := NewVersionContext()
	vc.SetTarget(releaseComponentKubernetes, "v1.28.0")
	vc.SetTarget(ComponentKubernetesMaster, "v1.33.1-of.2")
	if got := KubernetesTargetFromVersionContext(vc); got != "v1.33.1-of.2" {
		t.Fatalf("kubernetes target = %q, want v1.33.1-of.2", got)
	}
}

func TestClusterSpecHasUpgradeTargets_Kubernetes(t *testing.T) {
	vc := NewVersionContext()
	vc.SetTarget(ComponentKubernetesMaster, "v1.33.1-of.2")

	cluster := confv1beta1.Cluster{KubernetesVersion: "v1.33.1-of.1"}
	if !ClusterSpecHasUpgradeTargets(cluster, vc) {
		t.Fatal("expected mismatch for kubernetes target")
	}

	cluster.KubernetesVersion = "v1.33.1-of.2"
	if ClusterSpecHasUpgradeTargets(cluster, vc) {
		t.Fatal("expected no mismatch when spec matches kubernetes target")
	}
}

func TestClusterSpecHasUpgradeTargets(t *testing.T) {
	vc := NewVersionContext()
	vc.SetTarget(ComponentEtcd, "v3.6.7-of.8")

	cluster := confv1beta1.Cluster{EtcdVersion: "v3.6.7-of.1"}
	if !ClusterSpecHasUpgradeTargets(cluster, vc) {
		t.Fatal("expected mismatch for etcd target")
	}

	cluster.EtcdVersion = "v3.6.7-of.8"
	if ClusterSpecHasUpgradeTargets(cluster, vc) {
		t.Fatal("expected no mismatch when spec matches target")
	}
}
