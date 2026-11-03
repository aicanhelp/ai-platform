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

func TestApplyUpgradeHopToClusterStatus(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					OpenFuyaoVersion:  "v26.07",
					KubernetesVersion: "v1.33.1-of.2",
					EtcdVersion:       "v3.6.7-of.8",
					ContainerdVersion: "v2.1.2",
				},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			OpenFuyaoVersion:  "v26.06",
			KubernetesVersion: "v1.33.1-of.1",
			EtcdVersion:       "v3.6.7-of.1",
			ContainerdVersion: "v2.1.1",
		},
	}

	ApplyUpgradeHopToClusterStatus(bc, "v26.07")

	if bc.Status.OpenFuyaoVersion != "v26.07" {
		t.Fatalf("OpenFuyaoVersion: got %q want v26.07", bc.Status.OpenFuyaoVersion)
	}
	if bc.Status.KubernetesVersion != "v1.33.1-of.2" {
		t.Fatalf("KubernetesVersion: got %q", bc.Status.KubernetesVersion)
	}
	if bc.Status.EtcdVersion != "v3.6.7-of.8" {
		t.Fatalf("EtcdVersion: got %q", bc.Status.EtcdVersion)
	}
	if bc.Status.ContainerdVersion != "v2.1.2" {
		t.Fatalf("ContainerdVersion: got %q", bc.Status.ContainerdVersion)
	}
}

func TestApplyUpgradeHopToClusterSpec(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v26.06"},
			},
		},
	}

	ApplyUpgradeHopToClusterSpec(bc, "v26.07")

	if bc.Spec.ClusterConfig.Cluster.OpenFuyaoVersion != "v26.07" {
		t.Fatalf("spec OpenFuyaoVersion: got %q want v26.07", bc.Spec.ClusterConfig.Cluster.OpenFuyaoVersion)
	}
}

func TestApplyUpgradeHopToClusterStatusFallsBackToSpecOpenFuyao(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{OpenFuyaoVersion: "v26.07"},
			},
		},
		Status: confv1beta1.BKEClusterStatus{OpenFuyaoVersion: "v26.05"},
	}

	ApplyUpgradeHopToClusterStatus(bc, "")

	if bc.Status.OpenFuyaoVersion != "v26.07" {
		t.Fatalf("OpenFuyaoVersion: got %q want v26.07", bc.Status.OpenFuyaoVersion)
	}
}
