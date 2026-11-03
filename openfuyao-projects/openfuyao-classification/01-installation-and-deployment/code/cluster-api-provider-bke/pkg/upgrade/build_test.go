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

func TestBuildVersionContextFromBKECluster(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					EtcdVersion:       "3.5.12",
					KubernetesVersion: "v1.29.0",
					OpenFuyaoVersion:  "v2.6.0",
					ContainerdVersion: "1.7.0",
				},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			EtcdVersion:       "3.5.10",
			KubernetesVersion: "v1.28.0",
			OpenFuyaoVersion:  "v2.5.0",
			ContainerdVersion: "1.6.0",
		},
	}

	vc := BuildVersionContextFromBKECluster(bc)
	if vc.GetCurrent(ComponentEtcd) != "3.5.10" || vc.GetTarget(ComponentEtcd) != "3.5.12" {
		t.Fatalf("unexpected etcd versions: current=%q target=%q", vc.GetCurrent(ComponentEtcd), vc.GetTarget(ComponentEtcd))
	}
	if !vc.NeedsUpgrade(ComponentKubernetesMaster) || !vc.NeedsUpgrade(ComponentKubernetesWorker) {
		t.Fatal("expected kubernetes master/worker upgrade")
	}
	if vc.NeedsUpgrade(ComponentOpenFuyao) == false {
		t.Fatal("expected openfuyao upgrade")
	}
}

func TestBuildVersionContextFromBKECluster_Nil(t *testing.T) {
	vc := BuildVersionContextFromBKECluster(nil)
	if vc == nil || len(vc.Current) != 0 {
		t.Fatal("expected empty context for nil cluster")
	}
}
