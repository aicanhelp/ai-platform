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
	"strings"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

// ApplyUpgradeHopToClusterSpec sets the target openFuyao release on BKECluster spec
// before declarative DAG execution so downstream phases read the hop target version.
func ApplyUpgradeHopToClusterSpec(bc *bkev1beta1.BKECluster, hopOpenFuyaoVersion string) {
	if bc == nil || bc.Spec.ClusterConfig == nil {
		return
	}
	if v := strings.TrimSpace(hopOpenFuyaoVersion); v != "" {
		bc.Spec.ClusterConfig.Cluster.OpenFuyaoVersion = v
	}
}

// ApplyUpgradeHopToClusterStatus records installed component versions on BKECluster
// status after a declarative upgrade hop completes successfully.
func ApplyUpgradeHopToClusterStatus(bc *bkev1beta1.BKECluster, hopOpenFuyaoVersion string) {
	if bc == nil || bc.Spec.ClusterConfig == nil {
		return
	}
	spec := bc.Spec.ClusterConfig.Cluster

	v := strings.TrimSpace(hopOpenFuyaoVersion)
	if v != "" {
		bc.Status.OpenFuyaoVersion = v
	}
	if v == "" && spec.OpenFuyaoVersion != "" {
		bc.Status.OpenFuyaoVersion = spec.OpenFuyaoVersion
	}
	if spec.KubernetesVersion != "" {
		bc.Status.KubernetesVersion = spec.KubernetesVersion
	}
	if spec.EtcdVersion != "" {
		bc.Status.EtcdVersion = spec.EtcdVersion
	}
	if spec.ContainerdVersion != "" {
		bc.Status.ContainerdVersion = spec.ContainerdVersion
	}
}
