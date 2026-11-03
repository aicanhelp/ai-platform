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

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

const releaseComponentKubernetes = "kubernetes"

// ApplyVersionContextTargetsToClusterSpec copies declarative upgrade targets into
// BKECluster.Spec.ClusterConfig.Cluster so agent commands and BKEConfig reads stay aligned.
func ApplyVersionContextTargetsToClusterSpec(bc *bkev1beta1.BKECluster, vc *VersionContext) {
	if bc == nil || vc == nil || bc.Spec.ClusterConfig == nil {
		return
	}
	cluster := &bc.Spec.ClusterConfig.Cluster
	if v := vc.GetTarget(ComponentEtcd); v != "" {
		cluster.EtcdVersion = v
	}
	if v := vc.GetTarget(ComponentContainerd); v != "" {
		cluster.ContainerdVersion = v
	}
	if v := vc.GetTarget(ComponentNginxProxy); v != "" {
		cluster.NginxProxyVersion = v
	}
	if v := KubernetesTargetFromVersionContext(vc); v != "" {
		cluster.KubernetesVersion = v
	}
}

// KubernetesTargetFromVersionContext returns the kubernetes version target from VersionContext.
// Prefers kubernetes-master, then kubernetes-worker, then release.yaml "kubernetes".
func KubernetesTargetFromVersionContext(vc *VersionContext) string {
	if vc == nil {
		return ""
	}
	for _, name := range []string{
		ComponentKubernetesMaster,
		ComponentKubernetesWorker,
		releaseComponentKubernetes,
	} {
		if v := strings.TrimSpace(vc.GetTarget(name)); v != "" {
			return v
		}
	}
	return ""
}

// ClusterSpecHasUpgradeTargets reports whether etcd, containerd, or kubernetes targets differ from spec.
func ClusterSpecHasUpgradeTargets(cluster confv1beta1.Cluster, vc *VersionContext) bool {
	if vc == nil {
		return false
	}
	if v := vc.GetTarget(ComponentEtcd); v != "" && cluster.EtcdVersion != v {
		return true
	}
	if v := vc.GetTarget(ComponentContainerd); v != "" && cluster.ContainerdVersion != v {
		return true
	}
	if v := vc.GetTarget(ComponentNginxProxy); v != "" && cluster.NginxProxyVersion != v {
		return true
	}
	if v := KubernetesTargetFromVersionContext(vc); v != "" && cluster.KubernetesVersion != v {
		return true
	}
	return false
}
