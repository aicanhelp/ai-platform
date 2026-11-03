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

import bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"

// BuildVersionContextFromBKECluster fills Current from BKECluster status and Target from spec.
// Legacy path when no ReleaseImage bundle is available; declarative upgrade should use
// BuildVersionContextForUpgrade with a resolved release bundle.
// VersionContext keys match release.yaml component names.
func BuildVersionContextFromBKECluster(bc *bkev1beta1.BKECluster) *VersionContext {
	vc := NewVersionContext()
	if bc == nil {
		return vc
	}

	status := bc.Status
	if bc.Spec.ClusterConfig == nil {
		return vc
	}
	spec := bc.Spec.ClusterConfig.Cluster

	vc.SetCurrent(ComponentEtcd, status.EtcdVersion)
	vc.SetTarget(ComponentEtcd, spec.EtcdVersion)

	k8sCurrent, k8sTarget := status.KubernetesVersion, spec.KubernetesVersion
	vc.SetCurrent(ComponentKubernetesMaster, k8sCurrent)
	vc.SetTarget(ComponentKubernetesMaster, k8sTarget)
	vc.SetCurrent(ComponentKubernetesWorker, k8sCurrent)
	vc.SetTarget(ComponentKubernetesWorker, k8sTarget)

	vc.SetCurrent(ComponentOpenFuyao, status.OpenFuyaoVersion)
	vc.SetTarget(ComponentOpenFuyao, spec.OpenFuyaoVersion)

	vc.SetCurrent(ComponentContainerd, status.ContainerdVersion)
	vc.SetTarget(ComponentContainerd, spec.ContainerdVersion)

	return vc
}
