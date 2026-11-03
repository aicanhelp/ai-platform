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
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
)

// FillTargetFromBundle writes install and upgrade component versions from a release bundle into Target.
// Upgrade entries override install entries with the same name.
func FillTargetFromBundle(vc *VersionContext, bundle *releasemanifest.Bundle) {
	if vc == nil || bundle == nil {
		return
	}
	applyReleaseComponents(vc.SetTarget, bundle)
}

// FillCurrentFromBundle writes install and upgrade component versions from a release bundle into Current.
func FillCurrentFromBundle(vc *VersionContext, bundle *releasemanifest.Bundle) {
	if vc == nil || bundle == nil {
		return
	}
	applyReleaseComponents(vc.SetCurrent, bundle)
}

// BuildVersionContextForUpgrade builds VersionContext for declarative upgrade.
// Target comes from targetBundle (ReleaseImage). Current comes from currentBundle when set,
// otherwise from BKECluster status for components present in Target.
func BuildVersionContextForUpgrade(
	targetBundle *releasemanifest.Bundle,
	currentBundle *releasemanifest.Bundle,
	bc *bkev1beta1.BKECluster,
) *VersionContext {
	vc := NewVersionContext()
	if targetBundle != nil {
		FillTargetFromBundle(vc, targetBundle)
	}
	if targetBundle == nil && bc != nil {
		legacy := BuildVersionContextFromBKECluster(bc)
		for name, version := range legacy.Target {
			if version != "" {
				vc.SetTarget(name, version)
			}
		}
	}

	if currentBundle != nil {
		FillCurrentFromBundle(vc, currentBundle)
	} else {
		fillCurrentFromBKECluster(vc, bc)
	}
	return vc
}

func applyReleaseComponents(set func(name, version string), bundle *releasemanifest.Bundle) {
	ri := bundle.Release
	if ri.Spec.Install != nil {
		for _, c := range ri.Spec.Install.Components {
			if c.Name != "" && c.Version != "" {
				set(c.Name, c.Version)
			}
		}
	}
	if ri.Spec.Upgrade != nil {
		for _, c := range ri.Spec.Upgrade.Components {
			if c.Name != "" && c.Version != "" {
				set(c.Name, c.Version)
			}
		}
	}
}

func fillCurrentFromBKECluster(vc *VersionContext, bc *bkev1beta1.BKECluster) {
	if vc == nil || bc == nil {
		return
	}
	for _, name := range vc.TargetNames() {
		if vc.GetCurrent(name) != "" {
			continue
		}
		if cur := clusterCurrentForReleaseComponent(bc, name); cur != "" {
			vc.SetCurrent(name, cur)
		}
	}
}

// clusterCurrentForReleaseComponent maps a release component name to BKECluster status.
func clusterCurrentForReleaseComponent(bc *bkev1beta1.BKECluster, componentName string) string {
	if bc == nil {
		return ""
	}
	switch componentName {
	case ComponentEtcd:
		return bc.Status.EtcdVersion
	case ComponentKubernetesMaster, ComponentKubernetesWorker:
		return bc.Status.KubernetesVersion
	case ComponentContainerd:
		return bc.Status.ContainerdVersion
	case ComponentNginxProxy:
		return bc.Status.NginxProxyVersion
	case ComponentOpenFuyao:
		return bc.Status.OpenFuyaoVersion
	default:
		// fall through to addon/clusterComponentStatuses lookup below
	}
	for _, addon := range bc.Status.AddonStatus {
		if addon.Name == componentName && addon.Version != "" {
			return addon.Version
		}
	}
	if st, ok := bc.Status.ClusterComponentStatuses[componentName]; ok && st.CurrentVersion != "" {
		return st.CurrentVersion
	}
	return ""
}
