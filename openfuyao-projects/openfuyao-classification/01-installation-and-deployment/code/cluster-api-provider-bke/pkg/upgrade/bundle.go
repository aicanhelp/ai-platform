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
	"fmt"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
)

// BuildDAGFromBundle constructs an upgrade DAG from a resolved OCI release bundle.
func BuildDAGFromBundle(bundle *releasemanifest.Bundle, resolve topology.DependencyResolver) (*topology.UpgradeDAG, error) {
	components, err := UpgradeComponentsFromBundle(bundle)
	if err != nil {
		return nil, err
	}
	return topology.BuildUpgradeDAG(
		components,
		topology.MergeDependencyResolver(resolve, topology.DefaultDependencyResolver()),
	)
}

// UpgradeComponentsFromBundle returns upgrade components enriched from bundle ComponentVersions.
func UpgradeComponentsFromBundle(bundle *releasemanifest.Bundle) ([]cvv1alpha1.ReleaseImageUpgradeComponent, error) {
	if bundle == nil {
		return nil, fmt.Errorf("release bundle is nil")
	}
	if bundle.Release.Spec.Upgrade == nil || len(bundle.Release.Spec.Upgrade.Components) == 0 {
		return nil, fmt.Errorf("release bundle has no upgrade components")
	}
	out := make([]cvv1alpha1.ReleaseImageUpgradeComponent, len(bundle.Release.Spec.Upgrade.Components))
	for i, comp := range bundle.Release.Spec.Upgrade.Components {
		out[i] = enrichUpgradeComponent(comp, bundle)
	}
	return out, nil
}

// BundleDependencyResolver reads dependencies from bundle ComponentVersion CRs.
func BundleDependencyResolver(bundle *releasemanifest.Bundle) topology.DependencyResolver {
	return func(name, version string) ([]string, error) {
		if bundle == nil {
			return nil, nil
		}
		cv, ok := bundle.Components[releasemanifest.ComponentKey(name, version)]
		deps := make([]string, 0)
		if ok {
			deps = append(deps, topology.ComponentDependencyNames(cv.Spec.Dependencies)...)
		}
		return appendImplicitPreUpgradeDependency(bundle, name, deps), nil
	}
}

func appendImplicitPreUpgradeDependency(
	bundle *releasemanifest.Bundle,
	name string,
	deps []string,
) []string {
	if bundle == nil || name == ComponentPreUpgradeResources || !bundleHasUpgradeComponent(bundle, ComponentPreUpgradeResources) {
		return deps
	}
	for _, dep := range deps {
		if dep == ComponentPreUpgradeResources {
			return deps
		}
	}
	return append(deps, ComponentPreUpgradeResources)
}

func bundleHasUpgradeComponent(bundle *releasemanifest.Bundle, name string) bool {
	if bundle == nil || bundle.Release.Spec.Upgrade == nil {
		return false
	}
	for _, comp := range bundle.Release.Spec.Upgrade.Components {
		if comp.Name == name {
			return true
		}
	}
	return false
}

func enrichUpgradeComponent(
	comp cvv1alpha1.ReleaseImageUpgradeComponent,
	bundle *releasemanifest.Bundle,
) cvv1alpha1.ReleaseImageUpgradeComponent {
	if comp.Inline != nil || bundle == nil {
		return comp
	}
	cv, ok := bundle.Components[releasemanifest.ComponentKey(comp.Name, comp.Version)]
	if !ok || cv.Spec.Inline == nil {
		return comp
	}
	enriched := comp
	enriched.Inline = &cvv1alpha1.ReleaseImageUpgradeInline{
		Handler: cv.Spec.Inline.Handler,
		Version: cv.Spec.Inline.Version,
	}
	return enriched
}
