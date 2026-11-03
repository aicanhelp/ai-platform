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

package componentfactory

import (
	"fmt"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	releasemanifest "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

// NewFactoryFromBundle builds a per-reconcile ComponentFactory from a resolved release bundle.
// Inline handlers and versions come from release.yaml upgrade.components and component.yaml inline specs.
func NewFactoryFromBundle(bundle *releasemanifest.Bundle) (*ComponentFactory, error) {
	f := NewComponentFactory()
	if err := RegisterInlinePhasesFromBundle(f, bundle); err != nil {
		return nil, err
	}
	return f, nil
}

// RegisterInlinePhasesFromBundle registers inline Phase factories declared in the release bundle.
func RegisterInlinePhasesFromBundle(f *ComponentFactory, bundle *releasemanifest.Bundle) error {
	if f == nil {
		return fmt.Errorf("component factory is nil")
	}
	components, err := upgrade.UpgradeComponentsFromBundle(bundle)
	if err != nil {
		return err
	}

	seen := make(map[string]struct{})
	for _, comp := range components {
		if comp.Inline == nil || comp.Inline.Handler == "" {
			continue
		}
		version := comp.Inline.Version
		if version == "" {
			version = upgrade.ComponentManifestVersion
		}
		key := registryKey(comp.Inline.Handler, version)
		if _, ok := seen[key]; ok {
			continue
		}
		var componentVersion *cvv1alpha1.ComponentVersion
		if cv, ok := bundle.Components[releasemanifest.ComponentKey(comp.Name, comp.Version)]; ok {
			componentVersion = cv.DeepCopy()
		}
		if err := registerInlineComponent(f, comp.Inline.Handler, version, componentVersion); err != nil {
			return fmt.Errorf("register inline handler for component %q: %w", comp.Name, err)
		}
		seen[key] = struct{}{}
	}
	return nil
}
