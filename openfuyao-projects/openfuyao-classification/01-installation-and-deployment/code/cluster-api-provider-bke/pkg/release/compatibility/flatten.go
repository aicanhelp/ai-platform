/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package compatibility

import (
	"fmt"

	releasev1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/release/manifest"
)

type componentRef struct {
	Name    string
	Version string
}

func Flatten(bundle *manifest.Bundle) ([]ResolvedComponent, error) {
	if bundle == nil {
		return nil, fmt.Errorf("release bundle is nil")
	}
	visited := map[string]bool{}
	var result []ResolvedComponent

	var walk func(parent string, ref componentRef) error
	walk = func(parent string, ref componentRef) error {
		key := manifest.ComponentKey(ref.Name, ref.Version)
		if visited[key] {
			return nil
		}
		visited[key] = true

		cv, ok := bundle.Components[key]
		if !ok {
			return fmt.Errorf("component %s not found in release bundle", key)
		}
		if len(cv.Spec.SubComponents) > 0 {
			for _, sub := range cv.Spec.SubComponents {
				if err := walk(cv.Spec.Name, componentRef{Name: sub.Name, Version: sub.Version}); err != nil {
					return err
				}
			}
			return nil
		}

		result = append(result, ResolvedComponent{
			Name:        cv.Spec.Name,
			Version:     cv.Spec.Version,
			Parent:      parent,
			InstallType: cv.Spec.Type,
		})
		return nil
	}

	for _, ref := range releaseComponents(bundle.Release) {
		if err := walk("", ref); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func releaseComponents(ri releasev1.ReleaseImage) []componentRef {
	var refs []componentRef
	if ri.Spec.Install != nil {
		for _, c := range ri.Spec.Install.Components {
			refs = append(refs, componentRef{Name: c.Name, Version: c.Version})
		}
	}
	if ri.Spec.Upgrade != nil {
		for _, c := range ri.Spec.Upgrade.Components {
			refs = append(refs, componentRef{Name: c.Name, Version: c.Version})
		}
	}
	return refs
}
