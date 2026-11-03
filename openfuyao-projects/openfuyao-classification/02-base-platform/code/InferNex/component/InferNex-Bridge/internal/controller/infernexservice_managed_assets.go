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


package controller

import (
	"fmt"

	"k8s.io/utils/ptr"

	infernexv1alpha1 "gitcode.com/openFuyao/InferNex/api/v1alpha1"
	"gitcode.com/openFuyao/InferNex/internal/assets"
)

func resolveManagedComponent(
	component string,
	spec *infernexv1alpha1.EnabledComponentSpec,
) (*infernexv1alpha1.ComponentSpec, error) {
	if spec == nil || !enabled(spec.Enabled) {
		return nil, nil
	}

	assetSpec, err := assets.LoadManagedComponent(component)
	if err != nil {
		return nil, fmt.Errorf("load built-in component template for %q: %w", component, err)
	}

	return &infernexv1alpha1.ComponentSpec{
		Enabled:     ptr.To(true),
		Replicas:    1,
		ServicePort: assetSpec.ServicePort,
		Template:    assetSpec.Template.DeepCopy(),
	}, nil
}

func resolveCacheIndexerComponent(
	spec *infernexv1alpha1.CacheIndexerComponentSpec,
) (*infernexv1alpha1.ComponentSpec, error) {
	if spec == nil || !enabled(spec.Enabled) {
		return nil, nil
	}

	assetSpec, err := assets.LoadManagedComponent(cacheIndexerComponent)
	if err != nil {
		return nil, fmt.Errorf("load built-in component template for %q: %w", cacheIndexerComponent, err)
	}

	replicas := spec.Replicas
	if replicas <= 0 {
		replicas = 1
	}

	return &infernexv1alpha1.ComponentSpec{
		Enabled:     ptr.To(true),
		Replicas:    replicas,
		ServicePort: assetSpec.ServicePort,
		Template:    assetSpec.Template.DeepCopy(),
	}, nil
}
