/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package plugin

import (
	"context"
	"fmt"

	"k8s.io/klog/v2"

	"gitcode.com/openFuyao/ub-network-device-plugin/resource"
)

func New(ctx context.Context, opts ...Option) ([]Interface, error) {
	o := &options{}
	for _, opt := range opts {
		opt(o)
	}

	if o.resources == nil {
		klog.Warning("no default resources, returning a null manager")
		return nil, nil
	}

	resourceManagers, err := o.getResourceManagers()
	if err != nil {
		return nil, fmt.Errorf("failed to construct resource managers: %w", err)
	}

	var plugins []Interface
	for _, resourceManager := range resourceManagers {
		plugin, err := o.devicePluginForResource(ctx, resourceManager)
		if err != nil {
			return nil, fmt.Errorf("failed to create plugin: %w", err)
		}
		plugins = append(plugins, plugin)
	}
	return plugins, nil
}

// getResourceManager constructs a set of resource managers.
// Each resource manager maps to a specific named extended resource.
func (o *options) getResourceManagers() ([]resource.Manager, error) {
	return resource.NewUrmaResourceManagers(o.resources)
}
