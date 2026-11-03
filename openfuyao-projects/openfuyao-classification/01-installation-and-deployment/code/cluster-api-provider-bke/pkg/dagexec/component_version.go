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

package dagexec

import (
	"context"
	"fmt"

	"github.com/pkg/errors"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
)

// LoadComponentVersion resolves a node's version and validates the store result.
func LoadComponentVersion(ctx context.Context, store ComponentVersionStore, node *topology.ComponentNode,
	defaultVersion string) (*cvv1alpha1.ComponentVersion, error) {
	version := node.Version
	if version == "" {
		version = defaultVersion
	}
	componentVersion, err := store.GetComponentVersion(ctx, node.Name, version)
	if err != nil {
		return nil, errors.Wrapf(err, "%s: get component version", node.Name)
	}
	if componentVersion == nil {
		return nil, fmt.Errorf("%s: get component version: nil", node.Name)
	}
	return componentVersion, nil
}
