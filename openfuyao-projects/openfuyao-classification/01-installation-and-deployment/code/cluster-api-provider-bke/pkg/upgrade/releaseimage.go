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
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
)

// BuildDAGFromReleaseImage constructs an upgrade DAG from a ReleaseImage spec.
func BuildDAGFromReleaseImage(ri *cvv1alpha1.ReleaseImage, resolve topology.DependencyResolver) (*topology.UpgradeDAG, error) {
	if ri == nil {
		return nil, fmt.Errorf("release image is nil")
	}
	if ri.Spec.Upgrade == nil || len(ri.Spec.Upgrade.Components) == 0 {
		return nil, fmt.Errorf("release image %s has no upgrade components", ri.Name)
	}
	return topology.BuildUpgradeDAG(
		ri.Spec.Upgrade.Components,
		topology.MergeDependencyResolver(resolve, topology.DefaultDependencyResolver()),
	)
}
