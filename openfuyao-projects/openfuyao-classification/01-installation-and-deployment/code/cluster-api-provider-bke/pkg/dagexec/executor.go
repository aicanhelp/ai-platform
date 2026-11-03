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

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
)

// ComponentType is the executor dispatch key (inline / yaml / helm).
type ComponentType string

const (
	ComponentTypeInline ComponentType = "inline"
	ComponentTypeYAML   ComponentType = "yaml"
	ComponentTypeHelm   ComponentType = "helm"
)

// ComponentExecutor runs one upgrade component for a registered ComponentType.
type ComponentExecutor interface {
	ExecuteComponent(ctx context.Context, node *topology.ComponentNode, execCtx *ExecutionContext) error
	GetComponentType() ComponentType
}
