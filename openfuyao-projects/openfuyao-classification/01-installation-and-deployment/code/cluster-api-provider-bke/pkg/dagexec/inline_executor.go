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

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

// InlineComponentExecutor runs inline DAG nodes via InlineRunner.
type InlineComponentExecutor struct {
	Runner InlineRunner
}

// GetComponentType returns ComponentTypeInline.
func (e *InlineComponentExecutor) GetComponentType() ComponentType {
	return ComponentTypeInline
}

// ExecuteComponent validates node.Inline and delegates to InlineRunner.
// Skip / Upgrade is decided from ExecutionContext.VersionContext;
// matched versions return nil without invoking the runner.
func (e *InlineComponentExecutor) ExecuteComponent(
	ctx context.Context,
	node *topology.ComponentNode,
	execCtx *ExecutionContext,
) error {
	if e == nil || e.Runner == nil {
		return fmt.Errorf("inline runner is nil")
	}
	if node == nil {
		return fmt.Errorf("component node is nil")
	}
	if node.Inline == nil {
		return fmt.Errorf("inline component %q missing inline ref", node.Name)
	}

	var vc *upgrade.VersionContext
	if execCtx != nil {
		vc = execCtx.VersionContext
	}
	if !upgrade.NeedsExecution(vc, node.Name) {
		return nil
	}

	handler := node.Inline.Handler
	version := node.Inline.Version
	if handler == "" {
		return fmt.Errorf("inline component %q missing handler", node.Name)
	}
	if version == "" {
		version = defaultComponentVersion
	}
	if execCtx == nil {
		return e.Runner.Execute(ctx, nil, nil, handler, version)
	}
	return e.Runner.Execute(ctx, execCtx.OldCluster, execCtx.Cluster, handler, version)
}
