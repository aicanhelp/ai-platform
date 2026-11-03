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

package yamlinstaller

import (
	"context"
	"fmt"

	"github.com/pkg/errors"

	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/dagexec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

const defaultComponentVersion = "v1.0.0"

// YamlComponentExecutor adapts YamlInstaller to dagexec.ComponentExecutor.
// Lifecycle Pending/Installed/Failed is written by the DAG Scheduler on the
// registry path; this executor only decides Skip vs Apply and runs the engine.
type YamlComponentExecutor struct {
	Installer *YamlInstaller
	CVStore   dagexec.ComponentVersionStore
}

// GetComponentType returns dagexec.ComponentTypeYAML.
func (e *YamlComponentExecutor) GetComponentType() dagexec.ComponentType {
	return dagexec.ComponentTypeYAML
}

// ExecuteComponent skips when VersionContext says no upgrade is needed; otherwise
// loads ComponentVersion from CVStore and calls YamlInstaller.Apply.
func (e *YamlComponentExecutor) ExecuteComponent(
	ctx context.Context,
	node *topology.ComponentNode,
	execCtx *dagexec.ExecutionContext,
) error {
	if err := e.validateExecutorInputs(node); err != nil {
		return err
	}

	var vc *upgrade.VersionContext
	if execCtx != nil {
		vc = execCtx.VersionContext
	}
	if !upgrade.NeedsExecution(vc, node.Name) {
		return nil
	}

	cv, err := e.loadComponentVersion(ctx, node)
	if err != nil {
		return err
	}
	if cv.Spec.Type != "" && cv.Spec.Type != cvv1alpha1.ComponentTypeYAML {
		return fmt.Errorf("component %q type %q is not yaml", node.Name, cv.Spec.Type)
	}

	return e.Installer.Apply(ctx, cv, toApplyContext(execCtx))
}

// UninstallComponent loads ComponentVersion and calls YamlInstaller.Uninstall.
// Not used by the upgrade DAG path; exposed for explicit uninstall orchestration.
func (e *YamlComponentExecutor) UninstallComponent(
	ctx context.Context,
	node *topology.ComponentNode,
	execCtx *dagexec.ExecutionContext,
) error {
	if err := e.validateExecutorInputs(node); err != nil {
		return err
	}

	cv, err := e.loadComponentVersion(ctx, node)
	if err != nil {
		return err
	}
	return e.Installer.Uninstall(ctx, cv, toApplyContext(execCtx))
}

func (e *YamlComponentExecutor) validateExecutorInputs(node *topology.ComponentNode) error {
	if e == nil || e.Installer == nil {
		return errors.New("yaml installer is nil")
	}
	if e.CVStore == nil {
		return errors.New("component version store is nil")
	}
	if node == nil {
		return errors.New("component node is nil")
	}
	return nil
}

func (e *YamlComponentExecutor) loadComponentVersion(
	ctx context.Context,
	node *topology.ComponentNode,
) (*cvv1alpha1.ComponentVersion, error) {
	return dagexec.LoadComponentVersion(ctx, e.CVStore, node, defaultComponentVersion)
}

func toApplyContext(execCtx *dagexec.ExecutionContext) *ApplyContext {
	if execCtx == nil {
		return &ApplyContext{}
	}
	return &ApplyContext{
		TemplateContext: execCtx.TemplateContext,
		TargetClient:    execCtx.TargetClient,
	}
}

// Ensure YamlComponentExecutor implements dagexec.ComponentExecutor.
var _ dagexec.ComponentExecutor = (*YamlComponentExecutor)(nil)
