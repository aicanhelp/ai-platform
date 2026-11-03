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

package helminstaller

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

// Action is the Helm operation selected by ComponentVersion or release existence.
type Action string

const (
	ActionInstall Action = Action(cvv1alpha1.HelmModeInstall)
	ActionUpgrade Action = Action(cvv1alpha1.HelmModeUpgrade)
)

// HelmComponentExecutor adapts HelmInstaller to dagexec.ComponentExecutor.
// Scheduler owns clusterComponentStatuses; this executor only decides skip/action.
type HelmComponentExecutor struct {
	Installer *HelmInstaller
	CVStore   dagexec.ComponentVersionStore
}

// GetComponentType returns dagexec.ComponentTypeHelm.
func (e *HelmComponentExecutor) GetComponentType() dagexec.ComponentType {
	return dagexec.ComponentTypeHelm
}

// ExecuteComponent skips unchanged components, then installs or upgrades Helm release.
func (e *HelmComponentExecutor) ExecuteComponent(
	ctx context.Context,
	node *topology.ComponentNode,
	execCtx *dagexec.ExecutionContext,
) error {
	cv, execute, err := e.componentVersionForExecution(ctx, node, execCtx)
	if err != nil {
		return err
	}
	if !execute {
		return nil
	}
	if cv.Spec.Type != cvv1alpha1.ComponentTypeHelm {
		return fmt.Errorf("component %q type %q is not helm", node.Name, cv.Spec.Type)
	}
	if cv.Spec.Helm == nil {
		return fmt.Errorf("component %q spec.helm is required", node.Name)
	}

	action, err := actionFromMode(cv.Spec.Helm.Strategy.Mode)
	if err != nil {
		return fmt.Errorf("component %q: %w", node.Name, err)
	}
	return e.Installer.Apply(ctx, cv, toApplyContext(execCtx), action)
}

func (e *HelmComponentExecutor) componentVersionForExecution(
	ctx context.Context,
	node *topology.ComponentNode,
	execCtx *dagexec.ExecutionContext,
) (*cvv1alpha1.ComponentVersion, bool, error) {
	if err := e.validateExecutorInputs(node); err != nil {
		return nil, false, err
	}
	var versionContext *upgrade.VersionContext
	if execCtx != nil {
		versionContext = execCtx.VersionContext
	}
	if !upgrade.NeedsExecution(versionContext, node.Name) {
		return nil, false, nil
	}
	cv, err := e.loadComponentVersion(ctx, node)
	return cv, err == nil, err
}

func (e *HelmComponentExecutor) validateExecutorInputs(node *topology.ComponentNode) error {
	if e == nil || e.Installer == nil {
		return errors.New("helm installer is nil")
	}
	if e.CVStore == nil {
		return errors.New("component version store is nil")
	}
	if node == nil {
		return errors.New("component node is nil")
	}
	return nil
}

func (e *HelmComponentExecutor) loadComponentVersion(
	ctx context.Context,
	node *topology.ComponentNode,
) (*cvv1alpha1.ComponentVersion, error) {
	return dagexec.LoadComponentVersion(ctx, e.CVStore, node, defaultComponentVersion)
}

func actionFromMode(mode string) (Action, error) {
	switch mode {
	case "":
		return "", nil
	case cvv1alpha1.HelmModeInstall:
		return ActionInstall, nil
	case cvv1alpha1.HelmModeUpgrade:
		return ActionUpgrade, nil
	default:
		return "", fmt.Errorf("unsupported helm strategy mode %q", mode)
	}
}

func toApplyContext(execCtx *dagexec.ExecutionContext) *ApplyContext {
	result := &ApplyContext{}
	if execCtx == nil {
		return result
	}
	result.TemplateContext = execCtx.TemplateContext
	result.TargetClient = execCtx.TargetClient
	result.TargetRestConfig = execCtx.TargetRestConfig
	return result
}

// Ensure HelmComponentExecutor implements dagexec.ComponentExecutor.
var _ dagexec.ComponentExecutor = (*HelmComponentExecutor)(nil)
