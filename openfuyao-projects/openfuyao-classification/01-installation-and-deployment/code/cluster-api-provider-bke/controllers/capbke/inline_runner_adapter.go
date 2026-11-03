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

package capbke

import (
	"context"
	"fmt"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/componentfactory"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/dagexec"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
)

// InlinePhaseRunnerAdapter bridges componentfactory.PhaseRunner to dagexec.InlineRunner.
// PhaseContext stays in controllers; dagexec only sees the InlineRunner interface.
type InlinePhaseRunnerAdapter struct {
	phaseCtx *phaseframe.PhaseContext
	runner   *componentfactory.PhaseRunner
}

// NewInlinePhaseRunnerAdapter creates an InlineRunner backed by PhaseRunner + PhaseContext.
func NewInlinePhaseRunnerAdapter(phaseCtx *phaseframe.PhaseContext, runner *componentfactory.PhaseRunner) dagexec.InlineRunner {
	return &InlinePhaseRunnerAdapter{phaseCtx: phaseCtx, runner: runner}
}

// Execute delegates to PhaseRunner with the held PhaseContext.
func (a *InlinePhaseRunnerAdapter) Execute(
	_ context.Context,
	oldCluster, newCluster *bkev1beta1.BKECluster,
	handler, version string,
) error {
	if a == nil || a.runner == nil {
		return fmt.Errorf("inline phase runner adapter is nil")
	}
	return a.runner.Execute(a.phaseCtx, oldCluster, newCluster, handler, version)
}

type executionContextBuildOptions struct {
	phaseCtx         *phaseframe.PhaseContext
	oldCluster       *bkev1beta1.BKECluster
	newCluster       *bkev1beta1.BKECluster
	log              *bkev1beta1.BKELogger
	targetClient     kubernetes.Interface
	targetRestConfig *rest.Config
}

// buildExecutionContext assembles a phaseframe-free ExecutionContext for DAG scheduling.
func buildExecutionContext(opts executionContextBuildOptions) *dagexec.ExecutionContext {
	var (
		vc      *upgrade.VersionContext
		cli     client.Client
		cluster = opts.newCluster
		log     = opts.log
	)
	if opts.phaseCtx != nil {
		vc = opts.phaseCtx.VersionContext
		cli = opts.phaseCtx.Client
		if cluster == nil {
			cluster = opts.phaseCtx.BKECluster
		}
		if log == nil {
			log = opts.phaseCtx.Log
		}
	}
	return dagexec.NewExecutionContext(dagexec.NewExecutionContextOptions{
		OldCluster:             opts.oldCluster,
		Cluster:                cluster,
		Log:                    log,
		VersionContext:         vc,
		TargetClient:           opts.targetClient,
		TargetRestConfig:       opts.targetRestConfig,
		Client:                 cli,
		ComponentStatusUpdater: newComponentStatusUpdater(cli),
	})
}

func newComponentStatusUpdater(cli client.Client) dagexec.ComponentStatusUpdater {
	if cli == nil {
		return nil
	}
	return dagexec.NewBKEComponentStatusUpdater(cli)
}
