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

package phases

import (
	"context"
	"fmt"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/scriptutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureNodesPostProcessName confv1beta1.BKEClusterPhase = "EnsureNodesPostProcess"
)

type EnsureNodesPostProcess struct {
	phaseframe.BasePhase
	nodes bkenode.Nodes
}

func NewEnsureNodesPostProcess(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureNodesPostProcessName)
	return &EnsureNodesPostProcess{BasePhase: base}
}

func (e *EnsureNodesPostProcess) Execute() (ctrl.Result, error) {
	return e.CheckOrRunPostProcess()
}

func (e *EnsureNodesPostProcess) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	// Use NodeFetcher to get BKENodes from API server (in controller context)
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, new)
	if err != nil {
		return false
	}
	needExecute := phaseutil.GetNeedPostProcessNodesWithBKENodes(new, bkeNodes).Length() > 0
	// 初次部署场景：尚无节点完成 bootstrap，但存在待加入的节点，
	// 预先加入执行计划，Execute 阶段会在前序 Phase 完成后重新获取节点状态
	if !needExecute {
		needExecute = phaseutil.GetNeedJoinNodesWithBKENodes(new, bkeNodes).Length() > 0
	}
	if needExecute {
		e.SetStatus(bkev1beta1.PhaseWaiting)
	}
	return needExecute
}

func (e *EnsureNodesPostProcess) CheckOrRunPostProcess() (ctrl.Result, error) {
	phaseContextFields79 := e.Ctx.Untie()
	_, _, bkeCluster, _, log := phaseContextFields79.Context,
		phaseContextFields79.Client,
		phaseContextFields79.BKECluster,
		phaseContextFields79.Scheme,
		phaseContextFields79.Log
	log.Info(constant.NodesPostProcessCheckingReason, "Start post process scripts")

	// Use NodeFetcher to get BKENodes from API server (in controller context)
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, bkeCluster)
	if err != nil {
		return ctrl.Result{}, err
	}
	nodes := phaseutil.GetNeedPostProcessNodesWithBKENodes(bkeCluster, bkeNodes)
	if nodes.Length() == 0 {
		condition.ConditionMark(bkeCluster, bkev1beta1.NodesPostProcessCondition, confv1beta1.ConditionTrue, constant.NodesPostProcessReadyReason, "")
		log.Info(constant.NodesPostProcessCheckingReason, "No nodes need post process")
		return ctrl.Result{}, nil
	}
	e.nodes = nodes

	condition.ConditionMark(bkeCluster, bkev1beta1.NodesPostProcessCondition, confv1beta1.ConditionFalse, constant.NodesPostProcessNotReadyReason, "")
	if err := e.executeNodePostProcessScripts(); err != nil {
		return ctrl.Result{}, err
	}

	condition.ConditionMark(bkeCluster, bkev1beta1.NodesPostProcessCondition, confv1beta1.ConditionTrue, constant.NodesPostProcessReadyReason, "")
	log.Info(constant.NodesPostProcessReadyReason, "Post process scripts complete")

	return ctrl.Result{}, nil
}

func (e *EnsureNodesPostProcess) executeNodePostProcessScripts() error {
	phaseContextFields107 := e.Ctx.Untie()
	ctx, c, bkeCluster, scheme, log := phaseContextFields107.Context,
		phaseContextFields107.Client,
		phaseContextFields107.BKECluster,
		phaseContextFields107.Scheme,
		phaseContextFields107.Log
	nodes := e.nodes
	log.Info(constant.NodesPostProcessCheckingReason, "Checking post process config, total nodes=%d", len(nodes))

	var nodesWithConfig bkenode.Nodes
	var nodesWithoutConfig bkenode.Nodes // 新增：收集没有配置的节点
	for _, node := range nodes {
		if node.IP == "" {
			log.Debug("nodeIP empty, skip postprocess")
			continue
		}
		hasConfig := e.checkPostProcessConfigExists(ctx, c, log, node.IP)
		if !hasConfig {
			nodesWithoutConfig = append(nodesWithoutConfig, node)
			log.Debug("node %s has no postprocess config, skip", node.IP)
			continue
		}
		nodesWithConfig = append(nodesWithConfig, node)
	}

	for _, node := range nodesWithoutConfig {
		nodeFetcher := e.Ctx.NodeFetcher()
		if err := nodeFetcher.UpdateNodeStatusByIPForCluster(ctx, bkeCluster, node.IP, func(status *confv1beta1.BKENodeStatus) {
			status.State = confv1beta1.NodeReady
			status.StateCode |= bkev1beta1.NodePostProcessFlag
			status.Message = "Post process skipped (no config)"
		}); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to update postprocess skipped status for %s: %v", node.IP, err)
		}
	}

	log.Info(constant.NodesPostProcessCheckingReason, "postprocess config check done, total=%d, hit=%d", len(nodes), len(nodesWithConfig))
	if len(nodesWithConfig) == 0 {
		log.Info(constant.NodesPostProcessCheckingReason, "No nodes need post process, skip")
		return nil
	}

	log.Info(constant.NodesPostProcessCheckingReason, "create postprocess command, nodes=%d", len(nodesWithConfig))
	cmd, err := e.createPostProcessCommand(ctx, c, bkeCluster, scheme, nodesWithConfig)
	if err != nil {
		return errors.Wrapf(err, "create postprocess command failed")
	}
	log.Info(constant.NodesPostProcessCheckingReason, "postprocess command created, command=%s", cmd.CommandName)

	log.Info(constant.NodesPostProcessCheckingReason, "wait postprocess command, command=%s", cmd.CommandName)
	err, successNodes, failedNodes := cmd.Wait()
	log.Info(constant.NodesPostProcessCheckingReason, "postprocess command done, command=%s, success=%v, failed=%v", cmd.CommandName, successNodes, failedNodes)
	if cmd.Command != nil {
		phaseutil.LogCommandInfo(*cmd.Command, log, constant.NodesPostProcessCheckingReason)
	}
	if err != nil || len(failedNodes) > 0 {
		e.handleFailedNodes(cmd, failedNodes)
		failMsg := fmt.Sprintf("postprocess failed, success=%v, failed=%v", successNodes, failedNodes)
		condition.ConditionMark(bkeCluster, bkev1beta1.NodesPostProcessCondition,
			confv1beta1.ConditionFalse, constant.NodesPostProcessNotReadyReason, failMsg)
		return errors.New(failMsg)
	}

	e.markPostProcessSuccess(successNodes)
	return nil
}

func (e *EnsureNodesPostProcess) markPostProcessSuccess(successNodes []string) {
	phaseContextFields170 := e.Ctx.Untie()
	ctx, _, bkeCluster, _, log := phaseContextFields170.Context,
		phaseContextFields170.Client,
		phaseContextFields170.BKECluster,
		phaseContextFields170.Scheme,
		phaseContextFields170.Log
	nodeFetcher := e.Ctx.NodeFetcher()
	for _, node := range successNodes {
		nodeIP := phaseutil.GetNodeIPFromCommandWaitResult(node)
		if err := nodeFetcher.UpdateNodeStatusByIPForCluster(ctx, bkeCluster, nodeIP, func(status *confv1beta1.BKENodeStatus) {
			status.StateCode |= bkev1beta1.NodePostProcessFlag
			status.Message = "Post process scripts completed"
		}); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to update postprocess success status for %s: %v", nodeIP, err)
		}
	}
	log.Info(constant.NodesPostProcessReadyReason, "postprocess success nodes=%v", successNodes)
}

func (e *EnsureNodesPostProcess) handleFailedNodes(cmd *command.Custom, failedNodes []string) {
	phaseContextFields185 := e.Ctx.Untie()
	ctx, _, bkeCluster, _, log := phaseContextFields185.Context,
		phaseContextFields185.Client,
		phaseContextFields185.BKECluster,
		phaseContextFields185.Scheme,
		phaseContextFields185.Log
	nodeFetcher := e.Ctx.NodeFetcher()
	for _, node := range failedNodes {
		nodeIP := phaseutil.GetNodeIPFromCommandWaitResult(node)
		if err := nodeFetcher.UpdateNodeStatusByIPForCluster(ctx, bkeCluster, nodeIP, func(status *confv1beta1.BKENodeStatus) {
			status.State = bkev1beta1.NodeInitFailed
			status.StateCode |= bkev1beta1.NodePostProcessFlag
			status.Message = "Post process scripts failed"
		}); err != nil {
			log.Warn(constant.InternalErrorReason, "Failed to update postprocess failed status for %s: %v", nodeIP, err)
		}
	}
	if cmd.Command != nil {
		phaseutil.LogCommandFailed(*cmd.Command, failedNodes, log, constant.NodesPostProcessNotReadyReason)
	}
	log.Warn(constant.NodesPostProcessNotReadyReason, "postprocess failed nodes=%v", failedNodes)
}

func (e *EnsureNodesPostProcess) createPostProcessCommand(
	ctx context.Context,
	c client.Client,
	bkeCluster *bkev1beta1.BKECluster,
	scheme *runtime.Scheme,
	nodes bkenode.Nodes,
) (*command.Custom, error) {
	commandSpec := command.GenerateDefaultCommandSpec()
	execCommands := []agentv1beta1.ExecCommand{
		{
			ID: "execute-postprocess-scripts",
			Command: []string{
				"Postprocess",
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: false,
		},
	}
	commandSpec.Commands = execCommands

	commandName := fmt.Sprintf("postprocess-all-nodes-%d", time.Now().Unix())
	customCmd := &command.Custom{
		BaseCommand: command.BaseCommand{
			Ctx:             ctx,
			Client:          c,
			NameSpace:       bkeCluster.Namespace,
			Scheme:          scheme,
			OwnerObj:        bkeCluster,
			ClusterName:     bkeCluster.Name,
			Unique:          false,
			RemoveAfterWait: true,
			WaitTimeout:     30 * time.Minute,
		},
		Nodes:        nodes,
		CommandName:  commandName,
		CommandSpec:  commandSpec,
		CommandLabel: "bke.postprocess.node",
	}

	if err := customCmd.New(); err != nil {
		return nil, errors.Wrapf(err, "create postprocess command failed")
	}
	return customCmd, nil
}

func (e *EnsureNodesPostProcess) checkPostProcessConfigExists(ctx context.Context, c client.Client, log *bkev1beta1.BKELogger, nodeIP string) bool {
	// 1. global config
	globalConfigCM := &corev1.ConfigMap{}
	globalConfigKey := client.ObjectKey{Namespace: "user-system", Name: "postprocess-all-config"}
	if err := c.Get(ctx, globalConfigKey, globalConfigCM); err == nil {
		log.Info(constant.NodesPostProcessCheckingReason, "hit postprocess global config, nodeIP=%s", nodeIP)
		return true
	}

	batchConfig, err := scriptutil.LoadBatchConfig(ctx, c, scriptutil.PostprocessBatchConfig, nodeIP)
	if err != nil {
		log.Warn(constant.NodesPostProcessCheckingReason, "failed to load postprocess batch config, nodeIP=%s, err=%v", nodeIP, err)
	}
	if batchConfig != nil {
		log.Info(constant.NodesPostProcessCheckingReason, "hit postprocess batch config %s, nodeIP=%s", batchConfig.Name, nodeIP)
		return true
	}

	// 3. node config
	nodeConfigCM := &corev1.ConfigMap{}
	nodeConfigKey := client.ObjectKey{
		Namespace: "user-system",
		Name:      fmt.Sprintf("postprocess-config-node-%s", nodeIP),
	}
	if err := c.Get(ctx, nodeConfigKey, nodeConfigCM); err == nil {
		log.Info(constant.NodesPostProcessCheckingReason, "hit postprocess node config %s", nodeConfigKey.Name)
		return true
	}
	log.Info(constant.NodesPostProcessCheckingReason, "no postprocess config, nodeIP=%s", nodeIP)
	return false
}
