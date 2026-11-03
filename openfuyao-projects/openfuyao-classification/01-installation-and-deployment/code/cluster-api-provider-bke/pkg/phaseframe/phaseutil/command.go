/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseutil

import (
	"context"
	"fmt"

	"github.com/pkg/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
)

// LogCommandFailedParams holds parameters for LogCommandFailed function
type LogCommandFailedParams struct {
	Cmd        agentv1beta1.Command
	FailedNods []string
	Log        *bkev1beta1.BKELogger
	Reason     string
}

func LogCommandFailed(cmd agentv1beta1.Command, failedNods []string, log *bkev1beta1.BKELogger, reason string) (map[string][]string, error) {
	if len(failedNods) == 0 {
		return nil, nil
	}
	var commandErrs map[string][]string
	errs := make([]error, 0)
	commandErrs = make(map[string][]string)
	for _, node := range failedNods {
		if v, ok := cmd.Status[node]; ok {
			nodeErrs, nodeErr := processNodeConditions(v.Conditions, node, &cmd, log, reason)
			if nodeErr != nil {
				errs = append(errs, nodeErr)
			}
			if commandErrs != nil && len(nodeErrs) > 0 {
				commandErrs[node] = append(commandErrs[node], nodeErrs...)
			}
		}
	}
	return commandErrs, kerrors.NewAggregate(errs)
}

// ProcessNodeConditionsParams holds parameters for processNodeConditions function
type ProcessNodeConditionsParams struct {
	Conditions []*agentv1beta1.Condition
	Node       string
	Cmd        *agentv1beta1.Command
	Log        *bkev1beta1.BKELogger
	Reason     string
}

// processNodeConditions processes the conditions for a specific node
func processNodeConditions(conditions []*agentv1beta1.Condition, node string, cmd *agentv1beta1.Command, log *bkev1beta1.BKELogger, reason string) ([]string, error) {
	params := ProcessNodeConditionsParams{
		Conditions: conditions,
		Node:       node,
		Cmd:        cmd,
		Log:        log,
		Reason:     reason,
	}
	return processNodeConditionsWithParams(params)
}

// processNodeConditionsWithParams processes the conditions for a specific node with parameters struct
func processNodeConditionsWithParams(params ProcessNodeConditionsParams) ([]string, error) {
	var nodeErrs []string
	var aggregateErr error

	for _, condition := range params.Conditions {
		if condition.Status == metav1.ConditionFalse && condition.StdErr != nil && len(condition.StdErr) > 0 {
			errInfo := fmt.Sprintf("Node %q, Command %s, sub ID %q, err: %s", params.Node, utils.ClientObjNS(params.Cmd), condition.ID, condition.StdErr[len(condition.StdErr)-1])
			nodeErrs = append(nodeErrs, errInfo)
			err := errors.New(errInfo)
			aggregateErr = err
			// 中间层只收集错误上抛，由边界层统一打 Error；此处用 Debug 保留节点细节便于排障
			if params.Log != nil {
				params.Log.Debug(params.Reason, errInfo)
			}
		}
	}
	return nodeErrs, aggregateErr
}

func MarkNodeStatusByCommandErrs(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, commandErrs map[string][]string) {
	if len(commandErrs) == 0 || commandErrs == nil || bkeCluster == nil {
		return
	}
	tmpBKENodes, err := nodeutil.GetBKENodesFromClient(ctx, c, bkeCluster.Namespace, bkeCluster.Name)
	if err != nil {
		return
	}
	bkeNodes := bkev1beta1.BKENodes(tmpBKENodes)
	for nodeIp, errInfos := range commandErrs {
		bkeNodes.SetNodeStateMessage(nodeIp, fmt.Sprintf("%v", errInfos))
	}
}

func LogCommandInfo(cmd agentv1beta1.Command, log *bkev1beta1.BKELogger, reason string) {
	for nodeInfo, nodeCondition := range cmd.Status {
		for _, condition := range nodeCondition.Conditions {
			if condition.Status == metav1.ConditionTrue && (condition.StdOut != nil || len(condition.StdOut) > 0) {
				log.Info(reason, fmt.Sprintf("Node %q, Command %s, sub ID %q, info: %v", nodeInfo, utils.ClientObjNS(&cmd), condition.ID, condition.StdOut))
			}
		}
	}
}

func GetMasterInitCommand(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
	commandsLi := agentv1beta1.CommandList{}
	filters := GetListFiltersByBKECluster(bkeCluster)

	if err := c.List(ctx, &commandsLi, filters...); err != nil {
		return nil, err
	}

	for _, cmd := range commandsLi.Items {
		if labelhelper.HasLabel(&cmd, command.MasterInitCommandLabel) {
			return &cmd, nil
		}
	}
	return nil, errors.New("master init command not found")
}

func GetMasterJoinCommand(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
	commandsLi := agentv1beta1.CommandList{}
	filters := GetListFiltersByBKECluster(bkeCluster)

	if err := c.List(ctx, &commandsLi, filters...); err != nil {
		return nil, err
	}

	for _, cmd := range commandsLi.Items {
		if labelhelper.HasLabel(&cmd, command.MasterJoinCommandLabel) {
			return &cmd, nil
		}
	}
	return nil, errors.New("master join command not found")
}

func GetWorkerJoinCommand(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (*agentv1beta1.Command, error) {
	commandsLi := agentv1beta1.CommandList{}
	filters := GetListFiltersByBKECluster(bkeCluster)

	if err := c.List(ctx, &commandsLi, filters...); err != nil {
		return nil, err
	}

	for _, cmd := range commandsLi.Items {
		if labelhelper.HasLabel(&cmd, command.WorkerJoinCommandLabel) {
			return &cmd, nil
		}
	}
	return nil, errors.New("worker join command not found")
}

func GetNodeIPFromCommandWaitResult(result string) string {
	return utils.GetNodeIPFromCommandWaitResult(result)
}

// GetNotSkipFailedNode counts failed nodes that are not marked as needSkip.
// Deprecated: In controller context, use GetNotSkipFailedNodeWithBKENodes instead.
func GetNotSkipFailedNode(bkeCluster *bkev1beta1.BKECluster, failedNodesInfo []string) int {
	bkeNodes := GetBKENodesFromCluster(bkeCluster)
	return GetNotSkipFailedNodeWithBKENodes(bkeNodes, failedNodesInfo)
}

// GetNotSkipFailedNodeWithBKENodes counts failed nodes that are not marked as needSkip.
// Use this in controller context where BKENodes are fetched via NodeFetcher.
func GetNotSkipFailedNodeWithBKENodes(bkeNodes bkev1beta1.BKENodes, failedNodesInfo []string) int {
	nodeStatusByIP := make(map[string]*v1beta1.BKENode, len(bkeNodes))
	for i := range bkeNodes {
		bkenode := &bkeNodes[i]
		if bkenode.Spec.IP != "" {
			nodeStatusByIP[bkenode.Spec.IP] = bkenode
		}
	}

	notSkipCnt := 0
	for _, node := range failedNodesInfo {
		nodeIP := GetNodeIPFromCommandWaitResult(node)
		if bkenode, exists := nodeStatusByIP[nodeIP]; exists {
			if !bkenode.Status.NeedSkip {
				notSkipCnt++
			}
		}
	}

	return notSkipCnt
}
