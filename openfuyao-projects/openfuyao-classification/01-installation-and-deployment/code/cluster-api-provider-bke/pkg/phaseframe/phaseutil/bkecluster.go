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
	"sort"
	"time"

	"github.com/pkg/errors"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// IsPaused 判断bkecluster是否暂停
func IsPaused(bkeCluster *bkev1beta1.BKECluster) bool {
	v, ok := annotation.HasAnnotation(bkeCluster, annotation.BKEClusterPauseAnnotationKey)
	flag := ok && v == "true"

	return bkeCluster.Spec.Pause == flag
}

func IsDeleteOrReset(bkeCluster *bkev1beta1.BKECluster) bool {
	return !bkeCluster.DeletionTimestamp.IsZero() || bkeCluster.Spec.Reset
}

// GenerateBKEAgentStatus updates AgentStatus from ping results against all cluster nodes.
// When bkeNodes is set, nodes not in pingNodes use NodeAgentReadyFlag; pinged nodes use ping results only.
func GenerateBKEAgentStatus(
	success []string,
	bkeCluster *bkev1beta1.BKECluster,
	nodes node.Nodes,
	bkeNodes bkev1beta1.BKENodes,
	pingNodes node.Nodes,
) {
	available := len(success)
	if bkeNodes != nil {
		available = countAvailableAgentNodes(bkeNodes, pingNodes, success)
	}
	total := len(nodes)
	bkeCluster.Status.AgentStatus.Replies = int32(total)
	bkeCluster.Status.AgentStatus.UnavailableReplies = int32(total - available)
	bkeCluster.Status.AgentStatus.Status = fmt.Sprintf("%d/%d", available, total)
}

// GetBKEClusterAssociateMachines 获取bkecluster 关联的所有machine
func GetBKEClusterAssociateMachines(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) ([]clusterv1.Machine, error) {
	// 获取bkecluster 关联的machine
	machineList := &clusterv1.MachineList{}
	filters := GetListFiltersByBKECluster(bkeCluster)

	if err := c.List(ctx, machineList, filters...); err != nil {
		return nil, err
	}
	return machineList.Items, nil
}

// GetBKEClusterAssociateMasterMachines 获取bkecluster 关联的master machine
func GetBKEClusterAssociateMasterMachines(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) ([]clusterv1.Machine, error) {
	machineList, err := GetBKEClusterAssociateMachines(ctx, c, bkeCluster)
	if err != nil {
		return nil, err
	}
	var masterMachines []clusterv1.Machine
	for _, machine := range machineList {
		if util.IsControlPlaneMachine(&machine) {
			masterMachines = append(masterMachines, machine)
		}
	}
	return masterMachines, nil
}

// GetBKEClusterAssociateWorkerMachines 获取bkecluster 关联的worker machine
func GetBKEClusterAssociateWorkerMachines(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) ([]clusterv1.Machine, error) {
	machineList, err := GetBKEClusterAssociateMachines(ctx, c, bkeCluster)
	if err != nil {
		return nil, err
	}
	var workerMachines []clusterv1.Machine
	for _, machine := range machineList {
		if !util.IsControlPlaneMachine(&machine) {
			workerMachines = append(workerMachines, machine)
		}
	}
	return workerMachines, nil
}

// GetBKEClusterAssociateBKEMachines 获取bkecluster 关联的bkeMachine
func GetBKEClusterAssociateBKEMachines(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) ([]bkev1beta1.BKEMachine, error) {
	bkeMachineList := &bkev1beta1.BKEMachineList{}
	filters := GetListFiltersByBKECluster(bkeCluster)
	if err := c.List(ctx, bkeMachineList, filters...); err != nil {
		return nil, err
	}
	return bkeMachineList.Items, nil
}

// GetBKEClusterAssociateCommands 获取bkecluster 关联的command
func GetBKEClusterAssociateCommands(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) ([]agentv1beta1.Command, error) {
	commandsLi := agentv1beta1.CommandList{}
	filters := GetListFiltersByBKECluster(bkeCluster)

	if err := c.List(ctx, &commandsLi, filters...); err != nil {
		return nil, err
	}
	var commands []agentv1beta1.Command
	for _, cmd := range commandsLi.Items {
		if !command.IsOwnerRefCommand(bkeCluster, cmd) {
			continue
		}
		if _, ok := cmd.Annotations[annotation.CommandReconciledAnnotationKey]; ok {
			continue
		}
		if err := command.ValidateCommand(&cmd); err != nil {
			log.Warnf("skip invalid command %s: %v", cmd.Name, err)
			continue
		}
		commands = append(commands, cmd)
	}
	// 按照创建时间排序
	sort.Slice(commands, func(i, j int) bool {
		return commands[i].CreationTimestamp.Before(&commands[j].CreationTimestamp)
	})
	return commands, nil
}

// GetListFiltersByBKECluster 获取bkecluster list关联资源的 client.ListOption
func GetListFiltersByBKECluster(bkecluster *bkev1beta1.BKECluster) []client.ListOption {
	return []client.ListOption{
		client.InNamespace(bkecluster.Namespace),
		client.MatchingLabels{
			clusterv1.ClusterNameLabel: bkecluster.Name,
		},
	}
}

// GetIngressConfig 获取bkecluster ingress(ELB) addon配置

const defaultLauncherWaitTimeout = 5 * time.Minute

// GetLauncherWaitTimeout returns the wait timeout for bkeagent-launcher pods to become Ready.
// Falls back to 5m when the annotation is absent or invalid.
func GetLauncherWaitTimeout(bkeCluster *bkev1beta1.BKECluster) time.Duration {
	v, found := annotation.HasAnnotation(bkeCluster, annotation.LauncherWaitTimeOutAnnotationKey)
	if !found || v == "" {
		return defaultLauncherWaitTimeout
	}
	timeout, err := time.ParseDuration(v)
	if err != nil {
		return defaultLauncherWaitTimeout
	}
	return timeout
}

func GetBootTimeOut(bkeCluster *bkev1beta1.BKECluster) (time.Duration, error) {
	v, found := annotation.HasAnnotation(bkeCluster, annotation.NodeBootWaitTimeOutAnnotationKey)
	if !found {
		// todo 将默认配置写到common项目中
		return 10 * time.Minute, errors.Errorf("not found annotation %s, use default value 10m", annotation.NodeBootWaitTimeOutAnnotationKey)
	}
	timeout, err := time.ParseDuration(v)
	if err != nil {
		return 10 * time.Minute, fmt.Errorf("parse annotation %s value %s error: %w, use default value 10m", annotation.NodeBootWaitTimeOutAnnotationKey, v, err)
	}
	return timeout, nil
}
