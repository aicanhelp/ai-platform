/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

// DistributeKubeProxyKubeConfig distribute kube-proxy kubeconfig to node
// 在对存量bocloud集群进行节点扩容时，需要将kube-proxy的kubeconfig分发到新加入的节点上
// 注意存量集群的kube-proxy的kubeconfig是用的admin-kubeconfig，只是换了个名字，且改kubeconfig BKE不会创建了是直接使用的admin-kubeconfig
func DistributeKubeProxyKubeConfig(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodes bkenode.Nodes, log *bkev1beta1.BKELogger) error {
	// 使用bkeagent的 command做分发
	distributeCommandName := fmt.Sprintf("distribute-kubeproxy-kubeconfig-%d", time.Now().Unix())
	distributeCommandSpec := command.GenerateDefaultCommandSpec()
	distributeCommandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "distribute kube-proxy kubeconfig",
			Command: []string{
				fmt.Sprintf("secret:%s/%s-kubeconfig:ro:/etc/kubernetes/kube-proxy.kubeconfig", bkeCluster.Namespace, bkeCluster.Name),
			},
			Type:          agentv1beta1.CommandKubernetes,
			BackoffDelay:  3,
			BackoffIgnore: false,
		},
	}
	distributeCommand := command.Custom{
		BaseCommand: command.BaseCommand{
			Ctx:             ctx,
			Client:          c,
			Scheme:          c.Scheme(),
			OwnerObj:        bkeCluster,
			NameSpace:       bkeCluster.Namespace,
			ClusterName:     bkeCluster.Name,
			RemoveAfterWait: true,
			Unique:          true,
		},
		Nodes:        nodes,
		CommandName:  distributeCommandName,
		CommandSpec:  distributeCommandSpec,
		CommandLabel: command.BKEClusterLabel,
	}

	if err := distributeCommand.New(); err != nil {
		return err
	}

	err, _, failed := distributeCommand.Wait()
	if err != nil {
		return fmt.Errorf("failed to wait command %q: %w", distributeCommandName, err)
	}
	if failed != nil || len(failed) > 0 {
		commandErrs, cmdErr := LogCommandFailed(*distributeCommand.Command, failed, log, constant.BocloudClusterMasterCertDistributionFailedReason)
		MarkNodeStatusByCommandErrs(distributeCommand.Ctx, distributeCommand.Client, bkeCluster, commandErrs)
		if cmdErr != nil {
			return fmt.Errorf("failed to distribute kube-proxy kubeconfig on flow node %q: %w", failed, cmdErr)
		}
		return fmt.Errorf("failed to distribute kube-proxy kubeconfig on flow node %q", failed)
	}
	return nil
}
