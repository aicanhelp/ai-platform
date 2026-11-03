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

package command

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/pkg/errors"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

type Upgrade struct {
	BaseCommand
	Node        *confv1beta1.Node
	BKEConfig   string
	ClusterFrom string
	Phase       confv1beta1.BKEClusterPhase
	BackUpEtcd  bool
	// EtcdVersion is the declarative upgrade target from VersionContext (release bundle).
	// When set, the agent uses it for etcd static pod manifest rendering instead of BKECluster spec.
	EtcdVersion string
	// ControlPlaneConfigOldHash 旧的控制面 hash 结构体（来自 Status.ControlPlaneConfigHash）
	ControlPlaneConfigOldHash confv1beta1.ComponentConfigHash
	// ControlPlaneConfigNewHash 期望的控制面 hash 结构体（来自 Spec 计算）
	ControlPlaneConfigNewHash confv1beta1.ComponentConfigHash
}

func (u *Upgrade) Validate() error {
	if u.BKEConfig == "" {
		return errors.New("bkeConfig is empty")
	}
	if u.Node == nil {
		return errors.New("nodes is empty")
	}
	if u.ClusterFrom == "" {
		u.ClusterFrom = common.BKEClusterFromAnnotationValueBKE
	}
	return u.BaseCommand.validate()
}

func (u *Upgrade) New() error {
	if err := u.Validate(); err != nil {
		return err
	}
	commandName := fmt.Sprintf("%s%s-%d", UpgradeNodeCommandNamePrefix, u.Node.IP, time.Now().Unix())
	bkeConfig := fmt.Sprintf("bkeConfig=%s:%s", u.NameSpace, u.BKEConfig)
	phase := fmt.Sprintf("phase=%s", u.Phase)
	backUpEtcd := fmt.Sprintf("backUpEtcd=%t", u.BackUpEtcd)
	clusterType := fmt.Sprintf("clusterType=%s", u.ClusterFrom)
	execCommand := []string{
		"Kubeadm",
		phase,
		bkeConfig,
		clusterType,
		backUpEtcd,
	}
	if u.EtcdVersion != "" {
		execCommand = append(execCommand, fmt.Sprintf("etcdVersion=%s", u.EtcdVersion))
	}
	if oldHashJSON, err := json.Marshal(u.ControlPlaneConfigOldHash); err == nil && string(oldHashJSON) != "{}" {
		execCommand = append(execCommand, fmt.Sprintf("controlPlaneConfigOldHash=%s", string(oldHashJSON)))
	}
	if newHashJSON, err := json.Marshal(u.ControlPlaneConfigNewHash); err == nil && string(newHashJSON) != "{}" {
		execCommand = append(execCommand, fmt.Sprintf("controlPlaneConfigNewHash=%s", string(newHashJSON)))
	}
	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID:            "upgrade",
			Command:       execCommand,
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffDelay:  3,
			BackoffIgnore: false,
		},
	}

	var nodes bkenode.Nodes
	nodes = append(nodes, *u.Node)
	commandSpec.NodeSelector = getNodeSelector(nodes)

	return u.newCommand(commandName, BKEClusterLabel, commandSpec)
}

// Wait wait for command execution completion displayed
func (u *Upgrade) Wait() (error, []string, []string) {
	err, _, nodes := u.waitCommandComplete()
	return err, nodes.SuccessNodes, nodes.FailedNodes
}
