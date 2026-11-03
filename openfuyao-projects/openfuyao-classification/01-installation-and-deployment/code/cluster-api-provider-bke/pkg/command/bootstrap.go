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
	"fmt"
	"time"

	"github.com/pkg/errors"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

type Bootstrap struct {
	BaseCommand

	Node      *confv1beta1.Node
	BKEConfig string
	Phase     confv1beta1.BKEClusterPhase
}

func (b *Bootstrap) Validate() error {

	if b.Phase == "" {
		return errors.New("phase is empty")
	}
	if b.BKEConfig == "" {
		return errors.New("bkeConfig is empty")
	}
	return b.BaseCommand.validate()
}

func (b *Bootstrap) New() error {
	if err := b.Validate(); err != nil {
		return err
	}
	commandName := fmt.Sprintf("%s%s-%d", BootstrapCommandNamePrefix, b.Node.IP, time.Now().Unix())
	bkeConfig := fmt.Sprintf("bkeConfig=%s:%s", b.NameSpace, b.BKEConfig)
	phase := fmt.Sprintf("phase=%s", b.Phase)

	customLabel := ""
	switch b.Phase {
	case bkev1beta1.InitControlPlane:
		customLabel = MasterInitCommandLabel
	case bkev1beta1.JoinControlPlane:
		customLabel = MasterJoinCommandLabel
	case bkev1beta1.JoinWorker:
		customLabel = WorkerJoinCommandLabel
	default:
		// Handle unexpected phase values
	}

	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "check container runtime",
			Command: []string{
				"K8sEnvInit",
				"init=false",
				"check=true",
				"scope=runtime",
				bkeConfig,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffDelay:  3,
			BackoffIgnore: false,
		},
		{
			ID: "bootstrap",
			Command: []string{
				"Kubeadm",
				phase,
				bkeConfig,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffDelay:  3,
			BackoffIgnore: false,
		},
	}
	commandSpec.TTLSecondsAfterFinished = 0

	var nodes bkenode.Nodes
	nodes = append(nodes, *b.Node)
	commandSpec.NodeSelector = getNodeSelector(nodes)

	return b.newCommand(commandName, BKEMachineLabel, commandSpec, customLabel)

}
