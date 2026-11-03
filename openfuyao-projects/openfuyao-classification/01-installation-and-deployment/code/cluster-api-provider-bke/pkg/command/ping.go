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

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

type Ping struct {
	BaseCommand
	Nodes bkenode.Nodes
}

func (p *Ping) Validate() error {
	return p.validate()
}

func (p *Ping) New() error {
	if err := p.Validate(); err != nil {
		return err
	}
	commandName := fmt.Sprintf("%s-%d", PingCommandNamePrefix, time.Now().Unix())
	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "ping",
			Command: []string{
				"Ping",
			},
			Type:         agentv1beta1.CommandBuiltIn,
			BackoffDelay: 3,
		},
	}

	commandSpec.NodeSelector = getNodeSelector(p.Nodes)
	return p.newCommand(commandName, BKEClusterLabel, commandSpec)
}

// Wait wait for command execution completion displayed
func (p *Ping) Wait() (error, []string, []string) {
	err, complete, nodes := p.waitCommandComplete()
	// means all command not executed
	if !complete && len(nodes.FailedNodes) == 0 {
		successIPs := make([]string, 0, len(nodes.SuccessNodes))
		for _, s := range nodes.SuccessNodes {
			successIPs = append(successIPs, utils.GetNodeIPFromCommandWaitResult(s))
		}
		for _, node := range p.Nodes {
			if utils.ContainsString(successIPs, node.IP) {
				continue
			}
			nodes.FailedNodes = append(nodes.FailedNodes, node.IP)
		}
	}
	return err, nodes.SuccessNodes, nodes.FailedNodes
}
