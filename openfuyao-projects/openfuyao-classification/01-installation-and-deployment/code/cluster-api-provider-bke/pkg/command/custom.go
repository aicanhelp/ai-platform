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
	"github.com/pkg/errors"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

type Custom struct {
	BaseCommand
	Nodes        bkenode.Nodes
	CommandName  string
	CommandSpec  *agentv1beta1.CommandSpec
	CommandLabel string
}

func (c *Custom) Validate() error {
	if err := c.BaseCommand.validate(); err != nil {
		return err
	}
	if len(c.Nodes) == 0 {
		return errors.New("nodes is empty")
	}
	if c.CommandName == "" {
		return errors.New("commandName is empty")
	}
	if c.CommandSpec == nil {
		return errors.New("commandSpec is empty")
	}
	return nil
}

func (c *Custom) New() error {
	if err := c.Validate(); err != nil {
		return err
	}
	commandName := c.CommandName
	commandSpec := c.CommandSpec
	commandSpec.NodeSelector = getNodeSelector(c.Nodes)
	return c.newCommand(commandName, c.CommandLabel, commandSpec)

}

func (c *Custom) Wait() (error, []string, []string) {
	err, complete, nodes := c.waitCommandComplete()
	// means all command not executed
	if !complete && len(nodes.FailedNodes) == 0 {
		for _, node := range c.Nodes {
			if utils.ContainsString(nodes.SuccessNodes, node.Hostname) {
				continue
			}
			nodes.FailedNodes = append(nodes.FailedNodes, node.Hostname)
		}
	}
	return err, nodes.SuccessNodes, nodes.FailedNodes
}
