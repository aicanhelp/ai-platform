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
	"strings"
	"time"

	"github.com/pkg/errors"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

type Reset struct {
	BaseCommand
	Node      *confv1beta1.Node
	BKEConfig string
	Extra     []string

	// DeepRestore is used to restore the node to the initial state
	DeepRestore bool
}

func (r *Reset) Validate() error {
	if r.BKEConfig == "" {
		return errors.New("bkeConfig is empty")
	}
	if r.Node == nil {
		return errors.New("nodes is empty")
	}
	return r.BaseCommand.validate()
}

func (r *Reset) New() error {
	if err := r.Validate(); err != nil {
		return err
	}
	commandName := fmt.Sprintf("%s%s-%d", ResetNodeCommandNamePrefix, r.Node.IP, time.Now().Unix())
	bkeConfig := fmt.Sprintf("bkeConfig=%s:%s", r.NameSpace, r.BKEConfig)
	extra := fmt.Sprintf("extra=%s", strings.Join(r.Extra, ","))

	scope := "scope=cert,manifests,kubelet,container,containerRuntime,source,extra,global-cert,shutdown"

	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "reset",
			Command: []string{
				"Reset",
				bkeConfig,
				scope,
				extra,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: false,
		},
	}

	var nodes bkenode.Nodes
	nodes = append(nodes, *r.Node)

	commandSpec.NodeSelector = getNodeSelector(nodes)
	return r.newCommand(commandName, BKEMachineLabel, commandSpec)
}

// Wait wait for command execution completion displayed
func (r *Reset) Wait() (error, []string, []string) {
	err, _, nodes := r.waitCommandComplete()
	return err, nodes.SuccessNodes, nodes.FailedNodes
}
