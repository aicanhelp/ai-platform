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
)

type Hosts struct {
	BaseCommand

	Nodes bkenode.Nodes

	BkeConfigName string
}

func (e *Hosts) Validate() error {
	return ValidateBkeCommand(e.Nodes, e.BkeConfigName, &e.BaseCommand)
}

func (e *Hosts) New() error {
	if err := e.Validate(); err != nil {
		return err
	}
	// aright start create new env command
	bkeConfigStr := GenerateBkeConfigStr(e.NameSpace, e.BkeConfigName)

	commandName := fmt.Sprintf("%s-%d", K8sHostsCommandName, time.Now().Unix())
	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		// init node hosts for node join or delete
		{
			ID: "init and check node env",
			Command: []string{
				"K8sEnvInit",
				"init=true",
				"check=true",
				"scope=hosts",
				bkeConfigStr,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffDelay:  5,
			BackoffIgnore: false,
		},
	}

	commandSpec.NodeSelector = getNodeSelector(e.Nodes)

	return e.newCommand(commandName, BKEClusterLabel, commandSpec)
}
