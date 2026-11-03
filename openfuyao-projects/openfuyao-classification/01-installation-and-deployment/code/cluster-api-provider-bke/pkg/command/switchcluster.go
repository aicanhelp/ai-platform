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

	"github.com/google/uuid"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

type Switch struct {
	BaseCommand

	Nodes       bkenode.Nodes
	ClusterName string
}

func (s *Switch) Validate() error {
	if s.ClusterName == "" {
		return fmt.Errorf("cluster name is empty")
	}
	return s.BaseCommand.validate()
}

func (s *Switch) New() error {
	if err := s.Validate(); err != nil {
		return err
	}

	kubeconfig := fmt.Sprintf("kubeconfig=%s/%s-kubeconfig", s.NameSpace, s.ClusterName)
	clusterName := fmt.Sprintf("clusterName=%s", s.ClusterName)
	commandName := fmt.Sprintf("%s-%s-%d", SwitchClusterCommandNamePrefix, s.ClusterName, time.Now().Unix())
	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: uuid.New().String(),
			Command: []string{
				"SwitchCluster",
				kubeconfig,
				clusterName,
			},
			Type: agentv1beta1.CommandBuiltIn,
		},
	}
	commandSpec.NodeSelector = getNodeSelector(s.Nodes)
	return s.newCommand(commandName, BKEClusterLabel, commandSpec)
}

func (s *Switch) Wait() (error, []string, []string) {
	err, complete, nodes := s.waitCommandComplete()
	// means all command not executed
	if !complete && len(nodes.SuccessNodes) == 0 && len(nodes.FailedNodes) == 0 {
		for _, node := range s.Nodes {
			nodes.FailedNodes = append(nodes.FailedNodes, node.Hostname)
		}
	}
	return err, nodes.SuccessNodes, nodes.FailedNodes
}
