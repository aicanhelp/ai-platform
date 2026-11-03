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
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

type Collect struct {
	BaseCommand

	Node                *confv1beta1.Node
	EtcdCertificatesDir string
	K8sCertificatesDir  string
}

func (c *Collect) Validate() error {
	if c.EtcdCertificatesDir == "" {
		return fmt.Errorf("etcdCertificatesDir is empty")
	}
	if c.K8sCertificatesDir == "" {
		return fmt.Errorf("k8sCertificatesDir is empty")
	}
	if c.Node == nil {
		return fmt.Errorf("node is empty")
	}
	if c.ClusterName == "" {
		return fmt.Errorf("cluster name is empty")
	}
	return c.BaseCommand.validate()
}

func (c *Collect) New() error {
	if err := c.Validate(); err != nil {
		return err
	}
	clusterName := fmt.Sprintf("clusterName=%s", c.ClusterName)
	nameSpace := fmt.Sprintf("namespace=%s", c.NameSpace)
	k8sCertDir := fmt.Sprintf("k8sCertDir=%s", c.K8sCertificatesDir)
	etcdCertDir := fmt.Sprintf("etcdCertDir=%s", c.EtcdCertificatesDir)
	commandName := fmt.Sprintf("%s-%s-%d", CollectCertCommandNamePrefix, c.ClusterName, time.Now().Unix())
	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: uuid.New().String(),
			Command: []string{
				"Collect",
				clusterName,
				nameSpace,
				k8sCertDir,
				etcdCertDir,
			},
			Type: agentv1beta1.CommandBuiltIn,
		},
	}

	var nodes bkenode.Nodes
	nodes = append(nodes, *c.Node)

	commandSpec.NodeSelector = getNodeSelector(nodes)
	return c.newCommand(commandName, BKEClusterLabel, commandSpec)
}

func (c *Collect) Wait() (error, []string, []string) {
	err, _, nodes := c.waitCommandComplete()
	return err, nodes.SuccessNodes, nodes.FailedNodes
}
