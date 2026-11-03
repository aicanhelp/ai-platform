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
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

type HA struct {
	BaseCommand
	MasterNodes              bkenode.Nodes
	IngressNodes             bkenode.Nodes
	IngressVIP               string
	ControlPlaneEndpointPort int32
	ControlPlaneEndpointVIP  string
	ThirdImageRepo           string
	FuyaoImageRepo           string
	ManifestsDir             string
	VirtualRouterId          string
	WaitVIP                  bool

	isMasterHa bool
}

func (l *HA) Validate() error {
	if l.ThirdImageRepo == "" {
		return errors.New("ThirdImageRepo is empty")
	}
	if l.FuyaoImageRepo == "" {
		return errors.New("FuyaoImageRepo is empty")
	}
	if l.ManifestsDir == "" {
		return errors.New("manifestsDir is empty")
	}

	if l.MasterNodes.Length() != 0 && l.IngressNodes.Length() != 0 {
		return errors.New("loadbalance command except configure one type of Ha, but both types (master ha,ingress ha) are configured")

	}

	if l.MasterNodes.Length() != 0 {
		l.isMasterHa = true
		if l.ControlPlaneEndpointPort == 0 {
			return errors.New("controlPlaneEndpointPort is empty")
		}
		if l.ControlPlaneEndpointVIP == "" {
			return errors.New("controlPlaneEndpointVIP is empty")
		}
	}
	if l.IngressNodes.Length() != 0 {
		l.isMasterHa = false
		if l.IngressVIP == "" {
			return errors.New("ingressVIP is empty")
		}
	}

	if l.MasterNodes.Length() == 0 && l.IngressNodes.Length() == 0 {
		return errors.New("loadbalance command except at least one node but got 0")
	}

	return l.BaseCommand.validate()
}

// New 创建负载均衡命令
func (l *HA) New() error {
	if err := l.Validate(); err != nil {
		return err
	}

	// 准备基本参数
	manifestsDirParam := fmt.Sprintf("manifestsDir=%s", l.ManifestsDir)
	thirdImageRepoParam := fmt.Sprintf("thirdImageRepo=%s", l.ThirdImageRepo)
	fuyaoImageRepoParam := fmt.Sprintf("fuyaoImageRepo=%s", l.FuyaoImageRepo)

	commandName := fmt.Sprintf("%s-%d", HACommandName, time.Now().Unix())
	commandSpec := GenerateDefaultCommandSpec()

	// 根据HA类型设置命令参数
	if l.isMasterHa {
		l.setupMasterHACommand(commandSpec, manifestsDirParam, thirdImageRepoParam, fuyaoImageRepoParam)
		commandSpec.NodeSelector = getNodeSelector(l.MasterNodes)
	} else {
		l.setupIngressHACommand(commandSpec, manifestsDirParam, thirdImageRepoParam, fuyaoImageRepoParam)
		commandSpec.NodeSelector = getNodeSelector(l.IngressNodes)
	}

	return l.newCommand(commandName, BKEClusterLabel, commandSpec)
}

// setupMasterHACommand 设置Master HA命令
func (l *HA) setupMasterHACommand(commandSpec *agentv1beta1.CommandSpec, manifestsDirParam, thirdImageRepoParam, fuyaoImageRepoParam string) {
	haNodesParam := l.getHaNodesParam(l.MasterNodes)
	controlPlaneEndpointPortParam := fmt.Sprintf("controlPlaneEndpointPort=%d", l.ControlPlaneEndpointPort)
	controlPlaneEndpointVIPParam := fmt.Sprintf("controlPlaneEndpointVIP=%s", l.ControlPlaneEndpointVIP)
	virtualRouterIdParam := fmt.Sprintf("virtualRouterId=%s", l.VirtualRouterId)
	waitVIPParam := fmt.Sprintf("wait=%t", l.WaitVIP)

	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "master-ha",
			Command: []string{
				"HA",
				haNodesParam,
				controlPlaneEndpointPortParam,
				controlPlaneEndpointVIPParam,
				virtualRouterIdParam,
				thirdImageRepoParam,
				fuyaoImageRepoParam,
				manifestsDirParam,
				waitVIPParam,
			},
			Type: agentv1beta1.CommandBuiltIn,
		},
	}
}

// setupIngressHACommand 设置Ingress HA命令
func (l *HA) setupIngressHACommand(commandSpec *agentv1beta1.CommandSpec, manifestsDirParam, thirdImageRepoParam, fuyaoImageRepoParam string) {
	haNodesParam := l.getHaNodesParam(l.IngressNodes)
	ingressVIPParam := fmt.Sprintf("ingressVIP=%s", l.IngressVIP)
	virtualRouterIdParam := fmt.Sprintf("virtualRouterId=%s", l.VirtualRouterId)
	waitVIPParam := fmt.Sprintf("wait=%t", l.WaitVIP)

	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "ingress-ha",
			Command: []string{
				"HA",
				haNodesParam,
				ingressVIPParam,
				thirdImageRepoParam,
				fuyaoImageRepoParam,
				virtualRouterIdParam,
				manifestsDirParam,
				waitVIPParam,
			},
			Type: agentv1beta1.CommandBuiltIn,
		},
	}
}

// getHaNodesParam 获取HA节点参数
func (l *HA) getHaNodesParam(nodes bkenode.Nodes) string {
	var haNodesList []string
	for _, node := range nodes {
		haNodesList = append(haNodesList, fmt.Sprintf("%s:%s", node.Hostname, node.IP))
	}
	return fmt.Sprintf("haNodes=%s", strings.Join(haNodesList, ","))
}

func (l *HA) Wait() (error, []string, []string) {
	err, complete, nodes := l.waitCommandComplete()
	// means all command not executed
	if !complete && len(nodes.FailedNodes) == 0 {
		for _, node := range l.MasterNodes {
			if utils.ContainsString(nodes.SuccessNodes, node.Hostname) {
				continue
			}
			nodes.FailedNodes = append(nodes.FailedNodes, node.Hostname)
		}
	}
	return err, nodes.SuccessNodes, nodes.FailedNodes
}
