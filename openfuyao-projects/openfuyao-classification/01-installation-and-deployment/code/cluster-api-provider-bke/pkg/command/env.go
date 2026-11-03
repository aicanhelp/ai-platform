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

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

type ENV struct {
	BaseCommand

	Nodes bkenode.Nodes

	BkeConfigName     string
	ContainerdVersion string

	Extra []string

	ExtraHosts []string

	DryRun bool

	PrePullImage bool

	DeepRestore bool

	IsImmutableOS bool // worker 节点是否为不可变 OS（KubeOS）
}

func (e *ENV) Validate() error {
	return ValidateBkeCommand(e.Nodes, e.BkeConfigName, &e.BaseCommand)
}

func (e *ENV) NewConatinerdReset() error {
	if err := e.Validate(); err != nil {
		return err
	}
	commandName := K8sContainerdResetCommandName
	bkeConfigStr := fmt.Sprintf("bkeConfig=%s:%s", e.NameSpace, e.BkeConfigName)
	extra := fmt.Sprintf("extra=%s", strings.Join(e.Extra, ","))

	commandName = fmt.Sprintf("%s-%d", commandName, time.Now().Unix())
	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.TTLSecondsAfterFinished = 0
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "reset",
			Command: []string{
				"Reset",
				bkeConfigStr,
				"scope=containerd-cfg",
				extra,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: true,
		},
	}
	commandSpec.NodeSelector = getNodeSelector(e.Nodes)

	return e.newCommand(commandName, BKEClusterLabel, commandSpec)
}

func (e *ENV) NewConatinerdRedeploy() error {
	if err := e.Validate(); err != nil {
		return err
	}
	commandName := K8sContainerdRedeployCommandName
	bkeConfigStr := fmt.Sprintf("bkeConfig=%s:%s", e.NameSpace, e.BkeConfigName)

	commandName = fmt.Sprintf("%s-%d", commandName, time.Now().Unix())
	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.TTLSecondsAfterFinished = 0
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		{
			ID: "init and check node env",
			Command: []string{
				"K8sEnvInit",
				"init=true",
				"check=true",
				"scope=runtime",
				bkeConfigStr,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: false,
			BackoffDelay:  5,
		},
	}
	if e.ContainerdVersion != "" {
		commandSpec.Commands[0].Command = append(commandSpec.Commands[0].Command,
			fmt.Sprintf("containerdVersion=%s", e.ContainerdVersion))
	}
	commandSpec.NodeSelector = getNodeSelector(e.Nodes)

	return e.newCommand(commandName, BKEClusterLabel, commandSpec)
}

// New 创建环境初始化命令，镜像预拉取已并入主初始化命令，避免再额外下发单独的预拉取命令。
func (e *ENV) New() error {
	if err := e.Validate(); err != nil {
		return err
	}

	commandName := e.getCommandName()
	bkeConfigStr := GenerateBkeConfigStr(e.NameSpace, e.BkeConfigName)
	extra := fmt.Sprintf("extra=%s", strings.Join(e.Extra, ","))
	extraHosts := fmt.Sprintf("extraHosts=%s", strings.Join(e.ExtraHosts, ","))

	scope := e.getScope()

	commandName = fmt.Sprintf("%s-%d", commandName, time.Now().Unix())
	commandSpec := e.buildCommandSpec(bkeConfigStr, extra, extraHosts, scope)

	if e.DryRun {
		commandSpec.Commands = commandSpec.Commands[:1]
	}

	commandSpec.NodeSelector = getNodeSelector(e.Nodes)

	return e.newCommand(commandName, BKEClusterLabel, commandSpec)
}

// getCommandName 获取命令名称，根据DryRun标志决定使用哪个名称
func (e *ENV) getCommandName() string {
	commandName := K8sEnvCommandName
	if e.DryRun {
		commandName = K8sEnvDryRunCommandName
	}
	return commandName
}

// getScope 获取 reset 作用域字符串
func (e *ENV) getScope() string {
	if e.IsImmutableOS {
		// 不可变 OS：跳过 containerRuntime 和 extra
		return "scope=cert,manifests,container,kubelet"
	}
	if e.DeepRestore {
		return "scope=cert,manifests,container,kubelet,containerRuntime,extra"
	}
	return "scope=cert,manifests,container,kubelet,extra"
}

// getInitScope 获取环境初始化作用域字符串，镜像预拉取合并在主初始化流程内执行。
func (e *ENV) getInitScope() string {
	if e.IsImmutableOS {
		// 不可变 OS：跳过 firewall/selinux/swap/httpRepo/extra，保留 image 以预拉取 Calico 节点镜像。
		return "scope=time,hosts,dns,kernel,runtime,iptables,registry,image"
	}
	return "scope=time,hosts,dns,kernel,firewall,selinux,swap,httpRepo,runtime,iptables,registry,image,extra"
}

// buildCommandSpec 构建环境初始化命令规格，主初始化 scope 包含 image 以覆盖 Calico 节点镜像预拉取。
func (e *ENV) buildCommandSpec(bkeConfigStr, extra, extraHosts, scope string) *agentv1beta1.CommandSpec {
	commandSpec := GenerateDefaultCommandSpec()
	commandSpec.TTLSecondsAfterFinished = 0
	commandSpec.Commands = []agentv1beta1.ExecCommand{
		// Check whether the host hardware resources are sufficient to run k8s
		{
			ID: "node hardware resources check",
			Command: []string{
				"K8sEnvInit",
				"init=true",
				"check=true",
				"scope=node",
				bkeConfigStr,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: false,
		},
		// reset node
		{
			ID: "reset",
			Command: []string{
				"Reset",
				bkeConfigStr,
				scope,
				extra,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffIgnore: true,
		},
		// init node env to run k8s
		{
			ID: "init and check node env",
			Command: []string{
				"K8sEnvInit",
				"init=true",
				"check=true",
				e.getInitScope(),
				bkeConfigStr,
				extraHosts,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffDelay:  5,
			BackoffIgnore: false,
		},
	}
	return commandSpec
}

// createPrePullImageCommand 创建预拉取镜像命令
func (e *ENV) createPrePullImageCommand(bkeConfigStr string) {
	// only send this command in first deploy
	// Failure to execute this command will not affect the deployment of the entire cluster
	prePullImagesCommandName := fmt.Sprintf("%s-%d", K8sImagePrePullCommandName, time.Now().Unix())
	prePullImagesCommandSpec := GenerateDefaultCommandSpec()
	prePullImagesCommandSpec.Commands = []agentv1beta1.ExecCommand{
		// pre pull images command
		{
			ID: "pre pull images",
			Command: []string{
				"K8sEnvInit",
				"init=true",
				"check=true",
				"scope=image",
				bkeConfigStr,
			},
			Type:          agentv1beta1.CommandBuiltIn,
			BackoffDelay:  15,
			BackoffIgnore: true,
		},
	}
	selector := getNodeSelector(e.Nodes)
	// exclude master node which will be init
	delete(selector.MatchLabels, e.Nodes.Master()[0].IP)
	prePullImagesCommandSpec.NodeSelector = selector

	// ignore error
	_ = e.newCommand(prePullImagesCommandName, BKEClusterLabel, prePullImagesCommandSpec)
}

func (e *ENV) Wait() (error, []string, []string) {
	err, complete, nodes := e.waitCommandComplete()
	// means all command not executed
	if !complete && len(nodes.FailedNodes) == 0 {
		for _, node := range e.Nodes {
			if utils.ContainsString(nodes.SuccessNodes, node.Hostname) {
				continue
			}
			nodes.FailedNodes = append(nodes.FailedNodes, node.Hostname)
		}
	}
	return err, nodes.SuccessNodes, nodes.FailedNodes
}
