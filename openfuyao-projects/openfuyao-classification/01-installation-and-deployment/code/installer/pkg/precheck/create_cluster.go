/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package precheck

import (
	"fmt"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
)

// CreateClusterFlightCheck 创建集群前飞行检查
type CreateClusterFlightCheck struct {
	EnableFuyaoSystem bool
	Nodes             cluster.Nodes

	FlightCheck *FlightCheck
}

// NewCreateClusterFlightCheck is new CreateClusterFlightCheck
func NewCreateClusterFlightCheck(nodes cluster.Nodes, enableFuyaoSystem bool) *CreateClusterFlightCheck {
	return &CreateClusterFlightCheck{
		EnableFuyaoSystem: enableFuyaoSystem,
		Nodes:             nodes,
		FlightCheck:       NewFlightCheck(nodes),
	}
}

func (c *CreateClusterFlightCheck) nodeMemberLimit() error {
	if len(c.Nodes.Masters) == constants.MasterCount {
		return fmt.Errorf("high availability deployment requires at least three control planes")
	}
	return nil
}

// allInOneCheck 单节点检查
func (c *CreateClusterFlightCheck) allInOneCheck() error {
	cpuStandard, memoryStandard := 2, 4
	if c.EnableFuyaoSystem {
		cpuStandard, memoryStandard = 4, 8
	}
	err := c.FlightCheck.allNodeCupCheck(cpuStandard)
	if err != nil {
		return err
	}

	err = c.FlightCheck.allNodeMemoryCheck(memoryStandard)
	if err != nil {
		return err
	}

	err = c.FlightCheck.allNodeToolCheck()
	if err != nil {
		return err
	}
	return nil
}

// isHasSameIP 检查节点ip地址是否相同
func (c *CreateClusterFlightCheck) isHasSameIP() error {
	ipMap := make(map[string]bool)
	nodes := make([]*cluster.Node, 0)
	nodes = append(nodes, c.Nodes.Workers...)
	nodes = append(nodes, c.Nodes.Masters...)

	return c.FlightCheck.isHasSameIP(nodes, ipMap)
}

// checkOrSetHostName 检查节点hostname是否相同
func (c *CreateClusterFlightCheck) checkOrSetHostName() error {
	hostnameMap := make(map[string]string)
	nodes := make([]*cluster.Node, 0)
	nodes = append(nodes, c.Nodes.Workers...)
	nodes = append(nodes, c.Nodes.Masters...)

	return c.FlightCheck.checkOrSetHostName(nodes, hostnameMap)
}

func (c *CreateClusterFlightCheck) manyNodesCheck() error {
	checkFuncList := []func() error{c.FlightCheck.allNodePingCheck, c.nodeMemberLimit, c.isHasSameIP,
		c.checkOrSetHostName, c.FlightCheck.checkALLNodesTimeDiff, c.FlightCheck.checkAllNodesOsName,
		c.FlightCheck.checkWorkerCPUMemory, c.FlightCheck.allNodeToolCheck}

	for _, checkFunc := range checkFuncList {
		if err := checkFunc(); err != nil {
			return err
		}
	}

	cpuStandard, memoryStandard := computeMasterNodeResource(len(c.Nodes.Workers))
	if err := c.FlightCheck.masterNodesCupCheck(cpuStandard); err != nil {
		return err
	}
	if err := c.FlightCheck.masterNodesMemoryCheck(memoryStandard); err != nil {
		return err
	}

	return nil
}

// StartCheck 执行安装飞行检查
func (c *CreateClusterFlightCheck) StartCheck() error {
	fmt.Println("[preflight] Running pre-flight checks")
	if len(c.Nodes.Masters) == 1 && len(c.Nodes.Workers) == 0 && len(c.Nodes.Registry) == 0 {
		return c.allInOneCheck()
	}

	for _, node := range c.Nodes.Masters {
		if node.PrivateKeyPath != "" || len(node.Password) != 0 {
			continue
		}
		node.InputPasswd()
	}

	for _, node := range c.Nodes.Workers {
		if node.PrivateKeyPath != "" || len(node.Password) != 0 {
			continue
		}
		node.InputPasswd()
	}

	return c.manyNodesCheck()
}
