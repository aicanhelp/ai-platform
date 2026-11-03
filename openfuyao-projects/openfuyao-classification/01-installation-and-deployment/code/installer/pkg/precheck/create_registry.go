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
	"installer/pkg/fuyao/v1beta1/cluster"
)

const (
	registryNodeCpuCount   = 2
	registryNodeMemorySize = 4
)

// CreateRegistryFlightCheck 创建镜像仓前飞行检查
type CreateRegistryFlightCheck struct {
	Nodes cluster.Nodes

	FlightCheck *FlightCheck
}

// NewCreateRegistryFlightCheck is new CreateRegistryFlightCheck
func NewCreateRegistryFlightCheck(nodes cluster.Nodes) *CreateRegistryFlightCheck {
	return &CreateRegistryFlightCheck{
		Nodes:       nodes,
		FlightCheck: NewFlightCheck(nodes),
	}
}

// StartCheck 执行飞行检查
func (c *CreateRegistryFlightCheck) StartCheck() error {
	fmt.Println("[preflight] Running pre-flight checks")

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

	if err := c.FlightCheck.registryNodesToolCheck(); err != nil {
		return err
	}

	if err := c.FlightCheck.registryNodesCupCheck(registryNodeCpuCount); err != nil {
		return err
	}

	if err := c.FlightCheck.registryNodesMemoryCheck(registryNodeMemorySize); err != nil {
		return err
	}

	return nil
}
