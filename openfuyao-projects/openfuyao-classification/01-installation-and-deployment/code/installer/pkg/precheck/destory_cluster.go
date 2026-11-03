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

// DestroyClusterFlightCheck 卸载集群检查
type DestroyClusterFlightCheck struct {
	Nodes       cluster.Nodes
	FlightCheck *FlightCheck
}

// NewDestroyClusterFlightCheck is new DestroyClusterFlightCheck
func NewDestroyClusterFlightCheck(nodes cluster.Nodes) *DestroyClusterFlightCheck {
	return &DestroyClusterFlightCheck{
		Nodes:       nodes,
		FlightCheck: NewFlightCheck(nodes),
	}
}

// StartCheck 添加节点检查
func (d *DestroyClusterFlightCheck) StartCheck() error {
	fmt.Println("[preflight] Running pre-flight checks")
	// 检查节点是否能ping通

	for _, node := range d.Nodes.Masters {
		if node.PrivateKeyPath != "" || len(node.Password) != 0 {
			continue
		}
		node.InputPasswd()
	}

	for _, node := range d.Nodes.Workers {
		if node.PrivateKeyPath != "" || len(node.Password) != 0 {
			continue
		}
		node.InputPasswd()
	}

	return d.FlightCheck.allNodePingCheck()
}
