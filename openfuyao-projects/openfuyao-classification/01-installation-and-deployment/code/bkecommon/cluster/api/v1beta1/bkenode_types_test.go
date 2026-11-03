/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package v1beta1

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBKENode_SetState(t *testing.T) {
	node := &BKENode{}

	node.SetState(NodeReady)
	assert.Equal(t, NodeReady, node.Status.State)

	node.SetState(NodeFailed)
	assert.Equal(t, NodeFailed, node.Status.State)
}

func TestBKENode_SetStateWithMessage(t *testing.T) {
	node := &BKENode{}

	node.SetStateWithMessage(NodeFailed, "connection timeout")
	assert.Equal(t, NodeFailed, node.Status.State)
	assert.Equal(t, "connection timeout", node.Status.Message)
}

func TestBKENode_ToNode(t *testing.T) {
	bkeNode := &BKENode{
		Spec: BKENodeSpec{
			Role:     []string{"master", "etcd"},
			IP:       "192.168.1.100",
			Port:     "22",
			Username: "root",
			Password: "encrypted_password",
			Hostname: "master-1",
			Labels: []Label{
				{Key: "env", Value: "prod"},
			},
		},
	}

	node := bkeNode.ToNode()

	assert.Equal(t, bkeNode.Spec.Role, node.Role)
	assert.Equal(t, bkeNode.Spec.IP, node.IP)
	assert.Equal(t, bkeNode.Spec.Port, node.Port)
	assert.Equal(t, bkeNode.Spec.Username, node.Username)
	assert.Equal(t, bkeNode.Spec.Password, node.Password)
	assert.Equal(t, bkeNode.Spec.Hostname, node.Hostname)
	assert.Equal(t, bkeNode.Spec.Labels, node.Labels)
}

func TestFromNode(t *testing.T) {
	node := Node{
		Role:     []string{"node"},
		IP:       "192.168.1.101",
		Port:     "22",
		Username: "admin",
		Password: "password",
		Hostname: "worker-1",
		Labels: []Label{
			{Key: "role", Value: "worker"},
		},
	}

	spec := FromNode(node)

	assert.Equal(t, node.Role, spec.Role)
	assert.Equal(t, node.IP, spec.IP)
	assert.Equal(t, node.Port, spec.Port)
	assert.Equal(t, node.Username, spec.Username)
	assert.Equal(t, node.Password, spec.Password)
	assert.Equal(t, node.Hostname, spec.Hostname)
	assert.Equal(t, node.Labels, spec.Labels)
}

func TestNodeStateConstants(t *testing.T) {
	// 验证state常量的值
	assert.Equal(t, NodeState("NotReady"), NodeNotReady)
	assert.Equal(t, NodeState("Ready"), NodeReady)
	assert.Equal(t, NodeState("Pending"), NodePending)
	assert.Equal(t, NodeState("Failed"), NodeFailed)
	assert.Equal(t, NodeState("Deleting"), NodeDeleting)
	assert.Equal(t, NodeState("Upgrading"), NodeUpgrading)
	assert.Equal(t, NodeState("Provisioned"), NodeProvisioned)
}
