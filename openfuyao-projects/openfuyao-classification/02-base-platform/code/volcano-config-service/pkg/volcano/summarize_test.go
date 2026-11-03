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

package volcano

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"volcano-config-service/pkg/controller"
)

func TestSummarizeNodePolicyStatus_EmptyNodes(t *testing.T) {
	result := summarizeNodePolicyStatus([]controller.NodeConfiguration{}, func(node controller.NodeConfiguration) bool {
		return node.EnableStaticCPUManager
	})
	assert.Equal(t, policyStatusClose, result)
}

func TestSummarizeNodePolicyStatus_AllEnabled(t *testing.T) {
	nodes := []controller.NodeConfiguration{
		{NodeName: "node1", EnableStaticCPUManager: true},
		{NodeName: "node2", EnableStaticCPUManager: true},
	}
	result := summarizeNodePolicyStatus(nodes, func(node controller.NodeConfiguration) bool {
		return node.EnableStaticCPUManager
	})
	assert.Equal(t, policyStatusOpen, result)
}

func TestSummarizeNodePolicyStatus_AllDisabled(t *testing.T) {
	nodes := []controller.NodeConfiguration{
		{NodeName: "node1", EnableStaticCPUManager: false},
		{NodeName: "node2", EnableStaticCPUManager: false},
	}
	result := summarizeNodePolicyStatus(nodes, func(node controller.NodeConfiguration) bool {
		return node.EnableStaticCPUManager
	})
	assert.Equal(t, policyStatusClose, result)
}

func TestSummarizeNodePolicyStatus_Partial(t *testing.T) {
	nodes := []controller.NodeConfiguration{
		{NodeName: "node1", EnableStaticCPUManager: true},
		{NodeName: "node2", EnableStaticCPUManager: false},
	}
	result := summarizeNodePolicyStatus(nodes, func(node controller.NodeConfiguration) bool {
		return node.EnableStaticCPUManager
	})
	assert.Equal(t, policyStatusPartial, result)
}

func TestSummarizeNodePolicyStatus_SingleEnabled(t *testing.T) {
	nodes := []controller.NodeConfiguration{
		{NodeName: "node1", EnableNUMADistance: true},
	}
	result := summarizeNodePolicyStatus(nodes, func(node controller.NodeConfiguration) bool {
		return node.EnableNUMADistance
	})
	assert.Equal(t, policyStatusOpen, result)
}

func TestSummarizeNodePolicyStatus_SingleDisabled(t *testing.T) {
	nodes := []controller.NodeConfiguration{
		{NodeName: "node1", EnableNUMADistance: false},
	}
	result := summarizeNodePolicyStatus(nodes, func(node controller.NodeConfiguration) bool {
		return node.EnableNUMADistance
	})
	assert.Equal(t, policyStatusClose, result)
}
