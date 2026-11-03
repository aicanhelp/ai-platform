/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package node

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkecommon/cluster/api/v1beta1"
)

var (
	tNodes = []v1beta1.Node{
		{
			Hostname: "node1",
			Role:     []string{"master", "worker"},
			IP:       "111",
		},
		{
			Hostname: "node2",
			Role:     []string{"master", "worker"},
			IP:       "222",
		},
	}
)

func assertNodeOperationCount(t *testing.T, got []*NodeTransfer, operate NodeOperate, expected int) {
	t.Helper()
	count := 0
	for _, g := range got {
		if g.Operate == operate {
			count++
		}
	}
	assert.Equal(t, expected, count, "expected %d %s operations, got %d", expected, operate, count)
}

func TestCompareBKEConfigNode_Create(t *testing.T) {
	args := []v1beta1.Node{
		{
			Hostname: "node1",
			Role:     []string{"master", "worker"},
			IP:       "111",
		},
		{
			Hostname: "node3",
			Role:     []string{"master", "worker"},
			IP:       "333",
		},
		{
			Hostname: "node5",
			Role:     []string{"master", "worker"},
			IP:       "666",
		},
	}

	got, ok := CompareBKEConfigNode(tNodes, args)
	assert.True(t, ok, "expected changes to be detected")
	assertNodeOperationCount(t, got, CreateNode, 2)
}

func TestCompareBKEConfigNode_Update(t *testing.T) {
	args := []v1beta1.Node{
		{
			Hostname: "node1",
			Role:     []string{"master", "worker"},
			IP:       "222",
		},
		{
			Hostname: "node3",
			Role:     []string{"master", "worker"},
			IP:       "333",
		},
	}

	got, ok := CompareBKEConfigNode(tNodes, args)
	assert.True(t, ok, "expected changes to be detected")
	assertNodeOperationCount(t, got, UpdateNode, 1)
}

func TestCompareBKEConfigNode_Delete(t *testing.T) {
	args := []v1beta1.Node{
		{
			Hostname: "node2",
			Role:     []string{"master", "worker"},
			IP:       "222",
		},
		{
			Hostname: "node3",
			Role:     []string{"master", "worker"},
			IP:       "333",
		},
	}

	got, ok := CompareBKEConfigNode(tNodes, args)
	assert.True(t, ok, "expected changes to be detected")
	assertNodeOperationCount(t, got, RemoveNode, 1)
}
