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

package phaseutil

import (
	"testing"

	"github.com/stretchr/testify/assert"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

func TestIsValidSplitResult(t *testing.T) {
	assert.True(t, isValidSplitResult([]string{"a", "b"}))
	assert.False(t, isValidSplitResult([]string{"a"}))
}

func TestIsMatchingNode(t *testing.T) {
	node := bkenode.Node{IP: "192.168.1.1"}
	assert.True(t, isMatchingNode(node, "192.168.1.1"))
	assert.False(t, isMatchingNode(node, "192.168.1.2"))
	assert.False(t, isMatchingNode(node, ""))
}

func TestIsHostnameAlreadySet(t *testing.T) {
	nodes := []confv1beta1.BKENode{
		{Spec: confv1beta1.BKENodeSpec{Hostname: "host1"}},
		{Spec: confv1beta1.BKENodeSpec{Hostname: ""}},
	}
	assert.True(t, isHostnameAlreadySet(nodes, 0))
	assert.False(t, isHostnameAlreadySet(nodes, 1))
}

func TestUpdateNodeHostname(t *testing.T) {
	nodes := []confv1beta1.BKENode{
		{Spec: confv1beta1.BKENodeSpec{Hostname: ""}},
	}
	updateNodeHostname(nodes, 0, "newhost")
	assert.Equal(t, "newhost", nodes[0].Spec.Hostname)
}

func TestCleanupKeys(t *testing.T) {
	keys := []string{"a", "b", "c", "d"}
	result := cleanupKeys(keys, []int{1, 3})
	assert.Equal(t, []string{"a", "c"}, result)
}

func TestRemoveNodesWithHostnameAndGetStdOutKeys(t *testing.T) {
	stdOut := map[string]map[string][]string{
		"host1/10.0.0.1": {"ping": {"ok"}},
		"host2/10.0.0.2": {"ping": {"ok"}},
	}
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{Hostname: "host1", IP: "10.0.0.1"}},
	}
	removeNodesWithHostname(bkeNodes, stdOut)
	assert.NotContains(t, stdOut, "host1/10.0.0.1")
	assert.Contains(t, stdOut, "host2/10.0.0.2")
	assert.Len(t, getStdOutKeys(stdOut), 1)
}

func TestProcessNodeHostnameUpdate(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.5", Hostname: ""}},
	}
	cluster := &bkev1beta1.BKECluster{}
	indices := processNodeHostnameUpdate(NodeHostnameUpdateParams{
		BKECluster: cluster,
		Keys:       []string{"worker/10.0.0.5"},
		NodeIndex:  0,
		Node:       bkenode.Node{IP: "10.0.0.5"},
		BKENodes:   bkeNodes,
	})
	assert.Equal(t, []int{0}, indices)
	assert.Equal(t, "worker", bkeNodes[0].Spec.Hostname)
}

func TestCountAvailableAgentNodes(t *testing.T) {
	bkeNodes := bkev1beta1.BKENodes{
		{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.1"}},
		{Spec: confv1beta1.BKENodeSpec{IP: "10.0.0.2"}},
	}
	pingNodes := bkenode.Nodes{{IP: "10.0.0.1"}, {IP: "10.0.0.2"}}
	count := countAvailableAgentNodes(bkeNodes, pingNodes, []string{"10.0.0.1"})
	assert.Equal(t, 1, count)
}
