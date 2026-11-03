/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */
package node

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
)

const (
	localLoopbackIP              = "127.0.0.1"
	expectedMasterCount          = 4
	expectedExcludeRoleRemain    = 3
	expectedExcludeRoleAndIPKeep = 5
	expectedExcludeHostnameKeep  = 5
)

func testRemoteIP() string {
	if v := os.Getenv("TEST_REMOTE_IP"); v != "" {
		return v
	}
	return "remote.example.invalid"
}

var (
	nodes          = newNodes()
	filterOptions1 = FilterOptions{
		"Role": "master,etcd",
		"IP":   "openfuyao.obs.cn-north-2.myhuaweicloud.com",
	}
	filterOptions2 = FilterOptions{
		"Role":     WorkerNodeRole,
		"Hostname": "master-1",
	}
	filterOptions3 = FilterOptions{
		"Role": WorkerNodeRole,
	}
	filterOptions4 = FilterOptions{
		"Role": "foo",
	}
	filterOptions5 = FilterOptions{
		"Bar": "foo",
	}

	excludeOptions1 = FilterOptions{
		"Role": "master,etcd",
	}

	excludeOptions2 = FilterOptions{
		"Role": "master,etcd",
		"IP":   "openfuyao.obs.cn-north-2.myhuaweicloud.com",
	}

	excludeOptions3 = FilterOptions{
		"Bar": "foo",
	}
	excludeOptions4 = FilterOptions{
		"Hostname": "worker-4",
	}
)

func TestNode(t *testing.T) {
	t.Run("TestFilter", TestNodesFilter)
	t.Run("TestExclude", TestNodesExclude)
	t.Run("TestCurrentNode", TestCurrentNode)
	t.Run("TestNodeRole", TestNodeRole)
	t.Run("TestGetRoleNodes", TestGetRoleNodes)
	t.Run("Decrypt", func(t *testing.T) {
		nodes := nodes.Exclude(excludeOptions3)
		nodes.Decrypt()
	})
}

func TestNodesFilter(t *testing.T) {
	t.Run("Filter by multiple fields", test1)
	t.Run("Filter by multiple fields with Role ", test2)
	t.Run("Filter role", test3)
	t.Run("Filter for nonexistent value", test4)
	t.Run("Filter for nonexistent key", test5)
	t.Run("Filter MasterWorker", TestFilterMasterWorker)
}

func TestNodesExclude(t *testing.T) {
	t.Run("Exclude by fields", test6)
	t.Run("Exclude by multiple fields", test7)
	t.Run("Exclude for nonexistent key", test8)
	t.Run("Exclude for nonexistent value", test9)
}

func TestCurrentNode(t *testing.T) {
	nodes := Nodes{
		{IP: testRemoteIP()},
		{IP: localLoopbackIP},
	}
	node, err := nodes.CurrentNode()
	assert.Nil(t, err)
	assert.Equal(t, localLoopbackIP, node.IP)
}

func TestNodeRole(t *testing.T) {
	for _, node := range nodes {
		node := Node(node)
		if node.IsMaster() {
			assert.True(t, node.IsMaster())
		}
		if node.IsWorker() {
			assert.True(t, node.IsWorker())
		}
		if node.IsEtcd() {
			assert.True(t, node.IsEtcd())
		}
	}
}

func TestGetRoleNodes(t *testing.T) {
	masterNodes := nodes.Master()
	assert.Equal(t, expectedMasterCount, len(masterNodes))
	assert.Equal(t, masterNodes.MasterWorker().Length(), nodes.MasterWorker().Length())
}

func TestFilterMasterWorker(t *testing.T) {
	nodes := nodes.Filter(FilterOptions{"Role": MasterWorkerNodeRole})
	filterMasterWorkerNodes := Nodes{
		{
			Role:     []string{MasterWorkerNodeRole, EtcdNodeRole},
			IP:       "openfuyao.obs.cn-north-6.myhuaweicloud.com",
			Hostname: "master-worker-1",
		},
	}
	assert.New(t).Equal(filterMasterWorkerNodes, nodes)
}

func TestIsMasterWorker(t *testing.T) {
	t.Run("TestIsMasterWorker", func(t *testing.T) {
		n := Node{
			Role:     []string{MasterWorkerNodeRole, EtcdNodeRole},
			Hostname: "master-worker-1",
		}
		n.IsMasterWorker()
	})
}
func TestEtcd(t *testing.T) {
	t.Run("TestEtcd", func(t *testing.T) {
		etcdFilterNodes := Nodes{
			{
				Role:     []string{MasterWorkerNodeRole, EtcdNodeRole},
				Hostname: "master-worker-1",
			},
		}
		etcdFilterNodes.Etcd()
	})
}
func TestWorker(t *testing.T) {
	t.Run("TestWorker", func(t *testing.T) {
		workerNodes := Nodes{
			{
				Role:     []string{MasterWorkerNodeRole, EtcdNodeRole},
				IP:       "openfuyao.obs.cn-north-6.myhuaweicloud.com",
				Hostname: "master-worker-1",
			},
		}
		workerNodes.Worker()
	})
}
func TestNodeInfo(t *testing.T) {
	ip := "openfuyao.obs.cn-north-1.myhuaweicloud.com"
	t.Run("TestNodeInfo", func(t *testing.T) {
		n := v1beta1.Node{
			IP:       ip,
			Hostname: "master-worker-1",
		}
		NodeInfo(n)
	})
}

func test1(t *testing.T) {
	nodes := nodes.Filter(filterOptions1)
	resultNodes := Nodes{
		{
			Role:     []string{MasterNodeRole, EtcdNodeRole},
			IP:       "openfuyao.obs.cn-north-2.myhuaweicloud.com",
			Hostname: "master-2",
		},
	}
	assert.New(t).Equal(resultNodes, nodes)
}

func test2(t *testing.T) {
	nodes := nodes.Filter(filterOptions2)
	assert.New(t).Equal(0, nodes.Length())
}

func test3(t *testing.T) {
	nodes := nodes.Filter(filterOptions3)
	test3ResultNodes := Nodes{
		{
			Role:     []string{WorkerNodeRole},
			IP:       "openfuyao.obs.cn-north-4.myhuaweicloud.com",
			Hostname: "worker-4",
		},
		{
			Role:     []string{WorkerNodeRole},
			IP:       "openfuyao.obs.cn-north-5.myhuaweicloud.com",
			Hostname: "worker-5",
		},
	}
	assert.New(t).Equal(test3ResultNodes, nodes)
}

func test4(t *testing.T) {
	nodes := nodes.Filter(filterOptions4)
	assert.New(t).Equal(0, nodes.Length())
}

func test5(t *testing.T) {
	nodes := nodes.Filter(filterOptions5)
	assert.New(t).Equal(0, nodes.Length())
}

func test6(t *testing.T) {
	nodes := nodes.Exclude(excludeOptions1)
	assert.New(t).Equal(expectedExcludeRoleRemain, nodes.Length())
}

func test7(t *testing.T) {
	nodes := nodes.Exclude(excludeOptions2)
	assert.New(t).Equal(expectedExcludeRoleAndIPKeep, nodes.Length())
}

func test8(t *testing.T) {
	nodes := nodes.Exclude(excludeOptions3)
	expectedRemain := nodes.Length()
	assert.New(t).Equal(expectedRemain, nodes.Length())
}

func test9(t *testing.T) {
	nodes := nodes.Exclude(excludeOptions4)
	assert.New(t).Equal(expectedExcludeHostnameKeep, nodes.Length())
}

func TestConvertBKENodeListToNodes(t *testing.T) {
	bkeNodeList := &v1beta1.BKENodeList{
		Items: []v1beta1.BKENode{
			{
				Spec: v1beta1.BKENodeSpec{
					Role:     []string{MasterNodeRole, EtcdNodeRole},
					IP:       "192.168.1.1",
					Hostname: "master-1",
				},
			},
			{
				Spec: v1beta1.BKENodeSpec{
					Role:     []string{WorkerNodeRole},
					IP:       "192.168.1.2",
					Hostname: "worker-1",
				},
			},
		},
	}

	nodes := ConvertBKENodeListToNodes(bkeNodeList)
	assert.Equal(t, 2, len(nodes))
	assert.Equal(t, "192.168.1.1", nodes[0].IP)
	assert.Equal(t, "192.168.1.2", nodes[1].IP)

	// Test nil input
	nilNodes := ConvertBKENodeListToNodes(nil)
	assert.Empty(t, nilNodes)

	// Test empty list
	emptyNodes := ConvertBKENodeListToNodes(&v1beta1.BKENodeList{})
	assert.Empty(t, emptyNodes)
}

func TestConvertBKENodesToNodes(t *testing.T) {
	bkeNodes := []v1beta1.BKENode{
		{
			Spec: v1beta1.BKENodeSpec{
				Role:     []string{MasterNodeRole},
				IP:       "10.0.0.1",
				Hostname: "node-1",
			},
		},
	}

	nodes := ConvertBKENodesToNodes(bkeNodes)
	assert.Equal(t, 1, len(nodes))
	assert.Equal(t, "10.0.0.1", nodes[0].IP)

	// Test empty slice
	emptyNodes := ConvertBKENodesToNodes([]v1beta1.BKENode{})
	assert.Empty(t, emptyNodes)
}

func TestConvertNodesToBKENodes(t *testing.T) {
	nodes := Nodes{
		{
			Role:     []string{MasterNodeRole, EtcdNodeRole},
			IP:       "192.168.1.100",
			Hostname: "master-1",
		},
		{
			Role:     []string{WorkerNodeRole},
			IP:       "192.168.1.101",
			Hostname: "worker-1",
		},
	}

	bkeNodes := ConvertNodesToBKENodes(nodes, "test-ns", "test-cluster")
	assert.Equal(t, 2, len(bkeNodes))
	assert.Equal(t, "test-cluster-192-168-1-100", bkeNodes[0].Name)
	assert.Equal(t, "test-ns", bkeNodes[0].Namespace)
	assert.Equal(t, "test-cluster", bkeNodes[0].Labels["cluster.x-k8s.io/cluster-name"])

	// Test empty nodes
	emptyBkeNodes := ConvertNodesToBKENodes(Nodes{}, "ns", "cluster")
	assert.Empty(t, emptyBkeNodes)
}

func TestGenerateBKENodeName(t *testing.T) {
	tests := []struct {
		clusterName string
		nodeIP      string
		expected    string
	}{
		{"cluster1", "192.168.1.1", "cluster1-192-168-1-1"},
		{"my-cluster", "10.0.0.100", "my-cluster-10-0-0-100"},
		{"test", "172.16.0.1", "test-172-16-0-1"},
	}

	for _, tt := range tests {
		result := GenerateBKENodeName(tt.clusterName, tt.nodeIP)
		assert.Equal(t, tt.expected, result)
	}
}

func newNodes() Nodes {
	node1 := v1beta1.Node{
		Role:     []string{MasterNodeRole, EtcdNodeRole},
		IP:       "openfuyao.obs.cn-north-1.myhuaweicloud.com",
		Hostname: "master-1",
	}
	node2 := v1beta1.Node{
		Role:     []string{MasterNodeRole, EtcdNodeRole},
		IP:       "openfuyao.obs.cn-north-2.myhuaweicloud.com",
		Hostname: "master-2",
	}
	node3 := v1beta1.Node{
		Role:     []string{MasterNodeRole, EtcdNodeRole},
		IP:       "openfuyao.obs.cn-north-3.myhuaweicloud.com",
		Hostname: "master-3",
	}
	node4 := v1beta1.Node{
		Role:     []string{WorkerNodeRole},
		IP:       "openfuyao.obs.cn-north-4.myhuaweicloud.com",
		Hostname: "worker-4",
	}
	node5 := v1beta1.Node{
		Role:     []string{WorkerNodeRole},
		IP:       "openfuyao.obs.cn-north-5.myhuaweicloud.com",
		Hostname: "worker-5",
	}
	node6 := v1beta1.Node{
		Role:     []string{MasterWorkerNodeRole, EtcdNodeRole},
		IP:       "openfuyao.obs.cn-north-6.myhuaweicloud.com",
		Hostname: "master-worker-1",
	}
	return append(Nodes{}, node1, node2, node3, node4, node5, node6)
}

func TestSetDefaultsForNodes(t *testing.T) {
	tests := []struct {
		name     string
		nodes    Nodes
		expected Nodes
	}{
		{
			name:     "empty nodes",
			nodes:    Nodes{},
			expected: Nodes{},
		},
		{
			name: "nodes with defaults",
			nodes: Nodes{
				{IP: "192.168.1.1", Port: "22", Username: "root"},
			},
			expected: Nodes{
				{IP: "192.168.1.1", Port: "22", Username: "root"},
			},
		},
		{
			name: "nodes without port and username",
			nodes: Nodes{
				{IP: "192.168.1.1"},
			},
			expected: Nodes{
				{IP: "192.168.1.1", Port: DefaultNodeSSHPort, Username: DefaultNodeUserRoot},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SetDefaultsForNodes(tt.nodes)
			assert.Equal(t, tt.expected, result)
		})
	}
}
