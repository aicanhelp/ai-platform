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

package join

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	"installer/pkg/fuyao/v1beta1/cluster"
)

// createTestNodeJoinClusterOperate 创建测试用的 NodeJoinClusterOperate
func createTestNodeJoinClusterOperate(t *testing.T) *NodeJoinClusterOperate {
	const fileMode = 0644
	// 创建临时目录用于测试
	tempDir := t.TempDir()

	// 创建临时的 kubeconfig 文件
	kbCfgPath := filepath.Join(tempDir, "kubeconfig")
	kbCfgContent := `apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://localhost:6443
  name: executor-test
contexts:
- context:
    cluster: executor-test
    user: executor-user-test
  name: context
current-context: context
users:
- name: executor-user-test
  user:
    token: test-token`

	err := os.WriteFile(kbCfgPath, []byte(kbCfgContent), fileMode)
	assert.NoError(t, err)

	// 创建模拟的组件配置
	component := &cluster.Component{
		Name:  "kubernetes",
		Group: "core",
	}

	operate := &NodeJoinClusterOperate{
		initCfgFile:    "",
		kubeConfigPath: kbCfgPath,
		component:      component,
		k8sClient:      nil, // 空客户端用于测试
		certDir:        filepath.Join(tempDir, "pki"),
		etcdCertDir:    filepath.Join(tempDir, "pki", "etcd"),
	}

	return operate
}

const (
	TestAds1 = "192" + ".168." + "1.10"
	TestAds2 = "192" + ".168." + "1.11"
	TestAds3 = "192" + ".168." + "1.20"
)

// createTestNodes 创建测试用的节点
func createTestNodes() cluster.Nodes {
	master1 := &cluster.Node{
		Hostname: "master1",
		User:     "test",
		Password: []byte("password"),
		IP:       TestAds1,
		Port:     22,
	}

	master2 := &cluster.Node{
		Hostname: "master2",
		User:     "test",
		Password: []byte("password"),
		IP:       TestAds2,
		Port:     22,
	}

	worker1 := &cluster.Node{
		Hostname: "worker1",
		User:     "test",
		Password: []byte("password"),
		IP:       TestAds3,
		Port:     22,
	}

	return cluster.Nodes{
		Masters: []*cluster.Node{master1, master2},
		Workers: []*cluster.Node{worker1},
	}
}

func TestNewNodeJoinExecutor(t *testing.T) {
	nodes := createTestNodes()
	operate := createTestNodeJoinClusterOperate(t)

	executor := NewNodeJoinExecutor(nodes, operate)

	assert.NotNil(t, executor)
	assert.Equal(t, nodes, executor.nodes)
	assert.Equal(t, operate, executor.operate)
	assert.NotNil(t, executor.RoutineMaximum)
}

func TestNodeJoinExecutor_executor(t *testing.T) {
	nodes := createTestNodes()
	operate := createTestNodeJoinClusterOperate(t)
	executor := NewNodeJoinExecutor(nodes, operate)

	// 创建测试函数
	executionOrder := make([]string, 0)
	testFunc1 := func(node *cluster.Node) error {
		executionOrder = append(executionOrder, "func1")
		assert.Equal(t, TestAds1, node.IP)
		return nil
	}

	testFunc2 := func(node *cluster.Node) error {
		executionOrder = append(executionOrder, "func2")
		assert.Equal(t, TestAds1, node.IP)
		return nil
	}

	var wg sync.WaitGroup
	wg.Add(1)
	executor.RoutineMaximum.Add()

	go executor.executor(nodes.Masters[0], []function{testFunc1, testFunc2}, &wg)
	wg.Wait()

	// 验证函数按顺序执行
	assert.Equal(t, []string{"func1", "func2"}, executionOrder)
}
