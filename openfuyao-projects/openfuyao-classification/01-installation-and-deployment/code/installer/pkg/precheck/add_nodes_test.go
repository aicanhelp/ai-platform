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
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"installer/pkg/clients/k8s"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/operate"
)

const (
	testMasterIP  = "127.0.0.1"
	testWorkerIP  = "127.0.0.2"
	testMasterIP2 = "127.0.0.3"
)

// 创建测试用的节点列表
func createTestNodes() cluster.Nodes {
	return cluster.Nodes{
		Masters: []*cluster.Node{
			{
				IP:       testMasterIP,
				Hostname: "master1",
				User:     "test-user",
				Password: []byte("test-password"),
				Port:     22,
			},
			{
				IP:       testMasterIP2,
				Hostname: "master2",
				User:     "test-user",
				Password: []byte("test-password"),
				Port:     22,
			},
		},
		Workers: []*cluster.Node{
			{
				IP:       testWorkerIP,
				Hostname: "worker1",
				User:     "test-user",
				Password: []byte("test-password"),
				Port:     22,
			},
		},
	}
}

// 创建测试用的 Kubernetes 节点
func createTestK8sNodes() *corev1.NodeList {
	return &corev1.NodeList{
		Items: []corev1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "existing-master",
					Labels: map[string]string{
						constants.DefaultMasterLabelName: constants.DefaultMasterLabelValue,
					},
				},
				Status: corev1.NodeStatus{
					Addresses: []corev1.NodeAddress{
						{
							Type:    corev1.NodeInternalIP,
							Address: "127.0.0.100",
						},
					},
					Capacity: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("4"),
						corev1.ResourceMemory: resource.MustParse("8Gi"),
					},
				},
			},
		},
	}
}

func TestNewAddNodesFlightCheck(t *testing.T) {
	nodes := createTestNodes()
	check := NewAddNodesFlightCheck(nodes)

	assert.NotNil(t, check)
	assert.Equal(t, nodes, check.Nodes)
	assert.NotNil(t, check.FlightCheck)
}

const (
	cpuStd = 2
	memStd = 4
)

func TestAddNodesFlightCheck_isMasterCPUMemorySatisfy_Success(t *testing.T) {
	check := &AddNodesFlightCheck{}

	cpuMemInfo := map[corev1.ResourceName]map[string]int64{
		corev1.ResourceCPU: {
			testMasterIP: 4,
		},
		corev1.ResourceMemory: {
			testMasterIP: 8,
		},
	}

	result := check.isMasterCPUMemorySatisfy(cpuStd, memStd, cpuMemInfo)
	assert.True(t, result)
}

func TestAddNodesFlightCheck_isMasterCPUMemorySatisfy_Insufficient(t *testing.T) {
	check := &AddNodesFlightCheck{}

	cpuMemInfo := map[corev1.ResourceName]map[string]int64{
		corev1.ResourceCPU: {
			testMasterIP: 1, // 不足
		},
		corev1.ResourceMemory: {
			testMasterIP: cpuStd, // 不足
		},
	}

	result := check.isMasterCPUMemorySatisfy(cpuStd, memStd, cpuMemInfo)
	assert.False(t, result)
}

func TestAddNodesFlightCheck_checkMasterCPUMemory_InsufficientResources(t *testing.T) {
	nodes := createTestNodes()

	// 创建资源不足的节点
	insufficientNodes := &corev1.NodeList{
		Items: []corev1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "insufficient-master",
					Labels: map[string]string{
						constants.DefaultMasterLabelName: constants.DefaultMasterLabelValue,
					},
				},
				Status: corev1.NodeStatus{
					Addresses: []corev1.NodeAddress{
						{
							Type:    corev1.NodeInternalIP,
							Address: testMasterIP,
						},
					},
					Capacity: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("1"),   // 不足
						corev1.ResourceMemory: resource.MustParse("1Gi"), // 不足
					},
				},
			},
		},
	}

	check := &AddNodesFlightCheck{
		Nodes:        nodes,
		clusterNodes: insufficientNodes,
		FlightCheck: &FlightCheck{
			NodesOperate: &operate.NodesOperate{},
		},
	}

	err := check.checkMasterCPUMemory()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "CPU or memory resources of the master node")
}

func TestAddNodesFlightCheck_canAddMasterCheck_SingleMaster(t *testing.T) {
	nodes := cluster.Nodes{
		Masters: []*cluster.Node{
			{IP: testMasterIP, Hostname: "master1"},
		},
	}

	check := &AddNodesFlightCheck{
		Nodes: nodes,
	}

	err := check.canAddMasterCheck()
	assert.NoError(t, err)
}

func TestAddNodesFlightCheck_canAddMasterCheck_HAExpansionFailed(t *testing.T) {
	nodes := createTestNodes() // 包含2个master

	// 创建只有1个master的集群（无法扩展为HA）
	singleMasterCluster := &corev1.NodeList{
		Items: []corev1.Node{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "single-master",
					Labels: map[string]string{
						constants.DefaultMasterLabelName: constants.DefaultMasterLabelValue,
					},
				},
			},
		},
	}

	check := &AddNodesFlightCheck{
		Nodes:        nodes,
		clusterNodes: singleMasterCluster,
	}

	err := check.canAddMasterCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "non-HA cluster cannot be expanded to HA")
}

func TestAddNodesFlightCheck_checkOrSetHostName_Success(t *testing.T) {
	nodes := createTestNodes()
	check := &AddNodesFlightCheck{
		Nodes:        nodes,
		clusterNodes: createTestK8sNodes(),
		FlightCheck: &FlightCheck{
			NodesOperate: &operate.NodesOperate{},
		},
	}

	err := check.checkOrSetHostName()
	assert.NoError(t, err)
}

func TestAddNodesFlightCheck_isHasSameIP_Success(t *testing.T) {
	nodes := createTestNodes()
	check := &AddNodesFlightCheck{
		Nodes:        nodes,
		clusterNodes: createTestK8sNodes(),
		FlightCheck: &FlightCheck{
			NodesOperate: &operate.NodesOperate{},
		},
	}

	err := check.isHasSameIP()
	assert.NoError(t, err)
}

func TestAddNodesFlightCheck_StartCheck_KubeConfigDownloadFailed(t *testing.T) {
	nodes := createTestNodes()
	check := NewAddNodesFlightCheck(nodes)

	// Mock k8s.DownLoadKubeConfig to return error
	patches := gomonkey.ApplyFunc(k8s.DownLoadKubeConfig,
		func(*cluster.Node) (string, error) {
			return "", errors.New("failed to download kubeconfig")
		})
	defer patches.Reset()

	err := check.StartCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to download kubeconfig")
}

func TestAddNodesFlightCheck_StartCheck_K8sClientCreationFailed(t *testing.T) {
	nodes := createTestNodes()
	check := NewAddNodesFlightCheck(nodes)

	// Mock successful kubeconfig download but failed client creation
	patches := gomonkey.ApplyFunc(k8s.DownLoadKubeConfig,
		func(*cluster.Node) (string, error) {
			return "/tmp/kubeconfig", nil
		})
	defer patches.Reset()

	patches.ApplyFunc(k8s.NewKubernetesCfg, func(string) *k8s.KubernetesCfg {
		return &k8s.KubernetesCfg{}
	})

	err := check.StartCheck()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get k8s client")
}
