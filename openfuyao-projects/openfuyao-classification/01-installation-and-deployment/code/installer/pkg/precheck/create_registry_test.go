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
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	"installer/pkg/fuyao/v1beta1/cluster"
)

func TestCreateRegistryFlightCheck_StartCheck_InputPasswd(t *testing.T) {
	// 创建模拟的 Kubernetes 客户端
	clientset := fake.NewSimpleClientset()

	// 创建模拟的节点
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
		Status: corev1.NodeStatus{
			Addresses: []corev1.NodeAddress{
				{
					Type:    corev1.NodeInternalIP,
					Address: "127.0.0.1",
				},
			},
		},
	}
	clientset.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})

	// 创建模拟的节点列表
	nodes := cluster.Nodes{
		Masters: []*cluster.Node{
			{
				IP:       "127.0.0.1",
				Hostname: "localhost",
				Port:     8080, // 使用本地端口
				Password: []byte("testpwd"),
			},
		},
		Workers: []*cluster.Node{
			{
				IP:       "127.0.0.2",
				Hostname: "localhost",
				Port:     8081, // 使用本地端口
				Password: []byte("testpwd"),
			},
		},
	}

	// 创建 CreateRegistryFlightCheck 实例
	check := NewCreateRegistryFlightCheck(nodes)

	// 调用 StartCheck 函数
	err := check.StartCheck()

	// 验证结果
	assert.NoError(t, err)
}
