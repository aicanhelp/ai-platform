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
package testutils

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func TestKubernetesAPIServerWithHTTPTest(t *testing.T) {
	basePod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "nginx",
					Image: "nginx:latest",
				},
			},
		},
	}
	nonExistentPod := basePod.DeepCopy()
	nonExistentPod.SetName("non-existent-pod")
	mapObjs := map[string]interface{}{
		"/api/v1/namespaces/default/pods/non-existent-pod": nonExistentPod,
		"/api/v1/namespaces/default/pods/test-pod":         basePod,
		"/api/v1/namespaces/default/pods": corev1.PodList{
			Items: []corev1.Pod{basePod, *nonExistentPod},
		},
	}

	// 1. 创建测试 HTTP 服务器
	rconfig, tserver := TestGetK8sServerHttp(mapObjs)
	defer tserver.Close()
	// 3. 创建 Kubernetes 客户端
	clientset, err := kubernetes.NewForConfig(rconfig)
	if err != nil {
		t.Log(err)
	}

	// 4. 测试获取 Pod
	t.Run("Get Pod", func(t *testing.T) {
		_, err := clientset.CoreV1().Pods("default").Get(context.Background(), "test-pod", metav1.GetOptions{})
		if err != nil {
			t.Log(err)
		}
	})

	// 5. 测试列出 Pods
	t.Run("server version", func(t *testing.T) {
		if _, err := clientset.ServerVersion(); err != nil {
			t.Log(err)
		}
	})

	t.Run("Not Found", func(t *testing.T) {
		if _, err := RestConfigToKubeConfig(rconfig, "test"); err != nil {
			t.Log(err)
		}
	})
}
