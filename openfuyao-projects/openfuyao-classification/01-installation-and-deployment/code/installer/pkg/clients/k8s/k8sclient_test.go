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

package k8s

import (
	"testing"

	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// createTestKubeConfig 创建用于测试的kubeconfig
func CreateTestKubeConfig() *rest.Config {
	config := clientcmdapi.NewConfig()
	config.Clusters["test-cluster"] = &clientcmdapi.Cluster{
		Server: "https://test-server:6443",
	}
	config.AuthInfos["test-user"] = &clientcmdapi.AuthInfo{
		Token: "test-token",
	}
	config.Contexts["test-context"] = &clientcmdapi.Context{
		Cluster:  "test-cluster",
		AuthInfo: "test-user",
	}
	config.CurrentContext = "test-context"

	clientConfig := clientcmd.NewDefaultClientConfig(*config, &clientcmd.ConfigOverrides{})
	restConfig, _ := clientConfig.ClientConfig()
	return restConfig
}

func TestNewKubernetesClient_Success(t *testing.T) {
	qps, burst := float32(100.0), 200
	// 准备测试数据
	kubeConfig := CreateTestKubeConfig()
	k8sCfg := &KubernetesCfg{
		KubeConfig: kubeConfig,
		QPS:        qps,
		Burst:      burst,
	}

	// 执行测试
	client, err := NewKubernetesClient(k8sCfg)

	// 验证结果
	if err != nil {
		t.Errorf("Expected no error, but got: %v", err)
	}
	if client == nil {
		t.Error("Expected client to be created, but got nil")
	}

	// 验证配置参数是否正确设置
	if k8sCfg.KubeConfig.QPS != qps {
		t.Errorf("Expected QPS to be 100, but got: %f", k8sCfg.KubeConfig.QPS)
	}
	if k8sCfg.KubeConfig.Burst != burst {
		t.Errorf("Expected Burst to be 200, but got: %d", k8sCfg.KubeConfig.Burst)
	}
}

func TestNewKubernetesClient_NilConfig(t *testing.T) {
	// 准备测试数据 - nil配置
	k8sCfg := &KubernetesCfg{
		KubeConfig: nil,
	}

	// 执行测试
	client, err := NewKubernetesClient(k8sCfg)

	// 验证结果
	if err == nil {
		t.Error("Expected error for nil config, but got none")
	}
	if client != nil {
		t.Error("Expected client to be nil, but got a valid client")
	}
	expectedError := "k8s config get nil"
	if err.Error() != expectedError {
		t.Errorf("Expected error message '%s', but got: %s", expectedError, err.Error())
	}
}

func TestKubernetesClients_Methods(t *testing.T) {
	// 准备测试数据
	kubeConfig := CreateTestKubeConfig()
	k8sCfg := &KubernetesCfg{
		KubeConfig: kubeConfig,
		QPS:        100,
		Burst:      200,
	}

	// 创建客户端
	client, err := NewKubernetesClient(k8sCfg)
	if err != nil {
		t.Fatalf("Failed to create kubernetes client: %v", err)
	}

	// 测试Kubernetes()方法
	t.Run("Kubernetes method", func(t *testing.T) {
		k8sClient := client.Kubernetes()
		if k8sClient == nil {
			t.Error("Expected kubernetes client to be non-nil")
		}
	})

	// 测试KubeConfig()方法
	t.Run("KubeConfig method", func(t *testing.T) {
		config := client.KubeConfig()
		if config == nil {
			t.Error("Expected kubeconfig to be non-nil")
		}
		if config != kubeConfig {
			t.Error("Expected kubeconfig to match the original config")
		}
	})

	// 测试Snapshot()方法
	t.Run("Snapshot method", func(t *testing.T) {
		snapshotClient := client.Snapshot()
		if snapshotClient == nil {
			t.Error("Expected snapshot client to be non-nil")
		}
	})

	// 测试ApiExtension()方法
	t.Run("ApiExtension method", func(t *testing.T) {
		apiExtClient := client.ApiExtension()
		if apiExtClient == nil {
			t.Error("Expected api extension client to be non-nil")
		}
	})

	// 测试Dynamic()方法
	t.Run("Dynamic method", func(t *testing.T) {
		dynamicClient := client.Dynamic()
		if dynamicClient == nil {
			t.Error("Expected dynamic client to be non-nil")
		}
	})
}

func TestKubernetesCfg_DefaultValues(t *testing.T) {
	// 测试默认QPS和Burst值的设置
	kubeConfig := CreateTestKubeConfig()
	k8sCfg := &KubernetesCfg{
		KubeConfig: kubeConfig,
		// 不设置QPS和Burst，使用默认值
	}

	client, err := NewKubernetesClient(k8sCfg)
	if err != nil {
		t.Fatalf("Failed to create kubernetes client: %v", err)
	}

	if client == nil {
		t.Error("Expected client to be created successfully")
	}

	// 验证配置被正确应用
	config := client.KubeConfig()
	if config.QPS != k8sCfg.QPS {
		t.Errorf("Expected QPS to be %f, but got: %f", k8sCfg.QPS, config.QPS)
	}
	if config.Burst != k8sCfg.Burst {
		t.Errorf("Expected Burst to be %d, but got: %d", k8sCfg.Burst, config.Burst)
	}
}
