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
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"

	"installer/pkg/clients/k8s"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
)

func createTestKubeClient() k8s.Clients {
	// 准备测试数据
	kubeConfig := clientcmdapi.NewConfig()
	kubeConfig.CurrentContext = "test-context"
	kubeConfig.Clusters["test-cluster"] = &clientcmdapi.Cluster{
		Server: "https://test-server:6443",
	}
	kubeConfig.AuthInfos["test-user"] = &clientcmdapi.AuthInfo{
		Token: "test-token",
	}
	kubeConfig.Contexts["test-context"] = &clientcmdapi.Context{
		Cluster:  "test-cluster",
		AuthInfo: "test-user",
	}
	clientConfig := clientcmd.NewDefaultClientConfig(*kubeConfig, &clientcmd.ConfigOverrides{})
	restConfig, _ := clientConfig.ClientConfig()
	k8sCfg := &k8s.KubernetesCfg{
		KubeConfig: restConfig,
		QPS:        100,
		Burst:      200,
	}

	// 执行测试
	client, _ := k8s.NewKubernetesClient(k8sCfg)
	return client
}

func TestNewNodeJoinClusterOperate(t *testing.T) {
	tempDir := t.TempDir()
	kubeConfigPath := filepath.Join(tempDir, "kubeconfig")
	component := &fuyaov1beta1.Component{
		Name:  "kubernetes",
		Group: "core",
	}

	// 创建模拟的 k8s 客户端
	mockK8sClient := createTestKubeClient()

	operate := NewNodeJoinClusterOperate("init.cfg", kubeConfigPath, component, mockK8sClient)

	assert.NotNil(t, operate)
	assert.Equal(t, "init.cfg", operate.initCfgFile)
	assert.Equal(t, kubeConfigPath, operate.kubeConfigPath)
	assert.Equal(t, component, operate.component)
	assert.Equal(t, mockK8sClient, operate.k8sClient)
}

func TestNodeJoinClusterOperate_removeLocalCert(t *testing.T) {
	// Mock os.RemoveAll 方法
	patchRemoveAll := gomonkey.NewPatches()
	defer patchRemoveAll.Reset()
	patchRemoveAll.ApplyFunc(os.RemoveAll, func(path string) error {
		return nil
	})

	operate := &NodeJoinClusterOperate{
		caTmpPath:     "/tmp/k8s-certs",
		etcdCaTmpPath: "/tmp/k8s-certs/etcd",
	}

	err := operate.removeLocalCert()
	assert.NoError(t, err)
}
