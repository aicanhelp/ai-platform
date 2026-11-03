/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package common

import (
	"os"
	"path/filepath"
	"testing"

	"installer/pkg/fuyao/v1beta1/cluster"
)

func TestArgument_ClearSensitiveData(t *testing.T) {
	tests := []struct {
		name string
		arg  *Argument
	}{
		{
			name: "清除敏感数据测试",
			arg: &Argument{
				RegistryPasswd:     []byte("test-passwd"),
				RegistryDBPasswd:   []byte("test-db-passwd"),
				RegistryNodePasswd: []byte("test-node-passwd"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.arg.ClearSensitiveData()

			if len(tt.arg.RegistryPasswd) != 0 {
				t.Error("RegistryPasswd 未被清除")
			}
			if len(tt.arg.RegistryDBPasswd) != 0 {
				t.Error("RegistryDBPasswd 未被清除")
			}
			if len(tt.arg.RegistryNodePasswd) != 0 {
				t.Error("RegistryNodePasswd 未被清除")
			}
		})
	}
}

func TestKubeRuntime_ClearSensitiveData(t *testing.T) {
	testNodes := cluster.Nodes{
		Masters: []*cluster.Node{
			{Password: []byte("test-master-pwd")},
		},
		Workers: []*cluster.Node{
			{Password: []byte("test-worker-pwd")},
		},
	}

	testRuntime := &KubeRuntime{
		Cluster: &cluster.Cluster{
			Spec: cluster.Spec{
				Nodes: testNodes,
			},
		},
		Arg: Argument{
			RegistryPasswd:     []byte("test-registry-pwd"),
			RegistryDBPasswd:   []byte("test-db-pwd"),
			RegistryNodePasswd: []byte("test-node-pwd"),
		},
	}

	t.Run("清除运行时敏感数据", func(t *testing.T) {
		testRuntime.ClearSensitiveData()

		// 检查节点密码
		for _, master := range testRuntime.Cluster.Spec.Nodes.Masters {
			if len(master.Password) != 0 {
				t.Error("Master节点密码未被清除")
			}
		}
		for _, worker := range testRuntime.Cluster.Spec.Nodes.Workers {
			if len(worker.Password) != 0 {
				t.Error("Worker节点密码未被清除")
			}
		}

		// 检查参数密码
		if len(testRuntime.Arg.RegistryPasswd) != 0 {
			t.Error("RegistryPasswd 未被清除")
		}
		if len(testRuntime.Arg.RegistryDBPasswd) != 0 {
			t.Error("RegistryDBPasswd 未被清除")
		}
		if len(testRuntime.Arg.RegistryNodePasswd) != 0 {
			t.Error("RegistryNodePasswd 未被清除")
		}
	})
}

func TestNewKubeRuntime(t *testing.T) {
	// 创建临时测试目录
	tmpDir := t.TempDir()
	testConfigPath := filepath.Join(tmpDir, "test-config.yaml")

	// 创建测试配置文件
	testConfig := `
metadata:
  name: test-cluster
spec:
  nodes:
    masters:
      - ip: "192.168.1.1"
    workers:
      - ip: "192.168.1.2"
`
	const fileMode = 0644
	if err := os.WriteFile(testConfigPath, []byte(testConfig), fileMode); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		arg     Argument
		wantErr bool
	}{
		{
			name: "正常创建运行时",
			arg: Argument{
				FilePath: testConfigPath,
			},
			wantErr: false,
		},
		{
			name: "无效的配置文件路径",
			arg: Argument{
				FilePath: "invalid/path",
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewKubeRuntime(tt.arg)
			if err == nil {
				t.Error("getPrivateKey() test fail")
			}
		})
	}
}

func TestGetServiceName(t *testing.T) {
	tests := []struct {
		name string
	}{
		{
			name: "获取服务名称",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetServiceName(); got == "" {
				t.Errorf("GetServiceName() fail")
			}
		})
	}
}
