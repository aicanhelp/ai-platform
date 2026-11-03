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

package common

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/clients"
	"installer/pkg/constants"
)

const clusterConfig = "cluster-config.yaml"

func mockComponentConfig(t *testing.T, saveDir string) {
	yamlString := `apiVersion: openfuyao.io/v1beta1
kind: build
metadata:
    name: example-component-manifests
spec:
    ossa:
        - os-name: openEuler
          version: "22.03"
          cpu-arch: arm64
        - os-name: openEuler
          version: "22.03"
          cpu-arch: amd64
    components:
        cni:
            - name: calico
              version: 3.27.3
              describe: ""
              dependencies: 'kubernetes:[1.28.0,), '
            - name: cni-plugins
              version: 1.4.1
              describe: ""
              dependencies: ""
        core:
            - name: kubernetes
              version: 1.28.8
              describe: ""
              dependencies: 'containerd:[1.7.2,), runc:[1.1.9,), cni-plugins:[1.3.0,), kube-vip:[0.6.2,), '
        cri:
            - name: containerd
              version: 1.7.14
              describe: ""
              dependencies: ""
        ha:
            - name: kube-vip
              version: 0.7.2
              describe: ""
              dependencies: 'containerd:[1.7.2,), '
        ingress:
            - name: ingress-nginx
              version: 1.9.4
              describe: ""
              dependencies: 'kubernetes:[1.28.0,), '
        manager:
            - name: openfuyao-system
              version: latest
              describe: ""
              dependencies: 'kubernetes:[1.28.0,), helm:[3.14.2,), calico:[3.27.3,), yq:[4.43.1,), ingress-nginx:[1.9.4,), '
        oci:
            - name: runc
              version: 1.1.12
              describe: ""
              dependencies: ""
        pkg-manager:
            - name: helm
              version: 3.14.2
              describe: ""
              dependencies: ""
        tools:
            - name: yq
              version: 4.43.1
              describe: ""
              dependencies: ""`

	err := ioutil.WriteFile(filepath.Join(saveDir, constants.ComponentManifest), []byte(yamlString), 0644)
	assert.NoError(t, err)
}

func mockClusterConfig(t *testing.T, saveDir string) {
	yamlString := `
apiVersion: openfuyao.io/v1beta1
kind: cluster
metadata:
    name: example-cluster
spec:
    nodes:
        masters:
            - hostname: fuyao-master
              user: root
              ip: 192.168.0.1
              port: 22
              privateKeyPath: /root/.ssh/id_rsa
              removeTaint: true
              labels:
                - name: label-name
                  value: label-value
        workers:
            - hostname: fuyao-worker
              user: root
              ip: 192.168.0.2
              port: 22
              privateKeyPath: /root/.ssh/id_rsa
              labels:
                - name: label-name
                  value: label-value
`
	err := ioutil.WriteFile(filepath.Join(saveDir, clusterConfig), []byte(yamlString), 0644)
	assert.NoError(t, err)
}

func TestDefaultLoader(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-default-loader-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// mock组件清单文件
	mockComponentConfig(t, tempDir)

	// mock has private key
	patches := gomonkey.ApplyFunc(isHasPrivateKey, func() error {
		return nil
	})
	defer patches.Reset()

	patchManifestsPath := gomonkey.ApplyFunc(GetComponentManifestsAbsPath, func() (string, error) {
		return filepath.Join(tempDir, constants.ComponentManifest), nil
	})
	defer patchManifestsPath.Reset()

	// 不包含fuyao-system
	arg := Argument{
		FuyaoZone: CN,
	}
	r, err := NewKubeRuntime(arg)
	assert.NoError(t, err)
	assert.Equal(t, 6, len(r.Cluster.Spec.Components))
	for _, component := range r.Cluster.Spec.Components {
		if component.Name == constants.FuyaoSystem {
			t.Error("openfuyao-system should not be included in default loader")
		}
	}

	// 全量组件
	arg = Argument{
		FuyaoZone:    CN,
		AllComponent: true,
	}
	loader := NewLoader(arg)
	clusterCfg, err := loader.Load()
	assert.NoError(t, err)
	assert.Equal(t, 9, len(clusterCfg.Spec.Components))
	names := make([]string, 0)
	for _, component := range clusterCfg.Spec.Components {
		names = append(names, component.Name)
	}
	assert.Contains(t, names, constants.FuyaoSystem)
}

func TestFileLoader(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-file-loader-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// mock组件清单文件
	mockComponentConfig(t, tempDir)

	// mock集群规划文件
	mockClusterConfig(t, tempDir)

	patchManifestsPath := gomonkey.ApplyFunc(GetComponentManifestsAbsPath, func() (string, error) {
		return filepath.Join(tempDir, constants.ComponentManifest), nil
	})
	defer patchManifestsPath.Reset()

	// 创建普通集群
	arg := Argument{
		FuyaoZone: CN,
		FilePath:  filepath.Join(tempDir, clusterConfig),
	}
	r, err := NewKubeRuntime(arg)
	assert.NoError(t, err)
	assert.Equal(t, 9, len(r.Cluster.Spec.Components))
	names := make([]string, 0)
	for _, component := range r.Cluster.Spec.Components {
		names = append(names, component.Name)
	}
	assert.Contains(t, names, constants.FuyaoSystem)
	assert.NotContains(t, names, "kube-vip")

	patchSudoExecCmd := gomonkey.NewPatches()
	defer patchSudoExecCmd.Reset()
	patchSudoExecCmd.ApplyPrivateMethod(
		reflect.TypeOf(&clients.Client{}), "SudoExecCmd", func(_ *clients.Client, _ string) (string, error) {
			return "2", nil
		},
	)

	// 集群内添加节点
	arg = Argument{
		FuyaoZone: CN,
		FilePath:  filepath.Join(tempDir, clusterConfig),
		AddNodes:  true,
	}
	loader := NewLoader(arg)
	clusterCfg, err := loader.Load()
	assert.NoError(t, err)
	names = make([]string, 0)
	for _, component := range clusterCfg.Spec.Components {
		names = append(names, component.Name)
	}
	assert.NotContains(t, names, constants.FuyaoSystem)
}
