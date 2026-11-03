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

package executor

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/clients"
	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/executor/core"
	"installer/pkg/executor/join"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
)

const (
	ip0 = "o.x.y.z"
	ip1 = "w.x.y.z"
)

func MockGraph() *core.DirectedGraph {
	// mock component
	containerd := fuyaov1beta1.Component{
		Name:          "containerd",
		Group:         constants.Cri,
		BootstrapNode: constants.All,
	}
	runc := fuyaov1beta1.Component{
		Name:          "runc",
		Group:         constants.Oci,
		BootstrapNode: constants.All,
	}
	calico := fuyaov1beta1.Component{
		Name:          "calico",
		Group:         constants.Cni,
		BootstrapNode: constants.Master,
	}
	kubernetes := fuyaov1beta1.Component{
		Name:          "kubernetes",
		Group:         constants.Core,
		BootstrapNode: constants.All,
	}
	components := []fuyaov1beta1.Component{containerd, runc, calico, kubernetes}

	// mock metadata
	dependKubernetes := metadata.Dependency{
		Name: "kubernetes",
	}
	dependRunc := metadata.Dependency{
		Name: "runc",
	}
	dependContainerd := metadata.Dependency{
		Name: "containerd",
	}

	containerdMetaData := metadata.MetaData{
		Name:         "containerd",
		Dependencies: []metadata.Dependency{},
	}
	runcMetaData := metadata.MetaData{
		Name:         "runc",
		Dependencies: []metadata.Dependency{},
	}
	kubernetesMetaData := metadata.MetaData{
		Name:         "kubernetes",
		Dependencies: []metadata.Dependency{dependContainerd, dependRunc},
	}
	calicoMetaData := metadata.MetaData{
		Name:         "calico",
		Dependencies: []metadata.Dependency{dependKubernetes},
	}
	// 组件的名字不会重复
	name2MetaData := make(map[string]metadata.MetaData)
	name2MetaData[containerd.Name] = containerdMetaData
	name2MetaData[runc.Name] = runcMetaData
	name2MetaData[kubernetes.Name] = kubernetesMetaData
	name2MetaData[calico.Name] = calicoMetaData

	// 构建图
	graph := &core.DirectedGraph{}
	NameToComponentMap := make(map[string]fuyaov1beta1.Component)

	// 图中加入顶点
	for _, component := range components {
		NameToComponentMap[component.Name] = component
		var vertex = &core.Vertex{
			Data:      component,
			OutDegree: 0,
			InDegree:  0,
		}
		graph.AddVertex(vertex)
	}

	// 构建图的边
	for _, component := range components {
		metaData := name2MetaData[component.Name]
		for _, depend := range metaData.Dependencies {
			// 增加依赖
			dependComponent := NameToComponentMap[depend.Name]
			graph.AddEdgeByData(component.Name, dependComponent.Name)
		}
	}

	return graph
}

func MockNodes() fuyaov1beta1.Nodes {
	master := &fuyaov1beta1.Node{
		Hostname: "master",
		User:     "fuyao",
		Password: []byte("000"),
		IP:       ip0,
		Port:     22,
	}

	worker := &fuyaov1beta1.Node{
		Hostname: "worker",
		User:     "fuyao",
		Password: []byte("000"),
		IP:       ip1,
		Port:     22,
	}

	nodes := fuyaov1beta1.Nodes{Masters: []*fuyaov1beta1.Node{master}, Workers: []*fuyaov1beta1.Node{worker}}
	return nodes
}

func TestInstallExecutor(t *testing.T) {
	// mock common.GetPackageName
	patchPkgName := gomonkey.ApplyFunc(common.GetPackageName, func(node *fuyaov1beta1.Node) (string, error) {
		return "fuyao-pkg", nil
	})
	defer patchPkgName.Reset()

	// mock clients.SudoExecCmd
	patchExecCmd := gomonkey.NewPatches()
	defer patchExecCmd.Reset()
	patchExecCmd.ApplyMethodFunc(
		reflect.TypeOf(&clients.Client{}), "SudoExecCmd", func(_ string) (string, error) {
			return "", nil
		},
	)

	// mock join.NodeJoinCluster
	patchNodeJoin := gomonkey.ApplyFunc(join.NodeJoinCluster, func(nodes fuyaov1beta1.Nodes,
		component *fuyaov1beta1.Component, initCfgFile, execOperate string) error {
		return nil
	})
	defer patchNodeJoin.Reset()

	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-test-install-exec-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// mock join.NodeJoinCluster
	patchAbsPath := gomonkey.ApplyFunc(common.GetAssetAbsPath, func() (string, error) {
		return tempDir, nil
	})
	defer patchAbsPath.Reset()

	// 构建图
	graph := MockGraph()
	// 构建节点
	nodes := MockNodes()
	// 构建安装器
	executor := NewExecutor(constants.Install, nodes, graph)
	// 执行安装
	err = executor.Install()
	assert.NoError(t, err)
	// 验证安装结果
	installComponentManifest := &manifests.InstallComponentManifest{}
	err = common.UnmarshalYaml[*manifests.InstallComponentManifest](installComponentManifest, filepath.Join(tempDir,
		constants.InstallComponentManifestFile))
	ips := make([]string, 0)

	for _, nodeInfo := range installComponentManifest.Spec.Manifests {
		ips = append(ips, nodeInfo.Node.Ip)
	}
	assert.Contains(t, ips, ip0)
	assert.Contains(t, ips, ip1)

	masterInstallComponents := make([]string, 0)
	workerInstallComponents := make([]string, 0)
	for _, nodeInfo := range installComponentManifest.Spec.Manifests {
		if nodeInfo.Node.Ip == ip0 {
			for _, component := range nodeInfo.Components {
				masterInstallComponents = append(masterInstallComponents, component.Name)
			}
		}
		if nodeInfo.Node.Ip == ip1 {
			for _, component := range nodeInfo.Components {
				workerInstallComponents = append(workerInstallComponents, component.Name)
			}
		}
	}
	assert.Contains(t, masterInstallComponents, "containerd")
	assert.Contains(t, masterInstallComponents, "runc")
	assert.Contains(t, masterInstallComponents, "calico")
	assert.Contains(t, masterInstallComponents, "kubernetes")
	assert.Contains(t, workerInstallComponents, "containerd")
	assert.Contains(t, workerInstallComponents, "runc")
	assert.NotContains(t, workerInstallComponents, "calico")
	assert.Contains(t, workerInstallComponents, "kubernetes")
}
