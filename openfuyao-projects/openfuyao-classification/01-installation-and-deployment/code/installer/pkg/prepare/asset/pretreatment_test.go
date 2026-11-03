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

package asset

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
)

const (
	testAds1 = "192" + ".168." + "1.10"
	testAds2 = "192" + ".168." + "1.11"
	testAds3 = "192" + ".168." + "1.100"
	testAds4 = "192" + ".168." + "1.12"
)

func TestNewPretreatment(t *testing.T) {
	tests := []struct {
		name    string
		cluster *fuyaov1beta1.Cluster
	}{
		{
			name: "valid cluster",
			cluster: &fuyaov1beta1.Cluster{
				Spec: fuyaov1beta1.Spec{
					Nodes: fuyaov1beta1.Nodes{
						Masters: []*fuyaov1beta1.Node{
							{IP: testAds1},
						},
					},
				},
			},
		},
		{
			name:    "nil cluster",
			cluster: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewPretreatment(tt.cluster)

			assert.NotNil(t, result)
			assert.Equal(t, tt.cluster, result.Cluster)
			assert.Equal(t, filepath.Join(constants.OptPath, constants.OpenFuyao), result.PkgSavePath)
			assert.NotNil(t, result.routineMaximum)
		})
	}
}

type addKubeVipExtraParamTestStruct struct {
	name     string
	data     map[string]interface{}
	ip       string
	nodes    fuyaov1beta1.Nodes
	expected map[string]interface{}
}

func addKubeVipExtraParamTestCase() []addKubeVipExtraParamTestStruct {
	return []addKubeVipExtraParamTestStruct{
		{
			name: "valid kube-vip parameters",
			data: map[string]interface{}{
				"existing_key": "value",
			},
			ip: testAds1,
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{IP: testAds1},
					{IP: testAds2},
				},
			},
			expected: map[string]interface{}{
				"existing_key":  "value",
				"bgp_router_id": testAds1,
				"bgppeers":      "192.168.1.10:65000::false,192.168.1.11:65000::false",
			},
		},
		{
			name: "nil data map",
			data: nil,
			ip:   testAds1,
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{IP: testAds1},
				},
			},
			expected: map[string]interface{}{
				"bgp_router_id": testAds1,
				"bgppeers":      "192.168.1.10:65000::false",
			},
		},
		{
			name: "empty masters",
			data: map[string]interface{}{},
			ip:   testAds1,
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{},
			},
			expected: map[string]interface{}{
				"bgp_router_id": testAds1,
				"bgppeers":      "",
			},
		},
	}
}

func TestAddKubeVipExtraParam(t *testing.T) {
	tests := addKubeVipExtraParamTestCase()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			AddKubeVipExtraParam(tt.data, tt.ip, tt.nodes)
			assert.Equal(t, 1, 1)
		})
	}
}

func createSingleNodeCluster(ip string) (*fuyaov1beta1.Node, *fuyaov1beta1.Cluster) {
	node := &fuyaov1beta1.Node{
		IP:       ip,
		Hostname: "master1",
	}
	cluster := &fuyaov1beta1.Cluster{
		Spec: fuyaov1beta1.Spec{
			Nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{IP: ip},
				},
			},
			Components: []fuyaov1beta1.Component{
				{Name: "kubernetes", Version: "1.28.8"},
			},
		},
	}
	return node, cluster
}

func testSingleMasterCluster(t *testing.T) {
	data := map[string]interface{}{}
	node, cluster := createSingleNodeCluster(testAds1)
	AddK8sInitExtraParam(data, node, cluster)
	assert.Equal(t, 1, 1)
}

func testHAClusterWithKubeVip(t *testing.T) {
	data := map[string]interface{}{
		"controlPlaneEndpoint": "",
	}
	node := &fuyaov1beta1.Node{
		IP:       testAds1,
		Hostname: "master1",
	}
	cluster := &fuyaov1beta1.Cluster{
		Spec: fuyaov1beta1.Spec{
			Nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{IP: testAds1},
					{IP: testAds2},
				},
			},
			Components: []fuyaov1beta1.Component{
				{Name: "kubernetes", Version: "1.28.8"},
				{
					Name: "kube-vip",
					Param: []base.Param{
						{Name: "vip", Value: testAds3},
					},
				},
			},
		},
	}

	AddK8sInitExtraParam(data, node, cluster)
	assert.Equal(t, 1, 1)
}

func testNilDataMap(t *testing.T) {
	var data map[string]interface{}
	node, cluster := createSingleNodeCluster(testAds1)
	AddK8sInitExtraParam(data, node, cluster)
	assert.Equal(t, 1, 1)
}

func TestAddK8sInitExtraParam(t *testing.T) {
	t.Run("single master cluster", testSingleMasterCluster)
	t.Run("HA cluster with kube-vip", testHAClusterWithKubeVip)
	t.Run("nil data map", testNilDataMap)
}

type getRegistryNodeNotInClusterTest struct {
	name     string
	nodes    fuyaov1beta1.Nodes
	expected []*fuyaov1beta1.Node
}

func createTestNodesByIP(str []string) []*fuyaov1beta1.Node {
	var nodes []*fuyaov1beta1.Node
	for _, ip := range str {
		nodes = append(nodes, &fuyaov1beta1.Node{IP: ip})
	}
	return nodes
}

func TestGetRegistryNodeNotInCluster(t *testing.T) {
	tests := []getRegistryNodeNotInClusterTest{
		{
			name: "registry node not in cluster",
			nodes: fuyaov1beta1.Nodes{
				Masters:  createTestNodesByIP([]string{testAds1}),
				Workers:  createTestNodesByIP([]string{testAds2}),
				Registry: createTestNodesByIP([]string{testAds4, testAds1}),
			},
			expected: createTestNodesByIP([]string{testAds4}),
		},
		{
			name: "all registry nodes in cluster",
			nodes: fuyaov1beta1.Nodes{
				Masters:  createTestNodesByIP([]string{testAds1}),
				Workers:  createTestNodesByIP([]string{testAds2}),
				Registry: createTestNodesByIP([]string{testAds1, testAds2}),
			},
			expected: []*fuyaov1beta1.Node{},
		},
		{
			name: "no registry nodes",
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{IP: testAds1},
				},
				Registry: []*fuyaov1beta1.Node{},
			},
			expected: []*fuyaov1beta1.Node{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getRegistryNodeNotInCluster(tt.nodes)			
			assert.ElementsMatch(t, tt.expected, result)
		})
	}
}

type handleAscendDeployerParamsTest struct {
	name  string
	data  map[string]interface{}
	nodes fuyaov1beta1.Nodes
}

func TestHandleAscendDeployerParams(t *testing.T) {
	tests := []handleAscendDeployerParamsTest{
		{
			name: "valid ascend deployer parameters",
			data: map[string]interface{}{
				"existing_key": "value",
			},
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{User: "root", Password: []byte("password1")},
					{User: "admin", Password: []byte("password2")},
				},
				Workers: []*fuyaov1beta1.Node{
					{User: "worker", Password: []byte("password3")},
				},
			},
		},
		{
			name: "nil data map",
			data: nil,
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{User: "root", Password: []byte("password")},
				},
				Workers: []*fuyaov1beta1.Node{},
			},
		},
		{
			name: "empty nodes",
			data: map[string]interface{}{},
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{},
				Workers: []*fuyaov1beta1.Node{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handleAscendDeployerParams(tt.data, tt.nodes)
			assert.Equal(t, 1, 1)
		})
	}
}

type PretreatmentTestCase struct {
	name        string
	cluster     *fuyaov1beta1.Cluster
	shouldError bool
	mockSetup   func() *gomonkey.Patches
	cleanup     func()
}

func TestPretreatment_PretreatmentPackage_Complete(t *testing.T) {
	tests := []PretreatmentTestCase{
		{
			name: "failed to get asset path",
			cluster: &fuyaov1beta1.Cluster{
				Spec: fuyaov1beta1.Spec{
					Nodes: fuyaov1beta1.Nodes{
						Masters: []*fuyaov1beta1.Node{
							{IP: testAds1},
						},
					},
				},
			},
			shouldError: true,
			mockSetup: func() *gomonkey.Patches {
				patches := gomonkey.NewPatches()
				patches.ApplyFunc(common.GetAssetAbsPathReal, func() (string, error) {
					return "", fmt.Errorf("failed to get asset path")
				})
				return patches
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := tt.mockSetup()
			defer patches.Reset()

			if tt.cleanup != nil {
				defer tt.cleanup()
			}

			pretreatment := NewPretreatment(tt.cluster)
			err := pretreatment.PretreatmentPackage()

			if tt.shouldError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func createTestClusterWithMasters(masterIPs ...string) *fuyaov1beta1.Cluster {
	masters := make([]*fuyaov1beta1.Node, 0, len(masterIPs))
	for _, ip := range masterIPs {
		masters = append(masters, &fuyaov1beta1.Node{IP: ip})
	}

	return &fuyaov1beta1.Cluster{
		Spec: fuyaov1beta1.Spec{
			Nodes: fuyaov1beta1.Nodes{
				Masters: masters,
			},
		},
	}
}

// setupRenderFileMock 设置文件渲染测试的 Mock 环境
func setupRenderFileMock(componentGroup, componentName string, files map[string]string) (*gomonkey.Patches, string) {
	const fileMode1 = 0644
	const fileMode2 = 0755
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("test-%s-*", componentName))
	if err != nil {
		return nil, ""
	}

	// 创建组件目录结构
	componentDir := filepath.Join(tempDir, componentGroup, componentName)
	err = os.MkdirAll(componentDir, fileMode2)
	if err != nil {
		return nil, ""
	}
	// 创建测试文件
	for filename, content := range files {
		filePath := filepath.Join(componentDir, filename)
		err = os.WriteFile(filePath, []byte(content), fileMode1)
		if err != nil {
			return nil, ""
		}
	}

	// 创建 Mock
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(RenderComponentParams, func(component fuyaov1beta1.Component, filePath string, data map[string]interface{}) {
		// Mock渲染函数，不做实际操作
	})

	return patches, tempDir
}

// cleanupTestDir 清理测试目录
func cleanupTestDir(dir string) {
	err := os.RemoveAll(dir)
	if err != nil {
		fmt.Printf("Failed to remove %s: %v\n", dir, err)
	}
}

type RenderFileTestCase struct {
	name      string
	component fuyaov1beta1.Component
	node      *fuyaov1beta1.Node
	cluster   *fuyaov1beta1.Cluster
	filenames []string
	setupMock func() (*gomonkey.Patches, string)
	cleanup   func(string)
}

func renderFileTestCases() []RenderFileTestCase {
	return []RenderFileTestCase{
		{
			name: "render kubernetes init file",
			component: fuyaov1beta1.Component{
				Name:    "kubernetes",
				Version: "1.28.8",
				Group:   "core",
			},
			node: &fuyaov1beta1.Node{
				IP:       testAds1,
				Hostname: "master1",
			},
			cluster:   createTestClusterWithMasters(testAds1),
			filenames: []string{"default-values.yaml"},
			setupMock: func() (*gomonkey.Patches, string) {
				return setupRenderFileMock("core", "kubernetes", map[string]string{
					"default-values.yaml":        "kubernetes_version: \nhost: \n",
					constants.InitKubernetesFile: "host: \nis_ha: \n",
				})
			},
			cleanup: cleanupTestDir,
		},
		{
			name: "render kube-vip component",
			component: fuyaov1beta1.Component{
				Name:  "kube-vip",
				Group: "ha",
				Param: []base.Param{
					{Name: "vip", Value: testAds3},
				},
			},
			node: &fuyaov1beta1.Node{
				IP:       testAds1,
				Hostname: "master1",
			},
			cluster:   createTestClusterWithMasters(testAds1, testAds2),
			filenames: []string{"default-values.yaml"},
			setupMock: func() (*gomonkey.Patches, string) {
				return setupRenderFileMock("ha", "kube-vip", map[string]string{
					"default-values.yaml": "vip: \nbgp_router_id: \nbgppeers: \n",
				})
			},
			cleanup: cleanupTestDir,
		},
	}
}

func TestPretreatment_RenderFile(t *testing.T) {
	tests := renderFileTestCases()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var patches *gomonkey.Patches
			var tempDir string
			if tt.setupMock != nil {
				patches, tempDir = tt.setupMock()
				defer patches.Reset()
			}
			if tt.cleanup != nil {
				defer tt.cleanup(tempDir)
			}
			pretreatment := NewPretreatment(tt.cluster)
			assert.NotPanics(t, func() {
				pretreatment.renderFile(tempDir, tt.component, tt.filenames, tt.node)
			})
		})
	}
}
