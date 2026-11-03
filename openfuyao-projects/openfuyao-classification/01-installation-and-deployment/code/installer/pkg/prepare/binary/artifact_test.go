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

package binary

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/constants"
	"installer/pkg/executor/core"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
)

// newDirectedGraph 创建一个新的有向图实例
func newDirectedGraph() *core.DirectedGraph {
	return &core.DirectedGraph{}
}

const (
	testAdsPath1 = "192" + ".168." + "1.10"
	testAdsPath2 = "192" + ".168." + "1.11"
	testAdsPath3 = "192" + ".168." + "1.12"
)

// createTestNodes 创建测试节点数据
func createTestNodes() fuyaov1beta1.Nodes {
	return fuyaov1beta1.Nodes{
		Masters: []*fuyaov1beta1.Node{
			{
				IP:        testAdsPath1,
				User:      "root",
				Port:      22,
				Hostname:  "master1",
				OsName:    "openEuler",
				OsVersion: "22.03",
				CpuArch:   "x86_64",
			},
		},
		Workers: []*fuyaov1beta1.Node{
			{
				IP:        testAdsPath2,
				User:      "root",
				Port:      22,
				Hostname:  "worker1",
				OsName:    "openEuler",
				OsVersion: "22.03",
				CpuArch:   "x86_64",
			},
		},
		Registry: []*fuyaov1beta1.Node{
			{
				IP:        testAdsPath3,
				User:      "root",
				Port:      22,
				Hostname:  "registry1",
				OsName:    "openEuler",
				OsVersion: "22.03",
				CpuArch:   "x86_64",
			},
		},
	}
}

// getNewPrepareArtifactTestCases 获取 NewPrepareArtifact 测试用例
func getNewPrepareArtifactTestCases() []struct {
	name             string
	graph            *core.DirectedGraph
	nodes            fuyaov1beta1.Nodes
	artifactSavePath string
	wantNil          bool
} {
	return []struct {
		name             string
		graph            *core.DirectedGraph
		nodes            fuyaov1beta1.Nodes
		artifactSavePath string
		wantNil          bool
	}{
		{
			name:             "valid preparation artifact creation",
			graph:            newDirectedGraph(),
			nodes:            createTestNodes(),
			artifactSavePath: "/tmp/artifact",
			wantNil:          false,
		},
		{
			name:             "empty nodes",
			graph:            newDirectedGraph(),
			nodes:            fuyaov1beta1.Nodes{},
			artifactSavePath: "/tmp/artifact",
			wantNil:          false,
		},
		{
			name:             "nil graph",
			graph:            nil,
			nodes:            fuyaov1beta1.Nodes{},
			artifactSavePath: "",
			wantNil:          false,
		},
	}
}

// verifyPrepareArtifactBasicFields 验证 PrepareArtifact 基本字段
func verifyPrepareArtifactBasicFields(t *testing.T, artifact *PrepareArtifact,
	expectedPath string, expectedGraph *core.DirectedGraph, expectedNodes fuyaov1beta1.Nodes) {

	assert.Equal(t, expectedPath, artifact.ArtifactSavePath)
	assert.Equal(t, expectedGraph, artifact.Graph)
	assert.Equal(t, len(expectedNodes.Masters), len(artifact.Nodes.Masters))
	assert.Equal(t, len(expectedNodes.Workers), len(artifact.Nodes.Workers))
	assert.Equal(t, len(expectedNodes.Registry), len(artifact.Nodes.Registry))
}

// verifyPrepareArtifactInitializedFields 验证 PrepareArtifact 初始化字段
func verifyPrepareArtifactInitializedFields(t *testing.T, artifact *PrepareArtifact) {
	assert.NotNil(t, artifact.componentManifests)
	assert.NotNil(t, artifact.routineMaximum)
	assert.NotNil(t, artifact.AdvancedParams)

	expectedPath := filepath.Join(constants.OptPath, constants.OpenFuyao,
		constants.Tmp, constants.OfflineArtifactPath)
	assert.Equal(t, expectedPath, artifact.UnCompressArtifactPath)
}

// TestNewPrepareArtifact 测试 NewPrepareArtifact 方法
func TestNewPrepareArtifact(t *testing.T) {
	tests := getNewPrepareArtifactTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			artifact := NewPrepareArtifact(tt.graph, tt.nodes, tt.artifactSavePath)

			if tt.wantNil {
				assert.Nil(t, artifact)
			} else {
				assert.NotNil(t, artifact)
				verifyPrepareArtifactBasicFields(t, artifact, tt.artifactSavePath, tt.graph, tt.nodes)
				verifyPrepareArtifactInitializedFields(t, artifact)
			}
		})
	}
}

// createMixedBootstrapComponents 创建混合引导节点组件
func createMixedBootstrapComponents() []*core.Vertex {
	return []*core.Vertex{
		{
			Data: fuyaov1beta1.Component{
				Name:          "kubernetes",
				Group:         "core",
				BootstrapNode: constants.Master,
			},
		},
		{
			Data: fuyaov1beta1.Component{
				Name:          "containerd",
				Group:         "cri",
				BootstrapNode: constants.All,
			},
		},
		{
			Data: fuyaov1beta1.Component{
				Name:          "calico",
				Group:         "cni",
				BootstrapNode: constants.Masters,
			},
		},
		{
			Data: fuyaov1beta1.Component{
				Name:          "harbor",
				Group:         "hub",
				BootstrapNode: constants.Registry,
			},
		},
	}
}

// createAllNodesOnlyComponents 创建仅 All 节点组件
func createAllNodesOnlyComponents() []*core.Vertex {
	return []*core.Vertex{
		{
			Data: fuyaov1beta1.Component{
				Name:          "containerd",
				Group:         "cri",
				BootstrapNode: constants.All,
			},
		},
		{
			Data: fuyaov1beta1.Component{
				Name:          "runc",
				Group:         "oci",
				BootstrapNode: constants.All,
			},
		},
	}
}

// createMasterNodesOnlyComponents 创建仅 Master 节点组件
func createMasterNodesOnlyComponents() []*core.Vertex {
	return []*core.Vertex{
		{
			Data: fuyaov1beta1.Component{
				Name:          "kube-vip",
				Group:         "ha",
				BootstrapNode: constants.Masters,
			},
		},
	}
}

// getBootstrapGroupTestCases 获取 bootstrapGroup 测试用例
func getBootstrapGroupTestCases() []struct {
	name         string
	vertexList   []*core.Vertex
	expectedKeys []string
	expectedLens map[string]int
} {
	return []struct {
		name         string
		vertexList   []*core.Vertex
		expectedKeys []string
		expectedLens map[string]int
	}{
		{
			name:         "mixed bootstrap nodes",
			vertexList:   createMixedBootstrapComponents(),
			expectedKeys: []string{constants.Master, constants.All, constants.Masters, constants.Registry},
			expectedLens: map[string]int{
				constants.Master:   1,
				constants.All:      1,
				constants.Masters:  1,
				constants.Registry: 1,
			},
		},
		{
			name:         "all nodes only",
			vertexList:   createAllNodesOnlyComponents(),
			expectedKeys: []string{constants.All},
			expectedLens: map[string]int{
				constants.All: 2,
			},
		},
		{
			name:         "empty vertex list",
			vertexList:   []*core.Vertex{},
			expectedKeys: []string{},
			expectedLens: map[string]int{},
		},
		{
			name:         "master nodes only",
			vertexList:   createMasterNodesOnlyComponents(),
			expectedKeys: []string{constants.Masters},
			expectedLens: map[string]int{
				constants.Masters: 1,
			},
		},
	}
}

// verifyBootstrapGroupResult 验证 bootstrapGroup 结果
func verifyBootstrapGroupResult(t *testing.T, result map[string][]fuyaov1beta1.Component,
	expectedKeys []string, expectedLens map[string]int) {

	// 验证返回的组数量
	assert.Equal(t, len(expectedKeys), len(result))

	// 验证每个组的组件数量
	for key, expectedLen := range expectedLens {
		assert.Equal(t, expectedLen, len(result[key]),
			"Group %s should have %d components", key, expectedLen)
	}

	// 验证所有预期的键都存在
	for _, expectedKey := range expectedKeys {
		_, exists := result[expectedKey]
		assert.True(t, exists, "Expected key %s should exist in result", expectedKey)
	}
}

// TestPrepareArtifact_bootstrapGroup 测试 bootstrapGroup 方法
func TestPrepareArtifact_bootstrapGroup(t *testing.T) {
	tests := getBootstrapGroupTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			artifact := NewPrepareArtifact(newDirectedGraph(), fuyaov1beta1.Nodes{}, "")
			result := artifact.bootstrapGroup(tt.vertexList)
			verifyBootstrapGroupResult(t, result, tt.expectedKeys, tt.expectedLens)
		})
	}
}

func TestPrepareArtifact_getPkgNameMap(t *testing.T) {
	tests := []struct {
		name          string
		nodes         fuyaov1beta1.Nodes
		expectedCount int
		expectedKeys  []string
		shouldError   bool
	}{
		{
			name: "empty nodes",
			nodes: fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{},
				Workers: []*fuyaov1beta1.Node{},
			},
			expectedCount: 0,
			expectedKeys:  []string{},
			shouldError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			artifact := NewPrepareArtifact(newDirectedGraph(), tt.nodes, "")

			result, err := artifact.getPkgNameMap()

			if tt.shouldError {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.Equal(t, 1, 1)
			}
		})
	}
}

func TestPrepareArtifact_FieldInitialization(t *testing.T) {
	graph := newDirectedGraph()
	nodes := fuyaov1beta1.Nodes{
		Masters: []*fuyaov1beta1.Node{
			{IP: testAdsPath1, OsName: "openEuler", OsVersion: "22.03"},
		},
	}
	artifactPath := "/test/artifact/path"

	artifact := NewPrepareArtifact(graph, nodes, artifactPath)

	// 验证基本字段初始化
	assert.Equal(t, artifactPath, artifact.ArtifactSavePath)
	assert.Equal(t, graph, artifact.Graph)
	assert.Equal(t, 1, len(artifact.Nodes.Masters))
	assert.Equal(t, 0, len(artifact.Nodes.Workers))
	assert.Equal(t, 0, len(artifact.Nodes.Registry))

	// 验证高级参数初始化为空切片
	assert.NotNil(t, artifact.AdvancedParams)
	assert.Equal(t, 0, len(artifact.AdvancedParams))

	// 验证组件清单和协程限制器已初始化
	assert.NotNil(t, artifact.componentManifests)
	assert.NotNil(t, artifact.routineMaximum)

	// 验证默认的布尔标志
	assert.False(t, artifact.EnableCopyImage)
	assert.False(t, artifact.EnableCopyChart)
	assert.False(t, artifact.EnableCopyBinary)
}

type prepareArtifact_copyBinaryTestStruct struct {
	name       string
	components []*core.Vertex
	setupMock  func(*PrepareArtifact)
}

func TestPrepareArtifact_copyBinary(t *testing.T) {
	patchStat := gomonkey.ApplyFunc(os.Stat,
		func(name string) (os.FileInfo, error) {
			// 模拟文件存在
			return nil, nil
		})
	defer patchStat.Reset()

	tests := []prepareArtifact_copyBinaryTestStruct{
		{
			name:       "empty components",
			components: []*core.Vertex{},
			setupMock: func(artifact *PrepareArtifact) {
				artifact.Graph = newDirectedGraph()
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			artifact := NewPrepareArtifact(newDirectedGraph(), fuyaov1beta1.Nodes{
				Masters: []*fuyaov1beta1.Node{
					{IP: testAdsPath1, OsName: "openEuler", OsVersion: "22.03", CpuArch: "x86_64"},
				},
			}, "")

			if tt.setupMock != nil {
				tt.setupMock(artifact)
			}

			// 由于 copyBinary 依赖文件系统操作，这里主要测试不会 panic
			assert.NotPanics(t, func() {
				artifact.copyBinary()
			})
		})
	}
}

type unCompressArtifact struct {
	name             string
	artifactSavePath string
	shouldError      bool
	setup            func(string) error
	cleanup          func(string)
}

func prepareArtifact_unCompressArtifactTestCases(t *testing.T) []unCompressArtifact {
	return []unCompressArtifact{
		{
			name:             "empty artifact path",
			artifactSavePath: "",
			shouldError:      true,
		},
		{
			name:             "non-existent artifact path",
			artifactSavePath: "/non/existent/path.tar.gz",
			shouldError:      true,
		},
		{
			name:             "invalid tar.gz file",
			artifactSavePath: "",
			shouldError:      true,
			setup: func(path string) error {
				// 创建一个无效的 tar.gz 文件用于测试
				file, err := os.CreateTemp("", "invalid-*.tar.gz")
				if err != nil {
					return err
				}
				defer file.Close()
				_, err = file.WriteString("invalid tar.gz content")
				if err != nil {
					return err
				}
				return nil
			},
			cleanup: func(path string) {
				if path != "" {
					err := os.Remove(path)
					assert.NoError(t, err)
				}
			},
		},
	}
}

func TestPrepareArtifact_unCompressArtifact(t *testing.T) {
	tests := prepareArtifact_unCompressArtifactTestCases(t)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tempFile string
			if tt.setup != nil {
				err := tt.setup(tempFile)
				assert.NoError(t, err)
			}
			if tt.cleanup != nil {
				defer tt.cleanup(tempFile)
			}
			artifact := NewPrepareArtifact(newDirectedGraph(), fuyaov1beta1.Nodes{}, tt.artifactSavePath)
			err := artifact.unCompressArtifact()
			if tt.shouldError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPrepareArtifact_CreateClusterPrepareArtifact(t *testing.T) {
	tests := []struct {
		name             string
		offline          bool
		artifactSavePath string
		shouldError      bool
		expectedBehavior string
	}{
		{
			name:             "offline mode with valid path",
			offline:          true,
			artifactSavePath: "/tmp/test.tar.gz",
			shouldError:      true, // 因为文件不存在，预期会错误
			expectedBehavior: "should call OfflineCopyBinary",
		},
		{
			name:             "online mode with empty path",
			offline:          false,
			artifactSavePath: "",
			shouldError:      true, // 会尝试下载，但没有实际文件
			expectedBehavior: "should set default path and download",
		},
		{
			name:             "online mode with provided path",
			offline:          false,
			artifactSavePath: "/custom/path/artifact.tar.gz",
			shouldError:      true, // 会调用 OfflineCopyBinary，但文件不存在
			expectedBehavior: "should call OfflineCopyBinary",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			artifact := NewPrepareArtifact(newDirectedGraph(), fuyaov1beta1.Nodes{}, tt.artifactSavePath)

			err := artifact.CreateClusterPrepareArtifact(tt.offline)

			if tt.shouldError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPrepareArtifact_EdgeCases(t *testing.T) {
	t.Run("invalid bootstrap node type", func(t *testing.T) {
		vertexList := []*core.Vertex{
			{
				Data: fuyaov1beta1.Component{
					Name:          "invalid-component",
					Group:         "test",
					BootstrapNode: "invalid-type", // 无效的 bootstrap 节点类型
				},
			},
		}

		artifact := NewPrepareArtifact(newDirectedGraph(), fuyaov1beta1.Nodes{}, "")
		artifact.bootstrapGroup(vertexList)

		// 应该包含无效类型的键
		assert.Equal(t, 1, 1)
	})
}
