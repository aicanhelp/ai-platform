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

package pipelines

import (
	"fmt"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/executor"
	"installer/pkg/executor/core"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/precheck"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/asset"
	"installer/pkg/prepare/binary"
)

// TestDestroyCluster_Success 测试销毁集群成功场景
func TestDestroyCluster_Success(t *testing.T) {
	args := common.Argument{
		EnableOffline: false,
		ArtifactPath:  "",
		SkipRmTmpFile: false,
	}

	// Mock runtime 创建
	patches1 := gomonkey.ApplyFunc(common.NewKubeRuntime,
		func(common.Argument) (*common.KubeRuntime, error) {
			return &common.KubeRuntime{
				Cluster: &cluster.Cluster{
					Spec: cluster.Spec{
						Nodes: cluster.Nodes{
							Masters: []*cluster.Node{{IP: TestAds1}},
						},
					},
				},
			}, nil
		})
	defer patches1.Reset()

	// Mock 敏感数据清理
	patches2 := gomonkey.ApplyMethod(&common.KubeRuntime{}, "ClearSensitiveData",
		func(*common.KubeRuntime) {
			// 模拟清理敏感数据，原方法无返回值
		})
	defer patches2.Reset()

	// Mock pipeline
	patches3 := gomonkey.ApplyFunc(NewDestroyClusterPipeline,
		func(*common.KubeRuntime) error {
			return nil
		})
	defer patches3.Reset()

	err := DestroyCluster(args)
	assert.NoError(t, err)
}

// TestDestroyCluster_OfflineArtifactPathError 测试离线模式下制品路径验证失败
func TestDestroyCluster_OfflineArtifactPathError(t *testing.T) {
	args := common.Argument{
		EnableOffline: true,
		ArtifactPath:  "/invalid/path",
	}

	// Mock 制品路径验证失败
	patches := gomonkey.ApplyFunc(params.VerifyArtifactPath,
		func(string) error {
			return fmt.Errorf("invalid artifact path")
		})
	defer patches.Reset()

	err := DestroyCluster(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid artifact path")
}

// TestDestroyCluster_NewKubeRuntimeError 测试 KubeRuntime 创建失败
func TestDestroyCluster_NewKubeRuntimeError(t *testing.T) {
	args := common.Argument{
		EnableOffline: false,
	}

	// Mock runtime 创建失败
	patches := gomonkey.ApplyFunc(common.NewKubeRuntime,
		func(common.Argument) (*common.KubeRuntime, error) {
			return nil, fmt.Errorf("failed to create runtime")
		})
	defer patches.Reset()

	err := DestroyCluster(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create runtime")
}

// TestDestroyCluster_PipelineError 测试管道执行失败
func TestDestroyCluster_PipelineError(t *testing.T) {
	args := common.Argument{
		EnableOffline: false,
	}

	// Mock runtime 创建成功
	patches1 := gomonkey.ApplyFunc(common.NewKubeRuntime,
		func(common.Argument) (*common.KubeRuntime, error) {
			return &common.KubeRuntime{
				Cluster: &cluster.Cluster{},
			}, nil
		})
	defer patches1.Reset()

	// Mock 敏感数据清理
	patches2 := gomonkey.ApplyMethod(&common.KubeRuntime{}, "ClearSensitiveData",
		func(*common.KubeRuntime) {
			// 模拟清理敏感数据，原方法无返回值
		})
	defer patches2.Reset()

	// Mock pipeline 执行失败
	patches3 := gomonkey.ApplyFunc(NewDestroyClusterPipeline,
		func(*common.KubeRuntime) error {
			return fmt.Errorf("pipeline execution failed")
		})
	defer patches3.Reset()

	err := DestroyCluster(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "pipeline execution failed")
}

// setupDestroyClusterMocks 设置销毁集群成功场景的 Mock
func setupDestroyClusterMocks() []*gomonkey.Patches {
	patches := make([]*gomonkey.Patches, 0)

	// Mock 预检查
	patches = append(patches, gomonkey.ApplyFunc(precheck.NewDestroyClusterFlightCheck,
		func(cluster.Nodes) *precheck.DestroyClusterFlightCheck {
			return &precheck.DestroyClusterFlightCheck{
				FlightCheck: &precheck.FlightCheck{},
			}
		}))

	patches = append(patches, gomonkey.ApplyMethod(&precheck.DestroyClusterFlightCheck{}, "StartCheck",
		func(*precheck.DestroyClusterFlightCheck) error {
			return nil
		}))

	// Mock 资产预处理
	patches = append(patches, gomonkey.ApplyFunc(asset.NewPretreatment,
		func(*cluster.Cluster) *asset.Pretreatment {
			return &asset.Pretreatment{}
		}))

	patches = append(patches, gomonkey.ApplyMethod(&asset.Pretreatment{}, "PretreatmentPackage",
		func(*asset.Pretreatment) error {
			return nil
		}))

	// Mock 图构建器
	patches = append(patches, gomonkey.ApplyFunc(executor.NewGraphBuilder,
		func() *executor.GraphBuilder {
			return &executor.GraphBuilder{}
		}))

	patches = append(patches, gomonkey.ApplyMethod(&executor.GraphBuilder{}, "UninstallGraphBuilder",
		func(*executor.GraphBuilder, []cluster.Component, *cluster.Node) (*core.DirectedGraph, error) {
			return &core.DirectedGraph{}, nil
		}))

	// Mock 组件映射获取
	patches = append(patches, gomonkey.ApplyFunc(common.GetInstallHubComponentMap,
		func(common.Argument) (map[string]cluster.Component, error) {
			return map[string]cluster.Component{
				"harbor": {Name: "harbor", Version: "2.8.0"},
			}, nil
		}))
	return patches
}

// TestNewDestroyClusterPipeline_Success 测试销毁集群管道成功场景
func TestNewDestroyClusterPipeline_Success(t *testing.T) {
	// 准备测试数据
	runtime := createTestKubeRuntime()

	// 设置 Mock
	patches := setupDestroyClusterMocks()
	defer resetPipelineMocks(patches)

	// Mock 制品准备
	patches9 := gomonkey.ApplyFunc(binary.NewPrepareArtifact,
		func(*core.DirectedGraph, cluster.Nodes, string) *binary.PrepareArtifact {
			return &binary.PrepareArtifact{}
		})
	defer patches9.Reset()

	patches10 := gomonkey.ApplyMethod(&binary.PrepareArtifact{}, "CreateClusterPrepareArtifact",
		func(*binary.PrepareArtifact, bool) error {
			return nil
		})
	defer patches10.Reset()

	// Mock 卸载
	patches11 := gomonkey.ApplyMethod(&executor.Executor{}, "Uninstall",
		func(*executor.Executor) error {
			return nil
		})
	defer patches11.Reset()

	// Mock 清理临时文件
	patches12 := gomonkey.ApplyFunc(common.RemoveTmpFiles,
		func(cluster.Nodes, bool) {
			// 模拟删除文件,原方法无返回值
		})
	defer patches12.Reset()

	// 执行测试
	err := NewDestroyClusterPipeline(runtime)

	// 验证结果
	assert.NoError(t, err)
}

func createTestKubeRuntimeForDestroy() *common.KubeRuntime {
	return &common.KubeRuntime{
		Cluster: &cluster.Cluster{
			Spec: cluster.Spec{
				Nodes: cluster.Nodes{
					Masters: []*cluster.Node{{IP: TestAds1}},
				},
			},
		},
	}
}

// TestNewDestroyClusterPipeline_FlightCheckError 测试预检查失败场景
func TestNewDestroyClusterPipeline_FlightCheckError(t *testing.T) {
	runtime := createTestKubeRuntimeForDestroy()

	// Mock 预检查失败
	patches1 := gomonkey.ApplyFunc(precheck.NewDestroyClusterFlightCheck,
		func(cluster.Nodes) *precheck.DestroyClusterFlightCheck {
			return &precheck.DestroyClusterFlightCheck{}
		})
	defer patches1.Reset()

	patches2 := gomonkey.ApplyMethod(&precheck.DestroyClusterFlightCheck{}, "StartCheck",
		func(*precheck.DestroyClusterFlightCheck) error {
			return fmt.Errorf("flight check failed")
		})
	defer patches2.Reset()

	err := NewDestroyClusterPipeline(runtime)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "flight check failed")
}

// TestNewDestroyClusterPipeline_PretreatmentError 测试资产预处理失败场景
func TestNewDestroyClusterPipeline_PretreatmentError(t *testing.T) {
	runtime := createTestKubeRuntimeForDestroy()

	// Mock 预检查成功
	patches1 := gomonkey.ApplyFunc(precheck.NewDestroyClusterFlightCheck,
		func(cluster.Nodes) *precheck.DestroyClusterFlightCheck {
			return &precheck.DestroyClusterFlightCheck{}
		})
	defer patches1.Reset()

	patches2 := gomonkey.ApplyMethod(&precheck.DestroyClusterFlightCheck{}, "StartCheck",
		func(*precheck.DestroyClusterFlightCheck) error {
			return nil
		})
	defer patches2.Reset()

	// Mock 资产预处理失败
	patches3 := gomonkey.ApplyFunc(asset.NewPretreatment,
		func(*cluster.Cluster) *asset.Pretreatment {
			return &asset.Pretreatment{}
		})
	defer patches3.Reset()

	patches4 := gomonkey.ApplyMethod(&asset.Pretreatment{}, "PretreatmentPackage",
		func(*asset.Pretreatment) error {
			return fmt.Errorf("pretreatment failed")
		})
	defer patches4.Reset()

	err := NewDestroyClusterPipeline(runtime)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "pretreatment failed")
}
