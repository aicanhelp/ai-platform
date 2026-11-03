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
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/executor"
	"installer/pkg/executor/core"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/operate"
	"installer/pkg/precheck"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/asset"
	"installer/pkg/prepare/binary"
	"installer/pkg/prepare/hub"
)

func setupMocks(runtime *common.KubeRuntime) *gomonkey.Patches {
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(precheck.NewAddNodesFlightCheck, func(nodes cluster.Nodes) *precheck.AddNodesFlightCheck {
		mockFlightCheck := &precheck.AddNodesFlightCheck{
			FlightCheck: &precheck.FlightCheck{
				NodesOperate: &operate.NodesOperate{},
			},
		}
		return mockFlightCheck
	})
	patches.ApplyMethod(&precheck.AddNodesFlightCheck{}, "StartCheck", func(*precheck.AddNodesFlightCheck) error {
		return nil
	})
	patches.ApplyFunc(common.UnCompressPkgToOptPath, func(*cluster.Node) error {
		return nil
	})
	patches.ApplyFunc(asset.NewPretreatment, func(*cluster.Cluster) *asset.Pretreatment {
		return &asset.Pretreatment{}
	})
	patches.ApplyMethod(&asset.Pretreatment{}, "PretreatmentPackage", func(*asset.Pretreatment) error {
		return nil
	})
	patches.ApplyFunc(hub.InstallHarborCfg, func(*common.KubeRuntime) error {
		return nil
	})
	patches.ApplyFunc(executor.NewGraphBuilder, func() *executor.GraphBuilder {
		return &executor.GraphBuilder{}
	})
	patches.ApplyMethod(&executor.GraphBuilder{}, "InstallGraphBuilder",
		func(*executor.GraphBuilder, []cluster.Component, *cluster.Node) (*core.DirectedGraph, error) {
			return &core.DirectedGraph{}, nil
		})
	patches.ApplyFunc(executor.NewExecutor, func(string, cluster.Nodes, *core.DirectedGraph) *executor.Executor {
		return &executor.Executor{}
	})
	patches.ApplyMethod(&executor.Executor{}, "FlightCheck", func(*executor.Executor) {
		// 模拟检查,原方法无返回值
	})
	patches.ApplyFunc(binary.NewPrepareArtifact,
		func(*core.DirectedGraph, cluster.Nodes, string) *binary.PrepareArtifact {
			return &binary.PrepareArtifact{}
		})
	return patches
}

func TestNewAddNodesPipeline_Success(t *testing.T) {
	// 准备测试数据
	runtime := createTestKubeRuntime()

	// Mock 所有外部依赖
	patches := setupMocks(runtime)
	defer patches.Reset()

	patches.ApplyMethod(&executor.Executor{}, "AddNodes", func(e *executor.Executor) error {
		return nil
	})

	patches.ApplyFunc(common.PostAddNodeCheck, func(*cluster.Node) {
		// 模拟检查,原方法无返回值
	})

	patches.ApplyFunc(common.RemoveTmpFiles, func(cluster.Nodes, bool) {
		// 模拟删除文件,原方法无返回值
	})
	patches.ApplyMethod(&binary.PrepareArtifact{}, "CreateClusterPrepareArtifact",
		func(*binary.PrepareArtifact, bool) error {
			return nil
		})

	// 执行测试
	err := NewAddNodesPipeline(runtime)

	// 验证结果
	assert.NoError(t, err)
}

func TestNewAddNodesPipeline_FlightCheckError(t *testing.T) {
	runtime := createTestKubeRuntimeForDestroy()

	// Mock FlightCheck 返回错误
	patches := setupAddNodesMocks(runtime)
	defer patches.Reset()

	patches.ApplyMethod(&precheck.AddNodesFlightCheck{}, "StartCheck", func(*precheck.AddNodesFlightCheck) error {
		return errors.New("flight check failed")
	})

	err := NewAddNodesPipeline(runtime)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "flight check failed")
}

func setupAddNodesMocks(runtime *common.KubeRuntime) *gomonkey.Patches {
	patches := gomonkey.ApplyFunc(precheck.NewAddNodesFlightCheck, func(cluster.Nodes) *precheck.AddNodesFlightCheck {
		return &precheck.AddNodesFlightCheck{
			FlightCheck: &precheck.FlightCheck{
				NodesOperate: &operate.NodesOperate{},
			},
		}
	})
	return patches
}

func TestNewAddNodesPipeline_UnCompressError(t *testing.T) {
	runtime := createTestKubeRuntimeForDestroy()

	patches := setupAddNodesMocks(runtime)
	defer patches.Reset()

	patches.ApplyMethod(&precheck.AddNodesFlightCheck{}, "StartCheck", func(*precheck.AddNodesFlightCheck) error {
		return nil
	})

	patches.ApplyFunc(common.UnCompressPkgToOptPath, func(*cluster.Node) error {
		return errors.New("uncompress failed")
	})

	err := NewAddNodesPipeline(runtime)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "uncompress failed")
}

func TestAddNodes_Success(t *testing.T) {
	args := common.Argument{
		FilePath:      "/tmp/config.yaml",
		FuyaoZone:     "cn",
		EnableOffline: false,
		ArtifactPath:  "",
		SkipRmTmpFile: false,
	}

	// Mock 参数验证
	patches := gomonkey.ApplyFunc(params.VerifyZone, func(string, string) (bool, error) {
		return true, nil
	})
	defer patches.Reset()

	// Mock runtime 创建
	patches.ApplyFunc(common.NewKubeRuntime, func(common.Argument) (*common.KubeRuntime, error) {
		return createTestKubeRuntimeForDestroy(), nil
	})

	// Mock ClearSensitiveData
	patches.ApplyMethod(&common.KubeRuntime{}, "ClearSensitiveData", func(*common.KubeRuntime) {
		// 模拟清理敏感数据,原方法无返回值
	})

	// Mock pipeline
	patches.ApplyFunc(NewAddNodesPipeline, func(*common.KubeRuntime) error {
		return nil
	})

	err := AddNodes(args)

	assert.NoError(t, err)
}

func TestAddNodes_InvalidZone(t *testing.T) {
	args := common.Argument{
		FuyaoZone: "invalid-zone",
	}

	patches := gomonkey.ApplyFunc(params.VerifyZone, func(string, string) (bool, error) {
		return false, errors.New("invalid zone")
	})
	defer patches.Reset()

	err := AddNodes(args)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid zone")
}

func TestAddNodes_OfflineWithInvalidArtifactPath(t *testing.T) {
	args := common.Argument{
		FuyaoZone:     "cn",
		EnableOffline: true,
		ArtifactPath:  "",
	}

	patches := gomonkey.ApplyFunc(params.VerifyZone, func(string, string) (bool, error) {
		return true, nil
	})
	defer patches.Reset()

	patches.ApplyFunc(params.VerifyArtifactPath, func(string) error {
		return errors.New("invalid artifact path")
	})

	err := AddNodes(args)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid artifact path")
}

func TestAddNodes_RuntimeCreationError(t *testing.T) {
	args := common.Argument{
		FuyaoZone:     "cn",
		EnableOffline: false,
	}

	patches := gomonkey.ApplyFunc(params.VerifyZone, func(string, string) (bool, error) {
		return true, nil
	})
	defer patches.Reset()

	patches.ApplyFunc(common.NewKubeRuntime, func(common.Argument) (*common.KubeRuntime, error) {
		return nil, errors.New("runtime creation failed")
	})

	err := AddNodes(args)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "runtime creation failed")
}
