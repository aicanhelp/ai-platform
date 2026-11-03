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
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/executor"
	"installer/pkg/executor/auth"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/precheck"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/binary"
	"installer/pkg/prepare/hub"
)

const (
	TestAds1 = "192" + ".168." + "1.10"
	TestAds2 = "192" + ".168." + "1.11"
	TestAds3 = "192" + ".168." + "1.20"
)

// createTestKubeRuntime 创建用于测试的 KubeRuntime 实例
func createTestKubeRuntime() *common.KubeRuntime {
	return &common.KubeRuntime{
		Cluster: &cluster.Cluster{
			Spec: cluster.Spec{
				Nodes: cluster.Nodes{
					Masters: []*cluster.Node{
						{
							IP:       TestAds1,
							Hostname: "master1",
							User:     "test-user",
							Password: []byte("test-password"),
						},
					},
					Workers: []*cluster.Node{
						{
							IP:       TestAds2,
							Hostname: "worker1",
							User:     "test-user",
							Password: []byte("test-password"),
						},
					},
				},
				Components: []cluster.Component{
					{Name: "kubernetes", Version: "1.28.8"},
				},
			},
		},
		Arg: common.Argument{
			ArtifactPath:  "/tmp/artifacts",
			EnableOffline: false,
			SkipRmTmpFile: false,
		},
	}
}

// setupCreateClusterMocks 设置创建集群测试所需的所有 Mock
func setupCreateClusterMocks() []*gomonkey.Patches {
	// 获取通用 Mock（需要导入 create_registry_test.go 中的方法）
	patches := setupCommonMocks()

	// 添加集群特有的 Mock
	// Mock 预检查
	patches = append(patches, gomonkey.ApplyFunc(precheck.NewCreateClusterFlightCheck,
		func(cluster.Nodes, bool) *precheck.CreateClusterFlightCheck {
			mockFlightCheck := &precheck.CreateClusterFlightCheck{
				FlightCheck: &precheck.FlightCheck{},
			}
			return mockFlightCheck
		}))
	patches = append(patches, gomonkey.ApplyMethod(&precheck.CreateClusterFlightCheck{}, "StartCheck",
		func(*precheck.CreateClusterFlightCheck) error { return nil }))

	// Mock 集群特有功能
	patches = append(patches, gomonkey.ApplyFunc(common.IsContainsComponentByCfg,
		func(*cluster.Cluster, string) bool { return true }))
	patches = append(patches, gomonkey.ApplyFunc(common.UnCompressPkgToOptPath,
		func(*cluster.Node) error { return nil }))
	patches = append(patches, gomonkey.ApplyFunc(hub.InstallHarborCfg,
		func(*common.KubeRuntime) error { return nil }))

	// Mock 制品准备集群特有方法
	patches = append(patches, gomonkey.ApplyMethod(&binary.PrepareArtifact{}, "CreateClusterPrepareArtifact",
		func(*binary.PrepareArtifact, bool) error {
			return nil
		}))

	return patches
}

// resetAllPatches 重置所有 patches
func resetAllPatches(patches []*gomonkey.Patches) {
	for _, patch := range patches {
		patch.Reset()
	}
}

func TestNewCreateClusterPipeline_Success(t *testing.T) {
	runtime := createTestKubeRuntime()
	patches := setupCreateClusterMocks()
	defer resetAllPatches(patches)

	// Mock 安装
	patches14 := gomonkey.ApplyMethod(&executor.Executor{}, "Install",
		func(*executor.Executor) error {
			return nil
		})
	defer patches14.Reset()

	// Mock 后续处理
	patches15 := gomonkey.ApplyFunc(auth.PrintDefaultUserInfo,
		func(*cluster.Cluster) error {
			return nil
		})
	defer patches15.Reset()

	patches16 := gomonkey.ApplyFunc(common.PostInstallCheck,
		func(*cluster.Node) {
			// 模拟后续检查,原方法无返回值
		})
	defer patches16.Reset()

	patches17 := gomonkey.ApplyFunc(common.RemoveTmpFiles,
		func(cluster.Nodes, bool) {
			// 模拟临时文件删除,原方法无返回值
		})
	defer patches17.Reset()

	// 执行测试
	err := NewCreateClusterPipeline(runtime)

	// 验证结果
	assert.NoError(t, err)
}

func TestNewCreateClusterPipeline_FlightCheckError(t *testing.T) {
	runtime := &common.KubeRuntime{
		Cluster: &cluster.Cluster{
			Spec: cluster.Spec{
				Nodes: cluster.Nodes{
					Masters: []*cluster.Node{{IP: TestAds1}},
				},
			},
		},
	}

	// Mock 预检查失败
	patches1 := gomonkey.ApplyFunc(precheck.NewCreateClusterFlightCheck,
		func(cluster.Nodes, bool) *precheck.CreateClusterFlightCheck {
			return &precheck.CreateClusterFlightCheck{}
		})
	defer patches1.Reset()

	patches2 := gomonkey.ApplyMethod(&precheck.CreateClusterFlightCheck{}, "StartCheck",
		func(*precheck.CreateClusterFlightCheck) error {
			return fmt.Errorf("flight check failed")
		})
	defer patches2.Reset()

	patches3 := gomonkey.ApplyFunc(common.IsContainsComponentByCfg,
		func(*cluster.Cluster, string) bool {
			return false
		})
	defer patches3.Reset()

	err := NewCreateClusterPipeline(runtime)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "flight check failed")
}

func TestCreateCluster_Success(t *testing.T) {
	args := common.Argument{
		InitCfg:       "",
		EnableOffline: false,
		FuyaoZone:     "cn",
		SkipRmTmpFile: false,
	}

	// Mock 参数验证
	patches1 := gomonkey.ApplyFunc(params.VerifyZone,
		func(string, string) (bool, error) {
			return true, nil
		})
	defer patches1.Reset()

	// Mock runtime 创建
	patches2 := gomonkey.ApplyFunc(common.NewKubeRuntime,
		func(common.Argument) (*common.KubeRuntime, error) {
			return &common.KubeRuntime{
				Cluster: &cluster.Cluster{},
			}, nil
		})
	defer patches2.Reset()

	// Mock ClearSensitiveData
	patches3 := gomonkey.ApplyMethod(&common.KubeRuntime{}, "ClearSensitiveData",
		func(*common.KubeRuntime) {
			// 模拟清理敏感数据,原方法无返回值
		})
	defer patches3.Reset()

	// Mock pipeline
	patches4 := gomonkey.ApplyFunc(NewCreateClusterPipeline,
		func(*common.KubeRuntime) error {
			return nil
		})
	defer patches4.Reset()

	err := CreateCluster(args)
	assert.NoError(t, err)
}

func TestCreateCluster_InitCfgNotExists(t *testing.T) {
	args := common.Argument{
		InitCfg: "/non/existent/file.yaml",
	}

	err := CreateCluster(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "Init config file is nor exixts")
}

func TestCreateCluster_OfflineArtifactPathError(t *testing.T) {
	// 创建临时文件作为存在的初始化配置
	tmpFile, err := os.CreateTemp("", "init-*.yaml")
	assert.NoError(t, err)
	defer os.Remove(tmpFile.Name())

	args := common.Argument{
		InitCfg:       tmpFile.Name(),
		EnableOffline: true,
		ArtifactPath:  "/invalid/path",
	}

	// Mock params.VerifyArtifactPath 返回错误
	patches := gomonkey.ApplyFunc(params.VerifyArtifactPath,
		func(string) error {
			return fmt.Errorf("invalid artifact path")
		})
	defer patches.Reset()

	err = CreateCluster(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid artifact path")
}

func TestCreateCluster_ZoneVerificationError(t *testing.T) {
	args := common.Argument{
		EnableOffline: false,
		FuyaoZone:     "invalid",
	}

	// Mock 区域验证失败
	patches := gomonkey.ApplyFunc(params.VerifyZone,
		func(string, string) (bool, error) {
			return false, fmt.Errorf("invalid zone")
		})
	defer patches.Reset()

	err := CreateCluster(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid zone")
}

func TestCreateCluster_NewKubeRuntimeError(t *testing.T) {
	args := common.Argument{
		EnableOffline: false,
		FuyaoZone:     "cn",
	}

	// Mock 参数验证成功
	patches1 := gomonkey.ApplyFunc(params.VerifyZone,
		func(string, string) (bool, error) {
			return true, nil
		})
	defer patches1.Reset()

	// Mock runtime 创建失败
	patches2 := gomonkey.ApplyFunc(common.NewKubeRuntime,
		func(common.Argument) (*common.KubeRuntime, error) {
			return nil, fmt.Errorf("failed to create runtime")
		})
	defer patches2.Reset()

	err := CreateCluster(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create runtime")
}
