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
	"installer/pkg/prepare/hub"
)

// createTestKubeRuntime 创建用于测试的 KubeRuntime 实例
func createTestKubeRuntimeForRegistry() *common.KubeRuntime {
	return &common.KubeRuntime{
		Cluster: &cluster.Cluster{
			Spec: cluster.Spec{
				Nodes: cluster.Nodes{
					Registry: []*cluster.Node{
						{
							IP:       TestAds3,
							Hostname: "registry1",
							User:     "test-user",
							Password: []byte("test-password"),
						},
					},
				},
				Components: []cluster.Component{
					{Name: "harbor", Version: "2.8.0"},
				},
			},
		},
		Arg: common.Argument{
			FilePath:      "/tmp/artifacts",
			SkipRmTmpFile: false,
		},
	}
}

// setupCommonMocks 设置通用的 Mock，可被创建集群和创建镜像仓库测试复用
func setupCommonMocks() []*gomonkey.Patches {
	var patches []*gomonkey.Patches

	// Mock 资产预处理
	patches = append(patches, gomonkey.ApplyFunc(asset.NewPretreatment,
		func(*cluster.Cluster) *asset.Pretreatment {
			return &asset.Pretreatment{}
		}))
	patches = append(patches, gomonkey.ApplyMethod(&asset.Pretreatment{}, "PretreatmentPackage",
		func(*asset.Pretreatment) error { return nil }))

	// Mock 图构建器
	patches = append(patches, gomonkey.ApplyFunc(executor.NewGraphBuilder,
		func() *executor.GraphBuilder {
			return &executor.GraphBuilder{}
		}))
	patches = append(patches, gomonkey.ApplyMethod(&executor.GraphBuilder{}, "InstallGraphBuilder",
		func(*executor.GraphBuilder, []cluster.Component, *cluster.Node) (*core.DirectedGraph, error) {
			return &core.DirectedGraph{}, nil
		}))

	// Mock 执行器
	patches = append(patches, gomonkey.ApplyFunc(executor.NewExecutor,
		func(string, cluster.Nodes, *core.DirectedGraph) *executor.Executor {
			return &executor.Executor{}
		}))
	patches = append(patches, gomonkey.ApplyMethod(&executor.Executor{}, "FlightCheck",
		func(*executor.Executor) {
			// 模拟检查,原方法无返回值
		}))

	// Mock 制品准备 - 通用部分
	patches = append(patches, gomonkey.ApplyFunc(binary.NewPrepareArtifact,
		func(*core.DirectedGraph, cluster.Nodes, string) *binary.PrepareArtifact {
			return &binary.PrepareArtifact{}
		}))

	return patches
}

// setupCreateRegistryMocks 设置创建镜像仓库测试所需的所有 Mock
func setupCreateRegistryMocks() []*gomonkey.Patches {
	// 获取通用 Mock
	patches := setupCommonMocks()

	// 添加镜像仓库特有的 Mock
	// Mock 文件解压
	patches = append(patches, gomonkey.ApplyFunc(common.UnCompressPkgToOptPath,
		func(*cluster.Node) error { return nil }))

	// Mock 预检查
	patches = append(patches, gomonkey.ApplyFunc(precheck.NewCreateRegistryFlightCheck,
		func(cluster.Nodes) *precheck.CreateRegistryFlightCheck {
			mockFlightCheck := &precheck.CreateRegistryFlightCheck{
				FlightCheck: &precheck.FlightCheck{},
			}
			return mockFlightCheck
		}))
	patches = append(patches, gomonkey.ApplyMethod(&precheck.CreateRegistryFlightCheck{}, "StartCheck",
		func(*precheck.CreateRegistryFlightCheck) error { return nil }))

	// Mock 执行器镜像仓库特有方法
	patches = append(patches, gomonkey.ApplyMethod(&executor.Executor{}, "RegistryInstall",
		func(*executor.Executor) {
			// 模拟安装操作，原方法无返回值
		}))

	// Mock 制品准备镜像仓库特有方法
	patches = append(patches, gomonkey.ApplyMethod(&binary.PrepareArtifact{}, "CreateRegistryCopyBinary",
		func(*binary.PrepareArtifact) error { return nil }))
	patches = append(patches, gomonkey.ApplyMethod(&binary.PrepareArtifact{}, "UploadImages",
		func(*binary.PrepareArtifact, string) error { return nil }))

	return patches
}

// resetAllRegistryPatches 重置所有 patches
func resetAllRegistryPatches(patches []*gomonkey.Patches) {
	for _, patch := range patches {
		patch.Reset()
	}
}

func TestNewCreateRegistryPipeline_Success(t *testing.T) {
	// 准备测试数据
	runtime := createTestKubeRuntimeForRegistry()

	// 设置所有 Mock
	patches := setupCreateRegistryMocks()
	defer resetAllRegistryPatches(patches)

	// Mock Hub 相关操作
	patches14 := gomonkey.ApplyFunc(hub.GetHubInfo,
		func([]cluster.Component) (string, string, string) {
			return "harbor.example.com", "admin123", TestAds3
		})
	defer patches14.Reset()

	patches15 := gomonkey.ApplyFunc(hub.UploadImagePrepare,
		func(string, string, cluster.Nodes) error {
			return nil
		})
	defer patches15.Reset()

	patches16 := gomonkey.ApplyFunc(hub.RecordHarborConfig,
		func(string, string, *cluster.Node) error {
			return nil
		})
	defer patches16.Reset()

	// Mock 清理临时文件
	patches17 := gomonkey.ApplyFunc(common.RemoveTmpFiles,
		func(cluster.Nodes, bool) {
			// 模拟删除临时文件，原方法无返回值
		})
	defer patches17.Reset()

	// 执行测试
	err := NewCreateRegistryPipeline(runtime)

	// 验证结果
	assert.NoError(t, err)
}

func TestNewCreateRegistryPipeline_UnCompressError(t *testing.T) {
	runtime := &common.KubeRuntime{
		Cluster: &cluster.Cluster{
			Spec: cluster.Spec{
				Nodes: cluster.Nodes{
					Registry: []*cluster.Node{{IP: TestAds3}},
				},
			},
		},
	}

	// Mock 文件解压失败
	patches := gomonkey.ApplyFunc(common.UnCompressPkgToOptPath,
		func(*cluster.Node) error {
			return fmt.Errorf("uncompress failed")
		})
	defer patches.Reset()

	err := NewCreateRegistryPipeline(runtime)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "uncompress failed")
}

func TestNewCreateRegistryPipeline_FlightCheckError(t *testing.T) {
	runtime := &common.KubeRuntime{
		Cluster: &cluster.Cluster{
			Spec: cluster.Spec{
				Nodes: cluster.Nodes{
					Registry: []*cluster.Node{{IP: TestAds3}},
				},
			},
		},
	}

	// Mock 文件解压成功
	patches1 := gomonkey.ApplyFunc(common.UnCompressPkgToOptPath,
		func(*cluster.Node) error {
			return nil
		})
	defer patches1.Reset()

	// Mock 预检查失败
	patches2 := gomonkey.ApplyFunc(precheck.NewCreateRegistryFlightCheck,
		func(cluster.Nodes) *precheck.CreateRegistryFlightCheck {
			return &precheck.CreateRegistryFlightCheck{}
		})
	defer patches2.Reset()

	patches3 := gomonkey.ApplyMethod(&precheck.CreateRegistryFlightCheck{}, "StartCheck",
		func(*precheck.CreateRegistryFlightCheck) error {
			return fmt.Errorf("flight check failed")
		})
	defer patches3.Reset()

	err := NewCreateRegistryPipeline(runtime)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "flight check failed")
}

func TestCreateRegistry_Success(t *testing.T) {
	args := common.Argument{
		RegistryPasswd:         []byte("TestPass123!"),
		RegistryDBPasswd:       []byte("DBPass123!"),
		RegistryCertExpiryDays: "365",
		RegistryDomain:         "https://harbor.example.com",
	}
	// 使用通用 Mock 方法
	patches := setupCreateRegistryValidateMocks()
	// Mock 域名验证失败
	patches = append(patches, gomonkey.ApplyFunc(params.VerifyDomain,
		func(string, string) (bool, error) {
			return true, nil
		}))
	defer resetAllRegistryPatches(patches)
	// Mock runtime 创建
	patches4 := gomonkey.ApplyFunc(common.NewKubeRuntime,
		func(common.Argument) (*common.KubeRuntime, error) {
			return &common.KubeRuntime{
				Cluster: &cluster.Cluster{},
			}, nil
		})
	defer patches4.Reset()

	// Mock 敏感数据清理
	patches5 := gomonkey.ApplyMethod(&common.Argument{}, "ClearSensitiveData",
		func(*common.Argument) {
			// 模拟清理敏感数据，原方法无返回值
		})
	defer patches5.Reset()

	patches6 := gomonkey.ApplyMethod(&common.KubeRuntime{}, "ClearSensitiveData",
		func(*common.KubeRuntime) {
			// 模拟清理敏感数据，原方法无返回值
		})
	defer patches6.Reset()

	// Mock pipeline
	patches7 := gomonkey.ApplyFunc(NewCreateRegistryPipeline,
		func(*common.KubeRuntime) error {
			return nil
		})
	defer patches7.Reset()

	err := CreateRegistry(args)
	assert.NoError(t, err)
}

func TestCreateRegistry_PasswordValidationError(t *testing.T) {
	args := common.Argument{
		RegistryPasswd:   []byte("weak"),
		RegistryDBPasswd: []byte("weak"),
	}

	// Mock 第一个密码验证失败
	patches := gomonkey.ApplyFunc(params.VerifyPasswd,
		func([]byte) (bool, error) {
			return false, fmt.Errorf("password too weak")
		})
	defer patches.Reset()

	err := CreateRegistry(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "password too weak")
}

func TestCreateRegistry_DBPasswordValidationError(t *testing.T) {
	args := common.Argument{
		RegistryPasswd:   []byte("StrongPass123!"),
		RegistryDBPasswd: []byte("weak"),
	}

	callCount := 0
	// Mock 第一次密码验证成功，第二次失败
	patches := gomonkey.ApplyFunc(params.VerifyPasswd,
		func([]byte) (bool, error) {
			callCount++
			if callCount == 1 {
				return true, nil
			}
			return false, fmt.Errorf("DB password too weak")
		})
	defer patches.Reset()

	err := CreateRegistry(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "DB password too weak")
}

func TestCreateRegistry_CertExpiryDaysError(t *testing.T) {
	args := common.Argument{
		RegistryPasswd:         []byte("StrongPass123!"),
		RegistryDBPasswd:       []byte("DBPass123!"),
		RegistryCertExpiryDays: "-1", // 无效的天数
	}

	// Mock 密码验证成功
	patches1 := gomonkey.ApplyFunc(params.VerifyPasswd,
		func([]byte) (bool, error) {
			return true, nil
		})
	defer patches1.Reset()

	// Mock 证书有效期验证失败
	patches2 := gomonkey.ApplyFunc(params.VerifyRegistryCertExpiryDays,
		func(string) error {
			return fmt.Errorf("invalid cert expiry days")
		})
	defer patches2.Reset()

	err := CreateRegistry(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid cert expiry days")
}

func setupCreateRegistryValidateMocks() []*gomonkey.Patches {
	var patches []*gomonkey.Patches
	// Mock 密码验证成功
	patches = append(patches, gomonkey.ApplyFunc(params.VerifyPasswd,
		func([]byte) (bool, error) {
			return true, nil
		}))

	// Mock 证书有效期验证成功
	patches = append(patches, gomonkey.ApplyFunc(params.VerifyRegistryCertExpiryDays,
		func(string) error {
			return nil
		}))

	return patches
}

func TestCreateRegistry_DomainValidationError(t *testing.T) {
	args := common.Argument{
		RegistryPasswd:         []byte("StrongPass123!"),
		RegistryDBPasswd:       []byte("DBPass123!"),
		RegistryCertExpiryDays: "365",
		RegistryDomain:         "invalid-domain",
	}
	// 使用通用 Mock 方法
	patches := setupCreateRegistryValidateMocks()

	// Mock 域名验证失败
	patches = append(patches, gomonkey.ApplyFunc(params.VerifyDomain,
		func(string, string) (bool, error) {
			return false, fmt.Errorf("invalid domain format")
		}))

	defer resetAllRegistryPatches(patches)

	err := CreateRegistry(args)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid domain format")
}
