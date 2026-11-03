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
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/asset"
	"installer/pkg/prepare/binary"
	"installer/pkg/utils"
)

// TestNewDownloadArtifact 测试 DownloadArtifact 构造函数
func TestNewDownloadArtifact(t *testing.T) {
	assetPath := "/test/asset"
	tmpPath := "/test/tmp"
	zone := "Asia/Shanghai"

	download := NewDownloadArtifact(assetPath, tmpPath, zone)

	assert.NotNil(t, download)
	assert.Equal(t, assetPath, download.AssetAbsPath)
	assert.Equal(t, tmpPath, download.TmpAssetPath)
	assert.Equal(t, zone, download.FuyaoZone)
	assert.NotNil(t, download.PkgNameMap)
	assert.Equal(t, 0, len(download.PkgNameMap))
}

const fileMode = 0755

type DownloadArtifactTestCase struct {
	name           string
	setupTempFiles func(string) error
	group          string
	componentName  string
	shouldError    bool
	expectedError  string
}

// buildExecDownloadTestCases 构建 execDownload 测试用例
func buildExecDownloadTestCases() []DownloadArtifactTestCase {
	return []DownloadArtifactTestCase{
		{
			name:          "successful download",
			group:         "cri",
			componentName: "containerd",
			setupTempFiles: func(basePath string) error {
				// 创建下载脚本文件
				downloadFile := filepath.Join(basePath, "cri", "containerd", constants.DownloadFile)
				if err := os.MkdirAll(filepath.Dir(downloadFile), fileMode); err != nil {
					return err
				}

				// 创建模拟的下载脚本
				script := `#!/bin/bash
echo "Downloading component..."
mkdir -p "$1"
echo "Component downloaded successfully"
`
				return os.WriteFile(downloadFile, []byte(script), fileMode)
			},
			shouldError: false,
		},
		{
			name:          "missing download script",
			group:         "cni",
			componentName: "calico",
			setupTempFiles: func(basePath string) error {
				// 不创建下载脚本，测试文件不存在的情况
				return os.MkdirAll(filepath.Join(basePath, "cni", "calico"), fileMode)
			},
			shouldError:   true,
			expectedError: "no such file or directory",
		},
	}
}

// setupExecDownloadMocks 设置 execDownload 测试所需的 Mock
func setupExecDownloadMocks(shouldError bool, testName string) (*gomonkey.Patches, *gomonkey.Patches) {
	// Mock asset.RenderComponentParam 避免模板渲染错误
	patches1 := gomonkey.ApplyFunc(asset.RenderComponentParam,
		func(string, map[string]interface{}) error {
			return nil
		})

	// Mock utils.ExecLocalCmd
	patches2 := gomonkey.ApplyFunc(utils.ExecLocalCmd,
		func(cmd string) (string, error) {
			// 模拟成功执行
			if shouldError && testName == "missing download script" {
				return "", fmt.Errorf("no such file or directory")
			}
			return "success", nil
		})

	return patches1, patches2
}

// resetExecDownloadMocks 重置 execDownload 测试的 Mock
func resetExecDownloadMocks(patches1, patches2 *gomonkey.Patches) {
	patches1.Reset()
	patches2.Reset()
}

// TestDownloadArtifact_execDownload 测试单个组件下载
func TestDownloadArtifact_execDownload(t *testing.T) {
	tests := buildExecDownloadTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 创建临时目录
			tempDir := createTempTestDir(t, "download-exec-test")
			defer os.RemoveAll(tempDir)

			pkgPath := filepath.Join(tempDir, "pkg")
			artifactSavePath := filepath.Join(tempDir, "artifact")

			// 设置测试文件
			if tt.setupTempFiles != nil {
				err := tt.setupTempFiles(pkgPath)
				require.NoError(t, err)
			}

			download := NewDownloadArtifact("", "", "Asia/Shanghai")

			// 设置 Mock
			patches1, patches2 := setupExecDownloadMocks(tt.shouldError, tt.name)
			defer resetExecDownloadMocks(patches1, patches2)

			err := download.execDownload(pkgPath, tt.group, tt.componentName, artifactSavePath)

			if tt.shouldError {
				assert.Error(t, err)
				if tt.expectedError != "" {
					assert.Contains(t, err.Error(), tt.expectedError)
				}
			} else {
				assert.NoError(t, err)
				// 验证目录是否创建
				savePath := filepath.Join(artifactSavePath, tt.group, tt.componentName)
				_, err := os.Stat(savePath)
				assert.NoError(t, err)
			}
		})
	}
}

// TestDownloadArtifact_downloadArtifact 测试完整下载流程
func TestDownloadArtifact_downloadArtifact(t *testing.T) {
	tempDir := createTempTestDir(t, "download-artifact-test")
	defer os.RemoveAll(tempDir)

	assetPath := filepath.Join(tempDir, "asset")
	tmpPath := filepath.Join(tempDir, "tmp")
	pkgName := "fuyao-openEuler-22.03-amd64"
	artifactSavePath := filepath.Join(tempDir, "artifact")

	// 创建必要的目录结构
	require.NoError(t, os.MkdirAll(assetPath, fileMode))
	require.NoError(t, os.MkdirAll(filepath.Join(tmpPath, pkgName), fileMode))

	// 创建组件清单文件
	setupComponentManifest(t, assetPath)

	// 创建模拟的组件目录和脚本
	setupMockComponents(t, filepath.Join(tmpPath, pkgName))

	download := NewDownloadArtifact(assetPath, tmpPath, "Asia/Shanghai")

	// Mock asset.RenderComponentParam
	patches := gomonkey.ApplyFunc(asset.RenderComponentParam,
		func(string, map[string]interface{}) error {
			return nil
		})
	defer patches.Reset()

	// Mock utils.ExecLocalCmd 模拟脚本执行成功
	patchesExec := gomonkey.ApplyFunc(utils.ExecLocalCmd,
		func(string) (string, error) {
			return "success", nil
		})
	defer patchesExec.Reset()

	// 新增：Mock common.SubDirectory.GetSubdirectoryNames 方法
	patchesSubdir := gomonkey.ApplyMethod(&common.SubDirectory{}, "GetSubdirectoryNames",
		func(*common.SubDirectory) ([]string, error) {
			// 返回模拟的 hub 组件子目录
			return []string{"harbor", "nerdctl"}, nil
		})
	defer patchesSubdir.Reset()

	err := download.downloadArtifact(pkgName, artifactSavePath)
	assert.NoError(t, err)
}

// TestDownloadArtifact_StartDownload 测试完整的下载启动流程
func TestDownloadArtifact_StartDownload(t *testing.T) {
	tempDir := createTempTestDir(t, "start-download-test")
	defer os.RemoveAll(tempDir)

	assetPath := filepath.Join(tempDir, "asset")
	tmpPath := filepath.Join(tempDir, "tmp")
	artifactSavePath := filepath.Join(tempDir, "output")

	pkgNames := []string{
		"fuyao-openEuler-22.03-amd64",
		"fuyao-Ubuntu-20.04-amd64",
	}
	// 创建资产目录和组件清单
	require.NoError(t, os.MkdirAll(assetPath, fileMode))
	setupComponentManifest(t, assetPath)

	// 为每个包创建模拟组件
	for _, pkgName := range pkgNames {
		pkgPath := filepath.Join(tmpPath, pkgName)
		require.NoError(t, os.MkdirAll(pkgPath, fileMode))
		setupMockComponents(t, pkgPath)
	}

	download := NewDownloadArtifact(assetPath, tmpPath, "Asia/Shanghai")
	download.PkgNameMap = map[string]string{
		"fuyao-openEuler-22.03": "fuyao-openEuler-22.03-amd64.tar.gz",
		"fuyao-Ubuntu-20.04":    "fuyao-Ubuntu-20.04-amd64.tar.gz",
	}

	// Mock asset.RenderComponentParam - 避免模板渲染
	patches1 := gomonkey.ApplyFunc(asset.RenderComponentParam,
		func(string, map[string]interface{}) error {
			return nil
		})
	defer patches1.Reset()

	// Mock utils.ExecLocalCmd - 模拟脚本执行成功
	patches2 := gomonkey.ApplyFunc(utils.ExecLocalCmd,
		func(string) (string, error) {
			return "success", nil
		})
	defer patches2.Reset()

	// Mock common.SubDirectory.GetSubdirectoryNames - 处理 hub 组件
	patches3 := gomonkey.ApplyMethod(&common.SubDirectory{}, "GetSubdirectoryNames",
		func(*common.SubDirectory) ([]string, error) {
			return []string{"harbor", "nerdctl"}, nil
		})
	defer patches3.Reset()

	// Mock 压缩函数
	patchesCompress := gomonkey.ApplyFunc(common.CompressTarGZ,
		func(srcPath, dstPath string) error {
			const fileModeLow = 0644
			// 创建一个空的压缩文件用于测试
			return os.WriteFile(dstPath, []byte("mock tar.gz content"), fileModeLow)
		})
	defer patchesCompress.Reset()

	err := download.StartDownload(artifactSavePath)
	assert.NoError(t, err)

	expectedTarFile := filepath.Join(artifactSavePath, constants.OfflineArtifactPath+constants.TarGz)
	_, err = os.Stat(expectedTarFile)
	assert.NoError(t, err)
}

// setupSuccessfulPipelineMocks 设置成功执行管道的 Mock
func setupSuccessfulPipelineMocks() []*gomonkey.Patches {
	patches := make([]*gomonkey.Patches, 0)

	// Mock 目录创建
	patches = append(patches, gomonkey.ApplyFunc(utils.MkdirAll,
		func(string, os.FileMode) error {
			return nil
		}))

	// Mock 获取资源绝对路径
	patches = append(patches, gomonkey.ApplyFunc(common.GetAssetAbsPath,
		func() (string, error) {
			return "/test/asset", nil
		}))

	// Mock 获取包名映射
	patches = append(patches, gomonkey.ApplyFunc(binary.GetPkgNameMap,
		func(string) (map[string]string, error) {
			return map[string]string{
				"fuyao-openEuler-22.03": "fuyao-openEuler-22.03-amd64.tar.gz",
			}, nil
		}))

	// Mock 下载制品准备
	patches = append(patches, gomonkey.ApplyMethod(
		&binary.DownLoadArtifactPrepare{}, "Prepare",
		func(*binary.DownLoadArtifactPrepare) error {
			return nil
		}))

	// Mock 下载开始
	patches = append(patches, gomonkey.ApplyMethod(
		&DownloadArtifact{}, "StartDownload",
		func(*DownloadArtifact, string) error {
			return nil
		}))

	// Mock 临时目录清理
	patches = append(patches, gomonkey.ApplyFunc(os.RemoveAll,
		func(string) error {
			return nil
		}))

	return patches
}

// setupAssetPathErrorMocks 设置资源路径错误的 Mock
func setupAssetPathErrorMocks() []*gomonkey.Patches {
	patches := make([]*gomonkey.Patches, 0)

	// Mock 目录创建 - 避免实际文件操作
	patches = append(patches, gomonkey.ApplyFunc(utils.MkdirAll,
		func(string, os.FileMode) error {
			return nil
		}))

	// Mock 获取资源绝对路径返回错误
	patches = append(patches, gomonkey.ApplyFunc(common.GetAssetAbsPath,
		func() (string, error) {
			return "", fmt.Errorf("asset path not found")
		}))

	return patches
}

// resetPipelineMocks 重置管道测试的 Mock
func resetPipelineMocks(patches []*gomonkey.Patches) {
	for _, patch := range patches {
		patch.Reset()
	}
}

type ExportArtifactPipline struct {
	name        string
	args        common.Argument
	setupMocks  func() []*gomonkey.Patches
	shouldError bool
	errorMsg    string
}

// TestNewExportArtifactPipline 测试导出管道
func TestNewExportArtifactPipline(t *testing.T) {
	tests := []ExportArtifactPipline{
		{
			name: "successful pipeline",
			args: common.Argument{
				FileSaveFolder: "/tmp/test",
				FuyaoZone:      "Asia/Shanghai",
			},
			setupMocks:  setupSuccessfulPipelineMocks,
			shouldError: false,
		},
		{
			name: "asset path error",
			args: common.Argument{
				FileSaveFolder: "/tmp/test",
				FuyaoZone:      "Asia/Shanghai",
			},
			setupMocks:  setupAssetPathErrorMocks,
			shouldError: true,
			errorMsg:    "asset path not found",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := tt.setupMocks()
			defer func() {
				for _, patch := range patches {
					patch.Reset()
				}
			}()

			err := NewExportArtifactPipline(tt.args)

			verifyTestResult(t, err, tt.shouldError, tt.errorMsg)
		})
	}
}

// verifyTestResult 通用测试结果验证方法
func verifyTestResult(t *testing.T, err error, shouldError bool, errorMsg string) {
	if shouldError {
		assert.Error(t, err)
		if errorMsg != "" {
			assert.Contains(t, err.Error(), errorMsg)
		}
	} else {
		assert.NoError(t, err)
	}
}

// setupCommonExportMocks 设置导出测试的通用 Mock
func setupCommonExportMocks() []*gomonkey.Patches {
	patches := make([]*gomonkey.Patches, 0)

	// Mock 目录创建
	patches = append(patches, gomonkey.ApplyFunc(utils.MkdirAll,
		func(string, os.FileMode) error {
			return nil
		}))

	return patches
}

// setupTimezoneAndToolMocks 设置时区验证和工具安装的 Mock
func setupTimezoneAndToolMocks(timezoneSuccess bool, timezoneError error, toolSuccess bool, toolError error) []*gomonkey.Patches {
	patches := make([]*gomonkey.Patches, 0)

	// Mock 时区验证
	patches = append(patches, gomonkey.ApplyFunc(params.VerifyZone,
		func(string, string) (bool, error) {
			if timezoneError != nil {
				return false, timezoneError
			}
			return timezoneSuccess, nil
		}))

	// 只有时区验证成功时才设置后续 Mock
	if timezoneSuccess && timezoneError == nil {
		// Mock 绝对路径获取
		patches = append(patches, gomonkey.ApplyFunc(filepath.Abs,
			func(string) (string, error) {
				return "/absolute/test-output", nil
			}))

		// Mock 工具安装
		patches = append(patches, gomonkey.ApplyFunc(utils.LocalInstallTool,
			func(string) error {
				if toolError != nil {
					return toolError
				}
				return nil
			}))

		// 只有工具安装成功时才设置管道执行 Mock
		if toolSuccess && toolError == nil {
			patches = append(patches, gomonkey.ApplyFunc(NewExportArtifactPipline,
				func(common.Argument) error {
					return nil
				}))
		}
	}

	return patches
}

// setupSuccessfulExportMocks 设置成功导出的 Mock
func setupSuccessfulExportMocks() []*gomonkey.Patches {
	patches := setupCommonExportMocks()

	// 添加成功场景的特定 Mock
	specificPatches := setupTimezoneAndToolMocks(true, nil, true, nil)
	patches = append(patches, specificPatches...)

	return patches
}

// setupTimezoneErrorMocks 设置时区验证错误的 Mock
func setupTimezoneErrorMocks() []*gomonkey.Patches {
	patches := setupCommonExportMocks()

	// 添加时区错误场景的特定 Mock
	timezoneError := fmt.Errorf("invalid timezone: InvalidZone")
	specificPatches := setupTimezoneAndToolMocks(false, timezoneError, false, nil)
	patches = append(patches, specificPatches...)

	return patches
}

// setupToolInstallErrorMocks 设置工具安装错误的 Mock
func setupToolInstallErrorMocks() []*gomonkey.Patches {
	patches := setupCommonExportMocks()

	// 添加工具安装错误场景的特定 Mock
	toolError := fmt.Errorf("tar tool not found")
	specificPatches := setupTimezoneAndToolMocks(true, nil, false, toolError)
	patches = append(patches, specificPatches...)

	return patches
}

type exportArtifactTestStruct struct {
	name        string
	args        common.Argument
	setupMocks  func() []*gomonkey.Patches
	shouldError bool
	errorMsg    string
}

func TestExportArtifact(t *testing.T) {
	tests := []exportArtifactTestStruct{
		{
			name: "successful export",
			args: common.Argument{
				FileSaveFolder: "test-output",
				FuyaoZone:      "Asia/Shanghai",
			},
			setupMocks:  setupSuccessfulExportMocks,
			shouldError: false,
		},
		{
			name: "invalid timezone",
			args: common.Argument{
				FileSaveFolder: "test-output",
				FuyaoZone:      "InvalidZone",
			},
			setupMocks:  setupTimezoneErrorMocks,
			shouldError: true,
			errorMsg:    "invalid timezone",
		},
		{
			name: "tool installation failed",
			args: common.Argument{
				FileSaveFolder: "test-output",
				FuyaoZone:      "Asia/Shanghai",
			},
			setupMocks:  setupToolInstallErrorMocks,
			shouldError: true,
			errorMsg:    "tar tool not found",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := tt.setupMocks()
			defer resetPipelineMocks(patches)
			err := ExportArtifact(tt.args)
			verifyTestResult(t, err, tt.shouldError, tt.errorMsg)
		})
	}
}

// createTempTestDir 创建测试用的临时目录
func createTempTestDir(t *testing.T, prefix string) string {
	tempDir, err := os.MkdirTemp("", prefix+"-"+time.Now().Format("20060102150405"))
	require.NoError(t, err)
	return tempDir
}

// setupComponentManifest 创建组件清单文件
func setupComponentManifest(t *testing.T, assetPath string) {
	manifest := &manifests.ComponentManifests{
		APIVersion: constants.ApiVersion,
		Kind:       constants.Build,
		Metadata: base.Metadata{
			Name: "test-component-manifests",
		},
		Spec: manifests.Spec{
			Ossa: []manifests.OsInfo{
				{
					OsName:  "openEuler",
					Version: "22.03",
					CpuArch: "amd64",
				},
				{
					OsName:  "Ubuntu",
					Version: "20.04",
					CpuArch: "amd64",
				},
			},
			Components: map[string][]manifests.Component{
				"cri": {
					{Name: "containerd", Version: "1.7.14"},
				},
				"cni": {
					{Name: "calico", Version: "3.27.3"},
				},
			},
		},
	}

	// 使用正确的常量名称（单数形式）
	manifestPath := filepath.Join(assetPath, constants.ComponentManifest)
	require.NoError(t, os.MkdirAll(filepath.Dir(manifestPath), fileMode))

	err := common.MarshalYaml[manifests.ComponentManifests](manifestPath, *manifest)
	require.NoError(t, err)
}

// setupMockComponents 创建模拟的组件目录和脚本
func setupMockComponents(t *testing.T, pkgPath string) {
	components := []struct {
		group string
		name  string
	}{
		{"cri", "containerd"},
		{"cni", "calico"},
		{"hub", "harbor"}, // 新增 hub 组件
		{"hub", "nerdctl"},
	}

	for _, comp := range components {
		compDir := filepath.Join(pkgPath, comp.group, comp.name)
		require.NoError(t, os.MkdirAll(compDir, fileMode))

		// 创建下载脚本
		downloadScript := filepath.Join(compDir, constants.DownloadFile)
		script := `#!/bin/bash
echo "Downloading ` + comp.name + `..."
mkdir -p "$1"
echo "Download completed"
`
		require.NoError(t, os.WriteFile(downloadScript, []byte(script), fileMode))
	}
}
