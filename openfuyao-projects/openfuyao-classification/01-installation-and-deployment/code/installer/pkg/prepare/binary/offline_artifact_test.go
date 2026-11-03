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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"installer/pkg/fuyao/v1beta1/manifests"
)

func TestNewDownLoadArtifactPrepare(t *testing.T) {
	tests := []struct {
		name         string
		assetAbsPath string
		tmpAssetPath string
		zone         string
	}{
		{
			name:         "valid parameters",
			assetAbsPath: "/opt/fuyao/asset",
			tmpAssetPath: "/tmp/fuyao",
			zone:         "Asia/Shanghai",
		},
		{
			name:         "empty parameters",
			assetAbsPath: "",
			tmpAssetPath: "",
			zone:         "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewDownLoadArtifactPrepare(tt.assetAbsPath, tt.tmpAssetPath, tt.zone)

			assert.NotNil(t, result)
			assert.Equal(t, tt.assetAbsPath, result.AssetAbsPath)
			assert.Equal(t, tt.tmpAssetPath, result.TmpAssetPath)
			assert.Equal(t, tt.zone, result.FuyaoZone)
			assert.NotNil(t, result.PkgNameMap)
			assert.Equal(t, 0, len(result.PkgNameMap))
		})
	}
}

func TestDownLoadArtifactPrepare_getOnePkgName(t *testing.T) {
	tests := []struct {
		name       string
		pkgNameMap map[string]string
		expected   string
	}{
		{
			name: "valid package map",
			pkgNameMap: map[string]string{
				"fuyao-openEuler-22.03": "fuyao-openEuler-22.03-amd64.tar.gz",
			},
			expected: "fuyao-openEuler-22.03-amd64",
		},
		{
			name: "multiple packages",
			pkgNameMap: map[string]string{
				"fuyao-openEuler-22.03": "fuyao-openEuler-22.03-amd64.tar.gz",
				"fuyao-Ubuntu-20.04":    "fuyao-Ubuntu-20.04-amd64.tar.gz",
			},
			expected: "fuyao-openEuler-22.03-amd64", // 返回第一个找到的
		},
		{
			name:       "empty package map",
			pkgNameMap: map[string]string{},
			expected:   "",
		},
		{
			name: "package without tar.gz suffix",
			pkgNameMap: map[string]string{
				"fuyao-openEuler-22.03": "fuyao-openEuler-22.03-amd64",
			},
			expected: "fuyao-openEuler-22.03-amd64",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := &DownLoadArtifactPrepare{
				PkgNameMap: tt.pkgNameMap,
			}

			result := d.getOnePkgName()

			if tt.expected == "" {
				assert.Empty(t, result)
			} else {
				// 由于 map 遍历顺序不确定，对于多个包的情况只验证非空
				if len(tt.pkgNameMap) > 1 {
					assert.NotEmpty(t, result)
				} else {
					assert.Equal(t, tt.expected, result)
				}
			}
		})
	}
}

type handleAscendDeployerParamsTest struct {
	name              string
	componentManifest *manifests.ComponentManifests
	componentName     string
	expectedGroup     string
	shouldError       bool
}

func getComponentGroupNameTestCase() []handleAscendDeployerParamsTest {
	return []handleAscendDeployerParamsTest{
		{
			name: "component found in core group",
			componentManifest: &manifests.ComponentManifests{
				Spec: manifests.Spec{
					Components: map[string][]manifests.Component{
						"core": {
							{Name: "kubernetes"},
							{Name: "etcd"},
						},
						"cni": {
							{Name: "calico"},
						},
					},
				},
			},
			componentName: "kubernetes",
			expectedGroup: "core",
			shouldError:   false,
		},
		{
			name: "component found in cni group",
			componentManifest: &manifests.ComponentManifests{
				Spec: manifests.Spec{
					Components: map[string][]manifests.Component{
						"core": {
							{Name: "kubernetes"},
						},
						"cni": {
							{Name: "calico"},
						},
					},
				},
			},
			componentName: "calico",
			expectedGroup: "cni",
			shouldError:   false,
		},
	}
}

func TestGetComponentGroupName(t *testing.T) {
	tests := getComponentGroupNameTestCase()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			group, err := getComponentGroupName(tt.componentManifest, tt.componentName)

			if tt.shouldError {
				assert.Error(t, err)
				assert.Empty(t, group)
				assert.Contains(t, err.Error(), "component not found")
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedGroup, group)
			}
		})
	}
}

type getComponentGroupNameTestStruct struct {
	name        string
	setupFiles  func(string) error
	shouldError bool
}

func createValidFuyaoPackagesTest() getComponentGroupNameTestStruct {
	return getComponentGroupNameTestStruct{
		name: "valid fuyao packages",
		setupFiles: func(dir string) error {
			files := []string{
				"fuyao-openEuler-22.03-amd64.tar.gz",
				"fuyao-Ubuntu-20.04-amd64.tar.gz",
				"other-file.txt",
				"not-tar-gz-file",
			}
			for _, file := range files {
				if fileTemp, err := os.Create(filepath.Join(dir, file)); err != nil {
					err = fileTemp.Chmod(0000) // 设置无权限
					return err
				}
			}
			return nil
		},
		shouldError: false,
	}
}
func createNoFuyaoPackagesTest() getComponentGroupNameTestStruct {
	return getComponentGroupNameTestStruct{
		name: "no fuyao packages",
		setupFiles: func(dir string) error {
			if err := createFilesWithNoPermission(dir); err != nil {
				return err
			}
			return nil
		},
		shouldError: false,
	}
}

func createFilesWithNoPermission(dir string) error {
	files := []string{
		"fuyao-openEuler-22.03-amd64.tar.gz",
		"regular-file.txt",
	}
	for _, file := range files {
		if fileTemp, err := os.Create(filepath.Join(dir, file)); err != nil {
			err = fileTemp.Chmod(0000) // 设置无权限
			return err
		}
	}
	return nil
}

func createMixedFilesWithSubdirectoriesTest() getComponentGroupNameTestStruct {
	const fileMode = 0755
	return getComponentGroupNameTestStruct{
		name: "mixed files with subdirectories",
		setupFiles: func(dir string) error {
			if err := createFilesWithNoPermission(dir); err != nil {
				return err
			}
			if err := os.Mkdir(filepath.Join(dir, "subdir"), fileMode); err != nil {
				return err
			}
			return nil
		},
		shouldError: false,
	}
}

func createEmptyDirectoryTest() getComponentGroupNameTestStruct {
	return getComponentGroupNameTestStruct{
		name: "empty directory",
		setupFiles: func(dir string) error {
			return nil // 不创建任何文件
		},
		shouldError: false,
	}
}
func getPkgNameMapTestCases() []getComponentGroupNameTestStruct {
	return []getComponentGroupNameTestStruct{
		createValidFuyaoPackagesTest(),
		createNoFuyaoPackagesTest(),
		createMixedFilesWithSubdirectoriesTest(),
		createEmptyDirectoryTest(),
	}
}

func TestGetPkgNameMap(t *testing.T) {
	tests := getPkgNameMapTestCases()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 创建临时目录
			tempDir, err := os.MkdirTemp("", "test-pkg-map-*")
			require.NoError(t, err)
			defer os.RemoveAll(tempDir)

			// 设置测试文件
			err = tt.setupFiles(tempDir)
			require.NoError(t, err)

			// 执行测试
			result, err := GetPkgNameMap(tempDir)

			if tt.shouldError {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				// 验证结果格式
				for key, value := range result {
					assert.Contains(t, key, "fuyao")
					assert.Contains(t, value, ".tar.gz")
					assert.Contains(t, value, "fuyao")
				}
			}
		})
	}
}

func TestGetPkgNameMap_NonExistentDirectory(t *testing.T) {
	nonExistentPath := filepath.Join(os.TempDir(), "non-existent-dir-"+time.Now().Format("20060102150405"))

	result, err := GetPkgNameMap(nonExistentPath)

	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "failed to read dir")
}

func TestGetPkgNameMap_DuplicateArchitectures(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "test-duplicate-arch-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// 创建具有相同前缀但不同架构的包
	files := []string{
		"fuyao-openEuler-22.03-amd64.tar.gz",
		"fuyao-openEuler-22.03-arm64.tar.gz",
	}

	for _, file := range files {
		fileTemp, err := os.Create(filepath.Join(tempDir, file))
		err = fileTemp.Chmod(0644) // 设置可读写权限
		require.NoError(t, err)
	}

	result, err := GetPkgNameMap(tempDir)

	assert.NoError(t, err)

	// 验证键值对应关系
	for key, value := range result {
		assert.Contains(t, value, key)
		assert.Contains(t, value, ".tar.gz")
	}
}
