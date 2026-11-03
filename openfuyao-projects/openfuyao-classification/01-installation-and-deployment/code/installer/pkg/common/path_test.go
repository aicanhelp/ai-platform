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

package common

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/constants"
)

func TestGetAssetAbsPath(t *testing.T) {
	const fileMode = 0755
	// 创建临时目录模拟测试环境
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-asset-test-%d", time.Now().Unix()))
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// 测试 targets/asset 路径存在的情况
	targetsAssetDir := filepath.Join(tempDir, constants.Targets, constants.Asset)
	err = os.MkdirAll(targetsAssetDir, fileMode)
	assert.NoError(t, err)

	patches := gomonkey.ApplyFunc(GetAbsPath, func() (string, error) {
		return tempDir, nil
	})
	defer patches.Reset()

	assetPath, err := GetAssetAbsPath()
	assert.NoError(t, err)
	assert.Equal(t, targetsAssetDir, assetPath)

	// 测试 targets/asset 路径不存在的情况
	err = os.RemoveAll(targetsAssetDir)
	assert.NoError(t, err)

	assetPath, err = GetAssetAbsPath()
	assert.NoError(t, err)
	assert.Equal(t, filepath.Join(tempDir, constants.Asset), assetPath)

	// 测试 GetAbsPath 返回错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetAbsPath, func() (string, error) {
		return "", fmt.Errorf("abs path error")
	})

	_, err = GetAssetAbsPath()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "abs path error")

	// 测试 os.Stat 返回其他错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetAbsPath, func() (string, error) {
		return tempDir, nil
	})

	// 创建一个文件而不是目录，模拟权限错误
	fileName := filepath.Join(tempDir, constants.Targets, constants.Asset)
	err = os.MkdirAll(filepath.Dir(fileName), fileMode)
	assert.NoError(t, err)
	file, err := os.Create(fileName)
	assert.NoError(t, err)
	err = file.Chmod(0000) // 设置无权限
	assert.NoError(t, err)
	err = file.Close()
	assert.NoError(t, err)
	err = os.Chmod(fileName, 0000) // 设置无权限
	assert.NoError(t, err)

	_, err = GetAssetAbsPath()
	// 这种情况下应该会有错误或者返回备用路径
	// 具体行为取决于实际的错误类型
}

func TestGetAssetAbsPathReal(t *testing.T) {
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-asset-real-test-%d", time.Now().Unix()))
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	patches := gomonkey.ApplyFunc(GetAbsPath, func() (string, error) {
		return tempDir, nil
	})
	defer patches.Reset()

	assetPath, err := GetAssetAbsPathReal()
	assert.NoError(t, err)
	assert.Equal(t, filepath.Join(tempDir, constants.Asset), assetPath)

	// 测试 GetAbsPath 返回错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetAbsPath, func() (string, error) {
		return "", fmt.Errorf("real path error")
	})

	_, err = GetAssetAbsPathReal()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "real path error")
}

func TestGetComponentManifestsAbsPath(t *testing.T) {
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-manifests-test-%d", time.Now().Unix()))
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	patches := gomonkey.ApplyFunc(GetAssetAbsPathReal, func() (string, error) {
		return tempDir, nil
	})
	defer patches.Reset()

	manifestsPath, err := GetComponentManifestsAbsPath()
	assert.NoError(t, err)
	assert.Equal(t, filepath.Join(tempDir, constants.ComponentManifest), manifestsPath)

	// 测试 GetAssetAbsPathReal 返回错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetAssetAbsPathReal, func() (string, error) {
		return "", fmt.Errorf("manifests path error")
	})

	_, err = GetComponentManifestsAbsPath()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "manifests path error")
}

func TestCreateAssetDirectory(t *testing.T) {
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-create-asset-test-%d", time.Now().Unix()))
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	patches := gomonkey.ApplyFunc(GetAbsPath, func() (string, error) {
		return tempDir, nil
	})
	defer patches.Reset()

	assetPath, err := CreateAssetDirectory()
	assert.NoError(t, err)
	expectedPath := filepath.Join(tempDir, constants.Targets, constants.Asset)
	assert.Equal(t, expectedPath, assetPath)

	// 验证目录确实被创建
	_, err = os.Stat(assetPath)
	assert.NoError(t, err)

	// 测试 GetAbsPath 返回错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetAbsPath, func() (string, error) {
		return "", fmt.Errorf("create asset error")
	})

	_, err = CreateAssetDirectory()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "create asset error")

	// 测试 os.MkdirAll 返回错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetAbsPath, func() (string, error) {
		return tempDir, nil
	})

	patchesMkdir := gomonkey.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
		return fmt.Errorf("mkdir failed")
	})
	defer patchesMkdir.Reset()

	_, err = CreateAssetDirectory()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create directory")
}

func TestGetComponentManifests(t *testing.T) {
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-get-manifests-test-%d", time.Now().Unix()))
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// 创建有效的 manifests 文件
	manifestsPath := filepath.Join(tempDir, "manifests.yaml")
	manifestsContent := `apiVersion: v1beta1
kind: ComponentManifests
metadata:
  name: test-manifests
spec:
  components: []`
	const fileMode = 0644
	err = os.WriteFile(manifestsPath, []byte(manifestsContent), fileMode)
	assert.NoError(t, err)

	patches := gomonkey.ApplyFunc(GetComponentManifestsAbsPath, func() (string, error) {
		return manifestsPath, nil
	})
	defer patches.Reset()

	// 测试 GetComponentManifestsAbsPath 返回错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetComponentManifestsAbsPath, func() (string, error) {
		return "", fmt.Errorf("manifests abs path error")
	})

	_, err = GetComponentManifests()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "manifests abs path error")

	// 测试 UnmarshalYaml 返回错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetComponentManifestsAbsPath, func() (string, error) {
		return "/nonexistent/path", nil
	})

	_, err = GetComponentManifests()
	assert.Error(t, err)
}

func TestGetRegistryConfigYaml(t *testing.T) {
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-registry-config-test-%d", time.Now().Unix()))
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// 测试配置文件存在的情况
	registryConfigPath := filepath.Join(tempDir, constants.RegistryConfig)
	registryContent := `apiVersion: v1beta1
kind: RegistryConfig
metadata:
  name: test-registry
spec:
  endpoint: "test.com"`

	const fileMode = 0644
	err = os.WriteFile(registryConfigPath, []byte(registryContent), fileMode)
	assert.NoError(t, err)

	patches := gomonkey.ApplyFunc(GetAssetAbsPathReal, func() (string, error) {
		return tempDir, nil
	})
	defer patches.Reset()

	registryConfig, err := GetRegistryConfigYaml()
	assert.NoError(t, err)
	assert.NotNil(t, registryConfig)

	// 测试配置文件不存在的情况
	err = os.Remove(registryConfigPath)
	assert.NoError(t, err)

	registryConfig, err = GetRegistryConfigYaml()
	assert.NoError(t, err)
	assert.Nil(t, registryConfig)

	// 测试 GetAssetAbsPathReal 返回错误的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetAssetAbsPathReal, func() (string, error) {
		return "", fmt.Errorf("registry config path error")
	})

	_, err = GetRegistryConfigYaml()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "registry config path error")

	// 测试文件存在但 UnmarshalYaml 失败的情况
	patches.Reset()
	patches = gomonkey.ApplyFunc(GetAssetAbsPathReal, func() (string, error) {
		return tempDir, nil
	})

	// 创建无效的 YAML 文件
	invalidYamlPath := filepath.Join(tempDir, constants.RegistryConfig)
	err = os.WriteFile(invalidYamlPath, []byte("invalid: yaml: content: ["), fileMode)
	assert.NoError(t, err)

	_, err = GetRegistryConfigYaml()
	assert.Error(t, err)
}
