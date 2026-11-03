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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/config"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/utils"
)

// setupComponentManifestMocks 设置组件配置清单测试所需的所有 Mock
func setupComponentManifestMocks() *gomonkey.Patches {
	patches := gomonkey.ApplyFunc(common.GetAbsPath, func() (string, error) {
		return "/test/path", nil
	})

	// Mock JSON 文件读取
	patches.ApplyFunc(common.UnmarshalJSON[*manifests.ManifestMetaData],
		func(*manifests.ManifestMetaData, string) error {
			return nil
		})

	// Mock 文件路径操作
	patches.ApplyFunc(filepath.Join, func(elem ...string) string {
		return "/mocked/path"
	})

	// Mock 文件系统操作
	patches.ApplyFunc(os.Stat, func(string) (os.FileInfo, error) {
		return nil, nil
	})

	patches.ApplyFunc(os.Open, func(string) (*os.File, error) {
		return nil, nil
	})

	patches.ApplyFunc(os.ReadFile, func(string) ([]byte, error) {
		mockJSON := `{
            "version": "1.0",
            "supportedOs": [],
            "components": {}
        }`
		return []byte(mockJSON), nil
	})

	// Mock JSON 解析
	patches.ApplyFunc(json.Unmarshal, func([]byte, interface{}) error {
		return nil
	})
	return patches
}

func TestNewCreateComponentManifestPipeline_Success(t *testing.T) {
	// 创建临时测试目录
	tmpDir := t.TempDir()

	args := common.Argument{
		FileSaveFolder: tmpDir,
	}

	// 设置 Mock
	patches := setupComponentManifestMocks()
	defer patches.Reset()

	// Mock ComponentManifest 创建和操作
	mockComponentManifest := &config.ComponentManifest{}
	patches.ApplyFunc(config.NewComponentManifest,
		func(*manifests.ManifestMetaData, string) *config.ComponentManifest {
			return mockComponentManifest
		})

	patches.ApplyMethod(mockComponentManifest, "GenerateOsInfo", func(*config.ComponentManifest) error {
		return nil
	})

	patches.ApplyMethod(mockComponentManifest, "GenerateComponents", func(*config.ComponentManifest) error {
		return nil
	})

	// Mock YAML 输出
	patches.ApplyFunc(common.MarshalYaml[manifests.ComponentManifests],
		func(string, manifests.ComponentManifests) error {
			return nil
		})

	err := NewCreateComponentManifestPipeline(args)

	assert.NoError(t, err)
}

func TestNewCreateComponentManifestPipeline_GetAbsPathError(t *testing.T) {
	// 创建临时测试目录
	tmpDir := t.TempDir()

	args := common.Argument{
		FileSaveFolder: tmpDir,
	}

	patches := gomonkey.ApplyFunc(common.GetAbsPath, func() (string, error) {
		return "", fmt.Errorf("failed to get absolute path")
	})
	defer patches.Reset()

	err := NewCreateComponentManifestPipeline(args)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get absolute path")
}

func TestNewCreateComponentManifestPipeline_UnmarshalJSONError(t *testing.T) {
	// 创建临时测试目录
	tmpDir := t.TempDir()

	args := common.Argument{
		FileSaveFolder: tmpDir,
	}

	patches := gomonkey.ApplyFunc(common.GetAbsPath, func() (string, error) {
		return "/test/path", nil
	})
	defer patches.Reset()

	patches.ApplyFunc(common.UnmarshalJSON[*manifests.ManifestMetaData],
		func(*manifests.ManifestMetaData, string) error {
			return fmt.Errorf("failed to unmarshal JSON")
		})

	err := NewCreateComponentManifestPipeline(args)

	assert.Error(t, err)
}

func TestCreateComponentConfigFile_Success(t *testing.T) {
	// 创建临时测试目录
	tmpDir := t.TempDir()

	args := common.Argument{
		FileSaveFolder: tmpDir,
	}

	patches := gomonkey.ApplyFunc(utils.MkdirAll, func(string, os.FileMode) error {
		return nil
	})
	defer patches.Reset()

	patches.ApplyFunc(NewCreateComponentManifestPipeline, func(common.Argument) error {
		return nil
	})

	err := CreateComponentConfigFile(args)

	assert.NoError(t, err)
}

func TestCreateComponentConfigFile_EmptyFileSaveFolder(t *testing.T) {
	args := common.Argument{
		FileSaveFolder: "",
	}

	patches := gomonkey.ApplyFunc(NewCreateComponentManifestPipeline, func(common.Argument) error {
		return nil
	})
	defer patches.Reset()

	err := CreateComponentConfigFile(args)

	assert.NoError(t, err)
}

func TestCreateComponentConfigFile_MkdirAllError(t *testing.T) {
	args := common.Argument{
		FileSaveFolder: "/tmp/test-config",
	}

	patches := gomonkey.ApplyFunc(utils.MkdirAll, func(string, os.FileMode) error {
		return fmt.Errorf("failed to create directory")
	})
	defer patches.Reset()

	err := CreateComponentConfigFile(args)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create folder")
	assert.Contains(t, err.Error(), "failed to create directory")
}

func TestCreateComponentConfigFile_PipelineError(t *testing.T) {
	args := common.Argument{
		FileSaveFolder: "/tmp/test-config",
	}

	patches := gomonkey.ApplyFunc(utils.MkdirAll, func(string, os.FileMode) error {
		return nil
	})
	defer patches.Reset()

	patches.ApplyFunc(NewCreateComponentManifestPipeline, func(common.Argument) error {
		return fmt.Errorf("pipeline execution failed")
	})

	err := CreateComponentConfigFile(args)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "pipeline execution failed")
}
