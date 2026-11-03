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

package pipelines

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/config"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
	"installer/pkg/utils"
)

func createComponentManifest(t *testing.T) manifests.ComponentManifests {
	// 直接读取的manifests真实路径
	manifestDir := filepath.Join("..", "..", constants.Manifests)
	_, err := os.Stat(manifestDir)
	assert.Equal(t, nil, err)

	ManifestMetaDataPath := filepath.Join(manifestDir, constants.MetaDataFile)
	_, err = os.Stat(ManifestMetaDataPath)
	assert.Equal(t, nil, err)

	ManifestMetaData := &manifests.ManifestMetaData{}
	err = common.UnmarshalJSON[*manifests.ManifestMetaData](ManifestMetaData, ManifestMetaDataPath)
	assert.Equal(t, nil, err)

	componentManifest := config.NewComponentManifest(ManifestMetaData, manifestDir)

	err = componentManifest.GenerateOsInfo()
	assert.NoError(t, err)
	err = componentManifest.GenerateComponents()
	assert.NoError(t, err)

	manifest := manifests.ComponentManifests{
		APIVersion: constants.ApiVersion,
		Kind:       constants.Build,
		Metadata:   base.Metadata{Name: exampleManifest},
		Spec: manifests.Spec{
			Ossa:       componentManifest.OsInfo,
			Components: componentManifest.Components,
		},
	}
	return manifest
}

func getManifestsMetaData(t *testing.T) *metadata.Version {
	manifestsMetaData := &metadata.Version{}
	metadataPath := filepath.Join("..", "..", constants.Manifests, constants.MetaDataFile)
	err := common.UnmarshalJSON[*metadata.Version](manifestsMetaData, metadataPath)
	assert.Equal(t, nil, err)
	return manifestsMetaData
}

func TestCreateSubComponentTarPackage(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-component-tar-pkg-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// 将清单文件存储到临时目录
	componentManifest := createComponentManifest(t)
	manifestFilePath := filepath.Join(tempDir, "component.yaml")
	err = common.MarshalYaml[manifests.ComponentManifests](manifestFilePath, componentManifest)
	assert.Equal(t, nil, err)

	// 将manifests目录拷贝到临时目录
	err = common.CopyLocalDir(filepath.Join("..", "..", constants.Manifests), filepath.Join(tempDir, constants.Manifests))
	assert.Equal(t, nil, err)

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyPrivateMethod(
		&ComponentPackage{}, "copyCommonFile", func(_ *ComponentPackage, _ map[string]string) error {
			return nil
		},
	)

	componentPkg := NewComponentPackage()
	componentPkg.manifestsMetaData = getManifestsMetaData(t)

	componentPkg.AbsPath = tempDir
	componentPkg.ManifestFile = manifestFilePath
	componentPkg.ComponentManifests = &componentManifest

	err = componentPkg.CreateSubComponentTarPackage()
	assert.Equal(t, nil, err)

	files, err := utils.GetDirFiles(filepath.Join(tempDir, constants.Targets))
	assert.Equal(t, nil, err)
	assert.Equal(t, 2, len(files))
}

func TestCreateUnifyComponentPackage(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", fmt.Sprintf("fuyao-component-tar-pkg-%d", time.Now().Unix()))
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// 将清单文件存储到临时目录
	componentManifest := createComponentManifest(t)
	manifestFilePath := filepath.Join(tempDir, "component.yaml")
	err = common.MarshalYaml[manifests.ComponentManifests](manifestFilePath, componentManifest)
	assert.Equal(t, nil, err)

	// 将manifests目录拷贝到临时目录
	err = common.CopyLocalDir(filepath.Join("..", "..", constants.Manifests), filepath.Join(tempDir, constants.Manifests))
	assert.Equal(t, nil, err)

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyPrivateMethod(
		&ComponentPackage{}, "copyCommonFile", func(_ *ComponentPackage, _ map[string]string) error {
			return nil
		},
	)

	assetPath := filepath.Join(tempDir, constants.Targets, constants.Asset)
	err = os.MkdirAll(assetPath, constants.RwxRxRx)
	assert.Equal(t, nil, err)

	componentPkg := NewComponentPackage()
	componentPkg.manifestsMetaData = getManifestsMetaData(t)
	componentPkg.AbsPath = tempDir
	componentPkg.ManifestFile = manifestFilePath
	componentPkg.ComponentManifests = &componentManifest
	componentPkg.AssetPath = assetPath

	err = componentPkg.CreateUnifyComponentPackage()
	assert.Equal(t, nil, err)

	files, err := utils.GetDirFiles(assetPath)
	assert.Equal(t, nil, err)
	assert.Equal(t, 4, len(files))
}
