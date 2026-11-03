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
	"path/filepath"

	"installer/pkg/common"
	"installer/pkg/config"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/utils"
)

const exampleManifest = "example-component-manifests"

// NewCreateComponentManifestPipeline 创建组件资产清单配置文件
func NewCreateComponentManifestPipeline(args common.Argument) error {
	absPath, err := common.GetAbsPath()
	if err != nil {
		return err
	}
	manifestsPath := filepath.Join(absPath, constants.Manifests)
	ManifestsMetaDataPath := filepath.Join(manifestsPath, constants.MetaDataFile)
	ManifestMetaData := &manifests.ManifestMetaData{}
	if err := common.UnmarshalJSON[*manifests.ManifestMetaData](ManifestMetaData, ManifestsMetaDataPath); err != nil {
		return err
	}

	componentManifest := config.NewComponentManifest(ManifestMetaData, manifestsPath)
	// 生成操作系统信息
	if err := componentManifest.GenerateOsInfo(); err != nil {
		return err
	}
	if err := componentManifest.GenerateComponents(); err != nil {
		return err
	}

	manifest := manifests.ComponentManifests{
		APIVersion: constants.ApiVersion,
		Kind:       constants.Build,
		Metadata:   base.Metadata{Name: exampleManifest},
		Spec: manifests.Spec{
			Ossa:       componentManifest.OsInfo,
			Components: componentManifest.Components,
		},
	}
	return common.MarshalYaml[manifests.ComponentManifests](filepath.Join(args.FileSaveFolder,
		constants.ComponentManifest), manifest)
}

// CreateComponentConfigFile 创建组件资产清单配置文件
func CreateComponentConfigFile(args common.Argument) error {
	if args.FileSaveFolder != "" {
		fileSaveFolder := filepath.Clean(args.FileSaveFolder)
		args.FileSaveFolder = fileSaveFolder
		err := utils.MkdirAll(args.FileSaveFolder, constants.RwxRxRx)
		if err != nil {
			return fmt.Errorf("failed to create folder: %v", err)
		}
	}

	if err := NewCreateComponentManifestPipeline(args); err != nil {
		return err
	}
	fmt.Println("The component configuration file is successfully generated.")
	return nil
}
