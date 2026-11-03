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
	"strings"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/precheck/params"
	"installer/pkg/prepare/asset"
	"installer/pkg/prepare/binary"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// DownloadArtifact is download offline install artifact
type DownloadArtifact struct {
	// 安装包名字 fuyao-openEuler-22.03: fuyao-openEuler-22.03-amd64
	PkgNameMap   map[string]string
	AssetAbsPath string
	TmpAssetPath string
	// FuyaoZone 时区
	FuyaoZone string
}

// NewDownloadArtifact is new DownloadArtifact
func NewDownloadArtifact(assetAbsPath, tmpAssetPath, zone string) *DownloadArtifact {
	return &DownloadArtifact{
		PkgNameMap:   make(map[string]string),
		AssetAbsPath: assetAbsPath,
		FuyaoZone:    zone,
		TmpAssetPath: tmpAssetPath,
	}
}

func (d *DownloadArtifact) execDownload(pkgPath string, group string, componentName string,
	artifactSavePath string) error {
	// 渲染组件参数
	basePath := filepath.Join(pkgPath, group, componentName)
	filePath := filepath.Join(basePath, constants.DownloadFile)
	var param = make(map[string]interface{})
	param[constants.FuyaoZone] = d.FuyaoZone
	err := asset.RenderComponentParam(filePath, param)
	if err != nil {
		return err
	}

	// 组件目录，该目录下放二进制和镜像
	// fuyao-offline-artifact/fuyao-openEuler-22.03/cri/containerd
	savePath := filepath.Join(artifactSavePath, group, componentName)
	if err := utils.MkdirAll(savePath, constants.RwxRxRx); err != nil {
		return err
	}

	_, err = utils.ExecLocalCmd(fmt.Sprintf("cd %s && bash %s -p %s -i true -c true -y true -b true",
		filepath.Join(pkgPath, group, componentName), constants.DownloadFile, savePath))
	if err != nil {
		return err
	}

	return nil
}

func (d *DownloadArtifact) downloadArtifact(pkgName string, artifactSavePath string) error {
	componentManifest := &manifests.ComponentManifests{}
	manifestsPath := filepath.Join(d.AssetAbsPath, constants.ComponentManifest)
	if err := common.UnmarshalYaml[*manifests.ComponentManifests](componentManifest, manifestsPath); err != nil {
		return err
	}

	pkgPath := filepath.Join(d.TmpAssetPath, pkgName)
	for group, components := range componentManifest.Spec.Components {
		for _, component := range components {
			if err := d.execDownload(pkgPath, group, component.Name, artifactSavePath); err != nil {
				return err
			}
		}
	}

	otherGroups := []string{constants.Hub}
	for _, group := range otherGroups {
		groupFolder := filepath.Join(pkgPath, group)
		subDirectory := common.SubDirectory{
			RootDir: groupFolder,
		}
		subdirectoryNames, err := subDirectory.GetSubdirectoryNames()
		if err != nil {
			return err
		}
		for _, subdirectoryName := range subdirectoryNames {
			if err := d.execDownload(pkgPath, group, subdirectoryName, artifactSavePath); err != nil {
				return err
			}
		}
	}
	return nil
}

// StartDownload is start download
func (d *DownloadArtifact) StartDownload(artifactSavePath string) error {
	artifactSavePath = filepath.Join(artifactSavePath, constants.OfflineArtifactPath)
	if err := utils.MkdirAll(artifactSavePath, constants.RwxRxRx); err != nil {
		return err
	}

	for name, pkgName := range d.PkgNameMap {
		// fuyao-offline-artifact/fuyao-openEuler-22.03
		savePath := filepath.Join(artifactSavePath, name)
		if err := utils.Mkdir(savePath, constants.RwxRxRx); err != nil {
			return err
		}
		err := d.downloadArtifact(strings.TrimRight(pkgName, constants.TarGz), savePath)
		if err != nil {
			return err
		}
	}

	if err := common.CompressTarGZ(artifactSavePath, artifactSavePath+constants.TarGz); err != nil {
		return err
	}
	return os.RemoveAll(artifactSavePath)
}

// NewExportArtifactPipline is ExportArtifact pipline
func NewExportArtifactPipline(args common.Argument) error {
	assetAbsPath, err := common.GetAssetAbsPath()
	if err != nil {
		return err
	}
	tmpAssetPath := filepath.Join(constants.OptPath, constants.OpenFuyao, constants.Tmp)
	if err := utils.MkdirAll(tmpAssetPath, constants.RwxRxRx); err != nil {
		return err
	}
	pkgNameMap, err := binary.GetPkgNameMap(assetAbsPath)
	if err != nil {
		return err
	}

	prepare := binary.NewDownLoadArtifactPrepare(assetAbsPath, tmpAssetPath, args.FuyaoZone)
	prepare.PkgNameMap = pkgNameMap
	if err := prepare.Prepare(); err != nil {
		return err
	}

	download := NewDownloadArtifact(assetAbsPath, tmpAssetPath, args.FuyaoZone)
	download.PkgNameMap = pkgNameMap
	if err := download.StartDownload(args.FileSaveFolder); err != nil {
		return err
	}

	if err := os.RemoveAll(tmpAssetPath); err != nil {
		zlog.Error("failed to remove tmp asset path", tmpAssetPath)
	}

	fmt.Println("The offline artifact is exported successfully.")
	return nil
}

// ExportArtifact 制备离线安装所需制品
func ExportArtifact(args common.Argument) error {
	if args.FileSaveFolder != "" {
		if err := utils.MkdirAll(args.FileSaveFolder, constants.RwxRxRx); err != nil {
			return err
		}
	}

	ok, err := params.VerifyZone(constants.Zone, args.FuyaoZone)
	if !ok {
		return err
	}

	absSavePath, err := filepath.Abs(args.FileSaveFolder)
	if err != nil {
		return fmt.Errorf("failed to get artifact save absolute path")
	}
	args.FileSaveFolder = absSavePath

	if err := utils.LocalInstallTool(constants.Tar); err != nil {
		return err
	}
	return NewExportArtifactPipline(args)
}
