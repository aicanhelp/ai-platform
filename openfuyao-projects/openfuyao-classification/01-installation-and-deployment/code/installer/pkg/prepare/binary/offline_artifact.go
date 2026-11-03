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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/prepare/asset"
	"installer/pkg/utils"
	"installer/pkg/zlog"
)

// DownLoadArtifactPrepare is download offline prepare
type DownLoadArtifactPrepare struct {
	// 安装包名字, eg: fuyao-openEuler-22.03: fuyao-openEuler-22.03-amd64.tar.gz
	PkgNameMap map[string]string
	// AssetAbsPath asset绝对路径
	AssetAbsPath string
	// TmpAssetPath 临时asset路径
	TmpAssetPath string
	// FuyaoZone 时区
	FuyaoZone string
}

// NewDownLoadArtifactPrepare is new DownLoadArtifactPrepare
func NewDownLoadArtifactPrepare(assetAbsPath, tmpAssetPath, zone string) *DownLoadArtifactPrepare {
	return &DownLoadArtifactPrepare{
		PkgNameMap:   make(map[string]string),
		AssetAbsPath: assetAbsPath,
		FuyaoZone:    zone,
		TmpAssetPath: tmpAssetPath,
	}
}

func (d *DownLoadArtifactPrepare) getOnePkgName() string {
	for _, pkgName := range d.PkgNameMap {
		return strings.TrimRight(pkgName, constants.TarGz)
	}
	return ""
}

// getBaseComponentInstallRes 获取基础组件的安装结果
func (d *DownLoadArtifactPrepare) getBaseComponentInstallRes() (map[string]map[string]string, error) {
	pkgName := d.getOnePkgName()
	if pkgName == "" {
		return nil, fmt.Errorf("the package name is empty or invalid")
	}

	preCheckFile := filepath.Join(d.TmpAssetPath, pkgName, constants.Utils, constants.ArtifactPreCheck)
	err := common.CopyLocalFile(preCheckFile, filepath.Join(d.TmpAssetPath, constants.ArtifactPreCheck))
	if err != nil {
		return nil, fmt.Errorf("failed copy %s to tmp path: %v", preCheckFile, err)
	}

	_, err = utils.ExecLocalCmd(fmt.Sprintf("cd %s && bash %s", d.TmpAssetPath, constants.ArtifactPreCheck))
	if err != nil {
		return nil, fmt.Errorf("failed to exec artifact-precheck.sh :%v", err)
	}

	// 从文件中读取安装结果
	jsonData, err := os.ReadFile(filepath.Join(d.TmpAssetPath, constants.ArtifactPreCheckRes))
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %v", err)
	}
	var result map[string]map[string]string
	err = json.Unmarshal([]byte(jsonData), &result)
	if err != nil {
		return nil, fmt.Errorf("error parsing JSON: %v", err)
	}

	return result, nil
}

func (d *DownLoadArtifactPrepare) checkAndInstallBaseComponent() error {
	result, err := d.getBaseComponentInstallRes()
	if err != nil {
		return err
	}

	err = d.installBaseComponent(result)
	if err != nil {
		return err
	}
	return nil
}

// 获取需要安装的组件
func (d *DownLoadArtifactPrepare) installBaseComponent(componentInfo map[string]map[string]string) error {
	var componentNames []string
	for name, installInfo := range componentInfo {
		if installInfo[constants.Installed] == constants.HasInstall {
			continue
		}
		componentNames = append(componentNames, name)
		zlog.Infof("need install %s", name)
	}

	if len(componentNames) == 0 {
		return nil
	}

	return d.startInstallComponent(componentNames)
}

func getComponentGroupName(componentManifest *manifests.ComponentManifests, name string) (string, error) {
	for group, components := range componentManifest.Spec.Components {
		for _, component := range components {
			if component.Name == name {
				return group, nil
			}
		}
	}
	return "", fmt.Errorf("%s component not found", name)
}

// startInstallComponent 在线去安装组件
func (d *DownLoadArtifactPrepare) startInstallComponent(components []string) error {
	pkgName := d.getOnePkgName()
	if pkgName == "" {
		return fmt.Errorf("the package name is empty or invalid")
	}

	pkgPath := filepath.Join(d.TmpAssetPath, pkgName)
	componentManifest, err := common.GetComponentManifestsWithHubTmp(d.AssetAbsPath, pkgPath)
	if err != nil {
		return err
	}

	for _, component := range components {
		group, err := getComponentGroupName(componentManifest, component)
		if err != nil {
			return err
		}
		// 渲染组件参数
		basePath := filepath.Join(pkgPath, group, component)
		filePath := filepath.Join(basePath, constants.OnlineInstallFile)
		var param = make(map[string]interface{})
		param[constants.FuyaoZone] = d.FuyaoZone

		err = asset.RenderComponentParam(filePath, param)
		if err != nil {
			return err
		}

		_, err = utils.ExecLocalCmd(fmt.Sprintf("cd %s && bash %s", basePath, constants.OnlineInstallFile))
		if err != nil {
			return err
		}
	}
	return nil
}

// unCompressPkg 解压对应的安装包到临时目录
func (d *DownLoadArtifactPrepare) unCompressPkg() error {
	for _, pkgName := range d.PkgNameMap {
		pkgPath := filepath.Join(d.AssetAbsPath, pkgName)
		err := common.UnCompressTarGZ(pkgPath, d.TmpAssetPath)
		if err != nil {
			return err
		}
	}
	return nil
}

// Prepare 执行准备
func (d *DownLoadArtifactPrepare) Prepare() error {
	if err := d.unCompressPkg(); err != nil {
		return err
	}

	if err := d.checkAndInstallBaseComponent(); err != nil {
		return err
	}
	return nil
}

// GetPkgNameMap 获取安装包map，根据CPU架构去重
func GetPkgNameMap(assetAbsPath string) (map[string]string, error) {
	files, err := os.ReadDir(assetAbsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read dir:%v", err)
	}

	pkgNameMap := make(map[string]string)
	for _, file := range files {
		fileName := file.Name()
		if !file.IsDir() && strings.Contains(fileName, constants.TarGz) && strings.Contains(fileName, constants.Fuyao) {
			fileNameSlice := strings.Split(fileName, constants.Minus)
			pkgNameMap[strings.Join(fileNameSlice[:len(fileNameSlice)-1], constants.Minus)] = fileName
		}
	}
	return pkgNameMap, nil
}
