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

package common

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
	"installer/pkg/zlog"
)

const (
	notResource = "No resources found"
)

// GetInstallHubComponentMap 获取安装hub所需组件
func GetInstallHubComponentMap(args Argument) (map[string]string, error) {
	componentMap := make(map[string]string)
	if !args.SkipDestroyHub || !args.EnableOffline {
		zlog.Info("skip remove harbor")
		return componentMap, nil
	}

	pkgPath, pkgName, err := UnCompressOnePackage()
	if err != nil {
		return nil, err
	}

	hubMetaDataFile := filepath.Join(pkgPath, pkgName, constants.Hub, constants.Harbor, constants.MetaDataFile)
	metaData := &metadata.MetaData{}
	if err := UnmarshalJSON[*metadata.MetaData](metaData, hubMetaDataFile); err != nil {
		return componentMap, err
	}

	componentMap[constants.Harbor] = metaData.Version
	for _, depend := range metaData.Dependencies {
		componentMap[depend.Name] = depend.Version
	}
	return componentMap, nil
}

// GetComponentManifestsWithHubTmp 获取带有hub组件的组件清单
func GetComponentManifestsWithHubTmp(assetAbsPath, pkgPath string) (*manifests.ComponentManifests, error) {
	var err error
	if assetAbsPath == "" {
		assetAbsPath, err = GetAssetAbsPath()
		if err != nil {
			return nil, err
		}
	}

	componentManifests := &manifests.ComponentManifests{}
	manifestsPath := filepath.Join(assetAbsPath, constants.ComponentManifest)
	if err := UnmarshalYaml[*manifests.ComponentManifests](componentManifests, manifestsPath); err != nil {
		return nil, err
	}

	hubPath := filepath.Join(pkgPath, constants.Hub)
	folderNames, err := GetImmediateSubdirectories(hubPath)
	if err != nil {
		return nil, err
	}

	// 将hub分组下的组件加入到清单中
	components := make([]manifests.Component, 0)
	for _, folderName := range folderNames {
		metadataPath := filepath.Join(hubPath, folderName, constants.MetaDataFile)
		metaData := &metadata.MetaData{}
		if err := UnmarshalJSON[*metadata.MetaData](metaData, metadataPath); err != nil {
			return nil, err
		}
		component := manifests.Component{
			Name:    metaData.Name,
			Version: metaData.Version,
		}
		components = append(components, component)
	}
	if len(components) > 0 {
		componentManifests.Spec.Components[constants.Hub] = components
	}
	return componentManifests, nil
}

// IsContainsComponentByCfg 判断配置文件中是否包含某个组件
func IsContainsComponentByCfg(cluster2 *cluster.Cluster, name string) bool {
	for _, component := range cluster2.Spec.Components {
		if component.Name == name {
			return true
		}
	}
	return false
}

// IsDeployComponentByNs 判断是否部署某一组件
func IsDeployComponentByNs(node *cluster.Node, namespace string) (bool, error) {
	client := node.Client()

	output, err := client.SudoExecCmd(fmt.Sprintf("kubectl get pod -n %s | wc -l", namespace))
	if err != nil {
		return false, err
	}
	if strings.Contains(output, notResource) {
		return false, nil
	}
	count, err := strconv.Atoi(strings.TrimSpace(output))
	if err != nil {
		return false, err
	}
	return count > 1, nil
}
