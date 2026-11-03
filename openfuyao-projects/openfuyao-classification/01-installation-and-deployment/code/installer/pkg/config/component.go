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

package config

import (
	"fmt"
	"os"
	"path/filepath"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
	"installer/pkg/zlog"
)

// ComponentManifest 组件清单
type ComponentManifest struct {
	ManifestMetaData *manifests.ManifestMetaData
	ManifestPath     string

	OsInfo     []manifests.OsInfo
	Components map[string][]manifests.Component
}

// NewComponentManifest is new ComponentManifest
func NewComponentManifest(manifestMetaData *manifests.ManifestMetaData, manifestPath string) *ComponentManifest {
	return &ComponentManifest{
		ManifestMetaData: manifestMetaData,
		OsInfo:           make([]manifests.OsInfo, 0),
		Components:       make(map[string][]manifests.Component),
		ManifestPath:     manifestPath,
	}
}

// GenerateOsInfo 生成支持的操作系统信息
func (c *ComponentManifest) GenerateOsInfo() error {
	for _, cpuArch := range c.ManifestMetaData.CpuArch {
		zlog.Info("current cpu arch is ", cpuArch)
		// eg: ./manifests/amd64
		cpuArchPath := filepath.Join(c.ManifestPath, cpuArch)
		osNames, err := common.GetImmediateSubdirectories(cpuArchPath)
		if err != nil {
			return fmt.Errorf("get subdirectory error: %v", err)
		}
		if err := c.generateOsInfo(cpuArch, osNames); err != nil {
			return err
		}
	}
	return nil
}

func (c *ComponentManifest) generateOsInfo(cpuArch string, osNames []string) error {
	cpuArchPath := filepath.Join(c.ManifestPath, cpuArch)
	for _, osName := range osNames {
		// eg: ./manifests/amd64/openEuler
		osNamePath := filepath.Join(cpuArchPath, osName)
		osVersions, err := common.GetImmediateSubdirectories(osNamePath)
		if err != nil {
			return fmt.Errorf("get subdirectory error: %v", err)
		}

		for _, osVersion := range osVersions {
			if osVersion != constants.CommonScripts {
				c.OsInfo = append(c.OsInfo, manifests.OsInfo{OsName: osName, CpuArch: cpuArch, Version: osVersion})
				continue
			}

			// 同一操作系统通用代码目录(./manifests/amd64/openEuler/common-scripts)，从metadata中读取支持的OS版本信息
			versionMetaData := &manifests.OsVersionMetaData{}
			metadataPath := filepath.Join(osNamePath, constants.CommonScripts, constants.MetaDataFile)
			err := common.UnmarshalJSON[*manifests.OsVersionMetaData](versionMetaData, metadataPath)
			if err != nil {
				return err
			}
			for _, version := range versionMetaData.OsVersion {
				c.OsInfo = append(c.OsInfo, manifests.OsInfo{OsName: osName, CpuArch: cpuArch, Version: version})
			}
		}
	}
	return nil
}

// GenerateComponents 生成组件信息
func (c *ComponentManifest) GenerateComponents() error {
	groupPath, err := c.getComponentGroupPath()
	if err != nil || groupPath == "" {
		return err
	}

	for _, group := range c.ManifestMetaData.ComponentGroup {
		// eg: ./manifests/amd64/openEuler/22.03/cri
		componentPath := filepath.Join(groupPath, group)
		componentNames, err := common.GetImmediateSubdirectories(componentPath)
		if err != nil {
			return err
		}

		componentManifests, err := c.generateComponentsInfo(componentNames, componentPath)
		if err != nil {
			return err
		}
		c.Components[group] = componentManifests
	}
	return nil
}

func (c *ComponentManifest) generateComponentsInfo(names []string,
	componentPath string) ([]manifests.Component, error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("not components")
	}

	var componentManifests []manifests.Component
	for _, componentName := range names {
		// eg: ./manifests/amd64/openEuler/22.03/cri/containerd
		versionPath := filepath.Join(componentPath, componentName)
		versions, err := common.GetImmediateSubdirectories(versionPath)
		if err != nil {
			return nil, err
		}
		for _, version := range versions {
			// eg: ./manifests/amd64/openEuler/22.03/cri/containerd/1.7.8/metadata.json
			componentMetaData := &metadata.MetaData{}
			metadataPath := filepath.Join(versionPath, version, constants.MetaDataFile)
			if err := common.UnmarshalJSON[*metadata.MetaData](componentMetaData, metadataPath); err != nil {
				return nil, err
			}
			dependencies := ""
			for _, depend := range componentMetaData.Dependencies {
				dependencies += depend.Name + ":" + depend.Version + ", "
			}
			componentManifests = append(componentManifests, manifests.Component{
				Name:         componentName,
				Version:      version,
				Describe:     componentMetaData.Describe,
				Dependencies: dependencies,
			})
		}
	}
	return componentManifests, nil
}

// getComponentGroupPath 获取组件分组文件目录
func (c *ComponentManifest) getComponentGroupPath() (string, error) {
	if len(c.OsInfo) == 0 {
		return "", fmt.Errorf("not os info")
	}
	osInfo := c.OsInfo[0]
	groupPath := filepath.Join(c.ManifestPath, osInfo.CpuArch, osInfo.OsName, osInfo.Version)
	_, err := os.Stat(groupPath)
	if err == nil {
		return groupPath, nil
	}
	if os.IsNotExist(err) {
		return filepath.Join(c.ManifestPath, osInfo.CpuArch, osInfo.OsName, constants.CommonScripts), nil
	}
	return "", err
}
