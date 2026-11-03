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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/manifests"
)

const (
	openEuler = "openEuler"
	ubuntu    = "Ubuntu"
)

func TestComponent(t *testing.T) {
	// 直接读取的manifests真实路径
	manifestDir := filepath.Join("..", "..", "manifests")
	_, err := os.Stat(manifestDir)
	assert.Equal(t, nil, err)

	ManifestMetaDataPath := filepath.Join(manifestDir, constants.MetaDataFile)
	_, err = os.Stat(ManifestMetaDataPath)
	assert.Equal(t, nil, err)

	ManifestMetaData := &manifests.ManifestMetaData{}
	err = common.UnmarshalJSON[*manifests.ManifestMetaData](ManifestMetaData, ManifestMetaDataPath)
	assert.Equal(t, nil, err)

	componentManifest := NewComponentManifest(ManifestMetaData, manifestDir)

	err = componentManifest.GenerateOsInfo()
	assert.NoError(t, err)
	err = componentManifest.GenerateComponents()
	assert.NoError(t, err)

	osInfoList := componentManifest.OsInfo
	osNames := make([]string, 0)
	osVersions := make([]string, 0)
	osArchs := make([]string, 0)
	for _, osInfo := range osInfoList {
		osNames = append(osNames, osInfo.OsName)
		osVersions = append(osVersions, osInfo.Version)
		osArchs = append(osArchs, osInfo.CpuArch)
	}
	assert.Contains(t, osNames, openEuler)
	assert.Contains(t, osNames, ubuntu)
	assert.Contains(t, osArchs, constants.ARM64)
	assert.Contains(t, osArchs, constants.AMD64)
	assert.Contains(t, osVersions, "22.03")
	assert.Contains(t, osVersions, "22.04")

	componentMap := componentManifest.Components
	assert.Contains(t, componentMap, constants.Ha)
	componentNames := make([]string, 0)
	for _, component := range componentMap[constants.Ha] {
		componentNames = append(componentNames, component.Name)
	}
	assert.Contains(t, componentNames, "kube-vip")

	assert.Contains(t, componentMap, constants.Core)
	componentNames = make([]string, 0)
	for _, component := range componentMap[constants.Core] {
		componentNames = append(componentNames, component.Name)
	}
	assert.Contains(t, componentNames, "kubernetes")

	assert.Contains(t, componentMap, constants.Cni)
	componentNames = make([]string, 0)
	for _, component := range componentMap[constants.Cni] {
		componentNames = append(componentNames, component.Name)
	}
	assert.Contains(t, componentNames, "cni-plugins")
	assert.Contains(t, componentNames, "calico")

	assert.Contains(t, componentMap, constants.Cri)
	componentNames = make([]string, 0)
	for _, component := range componentMap[constants.Cri] {
		componentNames = append(componentNames, component.Name)
	}
	assert.Contains(t, componentNames, "containerd")

	assert.Contains(t, componentMap, constants.Tools)
	componentNames = make([]string, 0)
	for _, component := range componentMap[constants.Tools] {
		componentNames = append(componentNames, component.Name)
	}
	assert.Contains(t, componentNames, "yq")

	assert.Contains(t, componentMap, constants.PkgManager)
	componentNames = make([]string, 0)
	for _, component := range componentMap[constants.PkgManager] {
		componentNames = append(componentNames, component.Name)
	}
	assert.Contains(t, componentNames, "helm")

	assert.Contains(t, componentMap, constants.Oci)
	componentNames = make([]string, 0)
	for _, component := range componentMap[constants.Oci] {
		componentNames = append(componentNames, component.Name)
	}
	assert.Contains(t, componentNames, "runc")

	assert.NotContains(t, componentMap, constants.Hub)
}
