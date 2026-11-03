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

	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
)

// GetAbsPath 获取当前可执行程序的绝对路径
func GetAbsPath() (string, error) {
	ex, err := os.Executable()
	if err != nil {
		return "", err
	}

	return filepath.Dir(ex), nil
}

// GetAssetAbsPath 获取asset文件夹的绝对路径
func GetAssetAbsPath() (string, error) {
	path, err := GetAbsPath()
	if err != nil {
		return "", err
	}

	filePath := filepath.Join(path, constants.Targets, constants.Asset)
	_, err = os.Stat(filePath)

	if err == nil {
		return filePath, nil
	} else if os.IsNotExist(err) {
		return filepath.Join(path, constants.Asset), nil
	} else {
		return "", err
	}
}

// GetAssetAbsPathReal 获取asset文件夹的绝对路径
func GetAssetAbsPathReal() (string, error) {
	path, err := GetAbsPath()
	if err != nil {
		return "", err
	}
	return filepath.Join(path, constants.Asset), nil
}

// GetComponentManifestsAbsPath 获取组件清单的绝对路径
func GetComponentManifestsAbsPath() (string, error) {
	path, err := GetAssetAbsPathReal()
	if err != nil {
		return "", err
	}

	return filepath.Join(path, constants.ComponentManifest), nil
}

// CreateAssetDirectory 创建asset目录
func CreateAssetDirectory() (string, error) {
	absPath, err := GetAbsPath()
	if err != nil {
		return "", err
	}

	assetPath := filepath.Join(absPath, constants.Targets, constants.Asset)
	err = os.MkdirAll(assetPath, constants.RwxRxRx)
	if err != nil {
		return "", fmt.Errorf("failed to create directory: %v", err)
	}
	return assetPath, nil
}

// GetComponentManifests 获取组件资产清单yaml
func GetComponentManifests() (*manifests.ComponentManifests, error) {
	absPath, err := GetComponentManifestsAbsPath()
	if err != nil {
		return nil, err
	}
	data := &manifests.ComponentManifests{}
	if err := UnmarshalYaml[*manifests.ComponentManifests](data, absPath); err != nil {
		return nil, err
	}
	return data, nil
}

// GetRegistryConfigYaml 获取镜像仓配置yaml
func GetRegistryConfigYaml() (*metadata.RegistryConfig, error) {
	absPath, err := GetAssetAbsPathReal()
	if err != nil {
		return nil, err
	}

	registryCfgPath := filepath.Join(absPath, constants.RegistryConfig)
	_, err = os.Stat(registryCfgPath)
	if err == nil {
		data := &metadata.RegistryConfig{}
		if err := UnmarshalYaml[*metadata.RegistryConfig](data, registryCfgPath); err != nil {
			return nil, err
		}
		return data, nil
	}
	return nil, nil
}
