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

	"gopkg.in/yaml.v3"

	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/manifests"
	"installer/pkg/fuyao/v1beta1/metadata"
	"installer/pkg/fuyao/v1beta1/values"
)

// UnmarshalYaml 通用yaml解析
func UnmarshalYaml[T *values.Values | *manifests.ComponentManifests | *fuyaov1beta1.Cluster | *metadata.RegistryConfig | *manifests.InstallComponentManifest](data T, filePath string) error {
	yamlFile, err := os.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("Failed to read yaml file, please check if the yaml file exists: %v\n", err)
	}

	err = yaml.Unmarshal(yamlFile, data)
	if err != nil {
		return fmt.Errorf("Unmarshal yaml file failed, please check the yaml file format: %v\n", err)
	}

	return nil
}

// MarshalYaml 通用yaml生成
func MarshalYaml[T manifests.ComponentManifests | fuyaov1beta1.Cluster | *metadata.RegistryConfig | metadata.RegistryConfig | *manifests.InstallComponentManifest](filePath string, yamlTemplate T) error {
	data, err := yaml.Marshal(yamlTemplate)
	if err != nil {
		return fmt.Errorf("Marshal YAML document failed: %v\n", err)
	}

	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("Failed to created yaml file: %v\n", err)
	}
	defer file.Close()

	_, err = file.Write(data)
	if err != nil {
		return fmt.Errorf("Failed to write data to file: %v\n", err)
	}

	return nil
}
