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

package manifests

import "installer/pkg/fuyao/v1beta1/base"

// Component 组件参数
type Component struct {
	Name         string `yaml:"name"`
	Version      string `yaml:"version"`
	Describe     string `yaml:"describe"`
	Dependencies string `yaml:"dependencies"`
}

// OsInfo is os info
type OsInfo struct {
	OsName  string `yaml:"os-name"`
	Version string `yaml:"version"`
	CpuArch string `yaml:"cpu-arch"`
}

// Spec is spce data
type Spec struct {
	Ossa       []OsInfo               `yaml:"ossa"`
	Components map[string][]Component `yaml:"components"`
}

// ComponentManifests 总的yaml
type ComponentManifests struct {
	APIVersion string        `yaml:"apiVersion"`
	Kind       string        `yaml:"kind"`
	Metadata   base.Metadata `yaml:"metadata"`
	Spec       Spec          `yaml:"spec"`
}

// ManifestMetaData manifests元数据
type ManifestMetaData struct {
	CpuArch        []string `json:"cpu-arch"`
	ComponentGroup []string `json:"component_group"`
}

type OsVersionMetaData struct {
	OsVersion []string `json:"os-version"`
}
