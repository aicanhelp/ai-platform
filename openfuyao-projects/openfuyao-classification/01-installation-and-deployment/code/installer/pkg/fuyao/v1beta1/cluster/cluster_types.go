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

package cluster

import (
	"installer/pkg/constants"
	"installer/pkg/fuyao/v1beta1/base"
)

// Component is component data
type Component struct {
	Name          string `yaml:"name"`
	BootstrapNode string `yaml:"bootstrapNode,omitempty"`
	Version       string `yaml:"version,omitempty"`
	// 组件所在分组
	Group string       `yaml:"group,omitempty"`
	Param []base.Param `yaml:"param"`
}

// Spec is spec
type Spec struct {
	Nodes          Nodes        `yaml:"nodes"`
	Components     []Component  `yaml:"components,omitempty"`
	AdvancedParams []base.Param `yaml:"advancedParams,omitempty"`
}

// Cluster 总的yaml
type Cluster struct {
	APIVersion string        `yaml:"apiVersion"`
	Kind       string        `yaml:"kind"`
	Metadata   base.Metadata `yaml:"metadata"`
	Spec       Spec          `yaml:"spec"`
}

// NewCluster is new cluster
func NewCluster() Cluster {
	return Cluster{
		APIVersion: constants.ApiVersion,
		Kind:       constants.Cluster,
		Metadata:   base.Metadata{},
		Spec:       Spec{},
	}
}
