/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package helm

import "application-management-service/pkg/server/param"

// DeployParam include all the parameters that are needed in deployment
type DeployParam struct {
	ClusterName string `json:"clusterName,omitempty"`
	// Release 安装应用名称
	Release string `json:"release,omitempty"`
	// Namespace 应用命名空间
	Namespace string `json:"namespace,omitempty"`
	// CreateNamespace 是否创建命名空间
	CreateNamespace bool `json:"createNamespace,omitempty"`
	// ChartName chart 名称
	ChartName string `json:"chartName,omitempty"`
	// RepoName 仓库名称
	RepoName string `json:"repoName,omitempty"`
	// Version chart 版本
	Version string `json:"version,omitempty"`
	// Values 配置信息
	Values string `json:"values,omitempty"`
}

// ChartSearchParam query param for list chart api
type ChartSearchParam struct {
	Repositories []string
	Chart        string
	Type         string
	Pagination   *param.Pagination
	SortType     string
	Scene        []string
}
