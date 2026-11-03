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

import (
	"helm.sh/helm/v3/pkg/repo"
	"helm.sh/helm/v3/pkg/time"
)

// ChartDetailResponse chart detail file map
type ChartDetailResponse struct {
	Readme string `json:"readme"`
	Values string `json:"values"`
	Chart  string `json:"chart"`
}

// ChartVersionResponseWithTag restful response chart version fields with tag
type ChartVersionResponseWithTag struct {
	// tag for the request
	Tag string `json:"tag"`
	// restful response chart version fields
	ChartVersionResponses []*ChartVersionResponse `json:"Charts"`
}

// ChartVersionResponse restful response chart version fields
type ChartVersionResponse struct {
	// Metadata chart info
	Metadata *repo.ChartVersion `json:"metadata"`
	// Repo repo name
	Repo string `json:"repo"`
	// RepoUrl repo url
	RepoUrl string `json:"repoUrl"`
}

// ChartTemplateResponse restful response template file in helm chart
type ChartTemplateResponse struct {
	// Name Template file name
	Name string `json:"name,omitempty"`
	// Data Template file data
	Data string `json:"data,omitempty"`
}

// ReleaseHistoryResponse release 历史记录模型
type ReleaseHistoryResponse struct {
	Name          string            `json:"name,omitempty"`
	Version       int               `json:"version,omitempty"`
	Namespace     string            `json:"namespace,omitempty"`
	FirstDeployed time.Time         `json:"firstDeployed,omitempty"`
	LastDeployed  time.Time         `json:"lastDeployed,omitempty"`
	Labels        map[string]string `json:"labels,omitempty"`
	Status        string            `json:"status,omitempty"`
	ChartName     string            `json:"chartName,omitempty"`
	ChartVersion  string            `json:"chartVersion,omitempty"`
	APIVersion    string            `json:"apiVersion,omitempty"`
	AppVersion    string            `json:"appVersion,omitempty"`
}

// ListResponse restful response for list response
type ListResponse struct {
	Items       []interface{} `json:"items"`
	TotalItems  int           `json:"totalItems"`
	CurrentPage int           `json:"currentPage"`
	TotalPages  int           `json:"totalPage"`
}

// RepoResponse struct for get repo rest api response
type RepoResponse struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}
