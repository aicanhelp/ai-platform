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
	"bytes"
	"mime/multipart"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"marketplace-service/pkg/models/helm"
	"marketplace-service/pkg/server/param"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/zlog"
)

// Operation all operations for helm
type Operation interface {
	// 仓库操作
	CreateRepo(repoEntry *helm.SafeRepoEntry) (*httputil.ResponseJson, int)
	UpdateRepo(repoEntry *helm.SafeRepoEntry) (*httputil.ResponseJson, int)
	DeleteRepo(repoName string) (*httputil.ResponseJson, int)
	ListRepo(query *param.Query, repo string) (*httputil.ResponseJson, int)
	GetRepo(repoName string) (*httputil.ResponseJson, int)
	SyncAllRepos() (*httputil.ResponseJson, int)
	SyncRepo(repoName string) (*httputil.ResponseJson, int)
	GetRepoSyncStatus(repoName string) (*httputil.ResponseJson, int)
	// chart 操作
	GetLatestCharts(searchParam *helm.ChartSearchParam) (*httputil.ResponseJson, int)
	GetChartsWithOfficialTags(tags []string, query *param.Query) (*httputil.ResponseJson, int)
	CountCharts() (*httputil.ResponseJson, int)
	UploadChart(formFile multipart.File, fileHeader *multipart.FileHeader) (*httputil.ResponseJson, int)
	DeleteChart(chartName string) (*httputil.ResponseJson, int)
	DeleteChartVersion(chartName, version string) (*httputil.ResponseJson, int)
	GetChartVersions(repoName, chartName string) (*httputil.ResponseJson, int)
	GetChartVersion(repoName, chartName, version string) (*httputil.ResponseJson, int)
	GetChartFiles(repoName, chartName, version, fileType string) (*httputil.ResponseJson, int)
	GetChartBytesByVersion(repoName, chartName, version string) (*bytes.Buffer, error)
}

type helmClient struct {
	kubeConfig    *rest.Config
	dynamicClient dynamic.Interface
	clientset     kubernetes.Interface
}

// dynamic client for test
func (c *helmClient) SetDynamicClient(client dynamic.Interface) {
	c.dynamicClient = client
}

// kubernetes client for test
func (c *helmClient) SetK8sClient(client kubernetes.Interface) {
	c.clientset = client
}

// NewHelmOperation helm operation requires client set&dynamic client，for kubernetes resource operation
func NewHelmOperation(kubeConfig *rest.Config) (Operation, error) {
	dynamicClient, err := dynamic.NewForConfig(kubeConfig)
	if err != nil {
		zlog.Errorf("error creating dynamic client, err: %v", err)
		return nil, err
	}

	clientset, err := kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		zlog.Errorf("error creating client set, err: %v", err)
		return nil, err
	}
	return &helmClient{
		kubeConfig:    kubeConfig,
		dynamicClient: dynamicClient,
		clientset:     clientset,
	}, nil
}
