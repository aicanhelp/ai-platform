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

// Package helm
package helm

import (
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"application-management-service/pkg/models/helm"
	"application-management-service/pkg/server/param"
	"application-management-service/pkg/utils/httputil"
	"application-management-service/pkg/zlog"
)

// Operation all operations for helm
type Operation interface {

	// release 操作
	GetReleases(parameter param.ReleaseParameter) (*httputil.ResponseJson, int)
	GetReleaseInfo(namespace, releaseName string) (*httputil.ResponseJson, int)
	Install(deploy helm.DeployParam) (*httputil.ResponseJson, int)
	Upgrade(deploy helm.DeployParam) (*httputil.ResponseJson, int)
	Uninstall(namespace, releaseName string) (*httputil.ResponseJson, int)
	GetReleaseHistory(namespace string, name string, history int) (*httputil.ResponseJson, int)
	RollbackRelease(namespace, releaseName, chartName string, version, maxHistory int) (*httputil.ResponseJson, int)
	GetManifest(namespace, releaseName string) (*httputil.ResponseJson, int)
	GetValues(namespace, releaseName string) (*httputil.ResponseJson, int)
	GetReleasePods(namespace, releaseName string) (*httputil.ResponseJson, int)
	CheckReleaseStatus(chartName string) (*httputil.ResponseJson, int)
}

type helmClient struct {
	kubeConfig    *rest.Config
	dynamicClient dynamic.Interface
	clientset     kubernetes.Interface

	chartGetter ChartGetter
}

// NewHelmOperation helm operation requires client set&dynamic client，for kubernetes resource operation
func NewHelmOperation(kubeConfig *rest.Config) (Operation, error) {
	return NewHelmOperationWithDeps(kubeConfig, nil)
}

// NewHelmOperationWithDeps creates a real helm.Operation with injectable dependencies.
func NewHelmOperationWithDeps(kubeConfig *rest.Config, chartGetter ChartGetter) (Operation, error) {
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

	if chartGetter == nil {
		chartGetter = rpcChartGetter{}
	}

	return &helmClient{
		kubeConfig:    kubeConfig,
		dynamicClient: dynamicClient,
		clientset:     clientset,
		chartGetter:   chartGetter,
	}, nil
}
