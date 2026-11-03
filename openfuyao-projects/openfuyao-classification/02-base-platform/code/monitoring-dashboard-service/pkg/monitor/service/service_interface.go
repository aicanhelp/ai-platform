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

// Package service 为monitor模块的Service层，支持具体的模块业务
package service

import (
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"monitoring-dashboard-service/pkg/monitor/model"
	"monitoring-dashboard-service/pkg/utils/httputil"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

// Operation 是Service层统一对上层提供的接口，由Service层函数实现
type Operation interface {
	/* Widget Operation */
	// CreateWidget Widget Operation
	CreateWidget(widgetVO *model.WidgetVO) (*httputil.ResponseJson, int)
	UpdateWidget(widgetVO *model.WidgetVO) (*httputil.ResponseJson, int)
	DeleteWidget(widgetID string) (*httputil.ResponseJson, int)
	GetIndicatorNameList(dataSource string) (*httputil.ResponseJson, int)
	GetInstanceMetricByDisplayName(dataSource, displayName string) (*httputil.ResponseJson, int)
	GetInstanceValueByInstances(instanceQuery *model.InstanceQuery) (
		*httputil.ResponseJson, int)

	/* Dashboard Operation */
	CreateDashboard(dashboardVO *model.DashboardVO) (*httputil.ResponseJson, int)
	GetDashboardByDashboardID(dashboardVO *model.DashboardVO) (*httputil.ResponseJson, int)
	GetDashboardList() (*httputil.ResponseJson, int)
	UpdateDashboard(dashboardVO *model.DashboardVO) (*httputil.ResponseJson, int)
	DeleteDashboard(dashboardID string) (*httputil.ResponseJson, int)
}

// dashboardClient 用于实现接口 Operation
type dashboardClient struct {
	kubeConfig    *rest.Config
	dynamicClient dynamic.Interface
	clientSet     kubernetes.Interface
}

// NewMonitorOperation 返回实现接口 Operation 的结构体 dashboardClient 指针
func NewMonitorOperation(kubeConfig *rest.Config) (Operation, error) {
	dynamicClient, err := dynamic.NewForConfig(kubeConfig)
	if err != nil {
		zlog.LogError("error creating dynamic client, err: %v", err)
		return nil, err
	}
	clientSet, err := kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		zlog.LogError("error creating client set, err: %v", err)
		return nil, err
	}
	return &dashboardClient{
		kubeConfig:    kubeConfig,
		dynamicClient: dynamicClient,
		clientSet:     clientSet,
	}, nil
}
