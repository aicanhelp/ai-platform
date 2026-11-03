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

/*
the apiserver of monitor module v1 beta1
*/

// Package controller 为monitor模块的controller层，用于接受http request请求并调用service层业务函数
package controller

import (
	"fmt"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/rest"

	"monitoring-dashboard-service/pkg/monitor/controller/param"
	"monitoring-dashboard-service/pkg/monitor/service"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

// BindMonitorRoute bind the api route of monitor module
func BindMonitorRoute(webService *restful.WebService, kubeConfig *rest.Config) {
	operation, err := service.NewMonitorOperation(kubeConfig)
	if err != nil {
		zlog.LogErrorLine("new monitor operation failed, err: %s", err)
	}
	handler := newHandler(operation)

	bindWidgetRoutes(webService, handler)
	bindDashboardRoutes(webService, handler)
	bindDashboardRoutes(webService, handler)
}

// bindWidgetRoutes bind the api route of widget service
func bindWidgetRoutes(webService *restful.WebService, handler *Handler) {
	webService.Route(webService.POST("/widgets").To(handler.postCreateWidget))
	webService.Route(webService.PATCH(fmt.Sprintf("/widgets/{%s}", param.WidgetID)).
		To(handler.patchUpdateWidget))
	webService.Route(webService.DELETE(fmt.Sprintf("/widgets/{%s}", param.WidgetID)).
		To(handler.deleteWidget))
	// indicators
	webService.Route(webService.GET("/widgets/indicators").To(handler.getIndicatorList))
	webService.Route(webService.GET(fmt.Sprintf("/widgets/indicators/{%s}", param.DisplayName)).
		To(handler.getInstanceMetricList))
	webService.Route(webService.POST("/widgets/indicators/instances").To(handler.getInstanceValueList))
}

// bindDashboardRoutes bind the api route of dashboard service
func bindDashboardRoutes(webService *restful.WebService, handler *Handler) {
	webService.Route(webService.POST("/dashboards").To(handler.postCreateDashboard))
	webService.Route(webService.GET(fmt.Sprintf("/dashboards/{%s}", param.DashboardID)).
		To(handler.getDashboardByDashboardID))
	webService.Route(webService.GET("/dashboards").To(handler.getDashboardList))
	webService.Route(webService.PATCH(fmt.Sprintf("/dashboards/{%s}", param.DashboardID)).
		To(handler.patchUpdateDashboard))
	webService.Route(webService.DELETE(fmt.Sprintf("/dashboards/{%s}", param.DashboardID)).
		To(handler.deleteDashboard))
}
