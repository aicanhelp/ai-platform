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

// Package controller 为monitor模块的controller层，用于接受http request请求并调用service层业务函数
package controller

import (
	"net/http"

	"github.com/emicklei/go-restful/v3"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/monitor/controller/param"
	"monitoring-dashboard-service/pkg/monitor/model"
	"monitoring-dashboard-service/pkg/monitor/service"
	"monitoring-dashboard-service/pkg/utils/httputil"
	"monitoring-dashboard-service/pkg/utils/uutil"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

// Handler helm handler that contains all the helm operations
type Handler struct {
	dashboardHandler service.Operation
}

func newHandler(handler service.Operation) *Handler {
	return &Handler{dashboardHandler: handler}

}

// controller apiserver
func (h *Handler) postCreateWidget(request *restful.Request, response *restful.Response) {
	widgetVO := &model.WidgetVO{}
	err := request.ReadEntity(widgetVO) // 从request中获取实体
	if err != nil {
		zlog.FormatError("invalid request body by client: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidRequestBody,
			Msg:  "invalid request body by client",
		})
		return
	}
	err = widgetVO.ValidateCreate()
	if err != nil {
		zlog.FormatError("client request contains invalid parameters: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.CreateWidget(widgetVO)
	_ = response.WriteHeaderAndEntity(status, result)
}

// patchUpdateWidget controller apiserver 接受前端请求，对入参进行合法性校验
func (h *Handler) patchUpdateWidget(request *restful.Request, response *restful.Response) {
	widgetVO := &model.WidgetVO{}
	err := request.ReadEntity(widgetVO)
	widgetVO.WidgetID = request.PathParameter(param.WidgetID)
	if err != nil {
		zlog.FormatError("invalid request body by client: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidRequestBody,
			Msg:  "invalid request body by client",
		})
		return
	}
	err = widgetVO.ValidatePatchUpdate()
	if err != nil {
		zlog.FormatError("client request contains invalid parameters: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.UpdateWidget(widgetVO)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) deleteWidget(request *restful.Request, response *restful.Response) {
	widgetID := request.PathParameter(param.WidgetID)
	err := uutil.ValidateID(widgetID)
	if err != nil {
		zlog.FormatError("client request contains invalid parameters: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.DeleteWidget(widgetID)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) postCreateDashboard(request *restful.Request, response *restful.Response) {
	dashboardVO := &model.DashboardVO{}
	errEntity := request.ReadEntity(dashboardVO)
	if errEntity != nil {
		zlog.FormatError("invalid request body by client: %v", errEntity)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidRequestBody,
			Msg:  "invalid request body by client",
		})
		return
	}
	errValidate := dashboardVO.ValidateCreate()
	if errValidate != nil {
		zlog.FormatError("client request contains invalid parameters: %v", errValidate)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.CreateDashboard(dashboardVO)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getDashboardByDashboardID(request *restful.Request, response *restful.Response) {
	dashboardVO := model.DashboardVO{
		DashboardID: request.PathParameter(param.DashboardID),
		Start:       request.QueryParameter(param.Start),
		End:         request.QueryParameter(param.End),
		Step:        request.QueryParameter(param.Step),
	}
	err := dashboardVO.ValidatePostQuery()
	if err != nil {
		zlog.FormatError("client request contains invalid parameters: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.GetDashboardByDashboardID(&dashboardVO)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getDashboardList(request *restful.Request, response *restful.Response) {
	result, status := h.dashboardHandler.GetDashboardList()
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) patchUpdateDashboard(request *restful.Request, response *restful.Response) {
	dashboardVO := &model.DashboardVO{}
	err := request.ReadEntity(dashboardVO)
	if err != nil {
		zlog.FormatError("invalid request body by client: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidRequestBody,
			Msg:  "invalid request body by client",
		})
		return
	}
	dashboardVO.DashboardID = request.PathParameter(param.DashboardID)
	err = dashboardVO.ValidatePatchUpdate()
	if err != nil {
		zlog.FormatError("client request contains invalid parameters: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	// 调用service层
	result, status := h.dashboardHandler.UpdateDashboard(dashboardVO)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) deleteDashboard(request *restful.Request, response *restful.Response) {
	dashboardID := request.PathParameter(param.DashboardID)
	err := uutil.ValidateID(dashboardID)
	if err != nil {
		zlog.FormatError("client request contains invalid parameters: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.DeleteDashboard(dashboardID)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getIndicatorList(request *restful.Request, response *restful.Response) {
	dataSource := request.QueryParameter(param.DataSource)
	err := uutil.ValidateHost(dataSource)
	if err != nil {
		zlog.FormatError("client request contains invalid parameters: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.GetIndicatorNameList(dataSource)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getInstanceMetricList(request *restful.Request, response *restful.Response) {
	dataSource := request.QueryParameter(param.DataSource)
	displayName := request.PathParameter(param.DisplayName)
	errHost := uutil.ValidateHost(dataSource)
	errName := uutil.ValidateDisplayName(displayName, global.MaxIndicatorNameLength)
	if errHost != nil || errName != nil {
		zlog.FormatError("client request contains invalid parameters: %v, %v", errHost, errName)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.GetInstanceMetricByDisplayName(dataSource, displayName)
	_ = response.WriteHeaderAndEntity(status, result)
}

func (h *Handler) getInstanceValueList(request *restful.Request, response *restful.Response) {
	var instances []string
	err := request.ReadEntity(&instances)
	if err != nil {
		zlog.FormatError("invalid request body by client: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidRequestBody,
			Msg:  "invalid request body by client",
		})
		return
	}
	instanceQuery := model.InstanceQuery{
		DataSource: request.QueryParameter(param.DataSource),
		Start:      request.QueryParameter(param.Start),
		End:        request.QueryParameter(param.End),
		Step:       request.QueryParameter(param.Step),
		Instances:  instances,
	}
	err = instanceQuery.Validate()
	if err != nil {
		zlog.FormatError("client request contains invalid parameters: %v", err)
		_ = response.WriteHeaderAndEntity(http.StatusBadRequest, httputil.ResponseJson{
			Code: global.ClientInvalidParameters,
			Msg:  "client request contains invalid parameters",
		})
		return
	}
	result, status := h.dashboardHandler.GetInstanceValueByInstances(&instanceQuery)
	_ = response.WriteHeaderAndEntity(status, result)
}
