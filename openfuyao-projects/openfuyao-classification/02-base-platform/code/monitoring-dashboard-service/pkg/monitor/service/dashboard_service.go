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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/monitor/model"
	"monitoring-dashboard-service/pkg/utils/httputil"
	"monitoring-dashboard-service/pkg/utils/k8sutil"
	"monitoring-dashboard-service/pkg/utils/uutil"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

// used for create widget crd
var (
	dashboardCRDGVR = schema.GroupVersionResource{
		Group:    global.CRDRepoGroup,
		Version:  global.CRDRepoVersion,
		Resource: global.DashboardRepoResource,
	}
)

func (c *dashboardClient) CreateDashboard(vo *model.DashboardVO) (*httputil.ResponseJson, int) {
	dashboards, err := c.getDashboardSpecList()
	if vo.IsDisplayNameExists(dashboards) {
		return &httputil.ResponseJson{
			Code: global.ClientDuplicateNameExist,
			Msg:  "display name already exists",
			Data: nil,
		}, http.StatusBadRequest
	}
	createDashboardCR, err := c.createDashboardCR(vo)
	if err != nil {
		zlog.FormatError("create dashboard cr failed: %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerCreateDashboardFailed,
			Msg:  "create dashboard cr failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "successfully created dashboard cr",
		Data: createDashboardCR,
	}, http.StatusCreated
}

func (c *dashboardClient) GetDashboardByDashboardID(vo *model.DashboardVO) (
	*httputil.ResponseJson, int) {
	dashboard, err := c.getDashboardByDashboardID(vo)
	if err != nil {
		zlog.FormatError("get dashboard cr failed: %s", err)
		return &httputil.ResponseJson{
			Code: global.ServerGetDashboardFailed,
			Msg:  "get dashboard cr failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "successfully access to dashboard cr",
		Data: dashboard,
	}, http.StatusOK
}

func (c *dashboardClient) GetDashboardList() (*httputil.ResponseJson, int) {
	dashboards, err := c.getDashboardSpecList()
	if err != nil {
		zlog.FormatError("get dashboard cr list failed: %s", err)
		return &httputil.ResponseJson{
			Code: global.ServerGetDashboardListFailed,
			Msg:  "get dashboard cr list failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "successfully access to dashboard cr list",
		Data: dashboards,
	}, http.StatusOK
}

func (c *dashboardClient) UpdateDashboard(vo *model.DashboardVO) (*httputil.ResponseJson, int) {
	dashboardCR, err := c.getDashboardCRByDashboardID(vo.DashboardID)
	if err != nil {
		return &httputil.ResponseJson{
			Code: global.ServerGetDashboardFailed,
			Msg:  "get dashboard cr by id failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	dashboards, err := c.getDashboardSpecList()
	if vo.IsDisplayNameExists(dashboards) {
		return &httputil.ResponseJson{
			Code: global.ClientDuplicateNameExist,
			Msg:  "display name already exists",
			Data: nil,
		}, http.StatusBadRequest
	}
	dashboardCR.Spec.DisplayName = vo.DisplayName
	dashboardCR.Spec.LastUpdateTime = time.Now().UTC().Format(time.RFC3339)
	err = c.updateDashboardCR(dashboardCR)
	if err != nil {
		zlog.FormatError("update dashboard cr failed: %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerUpdateDashboardFailed,
			Msg:  "update dashboard cr failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "successfully updated dashboard cr",
	}, http.StatusOK
}

func (c *dashboardClient) DeleteDashboard(dashboardID string) (*httputil.ResponseJson, int) {
	err := c.deleteDashboardCR(dashboardID)
	if err != nil {
		zlog.FormatError("delete dashboard failed: %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerDeleteDashboardFailed,
			Msg:  "delete dashboard failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "successful deleted dashboard",
	}, http.StatusOK
}

func (c *dashboardClient) deleteDashboardCR(dashboardID string) error {
	// 根据dashboardID查询是否有widget，级联删除
	widgetCRList, err := c.getWidgetCRListByDashboardID(dashboardID)
	if err != nil {
		zlog.FormatError("get widget cr list failed: %v", err)
		return err
	}
	for _, widgetCR := range widgetCRList {
		err = c.deleteWidgetCR(widgetCR.Spec.WidgetID)
		if err != nil {
			zlog.FormatError("delete widget cr failed: %v", err)
			return err
		}
	}
	// 删除dashboard
	err = c.dynamicClient.Resource(dashboardCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		Delete(context.TODO(), dashboardID, v1.DeleteOptions{})
	if err != nil {
		zlog.FormatError("delete dashboard cr failed: %v", err)
		return err
	}
	return nil
}

func (c *dashboardClient) updateDashboardCR(dashboardCR *model.DashboardCRD) error {
	patch := map[string]interface{}{
		"spec": map[string]interface{}{
			"displayName":    dashboardCR.Spec.DisplayName,
			"totalWidgets":   dashboardCR.Spec.TotalWidgets,
			"lastUpdateTime": dashboardCR.Spec.LastUpdateTime,
		},
	}
	patchBytes, err := json.Marshal(patch)
	if err != nil {
		zlog.FormatError("marshal patch to bytes failed: %v", err)
		return err
	}
	_, err = c.dynamicClient.Resource(dashboardCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		Patch(context.TODO(), dashboardCR.Spec.DashboardID, types.MergePatchType, patchBytes, v1.PatchOptions{})
	if err != nil {
		zlog.FormatError("patch to dashboard cr failed: %v", err)
		return err
	}
	return nil
}

func (c *dashboardClient) getDashboardSpecList() ([]*model.DashboardSpec, error) {
	// dashboardCRUList is the unstructured list of monitor CRs
	dashboardCRUList, err := c.dynamicClient.Resource(dashboardCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		List(context.TODO(), v1.ListOptions{})
	if err != nil {
		return nil, err
	}
	var dashboardSpecList []*model.DashboardSpec
	for _, dashboardCRU := range dashboardCRUList.Items {
		var dashboardCR model.DashboardCRD
		err = runtime.DefaultUnstructuredConverter.FromUnstructured(dashboardCRU.Object, &dashboardCR)
		if err != nil {
			zlog.FormatError("Error converting to %s: %s", global.DashboardRepoKind, dashboardCR.GetName())
			return nil, err
		}
		dashboardSpec := &model.DashboardSpec{
			DashboardID:    dashboardCR.Spec.DashboardID,
			DisplayName:    dashboardCR.Spec.DisplayName,
			Creator:        dashboardCR.Spec.Creator,
			TotalWidgets:   dashboardCR.Spec.TotalWidgets,
			LastUpdateTime: dashboardCR.Spec.LastUpdateTime,
		}
		dashboardSpecList = append(dashboardSpecList, dashboardSpec)
	}
	return dashboardSpecList, err
}

func (c *dashboardClient) getDashboardByDashboardID(vo *model.DashboardVO) (
	*model.Dashboard, error) {
	// 根据ID获取dashboard CR
	dashboardCR, err := c.getDashboardCRByDashboardID(vo.DashboardID)
	if err != nil {
		return nil, err
	}
	// 根据 monitor ID 获取 Widget CR
	// 装填业务层 Dashboard 结构体
	widgets, err := c.getWidgetListByDashboardID(vo)
	if err != nil {
		zlog.FormatError("get widget list failed: %s", err)
		return nil, err
	}
	dashboard := &model.Dashboard{
		DashboardID:    dashboardCR.Spec.DashboardID,
		DisplayName:    dashboardCR.Spec.DisplayName,
		Creator:        dashboardCR.Spec.Creator,
		TotalWidgets:   len(widgets),
		LastUpdateTime: time.Now().UTC().Format(time.RFC3339),
		Widgets:        widgets,
	}
	return dashboard, nil
}

func (c *dashboardClient) createDashboardCR(vo *model.DashboardVO) (*unstructured.Unstructured, error) {
	dID := uutil.GenerateID()
	totalWidgets := 0
	lastUpdateTime := time.Now().UTC().Format(time.RFC3339)

	newDashboard := &model.DashboardCRD{
		TypeMeta:   getDashboardCRTypeMeta(),
		ObjectMeta: getDashboardCRObjectMeta(strings.ToLower(dID)),
		Spec: model.DashboardSpec{
			DashboardID:    dID,
			DisplayName:    vo.DisplayName,
			Creator:        vo.Creator,
			TotalWidgets:   totalWidgets,
			LastUpdateTime: lastUpdateTime,
		},
	}

	dashboardUnstructured, err := k8sutil.StructToUnstructured(newDashboard)
	if err != nil {
		zlog.FormatError("convert dashboard cr to unstructured error: %v", err)
		return nil, err
	}
	dashboardUnstructured, err = c.dynamicClient.Resource(dashboardCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		Create(context.TODO(), dashboardUnstructured, v1.CreateOptions{})
	if err != nil {
		zlog.FormatInfo("dashboard cr created failed %s", err)
		return nil, err
	}
	zlog.FormatInfo("dashboard cr: %s created", newDashboard.Name)
	return dashboardUnstructured, nil
}

func getDashboardCRObjectMeta(repoName string) v1.ObjectMeta {
	objectMeta := v1.ObjectMeta{
		Name:      repoName,
		Namespace: global.MonitorServiceDefaultNamespace,
		Annotations: map[string]string{
			fmt.Sprintf("%s/%s", global.MonitorServiceDefaultOrgName,
				global.ModificationTimestamp): time.Now().UTC().Format(time.RFC3339),
		},
	}
	return objectMeta
}

func getDashboardCRTypeMeta() v1.TypeMeta {
	typeMeta := v1.TypeMeta{
		Kind:       global.DashboardRepoKind,
		APIVersion: fmt.Sprintf("%s/%s", global.CRDRepoGroup, global.CRDRepoVersion),
	}
	return typeMeta
}

func (c *dashboardClient) getDashboardCRByDashboardID(dashboardID string) (*model.DashboardCRD, error) {
	// 根据ID获取dashboard CR
	dashboardUnstructured, err := c.dynamicClient.Resource(dashboardCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		Get(context.TODO(), dashboardID, v1.GetOptions{})
	if err != nil {
		zlog.FormatError("get dashboard cr failed %s", err)
		return nil, err
	}
	// 转换为结构体
	var dashboardCR model.DashboardCRD
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(dashboardUnstructured.Object, &dashboardCR)
	if err != nil {
		zlog.FormatError("Error converting to %s: %s", global.DashboardRepoKind, dashboardID)
		return nil, err
	}
	return &dashboardCR, nil
}
