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
	"errors"
	"fmt"
	"net/http"
	"strconv"
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
	"monitoring-dashboard-service/pkg/utils/zlog"
)

// used for create widget crd
var (
	widgetCRDGVR = schema.GroupVersionResource{
		Group:    global.CRDRepoGroup,
		Version:  global.CRDRepoVersion,
		Resource: global.WidgetCRDRepoResource,
	}
)

// implement operation
func (c *dashboardClient) CreateWidget(vo *model.WidgetVO) (*httputil.ResponseJson, int) {
	createWidgetCR, err := c.createWidgetCR(vo)
	if err != nil {
		zlog.FormatError("create widget cr failed: %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerCreateWidgetFailed,
			Msg:  "create widget cr failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "successfully created widget cr",
		Data: createWidgetCR,
	}, http.StatusCreated
}

func (c *dashboardClient) UpdateWidget(vo *model.WidgetVO) (*httputil.ResponseJson, int) {
	err := c.updateWidgetCR(vo)
	if err != nil {
		zlog.FormatError("update widget cr failed: %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerUpdateWidgetFailed,
			Msg:  "update widget cr failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "successfully updated widget cr",
	}, http.StatusOK
}

func (c *dashboardClient) DeleteWidget(widgetID string) (*httputil.ResponseJson, int) {
	err := c.deleteWidgetCR(widgetID)
	if err != nil {
		zlog.FormatError("delete widget cr failed: %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerDeleteWidgetFailed,
			Msg:  "delete widget cr failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "successfully deleted widget cr",
	}, http.StatusOK
}

func (c *dashboardClient) GetIndicatorNameList(dataSource string) (*httputil.ResponseJson, int) {
	nameList, err := c.getIndicatorNameList(dataSource)
	if err != nil {
		zlog.FormatError("get indicator list failed: %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerGetIndicatorNameListFailed,
			Msg:  "get indicator list failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "get indicator name list success",
		Data: nameList,
	}, http.StatusOK
}

func (c *dashboardClient) GetInstanceMetricByDisplayName(dataSource string, displayName string) (
	*httputil.ResponseJson, int) {
	indicator, err := c.getIndicatorMetricByDisplayName(dataSource, displayName)
	if err != nil {
		zlog.FormatError("get instance by indicator name failed %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerGetInstancesMetricFailed,
			Msg:  "get instance by indicator name failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "get instance by indicator name success",
		Data: indicator,
	}, http.StatusOK
}

func (c *dashboardClient) GetInstanceValueByInstances(query *model.InstanceQuery) (
	*httputil.ResponseJson, int) {
	instances, err := c.getInstanceValueByInstances(query)
	if err != nil {
		zlog.FormatError("get instance value by instances failed %v", err)
		return &httputil.ResponseJson{
			Code: global.ServerGetInstancesFailed,
			Msg:  "get instance value by instances failed",
			Data: nil,
		}, http.StatusInternalServerError
	}
	return &httputil.ResponseJson{
		Code: global.Success,
		Msg:  "get instance value by instances success",
		Data: instances,
	}, http.StatusCreated
}

func (c *dashboardClient) getInstanceValueByInstances(query *model.InstanceQuery) (
	[]model.Instance, error) {
	// 解析JSON转为键值对Metrics
	var instances []model.Instance
	for _, instanceJSON := range query.Instances {
		var instance model.Instance
		var metric map[string]interface{}
		err := json.Unmarshal([]byte(instanceJSON), &metric)
		if err != nil {
			zlog.FormatError("unmarshal instance metrics failed %v", err)
			return nil, err
		}
		instance.Metrics = metric
		// 拿着Metrics查询values并装配到instance
		err = c.assembleInstanceValue(&instance, query.DataSource, query.Start, query.End, query.Step)
		if err != nil {
			zlog.FormatError("assemble instance value failed %v", err)
			return nil, err
		}
		instances = append(instances, instance)
	}
	return instances, nil
}

func (c *dashboardClient) getIndicatorMetricByDisplayName(dataSource, displayName string) (
	*model.Indicator, error) {
	// 拼接URI向Prometheus发送请求
	uri := fmt.Sprintf("%s%s/query?query=%s", dataSource, global.PrometheusAPIPath, displayName)
	body, err := httputil.GetResponseJSONFromHTTPGet(uri)
	if err != nil || body == nil {
		return nil, err
	}
	data := httputil.GetMapFromMapByKey(body, "data") // map[string]interface{}
	if data == nil {
		return nil, nil
	}
	result := httputil.GetMapListFromMapByKey(data, "result") // []interface{}
	if result == nil {
		return nil, nil
	}
	var instances []model.Instance
	for _, item := range result {
		metrics := httputil.GetMapFromMapByKey(item, "metric")
		if metrics == nil {
			metrics = make(map[string]interface{})
		}
		instance := model.Instance{
			Metrics: metrics,
			Values:  []interface{}{},
		}
		instances = append(instances, instance)
	}
	indicator := &model.Indicator{
		DisplayName: displayName,
		Instances:   instances,
	}
	return indicator, nil
}

func (c *dashboardClient) getIndicatorNameList(dataSource string) ([]interface{}, error) {
	uri := fmt.Sprintf("%s%s/label/__name__/values", dataSource, global.PrometheusAPIPath)
	body, err := httputil.GetResponseJSONFromHTTPGet(uri) // map[string]interface{}
	if err != nil || body == nil {
		return nil, err
	}
	data := httputil.GetInterfaceListFromMapByKey(body, "data")
	if data == nil {
		return []interface{}{}, nil
	}
	return data, nil
}

// service
func (c *dashboardClient) deleteWidgetCR(widgetID string) error {
	// 获取widgetCR
	widgetCR, err := c.getWidgetCRByWidgetID(widgetID)
	if err != nil {
		return err
	}
	// 获取dashboard
	dashboardCR, err := c.getDashboardCRByDashboardID(widgetCR.Spec.DashboardID)
	if err != nil {
		return err
	}
	// 删除widgetCR
	err = c.dynamicClient.Resource(widgetCRDGVR).Namespace(global.MonitorServiceDefaultNamespace).
		Delete(context.TODO(), widgetID, v1.DeleteOptions{})
	if err != nil {
		zlog.FormatError("delete widget cr failed %v", err)
		return err
	}
	// 更新dashboardCR
	dashboardCR.Spec.TotalWidgets--
	dashboardCR.Spec.LastUpdateTime = time.Now().UTC().Format(time.RFC3339)
	if dashboardCR.Spec.TotalWidgets < 0 {
		zlog.FormatError("widget count less than 0")
		return err
	}
	err = c.updateDashboardCR(dashboardCR)
	if err != nil {
		zlog.FormatError("update dashboard cr failed")
		return err
	}
	return nil
}

func (c *dashboardClient) updateWidgetCR(vo *model.WidgetVO) error {
	widgetCR, err := c.getWidgetCRByWidgetID(vo.WidgetID) // 获取widgetCR
	if err != nil {
		return err
	}
	dashboardCR, err := c.getDashboardCRByDashboardID(widgetCR.Spec.DashboardID) // 获取dashboard CR
	if err != nil {
		return err
	}
	patch, err := model.NewPatchUpdateVO(vo) // 获取要更新的内容patch
	if err != nil {
		return err
	}
	patchBytes, err := json.Marshal(patch)
	if err != nil {
		zlog.FormatError("marshal patch to bytes failed %v", err)
		return err
	}
	_, err = c.dynamicClient.Resource(widgetCRDGVR).Namespace(global.MonitorServiceDefaultNamespace).
		Patch(context.TODO(), vo.WidgetID, types.MergePatchType, patchBytes, v1.PatchOptions{})
	if err != nil {
		zlog.FormatError("patch to cr failed %v", err)
		return err
	}
	// 更新dashboard LastUpdateTime
	dashboardCR.Spec.LastUpdateTime = time.Now().UTC().Format(time.RFC3339)
	err = c.updateDashboardCR(dashboardCR)
	if err != nil {
		zlog.FormatError("update dashboard cr failed %s", err)
		return err
	}
	return nil
}

// CreateWidget PostCreateWidget WidgetCRD CRUD Operation
func (c *dashboardClient) createWidgetCR(vo *model.WidgetVO) (*unstructured.Unstructured, error) {
	// 根据dashboardID获取dashboard CR
	dashboardCR, err := c.getDashboardCRByDashboardID(vo.DashboardID)
	if err != nil {
		return nil, err
	}
	// 根据VO构建WidgetCR
	newWidgetCR, err := model.NewWidgetCRDByCreateVO(vo)
	if err != nil {
		return nil, err
	}
	widgetUnstructured, err := k8sutil.StructToUnstructured(newWidgetCR)
	if err != nil {
		return nil, err
	}
	widgetUnstructured, err = c.dynamicClient.Resource(widgetCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		Create(context.TODO(), widgetUnstructured, v1.CreateOptions{})
	if err != nil {
		return nil, err
	}
	zlog.FormatInfo("widget cr: %s created", newWidgetCR.Name)
	// 更新dashboard CR中的totalWidgets
	dashboardCR.Spec.TotalWidgets = dashboardCR.Spec.TotalWidgets + 1
	dashboardCR.Spec.LastUpdateTime = time.Now().UTC().Format(time.RFC3339)
	err = c.updateDashboardCR(dashboardCR)
	if err != nil {
		zlog.FormatError("update dashboard cr failed %s", err)
		return nil, err
	}
	return widgetUnstructured, nil
}

func (c *dashboardClient) getWidgetListByDashboardID(dashboardVO *model.DashboardVO) ([]model.Widget, error) {
	listOpts := v1.ListOptions{
		LabelSelector: fmt.Sprintf("dashboardID=%s", dashboardVO.DashboardID),
	}
	widgetCRList, err := c.dynamicClient.Resource(widgetCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		List(context.TODO(), listOpts)
	if err != nil {
		zlog.FormatError("Error getting %s: %s", global.WidgetCRDRepoListKind, dashboardVO.DashboardID)
		return nil, err
	}
	var widgets []model.Widget
	for _, widgetUnstructured := range widgetCRList.Items {
		var widgetCR model.WidgetCRD
		err = runtime.DefaultUnstructuredConverter.FromUnstructured(widgetUnstructured.Object, &widgetCR)
		if err != nil {
			zlog.FormatError("Error converting to %s: %s", global.WidgetCRDRepoKind, widgetCR.Spec.WidgetID)
			return nil, err
		}
		wIGraph, err := model.GraphFactoryByWidgetCRD(&widgetCR)
		if err != nil {
			return nil, err
		}
		widget := &model.Widget{
			WidgetID:    widgetCR.Spec.WidgetID,
			DashboardID: widgetCR.Spec.DashboardID,
			DisplayName: widgetCR.Spec.DisplayName,
			DataSource:  widgetCR.Spec.DataSource,
			WidgetType:  widgetCR.Spec.WidgetType,
			WidgetGraph: wIGraph,
		}
		err = c.assembleWidgetIndicatorMetrics(widget, &widgetCR)
		if err != nil {
			zlog.FormatError("assemble widget metrics failed %v", err)
			return nil, err
		}
		err = c.assembleWidgetIndicatorValues(widget, dashboardVO.Start, dashboardVO.End, dashboardVO.Step)
		if err != nil {
			zlog.FormatError("assemble widget values failed %v", err)
			return nil, err
		}
		widgets = append(widgets, *widget)
	}
	return widgets, nil
}

func (c *dashboardClient) assembleWidgetIndicatorMetrics(widget *model.Widget, widgetCR *model.WidgetCRD) error {
	for _, instancesJSON := range widgetCR.Spec.Instances {
		var instance model.Instance
		var metric map[string]interface{}
		err := json.Unmarshal([]byte(instancesJSON), &metric)
		if err != nil {
			zlog.FormatError("unmarshal instance metrics failed %v", err)
			return err
		}
		instance.Metrics = metric
		widget.Instances = append(widget.Instances, instance)
	}
	// 组装 Widget 的 Instance 并返回
	return nil
}

func (c *dashboardClient) assembleWidgetIndicatorValues(widget *model.Widget, start, end, step string) error {
	// 将 indicators 从 JSON String 解析为 Map 列表
	for index, instance := range widget.Instances {
		// 根据 JSON String 和 DataSource 查询 prometheus
		err := c.assembleInstanceValue(&instance, widget.DataSource, start, end, step)
		if err != nil {
			zlog.FormatError("assemble instance value failed %v", err)
			return err
		}
		widget.Instances[index].Values = instance.Values
	}
	return nil
}

func (c *dashboardClient) getWidgetCRListByDashboardID(dashboardID string) ([]model.WidgetCRD, error) {
	// 以 dashboardID 为标签获取 widgetCRList
	listOpts := v1.ListOptions{
		LabelSelector: fmt.Sprintf("dashboardID=%s", dashboardID),
	}
	widgetCRUList, err := c.dynamicClient.Resource(widgetCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		List(context.TODO(), listOpts)
	if err != nil {
		zlog.FormatError("Error getting %s: %s", global.WidgetCRDRepoListKind, dashboardID)
		return nil, err
	}
	var widgetCRList []model.WidgetCRD
	for _, widgetUnstructured := range widgetCRUList.Items {
		var widgetCR model.WidgetCRD
		err = runtime.DefaultUnstructuredConverter.FromUnstructured(widgetUnstructured.Object, &widgetCR)
		if err != nil {
			zlog.FormatError("Error converting to %s: %s", global.WidgetCRDRepoKind, widgetCR.Spec.WidgetID)
			return nil, err
		}
		widgetCRList = append(widgetCRList, widgetCR)
	}
	return widgetCRList, err
}

func (c *dashboardClient) assembleInstanceValue(instance *model.Instance, dataSource, start, end, step string,
) error {
	// 根据 JSON String 和 DataSource 查询 prometheus
	var labelQuery string
	for key, value := range instance.Metrics {
		if labelQuery == "" {
			labelQuery = fmt.Sprintf(`%s="%s"`, key, value)
			continue
		}
		label := fmt.Sprintf(`%s="%s"`, key, value)
		labelQuery = fmt.Sprintf("%s,%s", labelQuery, label)
	}
	uri := fmt.Sprintf("%s%s/query_range?query={%s}&start=%s&end=%s&step=%ss",
		dataSource, global.PrometheusAPIPath, labelQuery, start, end, step)

	body, err := httputil.GetResponseJSONFromHTTPGet(uri)
	if err != nil {
		return err
	}
	data := httputil.GetMapFromMapByKey(body, "data") // map[string]interface{}
	if data == nil {
		instance.Values = []interface{}{}
		return nil
	}
	result := httputil.GetMapListFromMapByKey(data, "result") // []map[string]interface{}
	if result == nil {
		zlog.FormatInfo("no result found from the data of prometheus")
		instance.Values = []interface{}{}
		return nil
	}
	firstResult := result[0]
	values := httputil.GetInterfaceListFromMapByKey(firstResult, "values")
	if values == nil {
		zlog.FormatInfo("no value found from the first result of prometheus")
		instance.Values = []interface{}{}
		return nil
	}
	standardValues, err := c.standardizeInstanceValue(values, start, end, step)
	if err != nil {
		return err
	}
	instance.Values = standardValues
	return nil
}

func (c *dashboardClient) standardizeInstanceValue(values []interface{}, start, end, step string) (
	[]interface{}, error) {
	startTime, err1 := time.Parse(time.RFC3339, start)
	endTime, err2 := time.Parse(time.RFC3339, end)
	if err1 != nil || err2 != nil {
		return nil, errors.New("start or end time is not valid")
	}
	startTimeStep := startTime.Unix()
	endTimeStep := endTime.Unix()
	stepInt, err3 := strconv.Atoi(step)
	if err3 != nil {
		return nil, errors.New("step is not valid")
	}
	var result []interface{}
	for i := startTimeStep; i <= endTimeStep; i += int64(stepInt) {
		result = append(result, []interface{}{i, nil})
	} // 初始化标准区间列表
	for i := 0; i < len(values); i += 1 {
		timeStepFloat, ok := values[i].([]interface{})[0].(float64)
		if !ok {
			return nil, errors.New("time step of instance is not float64")
		}
		timeStepInt := int64(timeStepFloat)
		index := (timeStepInt - startTimeStep) / int64(stepInt) // 计算values中的时间戳在标准时间区间中的位置
		if index >= int64(len(result)) || index < 0 {
			return nil, errors.New("time range of instance is not valid")
		}
		result[index] = values[i]
	}
	return result, nil
}

func (c *dashboardClient) getWidgetCRByWidgetID(widgetID string) (*model.WidgetCRD, error) {
	widgetUnstructured, err := c.dynamicClient.Resource(widgetCRDGVR).
		Namespace(global.MonitorServiceDefaultNamespace).
		Get(context.TODO(), widgetID, v1.GetOptions{})
	if err != nil {
		return nil, err
	}
	var widgetCR model.WidgetCRD
	err = runtime.DefaultUnstructuredConverter.FromUnstructured(widgetUnstructured.Object, &widgetCR)
	if err != nil {
		return nil, err
	}
	return &widgetCR, nil
}
