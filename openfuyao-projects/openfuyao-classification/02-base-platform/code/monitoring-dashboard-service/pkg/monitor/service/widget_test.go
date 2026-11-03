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

package service

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/monitor/model"
	"monitoring-dashboard-service/pkg/utils/httputil"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

// TestDashboardClientCreateWidget API:创建widget
func TestDashboardClientCreateWidget(t *testing.T) {
	client := newTestDashboardClient()
	testCases := []struct {
		testName   string
		client     *dashboardClient
		vo         *model.WidgetVO
		wantResp   httputil.ResponseJson
		wantStatus int
	}{
		{
			testName: "Widget创建成功",
			client:   client,
			vo: &model.WidgetVO{
				DashboardID: widgetDashboardID,
				DisplayName: "测试监控组件创建",
				DataSource:  validateWidgetDataSource,
				WidgetType:  "3",
				Instances: []string{
					`{"__name__":"alertmanager_alerts","container":"alertmanager","endpoint":"web","instance":"172.17.217.26:9093","job":"alertmanager-main","namespace":"monitoring","pod":"alertmanager-main-2","service":"alertmanager-main","state":"active"}`,
				},
				BoardMin: "10",
				BoardMax: "100",
			},
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "successfully created widget cr",
				Data: widgetCreateSuccess,
			},
			wantStatus: http.StatusCreated,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, s := tc.client.CreateWidget(tc.vo)
			if tc.wantStatus == http.StatusCreated {
				// resp的类型为unstructured，需要转为struct进行处理
				widgetCR := getWidgetCRFromUnstructured(t, resp.Data)
				// 将resp中的动态值设置为固定值
				widgetCR.Spec.WidgetID = createWidgetIDForTest
				resp.Data = widgetCR.Spec
			}
			if s != tc.wantStatus {
				t.Errorf("test case %s: expected status %d, got %d", tc.testName, tc.wantStatus, s)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("test case %s: response data mismatch", tc.testName)
			}
		})
	}
}

var widgetCreateSuccess = model.WidgetSpec{
	WidgetID:    createWidgetIDForTest,
	DashboardID: widgetDashboardID,
	DisplayName: "测试监控组件创建",
	DataSource:  validateWidgetDataSource,
	WidgetType:  model.DashboardGraphType,
	Instances: []string{
		`{"__name__":"alertmanager_alerts","container":"alertmanager","endpoint":"web","instance":"172.17.217.26:9093","job":"alertmanager-main","namespace":"monitoring","pod":"alertmanager-main-2","service":"alertmanager-main","state":"active"}`,
	},
	BoardMin: 10,
	BoardMax: 100,
}

// getWidgetCRFromUnstructured API:创建widget 辅助函数 将Unstructured转为struct
func getWidgetCRFromUnstructured(t *testing.T, data any) *model.WidgetCRD {
	respUnstructured, ok := data.(*unstructured.Unstructured)
	if !ok {
		t.Errorf("Test failed, resp type error")
	}
	var widgetCR model.WidgetCRD
	err := runtime.DefaultUnstructuredConverter.FromUnstructured(respUnstructured.Object, &widgetCR)
	if err != nil {
		t.Errorf("Test failed")
	}
	return &widgetCR
}

// TestDashboardClientUpdateWidget API:更新widget
func TestDashboardClientUpdateWidget(t *testing.T) {
	client := newTestDashboardClient()
	testCases := createUpdateWidgetTestCases(client)
	runUpdateWidgetTests(t, testCases)
}

func createUpdateWidgetTestCases(client *dashboardClient) []updateWidgetTestCase {
	return []updateWidgetTestCase{
		{
			testName: "Widget更新成功",
			client:   client,
			vo:       createUpdateWidgetVO(createWidgetID),
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "successfully updated widget cr",
				Data: nil,
			},
			wantStatus: http.StatusOK,
		},
		{
			testName: "Widget更新失败:WidgetID不存在",
			client:   client,
			vo:       createUpdateWidgetVO(getWrongWidgetID),
			wantResp: httputil.ResponseJson{
				Code: global.ServerUpdateWidgetFailed,
				Msg:  "update widget cr failed",
				Data: nil,
			},
			wantStatus: http.StatusInternalServerError,
		},
	}
}

type updateWidgetTestCase struct {
	testName   string
	client     *dashboardClient
	vo         *model.WidgetVO
	wantResp   httputil.ResponseJson
	wantStatus int
}

func createUpdateWidgetVO(widgetID string) *model.WidgetVO {
	return &model.WidgetVO{
		WidgetID:    widgetID,
		DisplayName: "测试监控组件更新",
		DataSource:  validateWidgetDataSource,
		WidgetType:  "3",
		Instances: []string{
			`{"__name__":"alertmanager_alerts","container":"alertmanager","endpoint":"web","instance":"172.17.217.26:9093","job":"alertmanager-main","namespace":"monitoring","pod":"alertmanager-main-2","service":"alertmanager-main","state":"active"}`,
		},
		BoardMin: "0",
		BoardMax: "20",
	}
}

func runUpdateWidgetTests(t *testing.T, testCases []updateWidgetTestCase) {
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.UpdateWidget(tc.vo)
			if status != tc.wantStatus {
				t.Errorf("Test %s failed, status code error", tc.testName)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("Test %s failed, resp data error", tc.testName)
			}
		})
	}
}

// TestDashboardClientDeleteWidget API:删除widget
func TestDashboardClientDeleteWidget(t *testing.T) {
	client := newTestDashboardClient()
	testCases := []struct {
		testName   string
		client     *dashboardClient
		widgetID   string
		wantResp   httputil.ResponseJson
		wantStatus int
	}{
		{
			testName: "Widget删除成功",
			client:   client,
			widgetID: createWidgetID,
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "successfully deleted widget cr",
			},
			wantStatus: http.StatusOK,
		}, {
			testName: "Widget删除失败:WidgetID不存在",
			client:   client,
			widgetID: createWidgetID,
			wantResp: httputil.ResponseJson{
				Code: global.ServerDeleteWidgetFailed,
				Msg:  "delete widget cr failed",
				Data: nil,
			},
			wantStatus: http.StatusInternalServerError,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.DeleteWidget(tc.widgetID)
			if status != tc.wantStatus {
				t.Errorf("Test %s failed, status code error", tc.testName)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("Test %s failed, resp data error", tc.testName)
			}
		})
	}
}

func TestDashboardClientGetIndicatorNameList(t *testing.T) {
	server := mockNameListServer()
	defer server.Close()
	client := newTestDashboardClient()
	testCases := []struct {
		testName   string
		client     *dashboardClient
		dataSource string
		wantResp   httputil.ResponseJson
		wantStatus int
	}{
		{
			testName:   "获取Indicator名称列表成功",
			client:     client,
			dataSource: server.URL,
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "get indicator name list success",
				Data: nameListSuccess,
			},
			wantStatus: http.StatusOK,
		}, {
			testName:   "获取Indicator名称列表失败:URL错误",
			client:     client,
			dataSource: validateWidgetDataSource,
			wantResp: httputil.ResponseJson{
				Code: global.ServerGetIndicatorNameListFailed,
				Msg:  "get indicator list failed",
				Data: nil,
			},
			wantStatus: http.StatusInternalServerError,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.GetIndicatorNameList(tc.dataSource)
			if status != tc.wantStatus {
				t.Errorf("Test %s failed, status code error", tc.testName)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("Test %s failed, resp data error", tc.testName)
			}
		})
	}
}

var nameListSuccess = []interface{}{
	":node_memory_MemAvailable_bytes:sum",
	"ALERTS",
	"ALERTS_FOR_STATE",
	"aggregator_discovery_aggregation_count_total",
	"aggregator_unavailable_apiservice",
	"alertmanager_alerts",
	"alertmanager_alerts_invalid_total",
	"alertmanager_alerts_received_total",
	"alertmanager_build_info",
	"alertmanager_cluster_alive_messages_total",
	"alertmanager_cluster_enabled",
	"alertmanager_cluster_failed_peers",
}

func mockNameListServer() *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"status": "success",
			"data":   nameListSuccess,
		}
		respJson, err := json.Marshal(resp)
		if err != nil {
			zlog.FormatError("marshal response failed: %v", err)
			return
		}
		if r.URL.Path == "/api/v1/label/__name__/values" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, err := w.Write(respJson)
			if err != nil {
				zlog.FormatError("write response failed: %v", err)
			}
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	return server
}

func TestDashboardClientGetInstanceMetricByDisplayName(t *testing.T) {
	server := mockMetricsServer()
	defer server.Close()
	client := newTestDashboardClient()
	testCases := []struct {
		testName    string
		client      *dashboardClient
		dataSource  string
		displayName string
		wantResp    httputil.ResponseJson
		wantStatus  int
	}{
		{
			testName:    "通过DisplayName获取Indicator列表成功",
			client:      client,
			dataSource:  server.URL,
			displayName: "kubelet_http_requests_total",
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "get instance by indicator name success",
				Data: indicatorSuccess,
			},
			wantStatus: http.StatusOK,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.GetInstanceMetricByDisplayName(tc.dataSource, tc.displayName)
			if status != tc.wantStatus {
				t.Errorf("Test %s failed, status code error", tc.testName)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("Test %s failed, resp data error", tc.testName)
			}
		})
	}
}

func createTestMetrics() map[string]interface{} {
	return map[string]interface{}{
		"__name__":     "kubelet_http_requests_total",
		"endpoint":     "https-metrics",
		"instance":     "this-is-a-test-instance",
		"job":          "kubelet",
		"long_running": "false",
		"method":       "GET",
		"metrics_path": "/metrics",
		"namespace":    "kube-system",
		"node":         "test2-xiezuyue",
		"path":         "metrics",
		"server_type":  "readwrite",
		"service":      "kubelet",
	}
}

var indicatorSuccess = &model.Indicator{
	DisplayName: "kubelet_http_requests_total",
	Instances: []model.Instance{
		{
			Metrics: createTestMetrics(),
			Values:  []interface{}{},
		},
	},
}

func mockMetricsServer() *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(handleMetricsRequest))
	return server
}

func handleMetricsRequest(w http.ResponseWriter, r *http.Request) {
	resp := createMetricsResponse()
	respJson, err := json.Marshal(resp)
	if err != nil {
		zlog.FormatError("marshal response failed: %v", err)
		return
	}
	if r.URL.Path == "/api/v1/query" || r.URL.Path == "/api/v1/query_range" {
		writeMetricsSuccessResponse(w, respJson)
	} else {
		w.WriteHeader(http.StatusNotFound)
	}
}

func createMetricsResponse() map[string]interface{} {
	return map[string]interface{}{
		"status": "success",
		"data": map[string]interface{}{
			"resultType": "matrix",
			"result": []map[string]interface{}{
				{
					"metric": createTestMetrics(),
					"values": [][]interface{}{
						{float64(1726574340), "150515"},
						{float64(1726574370), "150517"},
						{float64(1726574400), "150517"},
					},
				},
			},
		},
	}
}

func writeMetricsSuccessResponse(w http.ResponseWriter, respJson []byte) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, err := w.Write(respJson)
	if err != nil {
		zlog.FormatError("write response failed: %v", err)
	}
}

func TestDashboardClientGetInstanceValueByInstances(t *testing.T) {
	server := mockMetricsServer()
	defer server.Close()
	client := newTestDashboardClient()
	testCases := []struct {
		testName   string
		client     *dashboardClient
		query      *model.InstanceQuery
		wantResp   httputil.ResponseJson
		wantStatus int
	}{
		{
			testName: "通过Metrics获取Instance成功",
			client:   client,
			query: &model.InstanceQuery{
				DataSource: server.URL,
				Start:      "2024-09-17T11:59:00Z",
				End:        "2024-09-17T12:00:00Z",
				Step:       "30",
				Instances: []string{
					`{"__name__": "kubelet_http_requests_total","endpoint": "https-metrics","instance": "this-is-a-test-instance","job": "kubelet","long_running": "false","method": "GET","metrics_path": "/metrics","namespace": "kube-system","node": "test2-xiezuyue","path": "metrics","server_type": "readwrite","service": "kubelet"}`,
				},
			},
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "get instance value by instances success",
				Data: instancesSuccess,
			},
			wantStatus: http.StatusCreated,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.GetInstanceValueByInstances(tc.query)
			if status != tc.wantStatus {
				t.Errorf("Test %s failed, status code error", tc.testName)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("Test %s failed, resp data error", tc.testName)
			}
		})
	}
}

var instancesSuccess = []model.Instance{
	model.Instance{
		Metrics: createTestMetrics(),
		Values: []interface{}{
			[]interface{}{
				float64(1726574340),
				"150515",
			},
			[]interface{}{
				float64(1726574370),
				"150517",
			},
			[]interface{}{
				float64(1726574400),
				"150517",
			},
		},
	},
}
