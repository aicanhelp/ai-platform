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
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/mitchellh/mapstructure"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic/fake"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/monitor/model"
	"monitoring-dashboard-service/pkg/utils/httputil"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

const (
	createDashboardID        = "2024-09-10-07-08-11-f3d405a5-710a-49de-9a27-63db9760e1f1"
	createWidgetID           = "2024-09-10-12-36-59-ce2ad735-195b-459b-b429-8576949e1385"
	createWidgetIDForTest    = "2024-09-10-12-36-59-ce2ad735-195b-459b-b429-8576949e1386"
	getWrongWidgetID         = "2024-09-10-12-36-59-ce2ad735-195b-459b-b429-8576949e1310"
	getDashboardID           = "2024-09-10-07-08-11-f3d405a5-710a-49de-9a27-63db9760e1f2"
	widgetDashboardID        = "2024-09-10-07-08-11-f3d405a5-710a-49de-9a27-63db9760e1f4"
	getWrongDashboardID      = "2024-09-10-07-08-11-f3d405a5-710a-49de-9a27-63db9760e1f3"
	lastUpdateTime           = "2024-09-10T12:42:58Z"
	validateWidgetDataSource = "http://prometheus-k8s.monitoring.svc.cluster.local:9090"
)

var totalWidgetsNumber int64 = 1

// 模拟K8S集群中已经存在的CR资源
var createDashboardUnstructured = unstructured.Unstructured{
	Object: map[string]interface{}{
		"kind":       "Dashboard",
		"apiVersion": "monitoring-dashboard.openfuyao.com/v1beta1",
		"metadata": map[string]interface{}{
			"namespace":         global.MonitorServiceDefaultNamespace,
			"name":              createDashboardID,
			"creationTimestamp": nil,
			"annotations": map[string]interface{}{
				"openfuyao.com/modificationTimestamp": lastUpdateTime,
			},
		},
		"spec": map[string]interface{}{
			"dashboardID":    createDashboardID,
			"displayName":    "测试监控看板名字重复",
			"creator":        "admin",
			"totalWidgets":   totalWidgetsNumber,
			"lastUpdateTime": lastUpdateTime,
		},
	},
}

var getDashboardUnstructured = unstructured.Unstructured{
	Object: map[string]interface{}{
		"kind":       "Dashboard",
		"apiVersion": "monitoring-dashboard.openfuyao.com/v1beta1",
		"metadata": map[string]interface{}{
			"namespace":         global.MonitorServiceDefaultNamespace,
			"name":              getDashboardID,
			"creationTimestamp": nil,
			"annotations": map[string]interface{}{
				"openfuyao.com/modificationTimestamp": lastUpdateTime,
			},
		},
		"spec": map[string]interface{}{
			"dashboardID":    getDashboardID,
			"displayName":    "测试监控看板获取",
			"creator":        "admin",
			"totalWidgets":   totalWidgetsNumber,
			"lastUpdateTime": lastUpdateTime,
		},
	},
}

var widgetDashboardUnstructured = unstructured.Unstructured{
	Object: map[string]interface{}{
		"kind":       "Dashboard",
		"apiVersion": "monitoring-dashboard.openfuyao.com/v1beta1",
		"metadata": map[string]interface{}{
			"namespace":         global.MonitorServiceDefaultNamespace,
			"name":              widgetDashboardID,
			"creationTimestamp": nil,
			"annotations": map[string]interface{}{
				"openfuyao.com/modificationTimestamp": lastUpdateTime,
			},
		},
		"spec": map[string]interface{}{
			"dashboardID":    widgetDashboardID,
			"displayName":    "测试监控看板测试组件",
			"creator":        "admin",
			"totalWidgets":   int64(0),
			"lastUpdateTime": lastUpdateTime,
		},
	},
}

var testWidgetUnstructured = unstructured.Unstructured{
	Object: map[string]interface{}{
		"kind":       "Widget",
		"apiVersion": "monitoring-dashboard.openfuyao.com/v1beta1",
		"metadata": map[string]interface{}{
			"namespace":         global.MonitorServiceDefaultNamespace,
			"name":              createWidgetID,
			"creationTimestamp": nil,
			"annotations": map[string]interface{}{
				"openfuyao.com/modificationTimestamp": lastUpdateTime,
			},
			"labels": map[string]interface{}{
				"dashboardID": getDashboardID,
			},
		},
		"spec": map[string]interface{}{
			"widgetID":    createWidgetID,
			"dashboardID": getDashboardID,
			"displayName": "测试监控组件",
			"dataSource":  "http://prometheus-k8s.monitoring.svc.cluster.local:9090",
			"widgetType":  int64(3),
			"instances": []interface{}{
				`{"__name__":"alertmanager_alerts","container":"alertmanager","endpoint":"web","instance":"this-is-test-ip","job":"alertmanager-main","namespace":"monitoring","pod":"alertmanager-main-2","service":"alertmanager-main","state":"active"}`,
			},
			"boardMin": int64(10),
			"boardMax": int64(100),
		},
	},
}

// newFakeDashboardClient 模拟K8S的client调用
func newFakeDashboardClient() *fake.FakeDynamicClient {
	scheme := runtime.NewScheme()
	return fake.NewSimpleDynamicClientWithCustomListKinds(
		scheme,
		map[schema.GroupVersionResource]string{
			{
				Group:    "monitoring-dashboard.openfuyao.com",
				Version:  "v1beta1",
				Resource: "dashboards",
			}: "DashboardList",
			{
				Group:    "monitoring-dashboard.openfuyao.com",
				Version:  "v1beta1",
				Resource: "widgets",
			}: "WidgetList",
		},
		&createDashboardUnstructured,
		&getDashboardUnstructured,
		&widgetDashboardUnstructured,
		&testWidgetUnstructured,
	)
}

// newTestDashboardClient 通过模拟的client创建测试用的dashboardClient
func newTestDashboardClient() *dashboardClient {
	return &dashboardClient{
		kubeConfig:    nil,
		dynamicClient: newFakeDashboardClient(),
		clientSet:     nil,
	}
}

// TestDashboardClientGetDashboardList API: 获取集群中所有Dashboard
func TestDashboardClientGetDashboardList(t *testing.T) {
	client := newTestDashboardClient()
	dashboards := []*model.DashboardSpec{
		getDashboardSpecFromUnstructured(&createDashboardUnstructured),
		getDashboardSpecFromUnstructured(&getDashboardUnstructured),
		getDashboardSpecFromUnstructured(&widgetDashboardUnstructured),
	}
	testCases := []struct {
		testName   string
		client     *dashboardClient
		wantResp   httputil.ResponseJson
		wantStatus int
	}{
		{
			testName: "获取Dashboard列表成功",
			client:   client,
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "successfully access to dashboard cr list",
				Data: dashboards,
			},
			wantStatus: http.StatusOK,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.GetDashboardList()
			if status != tc.wantStatus {
				t.Errorf("Test %s failed, status code error", tc.testName)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("Test %s failed, resp data error", tc.testName)
			}
		})
	}
}

// getDashboardSpecFromUnstructured API: 获取集群中所有Dashboard 将unstructured转为DashboardSpec
func getDashboardSpecFromUnstructured(data *unstructured.Unstructured) *model.DashboardSpec {
	spec, _, _ := unstructured.NestedMap(data.Object, "spec")
	var dashboardSpec *model.DashboardSpec
	if err := mapstructure.Decode(spec, &dashboardSpec); err != nil {
		fmt.Printf("Error decoding spec: %v\n", err)
		return nil
	}
	return dashboardSpec
}

// CreateDashboard API:创建dashboard 创建成功的预期结果
var dashboardCreateSuccess = model.DashboardSpec{
	DashboardID:    createDashboardID,
	DisplayName:    "测试监控看板创建",
	Creator:        "admin",
	TotalWidgets:   0,
	LastUpdateTime: lastUpdateTime,
}

// TestDashboardClientCreateDashboard API:创建dashboard
func TestDashboardClientCreateDashboard(t *testing.T) {
	client := newTestDashboardClient()
	cases := buildCreateDashboardCases(client)
	runCreateDashboardCases(t, cases)
}

type createDashboardCase struct {
	testName   string
	client     *dashboardClient
	vo         model.DashboardVO
	wantResp   httputil.ResponseJson
	wantStatus int
}

func buildCreateDashboardCases(client *dashboardClient) []createDashboardCase {
	return []createDashboardCase{
		{
			testName:   "Dashboard创建成功",
			client:     client,
			vo:         createDashboardVO("测试监控看板创建"),
			wantResp:   httputil.ResponseJson{Code: global.Success, Msg: "successfully created dashboard cr", Data: dashboardCreateSuccess},
			wantStatus: http.StatusCreated,
		},
		{
			testName:   "Dashboard创建失败:DisplayName重复",
			client:     client,
			vo:         createDashboardVO("测试监控看板名字重复"),
			wantResp:   httputil.ResponseJson{Code: global.ClientDuplicateNameExist, Msg: "display name already exists", Data: nil},
			wantStatus: http.StatusBadRequest,
		},
	}
}

func createDashboardVO(name string) model.DashboardVO {
	return model.DashboardVO{DisplayName: name, Creator: "admin"}
}

func runCreateDashboardCases(t *testing.T, cases []createDashboardCase) {
	for _, tc := range cases {
		tc := tc
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.CreateDashboard(&tc.vo)
			if tc.wantStatus == http.StatusCreated {
				dashboardCR := getDashboardCRFromUnstructured(t, resp.Data)
				dashboardCR.Spec.DashboardID = createDashboardID
				dashboardCR.Spec.LastUpdateTime = lastUpdateTime
				resp.Data = dashboardCR.Spec
			}
			if status != tc.wantStatus {
				t.Errorf("test %s: status code mismatch, want %d, got %d", tc.testName, tc.wantStatus, status)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("test %s: response data does not match expected", tc.testName)
			}
		})
	}
}

// getDashboardCRFromUnstructured API:创建dashboard 辅助函数 将Unstructured转为struct
func getDashboardCRFromUnstructured(t *testing.T, data any) *model.DashboardCRD {
	respUnstructured, ok := data.(*unstructured.Unstructured)
	if !ok {
		t.Errorf("Test failed, resp type error")
	}
	var dashboardCR model.DashboardCRD
	err := runtime.DefaultUnstructuredConverter.FromUnstructured(respUnstructured.Object, &dashboardCR)
	if err != nil {
		t.Errorf("Test failed")
	}
	return &dashboardCR
}

// TestDashboardClientGetDashboardByDashboardID API:根据ID获取完整Dashboard
func TestDashboardClientGetDashboardByDashboardID(t *testing.T) {
	server := mockRepoServer()
	defer server.Close()
	setTestWidgetUnstructuredURI(server.URL)
	client := newTestDashboardClient()

	cases := buildGetDashboardCases(client)
	runGetDashboardCases(t, cases)
}

type getDashboardCase struct {
	testName   string
	client     *dashboardClient
	vo         model.DashboardVO
	wantResp   httputil.ResponseJson
	wantStatus int
}

func buildGetDashboardCases(client *dashboardClient) []getDashboardCase {
	return []getDashboardCase{
		{
			testName:   "根据ID获取Dashboard成功",
			client:     client,
			vo:         newDashboardQueryVO(getDashboardID),
			wantResp:   httputil.ResponseJson{Code: global.Success, Msg: "successfully access to dashboard cr", Data: dashboardGetByIDSuccess},
			wantStatus: http.StatusOK,
		},
		{
			testName:   "根据ID获取Dashboard失败:ID不存在",
			client:     client,
			vo:         newDashboardQueryVO("getWrongDashboardID"),
			wantResp:   httputil.ResponseJson{Code: global.ServerGetDashboardFailed, Msg: "get dashboard cr failed", Data: nil},
			wantStatus: http.StatusInternalServerError,
		},
	}
}

func newDashboardQueryVO(id string) model.DashboardVO {
	return model.DashboardVO{
		DashboardID: id,
		Start:       "2024-09-17T11:59:00Z",
		End:         "2024-09-17T12:00:00Z",
		Step:        "30",
	}
}

func runGetDashboardCases(t *testing.T, cases []getDashboardCase) {
	for _, tc := range cases {
		tc := tc
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.GetDashboardByDashboardID(&tc.vo)
			if tc.wantStatus == http.StatusOK {
				dashboard, ok := resp.Data.(*model.Dashboard)
				if !ok {
					t.Errorf("Test failed, resp type error")
				}
				dashboard.LastUpdateTime = lastUpdateTime
				resp.Data = dashboard
			}
			if status != tc.wantStatus {
				t.Errorf("%s: unexpected status code, expected %d but got %d", tc.testName, tc.wantStatus, status)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("%s: response content differs from expected", tc.testName)
			}
		})
	}
}

const (
	tBoardMin = 10
	tBoardMax = 100
)

// dashboardGetByIDSuccess API:根据ID获取完整Dashboard 获取成功的预期结果
var dashboardGetByIDSuccess = &model.Dashboard{
	DashboardID:    getDashboardID,
	DisplayName:    "测试监控看板获取",
	Creator:        "admin",
	TotalWidgets:   1,
	LastUpdateTime: lastUpdateTime,
	Widgets: []model.Widget{
		{
			WidgetID:    createWidgetID,
			DashboardID: getDashboardID,
			DisplayName: "测试监控组件",
			DataSource:  "http://prometheus-k8s.monitoring.svc.cluster.local:9090",
			WidgetType:  model.DashboardGraphType,
			Instances: []model.Instance{
				{
					Metrics: map[string]interface{}{
						"__name__": "alertmanager_alerts", "container": "alertmanager", "endpoint": "web", "instance": "this-is-test-ip", "job": "alertmanager-main", "namespace": "monitoring", "pod": "alertmanager-main-2", "service": "alertmanager-main", "state": "active",
					},
					Values: []interface{}{},
				},
			},
			WidgetGraph: model.NewBoardGraph(model.DashboardGraphType, tBoardMin, tBoardMax),
		},
	},
}

// mockRepoServer API:根据ID获取Dashboard 模拟 prometheus server
func mockRepoServer() *httptest.Server {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]interface{}{
			"status": "success",
			"data": map[string]interface{}{
				"resultType": "matrix",
				"result":     []interface{}{},
			}}
		respJson, err := json.Marshal(resp)
		if err != nil {
			zlog.FormatError("marshal response failed: %v", err)
			return
		}
		if r.URL.Path == "/api/v1/query_range" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, err = w.Write(respJson)
			if err != nil {
				zlog.FormatError("write response failed: %v", err)
				return
			}
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	return server
}

// setTestWidgetUnstructuredURI API:根据ID获取Dashboard 在mock server启动后设置prometheus uri
func setTestWidgetUnstructuredURI(uri string) {
	if obj, ok := testWidgetUnstructured.Object["spec"].(map[string]interface{}); ok {
		obj["dataSource"] = uri
	}
	dashboardGetByIDSuccess.Widgets[0].DataSource = uri
}

// TestDashboardClientUpdateDashboard API:根据ID修改Dashboard
func TestDashboardClientUpdateDashboard(t *testing.T) {
	client := newTestDashboardClient()
	testCases := buildUpdateDashboardCases(client)
	runUpdateDashboardCases(t, testCases)
}

type updateDashboardCase struct {
	testName   string
	client     *dashboardClient
	vo         model.DashboardVO
	wantResp   httputil.ResponseJson
	wantStatus int
}

func buildUpdateDashboardCases(client *dashboardClient) []updateDashboardCase {
	return []updateDashboardCase{
		{
			testName: "Dashboard修改成功",
			client:   client,
			vo:       createUpdateDashboardVO(getDashboardID, "测试监控看板修改"),
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "successfully updated dashboard cr",
			},
			wantStatus: http.StatusOK,
		},
		{
			testName: "Dashboard修改失败:ID不存在",
			client:   client,
			vo:       createUpdateDashboardVO(getWrongDashboardID, "测试监控看板修改ID"),
			wantResp: httputil.ResponseJson{
				Code: global.ServerGetDashboardFailed,
				Msg:  "get dashboard cr by id failed",
				Data: nil,
			},
			wantStatus: http.StatusInternalServerError,
		},
		{
			testName: "Dashboard修改失败:displayName已存在",
			client:   client,
			vo:       createUpdateDashboardVO(getDashboardID, "测试监控看板修改"),
			wantResp: httputil.ResponseJson{
				Code: global.ClientDuplicateNameExist,
				Msg:  "display name already exists",
				Data: nil,
			},
			wantStatus: http.StatusBadRequest,
		},
	}
}

func createUpdateDashboardVO(dashboardID, displayName string) model.DashboardVO {
	return model.DashboardVO{
		DashboardID: dashboardID,
		DisplayName: displayName,
	}
}

func runUpdateDashboardCases(t *testing.T, testCases []updateDashboardCase) {
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.UpdateDashboard(&tc.vo)
			if status != tc.wantStatus {
				t.Errorf("Test %s failed, status code error", tc.testName)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("Test %s failed, resp data error", tc.testName)
			}
		})
	}
}

// TestDashboardClientDeleteDashboard API:根据ID删除Dashboard
func TestDashboardClientDeleteDashboard(t *testing.T) {
	client := newTestDashboardClient()
	testCases := []struct {
		testName    string
		client      *dashboardClient
		dashboardID string
		wantResp    httputil.ResponseJson
		wantStatus  int
	}{
		{
			testName:    "Dashboard删除成功",
			client:      client,
			dashboardID: createDashboardID,
			wantResp: httputil.ResponseJson{
				Code: global.Success,
				Msg:  "successful deleted dashboard",
			},
			wantStatus: http.StatusOK,
		}, {
			testName:    "Dashboard删除失败:ID不存在",
			client:      client,
			dashboardID: createDashboardID,
			wantResp: httputil.ResponseJson{
				Code: global.ServerDeleteDashboardFailed,
				Msg:  "delete dashboard failed",
			},
			wantStatus: http.StatusInternalServerError,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			resp, status := tc.client.DeleteDashboard(tc.dashboardID)
			if status != tc.wantStatus {
				t.Errorf("Test %s failed, status code error", tc.testName)
			}
			if !reflect.DeepEqual(resp, &tc.wantResp) {
				t.Errorf("Test %s failed, resp data error", tc.testName)
			}
		})
	}
}
