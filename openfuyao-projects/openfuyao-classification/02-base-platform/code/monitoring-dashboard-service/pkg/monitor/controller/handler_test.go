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

package controller

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/config/runtime"
	"monitoring-dashboard-service/pkg/monitor/model"
	"monitoring-dashboard-service/pkg/utils/httputil"
)

const (
	testWidgetID    = "2024-09-10-12-36-59-ce2ad735-195b-459b-b429-8576949e1385"
	testDashboardID = "2024-09-10-07-08-11-f3d405a5-710a-49de-9a27-63db9760e1f2"
	testDataSource  = "http://prometheus-k8s.monitoring.svc.cluster.local:9090"
	testStart       = "2024-09-17T11:59:00Z"
	testEnd         = "2024-09-17T12:00:00Z"
	testStep        = "30"
)

// fakeOperation 模拟service.Operation，记录调用并返回预设响应
type fakeOperation struct {
	resp   *httputil.ResponseJson
	status int

	createWidgetCalled int
	createWidgetVO     *model.WidgetVO
	updateWidgetCalled int
	updateWidgetVO     *model.WidgetVO
	deleteWidgetID     string
	getIndicatorDS     string
	getMetricDS        string
	getMetricName      string
	instanceQuery      *model.InstanceQuery
	createDashboardVO  *model.DashboardVO
	getDashboardVO     *model.DashboardVO
	updateDashboardVO  *model.DashboardVO
	deleteDashboardID  string
}

func newDefaultFakeOperation() *fakeOperation {
	return &fakeOperation{
		resp:   &httputil.ResponseJson{Code: global.Success, Msg: "fake success"},
		status: http.StatusOK,
	}
}

func (f *fakeOperation) CreateWidget(widgetVO *model.WidgetVO) (*httputil.ResponseJson, int) {
	f.createWidgetCalled++
	f.createWidgetVO = widgetVO
	return f.resp, f.status
}

func (f *fakeOperation) UpdateWidget(widgetVO *model.WidgetVO) (*httputil.ResponseJson, int) {
	f.updateWidgetCalled++
	f.updateWidgetVO = widgetVO
	return f.resp, f.status
}

func (f *fakeOperation) DeleteWidget(widgetID string) (*httputil.ResponseJson, int) {
	f.deleteWidgetID = widgetID
	return f.resp, f.status
}

func (f *fakeOperation) GetIndicatorNameList(dataSource string) (*httputil.ResponseJson, int) {
	f.getIndicatorDS = dataSource
	return f.resp, f.status
}

func (f *fakeOperation) GetInstanceMetricByDisplayName(dataSource, displayName string) (
	*httputil.ResponseJson, int) {
	f.getMetricDS = dataSource
	f.getMetricName = displayName
	return f.resp, f.status
}

func (f *fakeOperation) GetInstanceValueByInstances(query *model.InstanceQuery) (
	*httputil.ResponseJson, int) {
	f.instanceQuery = query
	return f.resp, f.status
}

func (f *fakeOperation) CreateDashboard(dashboardVO *model.DashboardVO) (*httputil.ResponseJson, int) {
	f.createDashboardVO = dashboardVO
	return f.resp, f.status
}

func (f *fakeOperation) GetDashboardByDashboardID(dashboardVO *model.DashboardVO) (
	*httputil.ResponseJson, int) {
	f.getDashboardVO = dashboardVO
	return f.resp, f.status
}

func (f *fakeOperation) GetDashboardList() (*httputil.ResponseJson, int) {
	return f.resp, f.status
}

func (f *fakeOperation) UpdateDashboard(dashboardVO *model.DashboardVO) (*httputil.ResponseJson, int) {
	f.updateDashboardVO = dashboardVO
	return f.resp, f.status
}

func (f *fakeOperation) DeleteDashboard(dashboardID string) (*httputil.ResponseJson, int) {
	f.deleteDashboardID = dashboardID
	return f.resp, f.status
}

// newTestContainer 构建挂载了全部monitor路由的测试容器
func newTestContainer(op *fakeOperation) *restful.Container {
	handler := newHandler(op)
	ws := restful.WebService{}
	ws.Path("/test").Produces(restful.MIME_JSON)
	bindWidgetRoutes(&ws, handler)
	bindDashboardRoutes(&ws, handler)
	container := restful.NewContainer()
	container.Router(restful.CurlyRouter{})
	container.Add(&ws)
	return container
}

type httpRequest struct {
	method string
	path   string
	query  string
	body   string
}

// doRequest 发送请求并返回状态码、业务码与业务消息
func doRequest(t *testing.T, server *httptest.Server, req httpRequest) (int, int32, string) {
	t.Helper()
	var body io.Reader
	if req.body != "" {
		body = bytes.NewBufferString(req.body)
	}
	target := server.URL + req.path
	if req.query != "" {
		target += "?" + req.query
	}
	httpReq, err := http.NewRequest(req.method, target, body)
	if err != nil {
		t.Fatalf("build request failed: %v", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		t.Fatalf("do request failed: %v", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read response failed: %v", err)
	}
	var respJson httputil.ResponseJson
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &respJson); err != nil {
			t.Fatalf("unmarshal response failed: %v, body: %s", err, raw)
		}
	}
	return resp.StatusCode, respJson.Code, respJson.Msg
}

const validWidgetBody = `{
	"dashboardID": "` + testDashboardID + `",
	"displayName": "测试组件",
	"dataSource": "` + testDataSource + `",
	"widgetType": "3",
	"instances": ["{\"__name__\":\"metric\"}"],
	"boardMin": "10",
	"boardMax": "100"
}`

// TestPostCreateWidget 创建widget路由
func TestPostCreateWidget(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantStatus int
		wantCode   int32
		wantCalled bool
	}{
		{"创建成功", validWidgetBody, http.StatusOK, global.Success, true},
		{"非法请求体", `{invalid-json`, http.StatusBadRequest, global.ClientInvalidRequestBody, false},
		{
			"非法参数:dashboardID为空",
			`{"displayName":"x","dataSource":"` + testDataSource + `","widgetType":"3"}`,
			http.StatusBadRequest, global.ClientInvalidParameters, false,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodPost, path: "/test/widgets", body: tc.body,
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
			if op.createWidgetCalled > 0 != tc.wantCalled {
				t.Errorf("service called=%d, want called=%v", op.createWidgetCalled, tc.wantCalled)
			}
		})
	}
}

// TestPatchUpdateWidget 更新widget路由
func TestPatchUpdateWidget(t *testing.T) {
	cases := []struct {
		name       string
		widgetID   string
		body       string
		wantStatus int
		wantCode   int32
		wantVOID   string
	}{
		{"更新成功", testWidgetID, validWidgetBody, http.StatusOK, global.Success, testWidgetID},
		{"非法请求体", testWidgetID, `{invalid`, http.StatusBadRequest,
			global.ClientInvalidRequestBody, ""},
		{
			"非法参数:类型为空", testWidgetID,
			`{"displayName":"x","dataSource":"` + testDataSource + `"}`,
			http.StatusBadRequest, global.ClientInvalidParameters, "",
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodPatch,
				path:   fmt.Sprintf("/test/widgets/%s", tc.widgetID),
				body:   tc.body,
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
			if tc.wantVOID != "" && op.updateWidgetVO.WidgetID != tc.wantVOID {
				t.Errorf("expected widgetID %s, got %s", tc.wantVOID, op.updateWidgetVO.WidgetID)
			}
		})
	}
}

// TestDeleteWidget 删除widget路由
func TestDeleteWidget(t *testing.T) {
	cases := []struct {
		name       string
		widgetID   string
		wantStatus int
		wantCode   int32
	}{
		{"删除成功", testWidgetID, http.StatusOK, global.Success},
		{"非法ID", "short-id", http.StatusBadRequest, global.ClientInvalidParameters},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodDelete, path: fmt.Sprintf("/test/widgets/%s", tc.widgetID),
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
		})
	}
}

// TestGetIndicatorList 指标名列表路由
func TestGetIndicatorList(t *testing.T) {
	cases := []struct {
		name       string
		query      string
		wantStatus int
		wantCode   int32
		wantDS     string
	}{
		{"成功", "dataSource=" + testDataSource, http.StatusOK, global.Success, testDataSource},
		{"非法dataSource", "dataSource=http%3A%2F%2Flocalhost%3A9090", http.StatusBadRequest,
			global.ClientInvalidParameters, ""},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodGet, path: "/test/widgets/indicators", query: tc.query,
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
			if tc.wantDS != "" && op.getIndicatorDS != tc.wantDS {
				t.Errorf("expected dataSource %s, got %s", tc.wantDS, op.getIndicatorDS)
			}
		})
	}
}

// TestGetInstanceMetricList 按名称获取指标实例路由
func TestGetInstanceMetricList(t *testing.T) {
	cases := []struct {
		name        string
		displayName string
		query       string
		wantStatus  int
		wantCode    int32
	}{
		{"成功", "kubelet_metric", "dataSource=" + testDataSource, http.StatusOK, global.Success},
		{"非法dataSource", "kubelet_metric", "dataSource=http%3A%2F%2Flocalhost%3A9090",
			http.StatusBadRequest, global.ClientInvalidParameters},
		{"超长displayName", strings.Repeat("a", 128), "dataSource=" + testDataSource,
			http.StatusBadRequest, global.ClientInvalidParameters},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodGet,
				path:   fmt.Sprintf("/test/widgets/indicators/%s", tc.displayName),
				query:  tc.query,
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
		})
	}
}

// TestGetInstanceValueList 按实例查询时序值路由
func TestGetInstanceValueList(t *testing.T) {
	validQuery := fmt.Sprintf("dataSource=%s&start=%s&end=%s&step=%s",
		testDataSource, testStart, testEnd, testStep)
	cases := []struct {
		name       string
		query      string
		body       string
		wantStatus int
		wantCode   int32
	}{
		{"成功", validQuery, `["{\"__name__\":\"metric\"}"]`, http.StatusOK, global.Success},
		{"非法请求体", validQuery, `{invalid`, http.StatusBadRequest, global.ClientInvalidRequestBody},
		{"非法参数:step错误", "dataSource=" + testDataSource + "&start=" + testStart +
			"&end=" + testEnd + "&step=abc", `[]`, http.StatusBadRequest,
			global.ClientInvalidParameters},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodPost, path: "/test/widgets/indicators/instances",
				query:  tc.query, body: tc.body,
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
		})
	}
}

// TestPostCreateDashboard 创建dashboard路由
func TestPostCreateDashboard(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantStatus int
		wantCode   int32
	}{
		{"创建成功", `{"displayName":"看板","creator":"admin"}`, http.StatusOK, global.Success},
		{"非法请求体", `{invalid`, http.StatusBadRequest, global.ClientInvalidRequestBody},
		{"非法参数:名称为空", `{"displayName":"","creator":"admin"}`, http.StatusBadRequest,
			global.ClientInvalidParameters},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodPost, path: "/test/dashboards", body: tc.body,
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
		})
	}
}

// TestGetDashboardByDashboardID 按ID获取dashboard路由
func TestGetDashboardByDashboardID(t *testing.T) {
	validQuery := fmt.Sprintf("start=%s&end=%s&step=%s", testStart, testEnd, testStep)
	cases := []struct {
		name       string
		dashboard  string
		query      string
		wantStatus int
		wantCode   int32
	}{
		{"成功", testDashboardID, validQuery, http.StatusOK, global.Success},
		{"非法ID", "short-id", validQuery, http.StatusBadRequest, global.ClientInvalidParameters},
		{"非法step", testDashboardID,
			fmt.Sprintf("start=%s&end=%s&step=abc", testStart, testEnd),
			http.StatusBadRequest, global.ClientInvalidParameters},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodGet,
				path:   fmt.Sprintf("/test/dashboards/%s", tc.dashboard),
				query:  tc.query,
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
		})
	}
}

// TestGetDashboardList dashboard列表路由
func TestGetDashboardList(t *testing.T) {
	op := newDefaultFakeOperation()
	server := httptest.NewServer(newTestContainer(op))
	defer server.Close()
	status, code, _ := doRequest(t, server, httpRequest{
		method: http.MethodGet, path: "/test/dashboards",
	})
	if status != http.StatusOK || code != global.Success {
		t.Errorf("expected 200/%d, got %d/%d", global.Success, status, code)
	}
}

// TestPatchUpdateDashboard 更新dashboard路由
func TestPatchUpdateDashboard(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantStatus int
		wantCode   int32
	}{
		{"更新成功", `{"displayName":"新名字"}`, http.StatusOK, global.Success},
		{"非法请求体", `{invalid`, http.StatusBadRequest, global.ClientInvalidRequestBody},
		{"非法参数:名称为空", `{"displayName":""}`, http.StatusBadRequest,
			global.ClientInvalidParameters},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodPatch,
				path:   fmt.Sprintf("/test/dashboards/%s", testDashboardID),
				body:   tc.body,
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
			if tc.wantStatus == http.StatusOK &&
				op.updateDashboardVO.DashboardID != testDashboardID {
				t.Errorf("expected dashboardID from path, got %s", op.updateDashboardVO.DashboardID)
			}
		})
	}
}

// TestDeleteDashboard 删除dashboard路由
func TestDeleteDashboard(t *testing.T) {
	cases := []struct {
		name       string
		dashboard  string
		wantStatus int
		wantCode   int32
	}{
		{"删除成功", testDashboardID, http.StatusOK, global.Success},
		{"非法ID", "short-id", http.StatusBadRequest, global.ClientInvalidParameters},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			op := newDefaultFakeOperation()
			server := httptest.NewServer(newTestContainer(op))
			defer server.Close()
			status, code, _ := doRequest(t, server, httpRequest{
				method: http.MethodDelete,
				path:   fmt.Sprintf("/test/dashboards/%s", tc.dashboard),
			})
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if code != tc.wantCode {
				t.Errorf("expected code %d, got %d", tc.wantCode, code)
			}
		})
	}
}

// TestBindMonitorRouteSuccess 绑定monitor路由成功
func TestBindMonitorRouteSuccess(t *testing.T) {
	ws := runtime.NewRestfulWebService(schema.GroupVersion{
		Group:   global.MonitorServiceDefaultHost,
		Version: global.MonitorServiceDefaultAPIVersion,
	})
	BindMonitorRoute(ws, &rest.Config{Host: "http://127.0.0.1:6443"})
	if len(ws.Routes()) == 0 {
		t.Error("expected routes to be registered")
	}
}

// TestBindMonitorRouteOperationError service创建失败时仍完成路由注册
func TestBindMonitorRouteOperationError(t *testing.T) {
	ws := runtime.NewRestfulWebService(schema.GroupVersion{
		Group:   global.MonitorServiceDefaultHost,
		Version: global.MonitorServiceDefaultAPIVersion,
	})
	badTLS := &rest.Config{
		Host: "https://127.0.0.1:6443",
		TLSClientConfig: rest.TLSClientConfig{
			CAData: []byte("invalid-ca-data"),
		},
	}
	BindMonitorRoute(ws, badTLS)
	if len(ws.Routes()) == 0 {
		t.Error("expected routes to be registered even if operation creation fails")
	}
}
