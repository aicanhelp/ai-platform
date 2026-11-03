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
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/rest"
	k8stesting "k8s.io/client-go/testing"

	"monitoring-dashboard-service/pkg/config/global"
	"monitoring-dashboard-service/pkg/monitor/model"
)

const (
	extraDashboardID  = "2024-09-10-07-08-11-f3d405a5-710a-49de-9a27-63db9760e1f9"
	extraWidgetID     = "2024-09-10-12-36-59-ce2ad735-195b-459b-b429-8576949e1387"
	extraInstanceJSON = `{"__name__":"metric","instance":"172.17.0.1:9090"}`
)

// newDashboardObject 构建dashboard CR unstructured对象
func newDashboardObject(name string, totalWidgets int64) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       global.DashboardRepoKind,
			"apiVersion": global.CRDRepoGroup + "/" + global.CRDRepoVersion,
			"metadata": map[string]interface{}{
				"namespace": global.MonitorServiceDefaultNamespace,
				"name":      name,
			},
			"spec": map[string]interface{}{
				"dashboardID":    name,
				"displayName":    "看板-" + name,
				"creator":        "admin",
				"totalWidgets":   totalWidgets,
				"lastUpdateTime": lastUpdateTime,
			},
		},
	}
}

// newWidgetObject 构建widget CR unstructured对象，spec覆盖自定义字段
func newWidgetObject(name, dashboardID string, widgetType interface{},
	instances []interface{}) *unstructured.Unstructured {
	if instances == nil {
		instances = []interface{}{extraInstanceJSON}
	}
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       global.WidgetCRDRepoKind,
			"apiVersion": global.CRDRepoGroup + "/" + global.CRDRepoVersion,
			"metadata": map[string]interface{}{
				"namespace": global.MonitorServiceDefaultNamespace,
				"name":      name,
				"labels": map[string]interface{}{
					"dashboardID": dashboardID,
				},
			},
			"spec": map[string]interface{}{
				"widgetID":    name,
				"dashboardID": dashboardID,
				"displayName": "组件-" + name,
				"dataSource":  "http://prometheus-k8s.monitoring.svc.cluster.local:9090",
				"widgetType":  widgetType,
				"instances":   instances,
				"boardMin":    int64(0),
				"boardMax":    int64(100),
			},
		},
	}
}

// newExtraClient 构建可注入错误的测试client
func newExtraClient(objects ...runtime.Object) *dashboardClient {
	scheme := runtime.NewScheme()
	fakeClient := fake.NewSimpleDynamicClientWithCustomListKinds(
		scheme,
		map[schema.GroupVersionResource]string{
			{
				Group:    global.CRDRepoGroup,
				Version:  global.CRDRepoVersion,
				Resource: global.DashboardRepoResource,
			}: global.DashboardRepoListKind,
			{
				Group:    global.CRDRepoGroup,
				Version:  global.CRDRepoVersion,
				Resource: global.WidgetCRDRepoResource,
			}: global.WidgetCRDRepoListKind,
		},
		objects...,
	)
	return &dashboardClient{
		kubeConfig:    nil,
		dynamicClient: fakeClient,
		clientSet:     nil,
	}
}

// injectError 向fake client注入指定verb/resource的错误
func injectError(t *testing.T, c *dashboardClient, verb, resource string) {
	t.Helper()
	fakeClient, ok := c.dynamicClient.(*fake.FakeDynamicClient)
	if !ok {
		t.Fatalf("dynamicClient is not a fake client")
	}
	fakeClient.PrependReactor(verb, resource, func(action k8stesting.Action) (
		bool, runtime.Object, error) {
		return true, nil, errors.Errorf("injected %s %s error", verb, resource)
	})
}

// newExtraDashboardVO 构建创建看板请求VO指针
func newExtraDashboardVO(name string) *model.DashboardVO {
	return &model.DashboardVO{DisplayName: name, Creator: "admin"}
}

// newExtraUpdateDashboardVO 构建更新看板请求VO指针
func newExtraUpdateDashboardVO(dashboardID, displayName string) *model.DashboardVO {
	return &model.DashboardVO{DashboardID: dashboardID, DisplayName: displayName}
}

// newExtraQueryDashboardVO 构建查询看板请求VO指针
func newExtraQueryDashboardVO(dashboardID string) *model.DashboardVO {
	return &model.DashboardVO{
		DashboardID: dashboardID,
		Start:       "2024-09-17T11:59:00Z",
		End:         "2024-09-17T12:00:00Z",
		Step:        "30",
	}
}

func newExtraValidWidgetVO(dashboardID string) *model.WidgetVO {
	return &model.WidgetVO{
		DashboardID: dashboardID,
		DisplayName: "组件创建",
		DataSource:  "http://prometheus-k8s.monitoring.svc.cluster.local:9090",
		WidgetType:  "0",
		Instances:   []string{extraInstanceJSON},
	}
}

// TestNewMonitorOperation 创建service操作对象成功与失败
func TestNewMonitorOperation(t *testing.T) {
	op, err := NewMonitorOperation(&rest.Config{Host: "http://127.0.0.1:6443"})
	if err != nil {
		t.Fatalf("NewMonitorOperation failed: %v", err)
	}
	if op == nil {
		t.Fatal("expected non-nil operation")
	}

	badTLS := &rest.Config{
		Host:            "https://127.0.0.1:6443",
		TLSClientConfig: rest.TLSClientConfig{CAData: []byte("invalid-ca")},
	}
	if _, err := NewMonitorOperation(badTLS); err == nil {
		t.Error("expected error for invalid kube config, got nil")
	}
}

// TestCreateWidgetErrors CreateWidget失败分支
func TestCreateWidgetErrors(t *testing.T) {
	// dashboard不存在
	client := newExtraClient()
	resp, status := client.CreateWidget(newExtraValidWidgetVO("not-exist-dashboard"))
	if status != http.StatusInternalServerError || resp.Code != global.ServerCreateWidgetFailed {
		t.Errorf("expected 500/%d, got %d/%d", global.ServerCreateWidgetFailed, status, resp.Code)
	}

	// dashboard存在但instances为空(NewWidgetCRDByCreateVO失败)
	client = newExtraClient(newDashboardObject(extraDashboardID, 0))
	vo := newExtraValidWidgetVO(extraDashboardID)
	vo.Instances = nil
	resp, status = client.CreateWidget(vo)
	if status != http.StatusInternalServerError || resp.Code != global.ServerCreateWidgetFailed {
		t.Errorf("expected 500/%d, got %d/%d", global.ServerCreateWidgetFailed, status, resp.Code)
	}

	// 更新dashboard计数失败
	client = newExtraClient(newDashboardObject(extraDashboardID, 0))
	injectError(t, client, "patch", global.DashboardRepoResource)
	resp, status = client.CreateWidget(newExtraValidWidgetVO(extraDashboardID))
	if status != http.StatusInternalServerError || resp.Code != global.ServerCreateWidgetFailed {
		t.Errorf("expected 500/%d, got %d/%d", global.ServerCreateWidgetFailed, status, resp.Code)
	}
}

// TestUpdateWidgetErrors UpdateWidget失败分支
func TestUpdateWidgetErrors(t *testing.T) {
	// widget存在但dashboard不存在
	client := newExtraClient(newWidgetObject(extraWidgetID, "missing-dashboard", int64(0), nil))
	vo := newExtraValidWidgetVO("")
	vo.WidgetID = extraWidgetID
	resp, status := client.UpdateWidget(vo)
	if status != http.StatusInternalServerError || resp.Code != global.ServerUpdateWidgetFailed {
		t.Errorf("expected 500/%d, got %d/%d", global.ServerUpdateWidgetFailed, status, resp.Code)
	}

	// 非法widgetType(NewPatchUpdateVO失败)
	client = newExtraClient(
		newDashboardObject(extraDashboardID, 1),
		newWidgetObject(extraWidgetID, extraDashboardID, int64(0), nil),
	)
	vo = newExtraValidWidgetVO(extraDashboardID)
	vo.WidgetID = extraWidgetID
	vo.WidgetType = "6"
	resp, status = client.UpdateWidget(vo)
	if status != http.StatusInternalServerError || resp.Code != global.ServerUpdateWidgetFailed {
		t.Errorf("expected 500/%d, got %d/%d", global.ServerUpdateWidgetFailed, status, resp.Code)
	}

	// patch widget失败
	client = newExtraClient(
		newDashboardObject(extraDashboardID, 1),
		newWidgetObject(extraWidgetID, extraDashboardID, int64(0), nil),
	)
	vo = newExtraValidWidgetVO(extraDashboardID)
	vo.WidgetID = extraWidgetID
	injectError(t, client, "patch", global.WidgetCRDRepoResource)
	resp, status = client.UpdateWidget(vo)
	if status != http.StatusInternalServerError || resp.Code != global.ServerUpdateWidgetFailed {
		t.Errorf("expected 500/%d, got %d/%d", global.ServerUpdateWidgetFailed, status, resp.Code)
	}
}

// TestDeleteWidgetErrors DeleteWidget失败与边界分支
func TestDeleteWidgetErrors(t *testing.T) {
	// widget存在但dashboard不存在
	client := newExtraClient(newWidgetObject(extraWidgetID, "missing-dashboard", int64(0), nil))
	resp, status := client.DeleteWidget(extraWidgetID)
	if status != http.StatusInternalServerError || resp.Code != global.ServerDeleteWidgetFailed {
		t.Errorf("expected 500/%d, got %d/%d", global.ServerDeleteWidgetFailed, status, resp.Code)
	}

	// 删除widget失败
	client = newExtraClient(
		newDashboardObject(extraDashboardID, 1),
		newWidgetObject(extraWidgetID, extraDashboardID, int64(0), nil),
	)
	injectError(t, client, "delete", global.WidgetCRDRepoResource)
	resp, status = client.DeleteWidget(extraWidgetID)
	if status != http.StatusInternalServerError || resp.Code != global.ServerDeleteWidgetFailed {
		t.Errorf("expected 500/%d, got %d/%d", global.ServerDeleteWidgetFailed, status, resp.Code)
	}

	// dashboard计数为0时删除widget(totalWidgets减为负的边界分支)
	client = newExtraClient(
		newDashboardObject(extraDashboardID, 0),
		newWidgetObject(extraWidgetID, extraDashboardID, int64(0), nil),
	)
	resp, status = client.DeleteWidget(extraWidgetID)
	if status != http.StatusOK {
		t.Errorf("expected 200 for zero-counter delete, got %d", status)
	}
	if resp.Code != global.Success {
		t.Errorf("expected success code, got %d", resp.Code)
	}
}

// newJSONServer 构建返回固定JSON的mock server
func newJSONServer(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
}

func newInstanceQuery(dataSource string, instances ...string) *model.InstanceQuery {
	return &model.InstanceQuery{
		DataSource: dataSource,
		Start:      "2024-09-17T11:59:00Z",
		End:        "2024-09-17T12:00:00Z",
		Step:       "30",
		Instances:  instances,
	}
}

// TestGetInstanceMetricByDisplayNameBranches 各响应体分支
func TestGetInstanceMetricByDisplayNameBranches(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantStatus int
		wantNil    bool
	}{
		{"响应无data", `{"status":"success"}`, http.StatusOK, true},
		{"data无result", `{"data":{}}`, http.StatusOK, true},
		{"result无metric", `{"data":{"result":[{"other":1}]}}`, http.StatusOK, false},
		{"正常响应", `{"data":{"result":[{"metric":{"__name__":"m"}}]}}`, http.StatusOK, false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			server := newJSONServer(tc.body)
			defer server.Close()
			client := newExtraClient()
			resp, status := client.GetInstanceMetricByDisplayName(server.URL, "metric_name")
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if tc.wantNil {
				// data可能为untyped nil或类型化nil指针(*model.Indicator)(nil)
				if resp.Data != nil {
					if indicator, ok := resp.Data.(*model.Indicator); !ok || indicator != nil {
						t.Errorf("expected nil indicator data, got %v", resp.Data)
					}
				}
			}
			if !tc.wantNil && resp.Data == nil {
				t.Error("expected non-nil data")
			}
		})
	}

	// 数据源不可达
	client := newExtraClient()
	resp, status := client.GetInstanceMetricByDisplayName("http://127.0.0.1:1", "metric_name")
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerGetInstancesMetricFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerGetInstancesMetricFailed, status, resp.Code)
	}
}

// TestGetInstanceValueByInstancesBranches 实例值查询各分支
func TestGetInstanceValueByInstancesBranches(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantStatus int
	}{
		{"响应无data", `{}`, http.StatusCreated},
		{"data无result", `{"data":{}}`, http.StatusCreated},
		{"result无values", `{"data":{"result":[{"metric":{}}]}}`, http.StatusCreated},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			server := newJSONServer(tc.body)
			defer server.Close()
			client := newExtraClient()
			resp, status := client.GetInstanceValueByInstances(
				newInstanceQuery(server.URL, extraInstanceJSON))
			if status != tc.wantStatus {
				t.Errorf("expected status %d, got %d", tc.wantStatus, status)
			}
			if tc.wantStatus == http.StatusCreated && resp.Data == nil {
				t.Error("expected non-nil instances data")
			}
		})
	}

	// 非法instance JSON
	client := newExtraClient()
	resp, status := client.GetInstanceValueByInstances(
		newInstanceQuery("http://127.0.0.1:1", "not-json"))
	if status != http.StatusInternalServerError || resp.Code != global.ServerGetInstancesFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerGetInstancesFailed, status, resp.Code)
	}

	// 空instances列表
	server := newJSONServer(`{}`)
	defer server.Close()
	client = newExtraClient()
	resp, status = client.GetInstanceValueByInstances(newInstanceQuery(server.URL))
	if status != http.StatusCreated {
		t.Errorf("expected 201 for empty instances, got %d", status)
	}

	// 时序值时间戳越界
	server2 := newJSONServer(
		`{"data":{"result":[{"values":[[9999999999.0,"1"]]}]}}`)
	defer server2.Close()
	resp, status = client.GetInstanceValueByInstances(
		newInstanceQuery(server2.URL, extraInstanceJSON))
	if status != http.StatusInternalServerError {
		t.Errorf("expected 500 for out-of-range timestamp, got %d", status)
	}

	// 时序值时间戳类型非法
	server3 := newJSONServer(
		`{"data":{"result":[{"values":[["not-float","1"]]}]}}`)
	defer server3.Close()
	resp, status = client.GetInstanceValueByInstances(
		newInstanceQuery(server3.URL, extraInstanceJSON))
	if status != http.StatusInternalServerError {
		t.Errorf("expected 500 for non-float timestamp, got %d", status)
	}
}

// TestStandardizeInstanceValue 时序标准化函数分支
func TestStandardizeInstanceValue(t *testing.T) {
	client := newExtraClient()
	values := []interface{}{
		[]interface{}{float64(1726574340), "1"},
		[]interface{}{float64(1726574370), "2"},
	}

	result, err := client.standardizeInstanceValue(values,
		"2024-09-17T11:59:00Z", "2024-09-17T12:00:00Z", "30")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if len(result) != 3 {
		t.Errorf("expected 3 standardized values, got %d", len(result))
	}

	if _, err := client.standardizeInstanceValue(values,
		"bad-start", "2024-09-17T12:00:00Z", "30"); err == nil {
		t.Error("expected error for invalid start time")
	}
	if _, err := client.standardizeInstanceValue(values,
		"2024-09-17T11:59:00Z", "bad-end", "30"); err == nil {
		t.Error("expected error for invalid end time")
	}
	if _, err := client.standardizeInstanceValue(values,
		"2024-09-17T11:59:00Z", "2024-09-17T12:00:00Z", "abc"); err == nil {
		t.Error("expected error for invalid step")
	}
}

// TestGetWidgetListByDashboardIDBranches widget列表装配失败分支
func TestGetWidgetListByDashboardIDBranches(t *testing.T) {
	vo := &model.DashboardVO{
		DashboardID: extraDashboardID,
		Start:       "2024-09-17T11:59:00Z",
		End:         "2024-09-17T12:00:00Z",
		Step:        "30",
	}

	// widgetType无法转换为int
	client := newExtraClient(
		newWidgetObject(extraWidgetID, extraDashboardID, "not-int", nil))
	if _, err := client.getWidgetListByDashboardID(vo); err == nil {
		t.Error("expected error for bad widgetType in CR")
	}

	// widgetType未知
	client = newExtraClient(
		newWidgetObject(extraWidgetID, extraDashboardID, int64(99), nil))
	if _, err := client.getWidgetListByDashboardID(vo); err == nil {
		t.Error("expected error for unknown widgetType")
	}

	// instances JSON非法
	client = newExtraClient(
		newWidgetObject(extraWidgetID, extraDashboardID, int64(0),
			[]interface{}{"not-json"}))
	if _, err := client.getWidgetListByDashboardID(vo); err == nil {
		t.Error("expected error for invalid instance JSON")
	}

	// 数据源不可达
	badDSWidget := newWidgetObject(extraWidgetID, extraDashboardID, int64(0), nil)
	setUnstructuredSpec(badDSWidget, "dataSource", "http://127.0.0.1:1")
	client = newExtraClient(badDSWidget)
	if _, err := client.getWidgetListByDashboardID(vo); err == nil {
		t.Error("expected error for unreachable data source")
	}

	// 数据源可用但start非法
	server := newJSONServer(
		`{"data":{"result":[{"values":[[1726574340.0,"1"]]}]}}`)
	defer server.Close()
	goodDSWidget := newWidgetObject(extraWidgetID, extraDashboardID, int64(0), nil)
	setUnstructuredSpec(goodDSWidget, "dataSource", server.URL)
	client = newExtraClient(goodDSWidget)
	badTimeVO := &model.DashboardVO{
		DashboardID: extraDashboardID,
		Start:       "bad-start",
		End:         "2024-09-17T12:00:00Z",
		Step:        "30",
	}
	if _, err := client.getWidgetListByDashboardID(badTimeVO); err == nil {
		t.Error("expected error for invalid query time")
	}
}

// setUnstructuredSpec 修改unstructured对象spec字段
func setUnstructuredSpec(obj *unstructured.Unstructured, key string, value interface{}) {
	if spec, ok := obj.Object["spec"].(map[string]interface{}); ok {
		spec[key] = value
	}
}

// TestDashboardServiceErrorBranches dashboard服务错误分支
func TestDashboardServiceErrorBranches(t *testing.T) {
	// CreateDashboard: 创建CR失败
	client := newExtraClient(newDashboardObject(extraDashboardID, 0))
	injectError(t, client, "create", global.DashboardRepoResource)
	resp, status := client.CreateDashboard(newExtraDashboardVO("看板创建失败"))
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerCreateDashboardFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerCreateDashboardFailed, status, resp.Code)
	}

	// GetDashboardList: list失败
	client = newExtraClient()
	injectError(t, client, "list", global.DashboardRepoResource)
	resp, status = client.GetDashboardList()
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerGetDashboardListFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerGetDashboardListFailed, status, resp.Code)
	}

	// GetDashboardList: CR转换失败
	badDashboard := newDashboardObject("bad-type-dashboard", 0)
	setUnstructuredSpec(badDashboard, "totalWidgets", "not-int")
	client = newExtraClient(badDashboard)
	resp, status = client.GetDashboardList()
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerGetDashboardListFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerGetDashboardListFailed, status, resp.Code)
	}

	// UpdateDashboard: patch失败
	client = newExtraClient(newDashboardObject(extraDashboardID, 0))
	injectError(t, client, "patch", global.DashboardRepoResource)
	resp, status = client.UpdateDashboard(
		newExtraUpdateDashboardVO(extraDashboardID, "看板更新失败"))
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerUpdateDashboardFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerUpdateDashboardFailed, status, resp.Code)
	}

	// UpdateDashboard: CR转换失败
	badGetDashboard := newDashboardObject("bad-get-dashboard", 0)
	setUnstructuredSpec(badGetDashboard, "totalWidgets", "not-int")
	client = newExtraClient(badGetDashboard)
	resp, status = client.UpdateDashboard(
		newExtraUpdateDashboardVO("bad-get-dashboard", "看板更新失败"))
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerGetDashboardFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerGetDashboardFailed, status, resp.Code)
	}

	// GetDashboardByDashboardID: widget list失败
	client = newExtraClient(newDashboardObject(extraDashboardID, 0))
	injectError(t, client, "list", global.WidgetCRDRepoResource)
	resp, status = client.GetDashboardByDashboardID(
		newExtraQueryDashboardVO(extraDashboardID))
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerGetDashboardFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerGetDashboardFailed, status, resp.Code)
	}

	// DeleteDashboard: widget list失败
	client = newExtraClient(newDashboardObject(extraDashboardID, 0))
	injectError(t, client, "list", global.WidgetCRDRepoResource)
	resp, status = client.DeleteDashboard(extraDashboardID)
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerDeleteDashboardFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerDeleteDashboardFailed, status, resp.Code)
	}

	// DeleteDashboard: widget删除失败
	client = newExtraClient(
		newDashboardObject(extraDashboardID, 1),
		newWidgetObject(extraWidgetID, extraDashboardID, int64(0), nil))
	injectError(t, client, "delete", global.WidgetCRDRepoResource)
	resp, status = client.DeleteDashboard(extraDashboardID)
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerDeleteDashboardFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerDeleteDashboardFailed, status, resp.Code)
	}

	// DeleteDashboard: dashboard删除失败
	client = newExtraClient(newDashboardObject(extraDashboardID, 0))
	injectError(t, client, "delete", global.DashboardRepoResource)
	resp, status = client.DeleteDashboard(extraDashboardID)
	if status != http.StatusInternalServerError ||
		resp.Code != global.ServerDeleteDashboardFailed {
		t.Errorf("expected 500/%d, got %d/%d",
			global.ServerDeleteDashboardFailed, status, resp.Code)
	}

	// DeleteDashboard: 级联删除成功
	client = newExtraClient(
		newDashboardObject(extraDashboardID, 1),
		newWidgetObject(extraWidgetID, extraDashboardID, int64(0), nil))
	resp, status = client.DeleteDashboard(extraDashboardID)
	if status != http.StatusOK || resp.Code != global.Success {
		t.Errorf("expected 200/%d, got %d/%d", global.Success, status, resp.Code)
	}
}

// TestGetWidgetCRListByDashboardIDBranches widget CR列表错误分支
func TestGetWidgetCRListByDashboardIDBranches(t *testing.T) {
	// list失败
	client := newExtraClient()
	injectError(t, client, "list", global.WidgetCRDRepoResource)
	if _, err := client.getWidgetCRListByDashboardID(extraDashboardID); err == nil {
		t.Error("expected error for injected list failure")
	}

	// CR转换失败
	client = newExtraClient(
		newWidgetObject(extraWidgetID, extraDashboardID, "not-int", nil))
	if _, err := client.getWidgetCRListByDashboardID(extraDashboardID); err == nil {
		t.Error("expected error for bad widget CR conversion")
	}
}

// TestAssembleWidgetIndicatorMetricsError instances非法JSON装配失败
func TestAssembleWidgetIndicatorMetricsError(t *testing.T) {
	client := newExtraClient()
	widget := &model.Widget{}
	widgetCR := &model.WidgetCRD{
		Spec: model.WidgetSpec{
			Instances: []string{"not-json"},
		},
	}
	if err := client.assembleWidgetIndicatorMetrics(widget, widgetCR); err == nil {
		t.Error("expected error for invalid instance JSON")
	}
}

// TestGetWidgetCRByWidgetIDConvertError widget CR转换失败
func TestGetWidgetCRByWidgetIDConvertError(t *testing.T) {
	client := newExtraClient(
		newWidgetObject(extraWidgetID, extraDashboardID, "not-int", nil))
	if _, err := client.getWidgetCRByWidgetID(extraWidgetID); err == nil {
		t.Error("expected error for bad widget CR conversion")
	}
}

// TestCreateDashboardCRPatchError 创建dashboard后更新注解等路径回归
func TestCreateDashboardCRPatchError(t *testing.T) {
	client := newExtraClient()
	if _, err := client.createDashboardCR(newExtraDashboardVO("看板CR创建")); err != nil {
		t.Fatalf("createDashboardCR failed: %v", err)
	}
}

// TestDeleteDashboardCRGetWidgetError 删除dashboard时查询widget列表失败
func TestDeleteDashboardCRGetWidgetError(t *testing.T) {
	client := newExtraClient()
	injectError(t, client, "list", global.WidgetCRDRepoResource)
	if err := client.deleteDashboardCR(extraDashboardID); err == nil {
		t.Error("expected error for injected widget list failure")
	}
}
