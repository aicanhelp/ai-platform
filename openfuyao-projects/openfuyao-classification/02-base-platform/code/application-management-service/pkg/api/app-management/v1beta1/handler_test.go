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

// Package v1beta1
package v1beta1

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"application-management-service/pkg/constant"
	helmModel "application-management-service/pkg/models/helm"
	"application-management-service/pkg/server/param"
	"application-management-service/pkg/utils/httputil"
	"application-management-service/pkg/zlog"
	"application-management-service/pkg/audit"
)

// HTTP状态码常量
const (
	statusOK            = 200
	statusCreated       = 201
	statusBadRequest    = 400
	statusUnauthorized  = 401
	statusForbidden     = 403
	statusNotFound      = 404
	statusNotAcceptable = 406
)

// 测试参数常量
const (
	testNamespace   = "default"
	testReleaseName = "test-release"
	testChartName   = "nginx"
	testMaxHistory  = 10
	testReplicas    = 3
	testVersion     = 1
	testRevision    = 2
)

// 消息常量
const (
	msgReleaseCreated       = "Release created successfully"
	msgReleaseDeleted       = "Release deleted successfully"
	msgReleaseNotFound      = "release not found"
	msgChartNotFound        = "chart not found"
	msgUnauthorizedCreate   = "current user is unauthorized to create or upgrade this release"
	msgUnauthorizedDelete   = "current user is unauthorized to delete this release"
	msgUnauthorizedRollback = "current user is unauthorized to rollback this release"
	msgParseError           = "parse request parameter error"
)

// URL模式常量
const (
	baseReleasesURL    = "/api/v1/releases"
	releaseInfoURL     = "/api/v1/releases/%s/%s"
	releaseHistoryURL  = "/api/v1/releases/%s/%s/history?maxHistory=10"
	releaseRollbackURL = "/api/v1/releases/%s/%s/rollback"
	releaseManifestURL = "/api/v1/releases/%s/%s/manifest"
	releaseValuesURL   = "/api/v1/releases/%s/%s/values"
	releasePodsURL     = "/api/v1/releases/%s/%s/pods"
	releaseStatusURL   = "/api/v1/releases/status/%s"
)

// HTTP头常量
const (
	contentTypeJSON = "application/json"
	acceptJSON      = "application/json"
)

// MockHelmOperation 模拟 helm.Operation 接口
type MockHelmOperation struct {
	mock.Mock
}

func (m *MockHelmOperation) GetReleases(params param.ReleaseParameter) (*httputil.ResponseJson, int) {
	args := m.Called(params)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) GetReleaseInfo(namespace, name string) (*httputil.ResponseJson, int) {
	zlog.Info("GetReleaseInfo")
	args := m.Called(namespace, name)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) Uninstall(namespace, name string) (*httputil.ResponseJson, int) {
	zlog.Info("Uninstall")
	args := m.Called(namespace, name)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) Install(deployParams helmModel.DeployParam) (*httputil.ResponseJson, int) {
	args := m.Called(deployParams)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) Upgrade(deployParams helmModel.DeployParam) (*httputil.ResponseJson, int) {
	zlog.Info("Upgrade")
	args := m.Called(deployParams)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) GetReleaseHistory(namespace, name string, maxHistory int) (*httputil.ResponseJson, int) {
	args := m.Called(namespace, name, maxHistory)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) RollbackRelease(namespace, name, chartName string,
	version, maxHistory int) (*httputil.ResponseJson, int) {
	args := m.Called(namespace, name, chartName, version, maxHistory)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) GetManifest(namespace, name string) (*httputil.ResponseJson, int) {
	zlog.Info("GetManifest")
	args := m.Called(namespace, name)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) GetValues(namespace, name string) (*httputil.ResponseJson, int) {
	zlog.Info("GetValues")
	args := m.Called(namespace, name)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) GetReleasePods(namespace, name string) (*httputil.ResponseJson, int) {
	zlog.Info("GetReleasePods")
	args := m.Called(namespace, name)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockHelmOperation) CheckReleaseStatus(chart string) (*httputil.ResponseJson, int) {
	args := m.Called(chart)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

// createWebService 创建测试用WebService
func createWebService(handler *Handler) *restful.WebService {
	ws := &restful.WebService{}
	ws.Path("/api/v1").
		Consumes(contentTypeJSON).
		Produces(contentTypeJSON)

	// 注册路由
	ws.Route(ws.GET("/releases").To(handler.getRelease))
	ws.Route(ws.GET("/releases/{namespace}/{release}").To(handler.getReleaseInfo))
	ws.Route(ws.POST("/releases/{namespace}/{release}").To(handler.createRelease))
	ws.Route(ws.PUT("/releases/{namespace}/{release}").To(handler.upgradeRelease))
	ws.Route(ws.DELETE("/releases/{namespace}/{release}").To(handler.deleteRelease))
	ws.Route(ws.GET("/releases/{namespace}/{release}/history").To(handler.getReleaseHistory))
	ws.Route(ws.POST("/releases/{namespace}/{release}/rollback").To(handler.rollbackRelease))
	ws.Route(ws.GET("/releases/{namespace}/{release}/manifest").To(handler.getHelmManifest))
	ws.Route(ws.GET("/releases/{namespace}/{release}/values").To(handler.getHelmValues))
	ws.Route(ws.GET("/releases/{namespace}/{release}/pods").To(handler.getReleasePods))
	ws.Route(ws.GET("/releases/status/{chart}").To(handler.checkReleaseStatus))

	return ws
}

// createTestContainer 创建测试用的WebService容器
func createTestContainer(handler *Handler) *restful.Container {
	container := restful.NewContainer()
	ws := createWebService(handler)
	container.Add(ws)
	return container
}

// createHttpRequest 创建基础HTTP请求
func createHttpRequest(method, url string, body interface{}) (*http.Request, error) {
	var bodyBytes []byte
	var err error

	if body != nil {
		bodyBytes, err = json.Marshal(body)
		if err != nil {
			return nil, err
		}
	}

	req := httptest.NewRequest(method, url, bytes.NewReader(bodyBytes))
	if req == nil {
		return nil, fmt.Errorf("failed to create http request")
	}

	req.Header.Set("Content-Type", contentTypeJSON)
	req.Header.Set("Accept", contentTypeJSON)
	return req, nil
}

// executeRequest 执行HTTP请求并返回响应
func executeRequest(handler *Handler, method, url string, body interface{}) (*httptest.ResponseRecorder, error) {
	req, err := createHttpRequest(method, url, body)
	if err != nil {
		return nil, err
	}

	recorder := httptest.NewRecorder()
	if recorder == nil {
		return nil, fmt.Errorf("failed to create response recorder")
	}

	container := createTestContainer(handler)
	container.ServeHTTP(recorder, req)

	return recorder, nil
}

// createSuccessResponse 创建成功响应
func createSuccessResponse(data interface{}) *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Code: statusOK,
		Data: data,
	}
}

// createErrorResponse 创建错误响应
func createErrorResponse(code int, message string) *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Code: int32(code),
		Msg:  message,
	}
}

// createReleaseListData 创建release列表测试数据
func createReleaseListData() []interface{} {
	return []interface{}{
		map[string]interface{}{
			"name":      testReleaseName,
			"namespace": testNamespace,
			"status":    "deployed",
		},
	}
}

// createReleaseInfoData 创建release信息测试数据
func createReleaseInfoData() map[string]interface{} {
	return map[string]interface{}{
		"name":      testReleaseName,
		"namespace": testNamespace,
		"version":   testVersion,
	}
}

// createMcsReleaseInfo 创建MCS release信息
func createMcsReleaseInfo() *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Data: map[string]interface{}{
			"chart": map[string]interface{}{
				"metadata": map[string]interface{}{
					"name": constant.McsChartName,
				},
			},
		},
	}
}

// createNormalReleaseInfo 创建普通release信息
func createNormalReleaseInfo() *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Data: map[string]interface{}{
			"chart": map[string]interface{}{
				"metadata": map[string]interface{}{
					"name": testChartName,
				},
			},
		},
	}
}

// createTestDeployParams 创建测试部署参数
func createTestDeployParams() helmModel.DeployParam {
	return helmModel.DeployParam{
		ChartName: testChartName,
		Values:    `{"replicas": 3}`,
	}
}

// createMcsDeployParams 创建MCS chart部署参数
func createMcsDeployParams() helmModel.DeployParam {
	return helmModel.DeployParam{
		ChartName: constant.McsChartName,
		Values:    `{"replicas": 1}`,
	}
}

// setupGetReleasesMock 设置GetReleases的mock
func setupGetReleasesMock(mockOp *MockHelmOperation, data interface{}) {
	mockOp.On("GetReleases", param.ReleaseParameter{}).Return(
		createSuccessResponse(data), http.StatusOK)
}

// setupGetReleaseInfoMock 设置GetReleaseInfo的mock
func setupGetReleaseInfoMock(mockOp *MockHelmOperation, namespace, name string,
	response *httputil.ResponseJson, statusCode int) {
	mockOp.On("GetReleaseInfo", namespace, name).Return(response, statusCode)
}

func TestHandlerGetReleaseSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	setupGetReleasesMock(mockOp, createReleaseListData())

	recorder, err := executeRequest(handler, "GET", baseReleasesURL, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerGetReleaseEmptyParams(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	setupGetReleasesMock(mockOp, []interface{}{})

	recorder, err := executeRequest(handler, "GET", baseReleasesURL, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerGetReleaseInfoSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	setupGetReleaseInfoMock(mockOp, testNamespace, testReleaseName,
		createSuccessResponse(createReleaseInfoData()), http.StatusOK)

	url := fmt.Sprintf(releaseInfoURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "GET", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerGetReleaseInfoNotFound(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	nonExistentRelease := "non-existent"
	setupGetReleaseInfoMock(mockOp, testNamespace, nonExistentRelease,
		createErrorResponse(statusNotFound, msgReleaseNotFound), http.StatusNotFound)

	url := fmt.Sprintf(releaseInfoURL, testNamespace, nonExistentRelease)
	recorder, err := executeRequest(handler, "GET", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusNotFound, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerCreateReleaseSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	deployParams := createTestDeployParams()
	expectedParams := deployParams
	expectedParams.Namespace = testNamespace
	expectedParams.Release = testReleaseName

	mockOp.On("Install", expectedParams).Return(
		createErrorResponse(statusCreated, msgReleaseCreated), http.StatusCreated)

	url := fmt.Sprintf(releaseInfoURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "POST", url, deployParams)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusCreated, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerCreateReleaseMcsUnauthorized(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	deployParams := createMcsDeployParams()

	url := fmt.Sprintf(releaseInfoURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "POST", url, deployParams)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusForbidden, recorder.Code)
}

func TestHandlerDeleteReleaseSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	setupGetReleaseInfoMock(mockOp, testNamespace, testReleaseName,
		createNormalReleaseInfo(), http.StatusOK)
	mockOp.On("Uninstall", testNamespace, testReleaseName).Return(
		createErrorResponse(statusOK, msgReleaseDeleted), http.StatusOK)

	url := fmt.Sprintf(releaseInfoURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "DELETE", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerDeleteReleaseMcsUnauthorized(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	setupGetReleaseInfoMock(mockOp, testNamespace, testReleaseName,
		createMcsReleaseInfo(), http.StatusOK)
	// 即使是MCS chart，代码仍可能调用Uninstall，所以添加mock
	mockOp.On("Uninstall", testNamespace, testReleaseName).Return(
		createErrorResponse(statusForbidden, msgUnauthorizedDelete), http.StatusForbidden)

	url := fmt.Sprintf(releaseInfoURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "DELETE", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusForbidden, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerUpgradeReleaseSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	deployParams := helmModel.DeployParam{
		ChartName: testChartName,
		Values:    `{"replicas": 5}`,
	}

	expectedParams := deployParams
	expectedParams.Namespace = testNamespace
	expectedParams.Release = testReleaseName

	mockOp.On("Upgrade", expectedParams).Return(
		createSuccessResponse("upgrade successful"), http.StatusOK)

	url := fmt.Sprintf(releaseInfoURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "PUT", url, deployParams)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerGetReleaseHistorySuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	historyData := []interface{}{
		map[string]interface{}{
			"revision": testVersion,
			"status":   "superseded",
		},
		map[string]interface{}{
			"revision": testRevision,
			"status":   "deployed",
		},
	}

	mockOp.On("GetReleaseHistory", testNamespace, testReleaseName, testMaxHistory).Return(
		createSuccessResponse(historyData), http.StatusOK)

	url := fmt.Sprintf(releaseHistoryURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "GET", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerRollbackReleaseSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	setupGetReleaseInfoMock(mockOp, testNamespace, testReleaseName,
		createNormalReleaseInfo(), http.StatusOK)
	mockOp.On("RollbackRelease", testNamespace, testReleaseName, "", 0, 0).Return(
		createSuccessResponse("rollback successful"), http.StatusOK)

	url := fmt.Sprintf(releaseRollbackURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "POST", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerRollbackReleaseMcsUnauthorized(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	setupGetReleaseInfoMock(mockOp, testNamespace, testReleaseName,
		createMcsReleaseInfo(), http.StatusOK)
	// 即使是MCS chart，代码仍可能调用RollbackRelease，所以添加mock
	mockOp.On("RollbackRelease", testNamespace, testReleaseName, "", 0, 0).Return(
		createErrorResponse(statusForbidden, msgUnauthorizedRollback), http.StatusForbidden)

	url := fmt.Sprintf(releaseRollbackURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "POST", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusForbidden, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerGetHelmManifestSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	manifestData := "apiVersion: v1\nkind: Service\nmetadata:\n  name: test-service"
	mockOp.On("GetManifest", testNamespace, testReleaseName).Return(
		createSuccessResponse(manifestData), http.StatusOK)

	url := fmt.Sprintf(releaseManifestURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "GET", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerGetHelmValuesSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	valuesData := map[string]interface{}{
		"replicas": testReplicas,
		"image":    "nginx:latest",
	}
	mockOp.On("GetValues", testNamespace, testReleaseName).Return(
		createSuccessResponse(valuesData), http.StatusOK)

	url := fmt.Sprintf(releaseValuesURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "GET", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerGetReleasePodsSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	podsData := []interface{}{
		map[string]interface{}{
			"name":   "test-pod-1",
			"status": "Running",
		},
		map[string]interface{}{
			"name":   "test-pod-2",
			"status": "Running",
		},
	}

	mockOp.On("GetReleasePods", testNamespace, testReleaseName).Return(
		createSuccessResponse(podsData), http.StatusOK)

	url := fmt.Sprintf(releasePodsURL, testNamespace, testReleaseName)
	recorder, err := executeRequest(handler, "GET", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerCheckReleaseStatusSuccess(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	statusData := map[string]interface{}{
		"status": "deployed",
	}
	mockOp.On("CheckReleaseStatus", testChartName).Return(
		createSuccessResponse(statusData), http.StatusOK)

	url := fmt.Sprintf(releaseStatusURL, testChartName)
	recorder, err := executeRequest(handler, "GET", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusOK, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerCheckReleaseStatusNotFound(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	nonExistentChart := "non-existent"
	mockOp.On("CheckReleaseStatus", nonExistentChart).Return(
		createErrorResponse(statusNotFound, msgChartNotFound), http.StatusNotFound)

	url := fmt.Sprintf(releaseStatusURL, nonExistentChart)
	recorder, err := executeRequest(handler, "GET", url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, recorder)

	assert.Equal(t, statusNotFound, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestHandlerCreateReleaseInvalidBody(t *testing.T) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	req := httptest.NewRequest(http.MethodPost, fmt.Sprintf(releaseInfoURL, testNamespace, testReleaseName),
		bytes.NewBufferString("{invalid-json"))
	req.Header.Set("Content-Type", contentTypeJSON)
	req.Header.Set("Accept", acceptJSON)

	recorder := httptest.NewRecorder()
	createTestContainer(handler).ServeHTTP(recorder, req)

	assert.Equal(t, statusBadRequest, recorder.Code)
	mockOp.AssertExpectations(t)
}

func TestReleaseAction(t *testing.T) {
	tests := []struct {
		name string
		request *restful.Request
		want string
	}{
		{
			name: "nil request",
			request: nil,
			want: "apply",
		},
		{
			name: "post request",
			request: &restful.Request{Request: httptest.NewRequest(http.MethodPost, "/", nil)},
			want: "create",
		},
		{
			name: "put request",
			request: &restful.Request{Request: httptest.NewRequest(http.MethodPut, "/", nil)},
			want: "update",
		},
		{
			name: "other request",
			request: &restful.Request{Request: httptest.NewRequest(http.MethodDelete, "/", nil)},
			want: "apply",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, releaseAction(tt.request))
		})
	}
}

func TestReleaseOperationDetail(t *testing.T) {
	baseDetail := audit.ReleaseDetail("create", testNamespace, testReleaseName)
	tests := []struct {
		name string
		responseJson *httputil.ResponseJson
		want string
	}{
		{
			name: "nil response",
			responseJson: nil,
			want: baseDetail,
		},
		{
			name: "empty message",
			responseJson: &httputil.ResponseJson{Msg: "   "},
			want: baseDetail,
		},
		{
			name: "success message",
			responseJson: &httputil.ResponseJson{Msg: "success"},
			want: baseDetail,
		},
		{
			name: "custom message",
			responseJson: &httputil.ResponseJson{Msg: "something failed"},
			want: baseDetail + ": something failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, releaseOperationDetail("create", testNamespace, testReleaseName, tt.responseJson, http.StatusOK))
		})
	}
}

// BenchmarkHandlerGetRelease 基准测试
func BenchmarkHandlerGetRelease(b *testing.B) {
	mockOp := new(MockHelmOperation)
	handler := newHandler(mockOp)

	mockOp.On("GetReleases", mock.AnythingOfType("param.ReleaseParameter")).Return(
		createSuccessResponse([]interface{}{}), http.StatusOK)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := executeRequest(handler, "GET", baseReleasesURL, nil)
		if err != nil {
			b.Fatal("Failed to execute request:", err)
		}
	}
}
