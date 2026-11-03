/*
 *
 *  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

package ray

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"openfuyao-rayservice/pkg/utils/httputil"
)

// 创建 Operation 接口的模拟实现
type MockOperation struct {
	mock.Mock
}

func (m *MockOperation) MethodInitOperation() {
	m.Called()
}

func (m *MockOperation) MethodAllOverview() (*httputil.ResponseJson, int) {
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodListMetaCrd(crd string) (*httputil.ResponseJson, int) {
	args := m.Called(crd)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodListCrd(crd string, req *CrdListRequest) (*httputil.ResponseJson, int) {
	args := m.Called(crd, req)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodCreateCrd(crd, data, createBy string, isStart bool) (*httputil.ResponseJson, int) {
	createArgs := m.Called(crd, data, createBy, isStart)
	return createArgs.Get(0).(*httputil.ResponseJson), createArgs.Int(1)
}

func (m *MockOperation) MethodDeleteCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	deleteArgs := m.Called(crd, ns, name)
	return deleteArgs.Get(0).(*httputil.ResponseJson), deleteArgs.Int(1)
}
func (m *MockOperation) MethodUpdateCrd(crd, ns, data, name string) (*httputil.ResponseJson, int) {
	updateArgs := m.Called(crd, ns, data, name)
	return updateArgs.Get(0).(*httputil.ResponseJson), updateArgs.Int(1)
}
func (m *MockOperation) MethodGetCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	getArgs := m.Called(crd, ns, name)
	return getArgs.Get(0).(*httputil.ResponseJson), getArgs.Int(1)
}

func (m *MockOperation) MethodRestartCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	restartArgs := m.Called(crd, ns, name)
	return restartArgs.Get(0).(*httputil.ResponseJson), restartArgs.Int(1)
}

func (m *MockOperation) MethodStopCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	stopArgs := m.Called(crd, ns, name)
	return stopArgs.Get(0).(*httputil.ResponseJson), stopArgs.Int(1)
}

// 定义HTTP状态码常量
const (
	HTTPStatusOK = 200
)

// 定义测试常量
const (
	TestNamespace = "vcjob"
	TestCrdName   = "raycluster-autoscaler-1"
	TestCrdType   = "raycluster"
	TestCreateBy  = "createBy"
	TestAll       = "all"
)

// 创建测试容器和handler
func createTestContainer(mockOp *MockOperation) *restful.Container {
	handler := newHandler(mockOp)
	container := restful.NewContainer()
	ws := NewWebSevice(groupVersion)
	getInfoAddToContainer(ws, handler)
	container.Add(ws)
	return container
}

// 验证HTTP响应
func validateHTTPResponse(t *testing.T, w *httptest.ResponseRecorder,
	expectedStatus int, expectedBody map[string]interface{}) {
	assert.Equal(t, expectedStatus, w.Code, "状态码应该匹配")

	var responseBody map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &responseBody)
	assert.NoError(t, err, "应该能够解析JSON")

	assert.Equal(t, expectedBody, responseBody, "响应体应该匹配")
}

// 创建成功响应模板
func createSuccessResponse(dataKey, dataValue string) *httputil.ResponseJson {
	return &httputil.ResponseJson{
		Data: map[string]interface{}{dataKey: dataValue},
		Msg:  "success",
	}
}

// 创建预期的成功响应体
func createExpectedBody(dataKey, dataValue string) map[string]interface{} {
	return map[string]interface{}{
		"data": map[string]interface{}{dataKey: dataValue},
		"msg":  "success",
	}
}

// 定义GET测试用例类型
type GetTestCase struct {
	name           string
	path           string
	setupMock      func(mockOp *MockOperation)
	expectedStatus int
	expectedBody   map[string]interface{}
}

// 获取GET测试用例列表
func getGetTestCases() []GetTestCase {
	return []GetTestCase{
		{
			name: "Test_CrdGetGet",
			path: fmt.Sprintf("/rest/v1/%s/get/%s/%s", TestCrdType, TestNamespace, TestCrdName),
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodGetCrd", TestCrdType, TestNamespace, TestCrdName).Return(
					createSuccessResponse("name", TestCrdName),
					HTTPStatusOK,
				)
			},
			expectedStatus: HTTPStatusOK,
			expectedBody:   createExpectedBody("name", TestCrdName),
		},
		{
			name: "Test_CrdListMetaGet",
			path: fmt.Sprintf("/rest/v1/%s/list/meta", TestCrdType),
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodListMetaCrd", TestCrdType).Return(
					createSuccessResponse("name", TestCrdType),
					HTTPStatusOK,
				)
			},
			expectedStatus: HTTPStatusOK,
			expectedBody:   createExpectedBody("name", TestCrdType),
		},
		{
			name: "Test_OverviewAllGet",
			path: fmt.Sprintf("/rest/v1/overview/%s", TestAll),
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodAllOverview").Return(
					createSuccessResponse("name", TestAll),
					HTTPStatusOK,
				)
			},
			expectedStatus: HTTPStatusOK,
			expectedBody:   createExpectedBody("name", TestAll),
		},
		{
			name: "Test_CrdListGet",
			path: fmt.Sprintf("/rest/v1/%s/list", TestCrdType),
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodListCrd", TestCrdType, &CrdListRequest{}).Return(
					createSuccessResponse("name", TestCrdType),
					HTTPStatusOK,
				)
			},
			expectedStatus: HTTPStatusOK,
			expectedBody:   createExpectedBody("name", TestCrdType),
		},
	}
}

// 测试单个GET用例
func testGetCase(t *testing.T, testCase GetTestCase) {
	mockOp := new(MockOperation)
	testCase.setupMock(mockOp)

	container := createTestContainer(mockOp)

	req := httptest.NewRequest("GET", testCase.path, nil)
	req.Header.Set("Accept", restful.MIME_JSON)

	w := httptest.NewRecorder()
	container.ServeHTTP(w, req)

	validateHTTPResponse(t, w, testCase.expectedStatus, testCase.expectedBody)
	mockOp.AssertExpectations(t)
}

// 测试 GET 方法
func TestHandler_GET_Methods(t *testing.T) {
	testCases := getGetTestCases()

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			testGetCase(t, testCase)
		})
	}
}

// 定义POST测试用例类型
type PostTestCase struct {
	name           string
	path           string
	requestBody    map[string]string
	formFile       map[string][]string
	setupMock      func(mockOp *MockOperation)
	expectedStatus int
	submitType     string
	expectedBody   map[string]interface{}
}

// 创建CRD创建操作的测试用例
func getCrdCreateTestCase() PostTestCase {
	return PostTestCase{
		name: "Test_CrdCreatePost",
		path: fmt.Sprintf("/rest/v1/%s/create", TestCrdType),
		formFile: map[string][]string{
			"createBy": {TestCreateBy},
			"isStart":  {"false"},
			"config":   {"fileString"},
		},
		submitType: "form",
		setupMock: func(mockOp *MockOperation) {
			mockOp.On("MethodCreateCrd", TestCrdType, "fileString", TestCreateBy, false).Return(
				createSuccessResponse("name", TestCrdType),
				HTTPStatusOK,
			)
		},
		expectedStatus: HTTPStatusOK,
		expectedBody:   createExpectedBody("name", TestCrdType),
	}
}

// 创建CRD重启操作的测试用例
func getCrdRestartTestCase() PostTestCase {
	return PostTestCase{
		name: "Test_CrdRestartPost",
		path: fmt.Sprintf("/rest/v1/%s/restart/%s/%s", TestCrdType, TestNamespace, TestCrdName),
		requestBody: map[string]string{
			"name": TestCrdType,
		},
		submitType: "body",
		setupMock: func(mockOp *MockOperation) {
			mockOp.On("MethodRestartCrd", TestCrdType, TestNamespace, TestCrdName).Return(
				createSuccessResponse("name", TestCrdType),
				HTTPStatusOK,
			)
		},
		expectedStatus: HTTPStatusOK,
		expectedBody:   createExpectedBody("name", TestCrdType),
	}
}

// 创建CRD停止操作的测试用例
func getCrdStopTestCase() PostTestCase {
	return PostTestCase{
		name: "Test_CrdStopPost",
		path: fmt.Sprintf("/rest/v1/%s/stop/%s/%s", TestCrdType, TestNamespace, TestCrdName),
		requestBody: map[string]string{
			"name": TestCrdType,
		},
		submitType: "body",
		setupMock: func(mockOp *MockOperation) {
			mockOp.On("MethodStopCrd", TestCrdType, TestNamespace, TestCrdName).Return(
				createSuccessResponse("name", TestCrdType),
				HTTPStatusOK,
			)
		},
		expectedStatus: HTTPStatusOK,
		expectedBody:   createExpectedBody("name", TestCrdType),
	}
}

// 获取POST测试用例列表
func getPostTestCases() []PostTestCase {
	return []PostTestCase{
		getCrdCreateTestCase(),
		getCrdRestartTestCase(),
		getCrdStopTestCase(),
	}
}

func createPostRequest(t *testing.T, testCase PostTestCase) *http.Request {
	t.Helper()

	if testCase.submitType == "body" {
		requestBodyJSON, err := json.Marshal(testCase.requestBody)
		if err != nil {
			t.Fatalf("Failed to marshal request body: %v", err)
		}
		req := httptest.NewRequest("POST", testCase.path, bytes.NewBuffer(requestBodyJSON))
		req.Header.Set("Content-Type", restful.MIME_JSON)
		req.Header.Set("Accept", restful.MIME_JSON)
		return req
	} else {
		req := httptest.NewRequest("POST", testCase.path,
			strings.NewReader(url.Values(testCase.formFile).Encode()))
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		return req
	}
}

// 测试单个POST用例
func testPostCase(t *testing.T, testCase PostTestCase) {
	mockOp := new(MockOperation)
	testCase.setupMock(mockOp)

	container := createTestContainer(mockOp)

	req := createPostRequest(t, testCase)
	w := httptest.NewRecorder()
	container.ServeHTTP(w, req)

	validateHTTPResponse(t, w, testCase.expectedStatus, testCase.expectedBody)
	mockOp.AssertExpectations(t)
}

// 测试 POST 方法
func TestHandler_POST_Methods(t *testing.T) {
	testCases := getPostTestCases()

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			testPostCase(t, testCase)
		})
	}
}
