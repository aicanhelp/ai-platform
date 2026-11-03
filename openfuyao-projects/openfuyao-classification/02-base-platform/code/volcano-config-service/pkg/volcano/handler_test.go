/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

// Package volcano config
package volcano

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"volcano-config-service/pkg/constant"
	"volcano-config-service/pkg/utils/httputil"
	"volcano-config-service/pkg/zlog"
)

// 创建 Operation 接口的模拟实现
type MockOperation struct {
	mock.Mock
}

func (m *MockOperation) MethodBalanceGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodBalanceGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodBalancePut(policy string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodBalancePut")
	args := m.Called(policy)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodNumaAwareGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodNumaAwareGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodNumaAwarePut(policy string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodNumaAwarePut")
	args := m.Called(policy)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodNumaDistanceGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodNumaDistanceGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodNumaDistancePut(policy string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodNumaDistancePut")
	args := m.Called(policy)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodNumaFastGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodNumaFastGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodNumaFastPut(policy string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodNumaFastPut")
	args := m.Called(policy)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodBinpackGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodBinpackGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodBinpackPut(policy string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodBinpackPut")
	args := m.Called(policy)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodDrfGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodDrfGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodDrfPut(policy string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodDrfPut")
	args := m.Called(policy)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodGangGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodGangGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodGangPut(policy string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodGangPut")
	args := m.Called(policy)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodPriorityGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodPriorityGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodPriorityPut(policy string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodPriorityPut")
	args := m.Called(policy)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodDefault() (*httputil.ResponseJson, int) {
	zlog.Info("MethodDefault")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodNodeList() (*httputil.ResponseJson, int) {
	zlog.Info("MethodNodeList")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodNodeListByName(name string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodNodeListByName")
	args := m.Called(name)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodCRInfo(name string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodCRInfo")
	args := m.Called(name)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodUuidToName(uuid string) (*httputil.ResponseJson, int) {
	zlog.Info("MethodUuidToName")
	args := m.Called(uuid)
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

func (m *MockOperation) MethodSchedulerGet() (*httputil.ResponseJson, int) {
	zlog.Info("MethodSchedulerGet")
	args := m.Called()
	return args.Get(0).(*httputil.ResponseJson), args.Int(1)
}

// 测试 GET 方法
func TestHandler_GET_Methods(t *testing.T) {
	// 在测试中注册路由的部分，需要与源代码中的ConfigVolcano函数保持一致
	// 创建一个新的 Handler 和 Container
	// 创建一个新的 MockOperation
	mockOp := new(MockOperation)
	handler := newHandler(mockOp)

	// 注册路由 - 使用与ConfigVolcano函数相同的方式
	ws := NewWbeSevice(groupVersion)

	// 添加GET信息相关路由
	getInfoAddToContainer(ws, handler)

	// 添加NUMA相关路由
	numaAwareController(ws, handler)

	tests := []struct {
		name           string
		path           string
		setupMock      func(mockOp *MockOperation)
		expectedStatus int
		expectedBody   map[string]interface{}
	}{
		{
			name: "Test_NumaAwareGet",
			path: "/rest/scheduling/v1/numaaware",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodNumaAwareGet").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"policy": "enabled"},
						Msg:  "NumaAwareGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"policy": "enabled"},
				"msg":  "NumaAwareGet success",
			},
		},
		{
			name: "Test_NumaDistanceGet",
			path: "/rest/scheduling/v1/numaDistance",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodNumaDistanceGet").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"policy": "enabled"},
						Msg:  "NumaDistanceGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"policy": "enabled"},
				"msg":  "NumaDistanceGet success",
			},
		},
		{
			name: "Test_NumaFastGet",
			path: "/rest/scheduling/v1/numaFast",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodNumaFastGet").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"policy": "enabled"},
						Msg:  "NumaFastGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"policy": "enabled"},
				"msg":  "NumaFastGet success",
			},
		},
		{
			name: "Test_BinpackGet",
			path: "/rest/scheduling/v1/binpack",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodBinpackGet").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"policy": "enabled"},
						Msg:  "BinpackGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"policy": "enabled"},
				"msg":  "BinpackGet success",
			},
		},
		{
			name: "Test_DrfGet",
			path: "/rest/scheduling/v1/drf",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodDrfGet").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"policy": "enabled"},
						Msg:  "DrfGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"policy": "enabled"},
				"msg":  "DrfGet success",
			},
		},
		{
			name: "Test_GangGet",
			path: "/rest/scheduling/v1/gang",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodGangGet").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"policy": "enabled"},
						Msg:  "GangGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"policy": "enabled"},
				"msg":  "GangGet success",
			},
		},
		{
			name: "Test_PriorityGet",
			path: "/rest/scheduling/v1/priority",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodPriorityGet").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"policy": "enabled"},
						Msg:  "PriorityGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"policy": "enabled"},
				"msg":  "PriorityGet success",
			},
		},
		{
			name: "Test_NodeListGet",
			path: "/rest/scheduling/v1/nodeList",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodNodeList").Return(
					&httputil.ResponseJson{
						Data: []interface{}{
							map[string]interface{}{"name": "node1"},
							map[string]interface{}{"name": "node2"},
						},
						Msg: "NodeListGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": []interface{}{
					map[string]interface{}{"name": "node1"},
					map[string]interface{}{"name": "node2"},
				},
				"msg": "NodeListGet success",
			},
		},
		{
			name: "Test_NodeListGetByName",
			path: "/rest/scheduling/v1/nodeList/node1",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodNodeListByName", "node1").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"name": "node1"},
						Msg:  "NodeListGetByName success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"name": "node1"},
				"msg":  "NodeListGetByName success",
			},
		},
		{
			name: "Test_CRGet",
			path: "/rest/scheduling/v1/cr/mycr",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodCRInfo", "mycr").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"name": "mycr"},
						Msg:  "CRGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"name": "mycr"},
				"msg":  "CRGet success",
			},
		},
		{
			name: "Test_PodNameGet",
			path: "/rest/scheduling/v1/uuidToName/uuid1",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodUuidToName", "uuid1").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"name": "pod1"},
						Msg:  "PodNameGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"name": "pod1"},
				"msg":  "PodNameGet success",
			},
		},
		{
			name: "Test_SchedulerGet",
			path: "/rest/scheduling/v1/scheduler",
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodSchedulerGet").Return(
					&httputil.ResponseJson{
						Data: map[string]interface{}{"status": "running"},
						Msg:  "SchedulerGet success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": map[string]interface{}{"status": "running"},
				"msg":  "SchedulerGet success",
			},
		},
	}

	// test
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 创建一个新的 MockOperation
			mockOp := new(MockOperation)

			// 设置 mock 的预期行为
			tt.setupMock(mockOp)

			// 创建一个新的 Handler 和 Container
			handler := newHandler(mockOp)

			// 注册路由
			ws := NewWbeSevice(groupVersion)
			container := restful.NewContainer()

			// 添加路由 - 这里需要根据实际情况选择相应的方法
			getInfoAddToContainer(ws, handler)
			numaAwareController(ws, handler)

			// 必须在这里添加其他路由
			ws.Route(ws.GET("/binpack").
				Doc("get the state of binpack").
				To(handler.binpackGet))

			ws.Route(ws.GET("/drf").
				Doc("get the state of drf").
				To(handler.drfGet))

			ws.Route(ws.GET("/gang").
				Doc("get the state of gang").
				To(handler.gangGet))

			ws.Route(ws.GET("/priority").
				Doc("get the state of priority").
				To(handler.priorityGet))

			// 其他需要测试的路由...
			container.Add(ws)

			// 创建一个测试请求
			req := httptest.NewRequest("GET", tt.path, nil)
			req.Header.Set("Accept", restful.MIME_JSON) // 设置 Accept 头，解决 406 问题

			t.Logf("已注册的路由:")
			for _, route := range ws.Routes() {
				t.Logf("- %s %s", route.Method, route.Path)
			}

			// 然后对每个测试用例，打印完整的请求路径
			t.Logf("发送请求: %s %s", "GET", tt.path)

			// 创建一个 ResponseRecorder 来记录响应
			w := httptest.NewRecorder()

			// 发送请求
			container.ServeHTTP(w, req)

			// 验证状态码
			assert.Equal(t, tt.expectedStatus, w.Code, "状态码应该匹配")

			// 解析响应体
			var responseBody map[string]interface{}
			err := json.Unmarshal(w.Body.Bytes(), &responseBody)
			assert.NoError(t, err, "应该能够解析 JSON")

			// 验证响应体
			assert.Equal(t, tt.expectedBody, responseBody, "响应体应该匹配")

			// 验证 mock 预期已满足
			mockOp.AssertExpectations(t)
		})
	}
}

func TestHandler_PUT_Methods(t *testing.T) {
	tests := []struct {
		name           string
		path           string
		requestBody    map[string]string
		setupMock      func(mockOp *MockOperation)
		expectedStatus int
		expectedBody   map[string]interface{}
	}{
		{
			name: "Test_NumaAwarePut",
			path: "/rest/scheduling/v1/numaaware",
			requestBody: map[string]string{
				"policy": "enabled",
			},
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodNumaAwarePut", "enabled").Return(
					&httputil.ResponseJson{
						Data: "NumaAwarePut enabled",
						Msg:  "success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": "NumaAwarePut enabled",
				"msg":  "success",
			},
		},
		{
			name: "Test_NumaDistancePut",
			path: "/rest/scheduling/v1/numaDistance",
			requestBody: map[string]string{
				"policy": "enabled",
			},
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodNumaDistancePut", "enabled").Return(
					&httputil.ResponseJson{
						Data: "NumaDistancePut enabled",
						Msg:  "success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": "NumaDistancePut enabled",
				"msg":  "success",
			},
		},
		{
			name: "Test_NumaFastPut",
			path: "/rest/scheduling/v1/numaFast",
			requestBody: map[string]string{
				"policy": "enabled",
			},
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodNumaFastPut", "enabled").Return(
					&httputil.ResponseJson{
						Data: "NumaFastPut enabled",
						Msg:  "success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": "NumaFastPut enabled",
				"msg":  "success",
			},
		},
		{
			name: "Test_BinpackPut",
			path: "/rest/scheduling/v1/binpack",
			requestBody: map[string]string{
				"policy": "enabled",
			},
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodBinpackPut", "enabled").Return(
					&httputil.ResponseJson{
						Data: "BinpackPut enabled",
						Msg:  "success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": "BinpackPut enabled",
				"msg":  "success",
			},
		},
		{
			name: "Test_DrfPut",
			path: "/rest/scheduling/v1/drf",
			requestBody: map[string]string{
				"policy": "enabled",
			},
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodDrfPut", "enabled").Return(
					&httputil.ResponseJson{
						Data: "DrfPut enabled",
						Msg:  "success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": "DrfPut enabled",
				"msg":  "success",
			},
		},
		{
			name: "Test_GangPut",
			path: "/rest/scheduling/v1/gang",
			requestBody: map[string]string{
				"policy": "enabled",
			},
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodGangPut", "enabled").Return(
					&httputil.ResponseJson{
						Data: "GangPut enabled",
						Msg:  "success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": "GangPut enabled",
				"msg":  "success",
			},
		},
		{
			name: "Test_PriorityPut",
			path: "/rest/scheduling/v1/priority",
			requestBody: map[string]string{
				"policy": "enabled",
			},
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodPriorityPut", "enabled").Return(
					&httputil.ResponseJson{
						Data: "PriorityPut enabled",
						Msg:  "success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": "PriorityPut enabled",
				"msg":  "success",
			},
		},
		{
			name: "Test_DefaultPut",
			path: "/rest/scheduling/v1/default",
			requestBody: map[string]string{
				"policy": "default",
			},
			setupMock: func(mockOp *MockOperation) {
				mockOp.On("MethodDefault").Return(
					&httputil.ResponseJson{
						Data: "DefaultPut default",
						Msg:  "success",
					},
					constant.Success,
				)
			},
			expectedStatus: constant.Success,
			expectedBody: map[string]interface{}{
				"data": "DefaultPut default",
				"msg":  "success",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 创建一个新的 MockOperation
			mockOp := new(MockOperation)

			// 设置 mock 的预期行为
			tt.setupMock(mockOp)

			// 创建一个新的 Handler 和 Container
			handler := newHandler(mockOp)
			container := restful.NewContainer()

			// 注册路由
			ws := NewWbeSevice(groupVersion)

			// 添加路由 - 这里需要根据实际情况选择相应的方法
			getInfoAddToContainer(ws, handler)
			numaAwareController(ws, handler)

			ws.Route(ws.PUT("/binpack").
				Doc("modify the state of binpack").
				To(handler.binpackPut))

			ws.Route(ws.PUT("/drf").
				Doc("modify the state of drf").
				To(handler.drfPut))

			ws.Route(ws.PUT("/gang").
				Doc("modify the state of gang").
				To(handler.gangPut))

			ws.Route(ws.PUT("/priority").
				Doc("modify the state of priority").
				To(handler.priorityPut))

			ws.Route(ws.PUT("/default").
				Doc("modify the state of default").
				To(handler.defaultPut))

			// 其他需要测试的路由...
			container.Add(ws)

			// 序列化请求体
			requestBodyJSON, err := json.Marshal(tt.requestBody)
			assert.NoError(t, err, "JSON 序列化不应失败")

			// 创建一个测试请求
			req := httptest.NewRequest("PUT", tt.path, bytes.NewBuffer(requestBodyJSON))
			req.Header.Set("Content-Type", restful.MIME_JSON)
			req.Header.Set("Accept", restful.MIME_JSON) // 设置 Accept 头，解决 406 问题

			// 创建一个 ResponseRecorder 来记录响应
			w := httptest.NewRecorder()

			// 发送请求
			container.ServeHTTP(w, req)

			// 验证状态码
			assert.Equal(t, tt.expectedStatus, w.Code, "状态码应该匹配")

			// 解析响应体
			var responseBody map[string]interface{}
			err = json.Unmarshal(w.Body.Bytes(), &responseBody)
			assert.NoError(t, err, "应该能够解析 JSON")

			// 验证响应体
			assert.Equal(t, tt.expectedBody, responseBody, "响应体应该匹配")

			// 验证 mock 预期已满足
			mockOp.AssertExpectations(t)
		})
	}
}

// 测试 getPolicy 函数
func TestGetPolicy(t *testing.T) {
	testCases := []struct {
		name        string
		requestBody map[string]string
		expectError bool
		expected    string
	}{
		{
			name:        "Valid Policy",
			requestBody: map[string]string{"policy": "enabled"},
			expectError: false,
			expected:    "enabled",
		},
		{
			name:        "Empty Policy",
			requestBody: map[string]string{"policy": ""},
			expectError: false, // 改为 false，因为不期望错误
			expected:    "",    // 期望返回空字符串
		},
		{
			name:        "Missing Policy Field",
			requestBody: map[string]string{"other": "value"},
			expectError: false,
			expected:    "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 序列化请求体
			body, err := json.Marshal(tc.requestBody)
			assert.NoError(t, err)

			// 创建请求并记录响应
			req := httptest.NewRequest("PUT", "/test", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()

			// 创建 restful 请求和响应
			httpRequest := restful.NewRequest(req)
			httpResponse := restful.NewResponse(resp)

			// 调用测试函数
			policy, err := getPolicy(httpRequest, httpResponse)

			if tc.expectError {
				assert.Error(t, err)
				assert.Empty(t, policy)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected, policy)
			}
		})
	}
}
