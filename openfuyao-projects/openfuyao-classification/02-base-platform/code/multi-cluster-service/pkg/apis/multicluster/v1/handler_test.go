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
package v1

import (
	"context"
	"fmt"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/responsehandlers"
	"openfuyao.com/multi-cluster-service/pkg/cluster"
	"openfuyao.com/multi-cluster-service/pkg/utils"
)

// 定义通用的测试结构体类型
type (
	testFields struct {
		ClusterClient cluster.OperationsInterface
		ApiClient     *APIClient
	}

	testArgs struct {
		req  *restful.Request
		resp *restful.Response
	}

	testCase struct {
		name   string
		fields testFields
		args   testArgs
	}
)

type MockClusterClient struct{}

func (m *MockClusterClient) Inspect(ctx context.Context, opts cluster.InspectOptions) (*cluster.ListOfCluster, error) {
	panic("implement me")
}

func (m *MockClusterClient) JoinHost(ctx context.Context) {
	fmt.Printf("JoinHost")
}

func (m *MockClusterClient) Authorize(ctx context.Context, opts cluster.AuthorizeOptions) error {
	return nil
}

func (m *MockClusterClient) Join(ctx context.Context, opts cluster.JoinOptions) error     { return nil }
func (m *MockClusterClient) Unjoin(ctx context.Context, opts cluster.UnjoinOptions) error { return nil }
func (m *MockClusterClient) Edit(ctx context.Context, opts cluster.EditOptions) error     { return nil }

// 测试函数部分保持不变
func TestNewHandler(t *testing.T) {
	h := NewHandler()
	assert.NotNil(t, h)
	assert.Nil(t, h.ClusterClient)
	assert.Nil(t, h.ApiClient)
}

func TestHandlerSayHello(t *testing.T) {
	tests := []testCase{
		createTestCase("Test sayHello", testFields{ApiClient: &APIClient{}}, testArgs{
			req:  new(restful.Request),
			resp: restful.NewResponse(httptest.NewRecorder()),
		}),
	}
	runTestCases(t, tests, func(t *testing.T, tt testCase) {
		h := createTestHandler(tt.fields)
		h.sayHello(tt.args.req, tt.args.resp)
	})
}

func TestHandlerAutoRegisterHost(t *testing.T) {
	tests := []struct {
		name   string
		fields testFields
	}{
		createSimpleTestCase("Test autoRegisterHost", testFields{ClusterClient: &MockClusterClient{}}),
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			h := createTestHandler(tt.fields)
			h.autoRegisterHost()
		})
	}
}

func TestHandlerAutoSendHello(t *testing.T) {
	tests := []struct {
		name   string
		fields testFields
	}{
		createSimpleTestCase("Test autoSendHello", testFields{ApiClient: &APIClient{}}),
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := setupPostMock(t, tt.fields.ApiClient)
			defer patches.Reset()

			h := createTestHandler(tt.fields)
			h.autoSendHello()
		})
	}
}

func TestHandlerAuthPolicyEnforcer(t *testing.T) {
	tests := []testCase{
		createTestCase("Test authPolicyEnforcer", testFields{ClusterClient: &MockClusterClient{}}, testArgs{
			req: restful.NewRequest(httptest.NewRequest("POST", "/",
				strings.NewReader(`{"UserInfo": {"test-user":{"Name": "test-user", "Groups": ["group1", "group2"]}}}`))),
			resp: restful.NewResponse(httptest.NewRecorder()),
		}),
	}
	runTestCases(t, tests, func(t *testing.T, tt testCase) {
		patches := setupResponseHandlerMocks(t, "AuthPolicies Updated successfully !")
		defer patches.Reset()

		h := createTestHandler(tt.fields)
		h.authPolicyEnforcer(tt.args.req, tt.args.resp)
	})
}

func TestHandlerEditCluster(t *testing.T) {
	tests := []testCase{
		createTestCase("Test editCluster - valid request", testFields{ClusterClient: &MockClusterClient{}}, testArgs{
			req: createRestfulRequest("PUT", "/clusters/mock-cluster",
				`{"ClusterLabels": {"env": "production"}}`, "mock-cluster"),
			resp: restful.NewResponse(httptest.NewRecorder()),
		}),
	}
	runTestCases(t, tests, func(t *testing.T, tt testCase) {
		patches := setupEditClusterMocks(t)
		defer patches.Reset()

		h := createTestHandler(tt.fields)
		h.editCluster(tt.args.req, tt.args.resp)
	})
}

func TestHandlerUnregisterCluster(t *testing.T) {
	tests := []testCase{
		createTestCase("Test unregisterCluster - valid request", testFields{ClusterClient: &MockClusterClient{}}, testArgs{
			req:  createRestfulRequest("DELETE", "/clusters/mock-cluster", "", "mock-cluster"),
			resp: restful.NewResponse(httptest.NewRecorder()),
		}),
	}
	runTestCases(t, tests, func(t *testing.T, tt testCase) {
		patches := setupUnregisterClusterMocks(t)
		defer patches.Reset()

		h := createTestHandler(tt.fields)
		h.unregisterCluster(tt.args.req, tt.args.resp)
	})
}

// 辅助函数部分
func createTestHandler(fields testFields) *handler {
	return &handler{
		ClusterClient: fields.ClusterClient,
		ApiClient:     fields.ApiClient,
	}
}

func createTestCase(name string, fields testFields, args testArgs) testCase {
	return testCase{
		name:   name,
		fields: fields,
		args:   args,
	}
}

func createSimpleTestCase(name string, fields testFields) struct {
	name   string
	fields testFields
} {
	return struct {
		name   string
		fields testFields
	}{
		name:   name,
		fields: fields,
	}
}

func runTestCases(t *testing.T, tests []testCase, testFunc func(*testing.T, testCase)) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testFunc(t, tt)
		})
	}
}

func createRestfulRequest(method, path, body, clusterTag string) *restful.Request {
	req := restful.NewRequest(httptest.NewRequest(method, path, strings.NewReader(body)))
	if clusterTag != "" {
		req.PathParameters()["clusterTag"] = clusterTag
	}
	return req
}

func setupPostMock(t *testing.T, apiClient *APIClient) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(apiClient),
		"Post", func(_ *APIClient, url string, a interface{}, body string) ([]uint8, error) {
			assert.Equal(t, fmt.Sprintf("%s%s", UserManagerEndpoint, UserManagerSignalUrl), url)
			return nil, nil
		})
}

func setupResponseHandlerMocks(t *testing.T, msg string) *gomonkey.Patches {
	return gomonkey.ApplyFunc(responsehandlers.SendStatusBadRequest,
		func(resp *restful.Response, message string, err error) {
			assert.Equal(t, "Invalid request data", message)
		}).ApplyFunc(responsehandlers.SendStatusServerError,
		func(resp *restful.Response, message string, err error) {
			assert.Equal(t, "Server error", message)
		}).ApplyFunc(responsehandlers.SendStatusOk, func(resp *restful.Response, message string) {
		assert.Equal(t, msg, message)
	})
}

func setupEditClusterMocks(t *testing.T) *gomonkey.Patches {
	patches := setupResponseHandlerMocks(t, "Cluster tagged successfully")
	patches.ApplyFunc(utils.IsLabelsValid, func(labels map[string]string) bool {
		return true
	}).ApplyFunc(checkUserAccess, func(req *restful.Request,
		resp *restful.Response, client cluster.OperationsInterface, action string) bool {
		return true
	})
	return patches
}

func setupUnregisterClusterMocks(t *testing.T) *gomonkey.Patches {
	patches := gomonkey.ApplyFunc(responsehandlers.SendStatusBadRequest,
		func(resp *restful.Response, message string, err error) {
			assert.Equal(t, "Invalid cluster name", message)
		}).ApplyFunc(responsehandlers.SendStatusServerError,
		func(resp *restful.Response, message string, err error) {
			assert.Equal(t, "Server error", message)
		}).ApplyFunc(responsehandlers.SendStatusOk, func(resp *restful.Response, message string) {
		assert.Equal(t, "Cluster unregistered successfully", message)
	})
	patches.ApplyFunc(checkUserAccess, func(req *restful.Request, resp *restful.Response,
		client cluster.OperationsInterface, action string) bool {
		return true
	}).ApplyFunc(sync2UserMgr, func(apiClient *APIClient, req *restful.Request,
		msg interface{}, clusterTag, action string) []uint8 {
		return nil
	})
	return patches
}
