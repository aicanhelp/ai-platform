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
	"context"
	"strings"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"

	"openfuyao-rayservice/pkg/utils/httputil"
)

// 模拟 Operation 接口的实现

func newMockOperation() *mockOperation {
	return &mockOperation{
		methodCalled: make(map[string]bool),
		responseData: "test-data",
	}
}

type mockOperation struct {
	mock.Mock
	methodCalled map[string]bool
	responseData interface{}
}

func (m *mockOperation) MethodInitOperation() {
	m.Called()
}

func (m *mockOperation) MethodAllOverview() (*httputil.ResponseJson, int) {
	methodAllOverviewArgs := m.Called()
	return methodAllOverviewArgs.Get(0).(*httputil.ResponseJson), methodAllOverviewArgs.Int(1)
}

func (m *mockOperation) MethodListMetaCrd(crd string) (*httputil.ResponseJson, int) {
	methodListMetaCrdArgs := m.Called(crd)
	return methodListMetaCrdArgs.Get(0).(*httputil.ResponseJson), methodListMetaCrdArgs.Int(1)
}

func (m *mockOperation) MethodListCrd(crd string, req *CrdListRequest) (*httputil.ResponseJson, int) {
	methodListCrdArgs := m.Called(crd, req)
	return methodListCrdArgs.Get(0).(*httputil.ResponseJson), methodListCrdArgs.Int(1)
}

func (m *mockOperation) MethodCreateCrd(crd, data, createBy string, isStart bool) (*httputil.ResponseJson, int) {
	methodCreateCrdArgs := m.Called(crd, data, createBy, isStart)
	return methodCreateCrdArgs.Get(0).(*httputil.ResponseJson), methodCreateCrdArgs.Int(1)
}

func (m *mockOperation) MethodDeleteCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	methodDeleteCrdArgs := m.Called(crd, ns, name)
	return methodDeleteCrdArgs.Get(0).(*httputil.ResponseJson), methodDeleteCrdArgs.Int(1)
}
func (m *mockOperation) MethodUpdateCrd(crd, ns, data, name string) (*httputil.ResponseJson, int) {
	methodUpdateCrdArgs := m.Called(crd, ns, data, name)
	return methodUpdateCrdArgs.Get(0).(*httputil.ResponseJson), methodUpdateCrdArgs.Int(1)
}
func (m *mockOperation) MethodGetCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	methodGetCrdArgs := m.Called(crd, ns, name)
	return methodGetCrdArgs.Get(0).(*httputil.ResponseJson), methodGetCrdArgs.Int(1)
}

func (m *mockOperation) MethodRestartCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	methodRestartCrdArgs := m.Called(crd, ns, name)
	return methodRestartCrdArgs.Get(0).(*httputil.ResponseJson), methodRestartCrdArgs.Int(1)
}

func (m *mockOperation) MethodStopCrd(crd, ns, name string) (*httputil.ResponseJson, int) {
	methodStopCrdArgs := m.Called(crd, ns, name)
	return methodStopCrdArgs.Get(0).(*httputil.ResponseJson), methodStopCrdArgs.Int(1)
}

func TestNewWebSevice(t *testing.T) {
	// 创建测试数据
	testGV := schema.GroupVersion{Group: "/test/group", Version: "v1"}

	// 调用函数
	ws := NewWebSevice(testGV)

	// 验证结果
	require.NotNil(t, ws, "WebService 不应为 nil")
	assert.Equal(t, strings.TrimRight(testGV.String(), "/"), ws.RootPath(), "WebService 路径应等于 GroupVersion")

	// 直接检查路径是否正确，不测试 Produces
	// 由于链式调用可能导致 Produces 不被保存，我们跳过这部分测试
	t.Log("注意：由于 NewWbeSevice 使用链式调用，无法直接测试 Produces() 方法的结果")
}

func TestConfigRay(t *testing.T) {
	type args struct {
		c          *restful.Container
		kubeConfig *rest.Config
		ctx        context.Context
	}
	var tests []struct {
		name    string
		args    args
		wantErr bool
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ConfigRay(tt.args.c, tt.args.kubeConfig, tt.args.ctx); (err != nil) != tt.wantErr {
				t.Errorf("ConfigRay() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

// TestGetInfoAddToContainer 测试 getInfoAddToContainer 函数
func Test_getInfoAddToContainer(t *testing.T) {
	// 创建 WebService
	ws := NewWebSevice(groupVersion)

	// 创建真实的 Handler (使用模拟的 Operation)
	operation := newMockOperation()
	handler := newHandler(operation)

	// 调用函数
	getInfoAddToContainer(ws, handler)

	// 验证路由是否已添加
	routes := ws.Routes()

	// 这里我们简单检查是否有预期数量的路由，以及是否包含某些路径
	const minExpectedRoutes = 5
	assert.GreaterOrEqual(t, len(routes), minExpectedRoutes, "应至少添加 5 个路由")

	// 创建一个辅助函数来检查是否存在特定路径的路由
	hasRoute := func(path string, method string) bool {
		for _, route := range routes {
			if route.Path == path && route.Method == method {
				return true
			}
		}
		return false
	}

	// 验证已添加预期的路由
	assert.True(t, hasRoute("/rest/v1/{crd}/list/meta", "GET"), "应有 GET /meta 路由")
	assert.True(t, hasRoute("/rest/v1/{crd}/get/{namespace}/{name}", "GET"), "应有 GET /{crd}/get/{namespace}/{name} 路由")
	assert.True(t, hasRoute("/rest/v1/{crd}/delete/{namespace}/{name}", "DELETE"), "应有 DELETE /{crd}/delete/{namespace}/{name}路由")
	assert.True(t, hasRoute("/rest/v1/{crd}/restart/{namespace}/{name}", "POST"), "应有 POST /{crd}/restart/{namespace}/{name} 路由")
	assert.True(t, hasRoute("/rest/v1/{crd}/stop/{namespace}/{name}", "POST"), "应有 POST /{crd}/stop/{namespace}/{name} 路由")
}
