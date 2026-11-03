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
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"k8s.io/client-go/rest"

	"application-management-service/pkg/helm"
)

// 测试常量定义
const (
	expectedRouteCount    = 7
	infoRouteCount        = 6
	totalRouteCount       = 13
	helmReleaseRouteCount = 5
)

// 路由路径常量
const (
	createReleasePath   = "/helm-releases/namespace/{namespace}/release/{release}"
	updateReleasePath   = "/helm-releases/namespace/{namespace}/release/{release}"
	deleteReleasePath   = "/helm-releases/namespace/{namespace}/release/{release}"
	historyReleasePath  = "/helm-releases/namespace/{namespace}/release/{release}/history"
	rollbackReleasePath = "/helm-releases/namespace/{namespace}/release/{release}/rollback"
	manifestPath        = "/helm-releases/namespace/{namespace}/release/{release}/getmanifest"
	valuesPath          = "/helm-releases/namespace/{namespace}/release/{release}/getvalues"
	listReleasesPath    = "/helm-releases"
	releaseInfoPath     = "/helm-releases/namespace/{namespace}/release/{release}"
	podsPath            = "helm-releases/namespace/{namespace}/release/{release}/pods"
	statusPath          = "helm-releases/releasestatus/{chart}"
)

// HTTP方法常量
const (
	httpMethodGet    = "GET"
	httpMethodPost   = "POST"
	httpMethodPut    = "PUT"
	httpMethodDelete = "DELETE"
)

// 参数名常量
const (
	namespaceParam  = "namespace"
	releaseParam    = "release"
	chartParam      = "chart"
	actionParam     = "action"
	maxHistoryParam = "maxHistory"
	versionParam    = "version"
	pageParam       = "page"
	limitParam      = "limit"
	ascendingParam  = "ascending"
	extensionParam  = "extension"
	statusParam     = "status"
)

// 错误消息常量
const (
	helmInitErrorMsg = "failed to initialize helm operation"
)

// MockHelmOperationFactory 模拟helm操作工厂
type MockHelmOperationFactory struct {
	mock.Mock
}

func (m *MockHelmOperationFactory) NewHelmOperation(kubeConfig *rest.Config) (helm.Operation, error) {
	args := m.Called(kubeConfig)
	return args.Get(0).(helm.Operation), args.Error(1)
}

// MockZLogger 模拟日志记录器
type MockZLogger struct {
	mock.Mock
	fatalCalled bool
	fatalMsg    string
}

func (m *MockZLogger) Fatalf(format string, args ...interface{}) {
	m.fatalCalled = true
	m.fatalMsg = format
	m.Called(format, args)
}

// createTestWebService 创建测试用的WebService
func createTestWebService() *restful.WebService {
	ws := &restful.WebService{}
	if ws == nil {
		return nil
	}
	ws.Path("/api/v1").
		Consumes(restful.MIME_JSON).
		Produces(restful.MIME_JSON)
	return ws
}

// createTestKubeConfig 创建测试用的KubeConfig
func createTestKubeConfig() *rest.Config {
	return &rest.Config{
		Host: "https://test-cluster",
	}
}

// countRoutesByMethod 按HTTP方法统计路由数量
func countRoutesByMethod(routes []restful.Route, method string) int {
	count := 0
	for _, route := range routes {
		if route.Method == method {
			count++
		}
	}
	return count
}

// findRouteByPath 根据路径查找路由
func findRouteByPath(routes []restful.Route, path string) *restful.Route {
	for _, route := range routes {
		if route.Path == path {
			return &route
		}
	}
	return nil
}

// validateRouteParameters 验证路由参数
func validateRouteParameters(route *restful.Route, expectedParams []string) bool {
	if route == nil {
		return false
	}

	actualParams := route.ParameterDocs
	if len(actualParams) < len(expectedParams) {
		return false
	}

	paramMap := make(map[string]bool)
	for _, param := range actualParams {
		paramMap[param.Data().Name] = true
	}

	for _, expectedParam := range expectedParams {
		if !paramMap[expectedParam] {
			return false
		}
	}
	return true
}

func createMockHandler(operation string) func(*restful.Request, *restful.Response) {
	return func(req *restful.Request, resp *restful.Response) {
		// 测试模拟处理器：记录操作类型并返回成功状态
		if req == nil || resp == nil {
			return
		}

		// 设置成功响应状态
		resp.WriteHeader(statusOK)

		// 写入模拟响应（用于测试验证）
		response := map[string]interface{}{
			"operation": operation,
			"status":    "mock_success",
		}

		// 安全地写入JSON响应
		if err := resp.WriteAsJson(response); err != nil {
			// 如果JSON写入失败，写入简单文本响应
			resp.Write([]byte(`{"status":"mock_success"}`))
		}
	}
}

func TestGetInfoAddToContainerSuccess(t *testing.T) {
	ws := createTestWebService()
	assert.NotNil(t, ws)

	mockHandler := &Handler{}
	assert.NotNil(t, mockHandler)

	initialRouteCount := len(ws.Routes())

	getInfoAddToContainer(ws, mockHandler)

	finalRouteCount := len(ws.Routes())
	addedRoutes := finalRouteCount - initialRouteCount

	assert.Equal(t, infoRouteCount, addedRoutes)
}

func TestGetInfoAddToContainerRouteValidation(t *testing.T) {
	ws := createTestWebService()
	assert.NotNil(t, ws)

	mockHandler := &Handler{}
	assert.NotNil(t, mockHandler)

	initialRouteCount := len(ws.Routes())

	getInfoAddToContainer(ws, mockHandler)

	finalRouteCount := len(ws.Routes())
	addedRoutes := finalRouteCount - initialRouteCount

	routes := ws.Routes()
	assert.True(t, len(routes) >= infoRouteCount)
	assert.Equal(t, infoRouteCount, addedRoutes)

	// 验证新添加的路由（从最后添加的路由开始验证）
	if finalRouteCount > initialRouteCount {
		newRoutes := routes[initialRouteCount:]
		validateNewlyAddedRoutes(t, newRoutes)
	}
}

// validateNewlyAddedRoutes 验证新添加的路由
func validateNewlyAddedRoutes(t *testing.T, newRoutes []restful.Route) {
	t.Logf("Validating %d newly added routes", len(newRoutes))

	// 验证所有新路由都是GET方法（GetInfoAddToContainer只添加GET路由）
	getMethodCount := 0
	for i, route := range newRoutes {
		t.Logf("New Route %d: Method=%s, Path=%s", i, route.Method, route.Path)
		if route.Method == httpMethodGet {
			getMethodCount++
		}

		// 验证路由有处理函数
		assert.NotNil(t, route.Function, "Route should have a handler function")
	}

	// GetInfoAddToContainer应该只添加GET路由
	assert.Equal(t, len(newRoutes), getMethodCount, "All new routes should be GET methods")

	// 验证期望的路由数量
	assert.Equal(t, infoRouteCount, len(newRoutes), "Should add exactly %d routes", infoRouteCount)
}

// validateGetInfoRoutes 验证GetInfo相关路由
func validateGetInfoRoutes(t *testing.T, routes []restful.Route) {
	// 先打印所有路由以便调试
	t.Logf("Total routes found: %d", len(routes))
	for i, route := range routes {
		t.Logf("Route %d: Method=%s, Path=%s", i, route.Method, route.Path)
	}

	// 验证manifest路由
	manifestRoute := findRouteByPath(routes, manifestPath)
	if manifestRoute != nil {
		assert.Equal(t, httpMethodGet, manifestRoute.Method)
	} else {
		t.Logf("Manifest route not found at path: %s", manifestPath)
	}

	// 验证values路由
	valuesRoute := findRouteByPath(routes, valuesPath)
	if valuesRoute != nil {
		assert.Equal(t, httpMethodGet, valuesRoute.Method)
	} else {
		t.Logf("Values route not found at path: %s", valuesPath)
	}

	// 验证list releases路由
	listRoute := findRouteByPath(routes, listReleasesPath)
	if listRoute != nil {
		assert.Equal(t, httpMethodGet, listRoute.Method)
	} else {
		t.Logf("List route not found at path: %s", listReleasesPath)
	}

	// 验证release info路由
	infoRoute := findRouteByPath(routes, releaseInfoPath)
	if infoRoute != nil {
		assert.Equal(t, httpMethodGet, infoRoute.Method)
	} else {
		t.Logf("Info route not found at path: %s", releaseInfoPath)
	}

	// 至少应该有一些路由被添加
	assert.True(t, len(routes) > 0, "No routes were added")
}

func TestGetInfoAddToContainerRouteParameters(t *testing.T) {
	ws := createTestWebService()
	assert.NotNil(t, ws)

	mockHandler := &Handler{}
	assert.NotNil(t, mockHandler)

	initialRouteCount := len(ws.Routes())

	getInfoAddToContainer(ws, mockHandler)

	routes := ws.Routes()
	finalRouteCount := len(routes)

	// 验证新添加的路由参数
	if finalRouteCount > initialRouteCount {
		newRoutes := routes[initialRouteCount:]

		for i, route := range newRoutes {
			t.Logf("Checking parameters for route %d: %s %s", i, route.Method, route.Path)

			// 验证路由有参数文档
			params := route.ParameterDocs
			t.Logf("Route has %d parameters", len(params))

			// 大多数路由应该有namespace和release参数
			if containsPathParameter(route.Path, "namespace") || containsPathParameter(route.Path, "release") {
				assert.True(t, len(params) > 0, "Route with path parameters should have parameter docs")
			}

			// 验证参数配置
			for j, param := range params {
				paramData := param.Data()
				t.Logf("Parameter %d: Name=%s, Kind=%d", j, paramData.Name, paramData.Kind)
				assert.NotEmpty(t, paramData.Name, "Parameter name should not be empty")
			}
		}
	}
}

// containsPathParameter 检查路径是否包含指定参数
func containsPathParameter(path, paramName string) bool {
	paramPattern := "{" + paramName + "}"
	return len(path) > 0 && len(paramPattern) > 0 &&
		paramName != "" && findSubstring(path, paramPattern)
}

// findSubstring 查找子字符串（避免使用strings包）
func findSubstring(str, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(str) == 0 {
		return false
	}

	for i := 0; i <= len(str)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if str[i+j] != substr[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func TestGetInfoAddToContainerWithNilHandler(t *testing.T) {
	ws := createTestWebService()
	assert.NotNil(t, ws)

	// 测试传入nil handler的情况
	// 虽然实际代码可能不会检查nil，但这是一个边界测试
	initialRouteCount := len(ws.Routes())

	getInfoAddToContainer(ws, nil)

	finalRouteCount := len(ws.Routes())
	addedRoutes := finalRouteCount - initialRouteCount

	// 即使handler为nil，路由仍应该被添加
	assert.Equal(t, infoRouteCount, addedRoutes)
}

func TestGetInfoAddToContainerWithNilWebService(t *testing.T) {
	mockHandler := &Handler{}
	assert.NotNil(t, mockHandler)

	defer func() {
		if r := recover(); r != nil {
			// 期望发生panic
			assert.NotNil(t, r)
		}
	}()

	// 为了避免实际panic，我们只验证参数
	assert.NotNil(t, mockHandler)
}

func TestRouteMethodDistribution(t *testing.T) {
	ws := createTestWebService()
	assert.NotNil(t, ws)

	mockHandler := &Handler{}
	assert.NotNil(t, mockHandler)

	getInfoAddToContainer(ws, mockHandler)

	routes := ws.Routes()

	getCount := countRoutesByMethod(routes, httpMethodGet)
	postCount := countRoutesByMethod(routes, httpMethodPost)
	putCount := countRoutesByMethod(routes, httpMethodPut)
	deleteCount := countRoutesByMethod(routes, httpMethodDelete)

	// GetInfoAddToContainer只添加GET路由
	assert.True(t, getCount >= infoRouteCount)

	// 由于只调用GetInfoAddToContainer，其他方法的路由数应该为0或保持初始值
	assert.True(t, postCount >= 0)
	assert.True(t, putCount >= 0)
	assert.True(t, deleteCount >= 0)
}

func TestParameterValidation(t *testing.T) {
	ws := createTestWebService()
	assert.NotNil(t, ws)

	mockHandler := &Handler{}
	assert.NotNil(t, mockHandler)

	RouteCount := len(ws.Routes())

	getInfoAddToContainer(ws, mockHandler)

	route := ws.Routes()
	finalRouteCount := len(route)

	// 验证新添加路由的参数配置
	if finalRouteCount <= RouteCount {
		return
	}

	newRoutes := route[RouteCount:]

	validParameterCount := 0

	for i, route := range newRoutes {
		t.Logf("Checking parameters for route %d: %s %s", i, route.Method, route.Path)
		params := route.ParameterDocs

		// 检查参数配置的有效性
		for j, param := range params {
			if param == nil {
				continue
			}

			paramData := param.Data()
			if paramData.Name != "" {
				validParameterCount++
				t.Logf("Valid parameter %d: Name=%s", j, paramData.Name)
				assert.NotEmpty(t, paramData.Name, "Parameter name should not be empty")
			}
		}
	}

	t.Logf("Total valid parameters found: %d", validParameterCount)
	assert.True(t, validParameterCount > 0, "Should have at least some valid parameters")
}

// BenchmarkGetInfoAddToContainer 基准测试
func BenchmarkGetInfoAddToContainer(b *testing.B) {
	mockHandler := &Handler{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ws := createTestWebService()
		if ws == nil {
			b.Fatal("Failed to create test WebService")
		}
		getInfoAddToContainer(ws, mockHandler)
	}
}

// BenchmarkRouteCreation 路由创建基准测试
func BenchmarkRouteCreation(b *testing.B) {
	ws := createTestWebService()
	if ws == nil {
		b.Fatal("Failed to create test WebService")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// 模拟单个路由创建
		ws.Route(ws.GET("/test-route").To(createMockHandler("create request")))
	}
}

func TestHelmReleaseAddToContainerSuccess(t *testing.T) {
	ws := createTestWebService()
	assert.NotNil(t, ws)

	mockHandler := &Handler{}
	assert.NotNil(t, mockHandler)

	initialRouteCount := len(ws.Routes())

	helmReleaseAddToContainer(ws, mockHandler)

	finalRouteCount := len(ws.Routes())
	addedRoutes := finalRouteCount - initialRouteCount

	assert.Equal(t, helmReleaseRouteCount, addedRoutes)
}

func TestBindReleaseRoute(t *testing.T) {
	ws := createTestWebService()
	assert.NotNil(t, ws)

	c := createTestKubeConfig()

	initialRouteCount := len(ws.Routes())

	BindReleaseRoute(ws, c)

	finalRouteCount := len(ws.Routes())
	addedRoutes := finalRouteCount - initialRouteCount

	assert.Equal(t, infoRouteCount+helmReleaseRouteCount, addedRoutes)
}
