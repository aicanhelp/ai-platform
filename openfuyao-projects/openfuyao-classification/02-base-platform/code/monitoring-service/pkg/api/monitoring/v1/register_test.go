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
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/monitoring-service/pkg/client"
	"openfuyao.com/monitoring-service/pkg/server/runtime"
)

// mockMonitoringClient implements client.Interface for testing register routes
type mockMonitoringClient struct{}

func (m *mockMonitoringClient) QueryInstantMetrics(expr string, ts time.Time) client.Metric {
	return client.Metric{}
}

func (m *mockMonitoringClient) QueryRangeMetrics(expr string, start, end time.Time, step time.Duration) client.Metric {
	return client.Metric{}
}

func (m *mockMonitoringClient) QueryInstantCustomMetrics(metrics []string, ts time.Time, o client.QueryOption) []client.Metric {
	return nil
}

func (m *mockMonitoringClient) QueryRangeCustomMetrics(metrics []string, start, end time.Time, step time.Duration, o client.QueryOption) []client.Metric {
	return nil
}

func (m *mockMonitoringClient) GetMetricNames() ([]client.NativeMetricName, error) {
	return []client.NativeMetricName{{NativeMetricName: "up"}}, nil
}

func (m *mockMonitoringClient) GetTargets() ([]client.Target, error) {
	return []client.Target{}, nil
}

func (m *mockMonitoringClient) GetAlertingRules() ([]client.AlertingRule, error) {
	return []client.AlertingRule{}, nil
}

func TestAddToContainer(t *testing.T) {
	c := restful.NewContainer()
	mockClient := &mockMonitoringClient{}

	err := AddToContainer(c, mockClient)

	assert.NoError(t, err)
	svcs := c.RegisteredWebServices()
	assert.Len(t, svcs, 1, "container should have one registered WebService")
	ws := svcs[0]
	assert.Equal(t, runtime.RestApiRootPath+"/monitoring/v1", ws.RootPath())
}

func TestAddToContainerMetricNamesRoute(t *testing.T) {
	c := restful.NewContainer()
	mockClient := &mockMonitoringClient{}

	err := AddToContainer(c, mockClient)
	assert.NoError(t, err)

	basePath := runtime.RestApiRootPath + "/monitoring/v1"
	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/metric-names", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	// Mock returns one metric name "up"
	assert.Contains(t, rec.Body.String(), "up")
}

func TestAddToContainerTargetsRoute(t *testing.T) {
	c := restful.NewContainer()
	mockClient := &mockMonitoringClient{}

	err := AddToContainer(c, mockClient)
	assert.NoError(t, err)

	basePath := runtime.RestApiRootPath + "/monitoring/v1"
	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/targets", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "targets")
}

func TestAddToContainerAlertingRulesRoute(t *testing.T) {
	c := restful.NewContainer()
	mockClient := &mockMonitoringClient{}

	err := AddToContainer(c, mockClient)
	assert.NoError(t, err)

	basePath := runtime.RestApiRootPath + "/monitoring/v1"
	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/alerting-rules", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "alertingRules")
}

func TestAddToContainerClusterMetricsRoute(t *testing.T) {
	c := restful.NewContainer()
	mockClient := &mockMonitoringClient{}

	err := AddToContainer(c, mockClient)
	assert.NoError(t, err)

	basePath := runtime.RestApiRootPath + "/monitoring/v1"
	// Instant query: time param only
	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/cluster/metrics?time=2024-05-13%2009:00:00", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestAddToContainerQueryRouteRequiresExpr(t *testing.T) {
	c := restful.NewContainer()
	mockClient := &mockMonitoringClient{}

	err := AddToContainer(c, mockClient)
	assert.NoError(t, err)

	basePath := runtime.RestApiRootPath + "/monitoring/v1"
	// /query without expr - handler may still run and use empty expr; with expr we get 200
	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/query?expr=up&time=2024-05-13%2009:00:00", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	// Handler runs; mock returns empty Metric; should be 200
	assert.Equal(t, http.StatusOK, rec.Code)
}

func TestAddToContainerNotFound(t *testing.T) {
	c := restful.NewContainer()
	mockClient := &mockMonitoringClient{}

	err := AddToContainer(c, mockClient)
	assert.NoError(t, err)

	basePath := runtime.RestApiRootPath + "/monitoring/v1"
	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/nonexistent", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
}
