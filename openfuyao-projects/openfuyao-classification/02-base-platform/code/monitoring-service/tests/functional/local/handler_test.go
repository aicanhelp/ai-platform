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

// Functional tests for the monitoring handler (handler.go). Only the Prometheus HTTP API
// is mocked via a fake server; the real handler and real Prometheus client are used.

package local

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/monitoring-service/pkg/api/monitoring/v1"
	"openfuyao.com/monitoring-service/pkg/client/prometheus"
)

const basePath = "/rest/monitoring/v1"

func mockPrometheusAPIServer() *httptest.Server {
	mux := http.NewServeMux()

	instantBody := `{"status":"success","data":{"resultType":"vector","result":[{"metric":{"__name__":"up"},"value":[1714995000,"1"]}]}}`
	rangeBody := `{"status":"success","data":{"resultType":"matrix","result":[{"metric":{"__name__":"up"},"values":[[1714995000,"1"],[1714995015,"1"]]}]}}`

	mux.HandleFunc("/api/v1/query", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(instantBody))
	})
	mux.HandleFunc("/api/v1/query_range", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(rangeBody))
	})
	mux.HandleFunc("/api/v1/targets", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"success","data":{"activeTargets":[{"discoveredLabels":{"job":"test"},"scrapeUrl":"http://prometheus:9090","health":"up"}]}}`))
	})
	mux.HandleFunc("/api/v1/rules", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"success","data":{"groups":[{"name":"test.rules","rules":[{"state":"inactive","name":"TestAlert","query":"up","labels":{"severity":"warning"},"annotations":{"description":"test","summary":"test"},"type":"alerting"}]}]}}`))
	})
	mux.HandleFunc("/api/v1/label/__name__/values", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"success","data":["up","cluster_cpu_usage"]}`))
	})

	return httptest.NewServer(mux)
}

// newTestContainer creates a test container with real handler, real MonitoringOperator, real prometheus client,
// but with a mock Prometheus API server. This exercises the full call chain with only the Prometheus API mocked.
//
// Route Registration Flow:
//   v1.AddToContainer calls:
//     - runtime.NewRestWebService(apiGroupVersion) → creates WebService with path "/rest/monitoring/v1"
//     - addQueryRoute(ws, h) → registers route: GET("/query").To(h.handleQuery)
//     - c.Add(ws) → adds WebService to restful.Container
//
// When c.ServeHTTP(rec, req) is called:
//   - Container matches request path "/rest/monitoring/v1/query" to registered route
//   - Calls handler.handleQuery(req, resp)
func newTestContainer(t *testing.T) (*restful.Container, *httptest.Server) {
	srv := mockPrometheusAPIServer()
	// Real prometheus client that will call the mock API server
	client, err := prometheus.NewPrometheus(&prometheus.Options{Endpoint: srv.URL})
	assert.NoError(t, err)
	c := restful.NewContainer()
	// v1.AddToContainer registers routes: GET("/query").To(h.handleQuery), etc.
	// The container will route HTTP requests to these handler methods
	err = v1.AddToContainer(c, client)
	assert.NoError(t, err)
	return c, srv
}

func TestFunctionalHandleMetricNames(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/metric-names", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "names")
	assert.Contains(t, rec.Body.String(), "up")
	assert.Contains(t, rec.Body.String(), "cluster_cpu_usage")
}

func TestFunctionalHandleTargetsDiscovery(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/targets", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "targets")
}

func TestFunctionalHandleAlertingRules(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/alerting-rules", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "alertingRules")
	assert.Contains(t, rec.Body.String(), "TestAlert")
}

func TestFunctionalHandleClusterMetricsInstant(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/cluster/metrics?time=2024-05-13%2009:00:00", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "result")
}

func TestFunctionalHandleClusterMetricsRange(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/cluster/metrics?start=2024-05-13%2008:00:00&end=2024-05-13%2009:00:00&step=15s"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "results")
}

func TestFunctionalHandleNodeMetrics(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/nodes/node-1/metrics?time=2024-05-13%2009:00:00", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandleDeploymentMetrics(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/namespaces/default/deployments/my-deploy/metrics?time=2024-05-13%2009:00:00"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandleStatefulsetMetrics(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/namespaces/default/statefulsets/my-sts/metrics?time=2024-05-13%2009:00:00"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandleDaemonsetMetrics(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/namespaces/default/daemonsets/my-ds/metrics?time=2024-05-13%2009:00:00"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandlePodMetrics(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/namespaces/default/pods/my-pod/metrics?time=2024-05-13%2009:00:00"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandleContainerMetrics(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/namespaces/default/pods/my-pod/containers/my-ctr/metrics?time=2024-05-13%2009:00:00"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandleControlPlaneComponentsMetrics(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/control-plane-components/etcd?time=2024-05-13%2009:00:00&instance=10.0.0.1:2379"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandleNodeComponentsMetricsQuery(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/node-components/kubelet?time=2024-05-13%2009:00:00&instance=10.0.0.1:10250"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandleAddonsComponentsMetrics(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/add-ons-components/coredns?time=2024-05-13%2009:00:00&instance=10.0.0.1:9153"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
}

func TestFunctionalHandleQueryInstant(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/query?expr=up&time=2024-05-13%2009:00:00", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "result")
	assert.Contains(t, rec.Body.String(), "vector") // Instant query returns vector type
}

func TestFunctionalHandleQueryRange(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	url := "http://example.com" + basePath + "/query?expr=up&start=2024-05-13%2008:00:00&end=2024-05-13%2009:00:00&step=15s"
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusOK, rec.Code)
	assert.Contains(t, rec.Header().Get("Content-Type"), "application/json")
	assert.Contains(t, rec.Body.String(), "result")
	assert.Contains(t, rec.Body.String(), "matrix") // Range query returns matrix type
}

func TestFunctionalNotFound(t *testing.T) {
	c, srv := newTestContainer(t)
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "http://example.com"+basePath+"/unknown", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	assert.Equal(t, http.StatusNotFound, rec.Code)
}
