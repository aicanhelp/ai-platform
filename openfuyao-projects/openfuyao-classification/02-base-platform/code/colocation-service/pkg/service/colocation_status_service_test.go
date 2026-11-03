/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is Licensed under Mulan PSL v2.
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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	colocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
	colocationclientset "openfuyao.com/colocation-service/pkg/client/clientset/versioned/fake"
)

// 创建测试服务器
func createTestServerInStatus() *Server {
	clientset := fake.NewSimpleClientset()
	colocationClientset := colocationclientset.NewSimpleClientset()

	server := &Server{
		client:           clientset,
		colocationclient: colocationClientset,
		ws:               new(restful.WebService),
	}
	server.initColocationStatusRoute()
	return server
}

// 创建测试请求
func createTestRequestInStatus(method, path string) *restful.Request {
	req := httptest.NewRequest(method, path, nil)
	return restful.NewRequest(req)
}

// 创建测试响应
func createTestResponseInStatus() (*httptest.ResponseRecorder, *restful.Response) {
	recorder := httptest.NewRecorder()
	response := restful.NewResponse(recorder)
	response.SetRequestAccepts("application/json")
	return recorder, response
}

// 创建测试节点
func createTestNodeInStatus(clientset *fake.Clientset, name string, isColocated bool) {
	labels := map[string]string{}
	if isColocated {
		labels[colocationv1.ColocationNodeLabel] = "true"
	}

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
	}
	clientset.CoreV1().Nodes().Create(context.TODO(), node, metav1.CreateOptions{})
}

// 创建测试Pod
func createTestPod(clientset *fake.Clientset, name, namespace string, qosType colocationv1.QosType) {
	annotations := map[string]string{}
	if qosType != colocationv1.QosNONE {
		annotations[colocationv1.QosWorkloadAnn] = string(qosType)
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: annotations,
		},
	}
	clientset.CoreV1().Pods(namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
}

// 创建模拟Prometheus服务器
func createMockPrometheusServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(mockPrometheusHandler))
}

func mockPrometheusHandler(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("query")
	response := createPrometheusResponse(query)

	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(response)
	if err != nil {
		panic(err)
	}
}

func createPrometheusResponse(query string) PrometheusResponse {
	response := PrometheusResponse{
		Status: "success",
		Data: struct {
			ResultType string             `json:"resultType"`
			Result     []PrometheusMetric `json:"result"`
		}{
			ResultType: "matrix",
			Result:     []PrometheusMetric{},
		},
	}

	if strings.Contains(query, "node:") {
		// 确保返回的指标数据包含正确的实例名称
		response.Data.Result = []PrometheusMetric{
			{
				Metric: struct {
					Instance string `json:"instance"`
				}{
					Instance: "node1",
				},
				Value: [][]interface{}{
					{float64(time.Now().Unix()), "0.8"},
				},
			},
			{
				Metric: struct {
					Instance string `json:"instance"`
				}{
					Instance: "node2",
				},
				Value: [][]interface{}{
					{float64(time.Now().Unix()), "0.6"},
				},
			},
		}
	} else {
		response.Data.Result = []PrometheusMetric{
			{
				Metric: struct {
					Instance string `json:"instance"`
				}{
					Instance: "cluster",
				},
				Value: [][]interface{}{
					{float64(time.Now().Unix()), "0.75"},
				},
			},
		}
	}

	return response
}

// 创建节点级指标数据
func createNodeMetrics() []PrometheusMetric {
	return []PrometheusMetric{
		{
			Metric: struct {
				Instance string `json:"instance"`
			}{
				Instance: "node1",
			},
			Value: [][]interface{}{
				{time.Now().Unix(), "0.8"},
			},
		},
		{
			Metric: struct {
				Instance string `json:"instance"`
			}{
				Instance: "node2",
			},
			Value: [][]interface{}{
				{time.Now().Unix(), "0.6"},
			},
		},
	}
}

// 创建集群级指标数据
func createClusterMetrics() []PrometheusMetric {
	return []PrometheusMetric{
		{
			Metric: struct {
				Instance string `json:"instance"`
			}{
				Instance: "cluster",
			},
			Value: [][]interface{}{
				{time.Now().Unix(), "0.75"},
			},
		},
	}
}

func TestGetColocationStatus(t *testing.T) {
	// 1. 准备测试环境
	server := createTestServerInStatus()

	// 创建节点
	createTestNodeInStatus(server.client.(*fake.Clientset), "node1", true)  // 混部节点
	createTestNodeInStatus(server.client.(*fake.Clientset), "node2", true)  // 混部节点
	createTestNodeInStatus(server.client.(*fake.Clientset), "node3", false) // 非混部节点

	// 创建Pod
	createTestPod(server.client.(*fake.Clientset), "pod1", "ns1", colocationv1.QosHLS)  // 高优在线
	createTestPod(server.client.(*fake.Clientset), "pod2", "ns1", colocationv1.QosLS)   // 普通在线
	createTestPod(server.client.(*fake.Clientset), "pod3", "ns1", colocationv1.QosBE)   // 离线
	createTestPod(server.client.(*fake.Clientset), "pod4", "ns1", colocationv1.QosNONE) // 普通

	// 2. 创建请求和响应
	request := createTestRequestInStatus("GET", "/colocation-status/")
	recorder, response := createTestResponseInStatus()

	// 3. 执行请求
	server.getColocationStatus(request, response)

	// 4. 验证响应
	assert.Equal(t, http.StatusOK, recorder.Code)

	var result struct {
		Data ColocationStatus `json:"data"`
	}
	err := json.Unmarshal(recorder.Body.Bytes(), &result)
	assert.NoError(t, err)

	// 验证节点统计
	totalNodeCount := 3
	colocationNodeCount := 2
	nonColotaionNodeCount := 1
	assert.Equal(t, totalNodeCount, result.Data.TotalNodeCount)
	assert.Equal(t, colocationNodeCount, result.Data.ColocationNodeCount)
	assert.Equal(t, nonColotaionNodeCount, result.Data.NonColocationNodeCount)

	// 验证Pod统计
	assert.Equal(t, 1, result.Data.PodStat.HLSCount)
	assert.Equal(t, 1, result.Data.PodStat.LSCount)
	assert.Equal(t, 1, result.Data.PodStat.BECount)
	assert.Equal(t, 1, result.Data.PodStat.NormalCount)
}

func TestGetColocationClusterMetrics(t *testing.T) {
	// 1. 准备测试环境
	server := createTestServerInStatus()

	// 创建模拟Prometheus服务器
	promServer := createMockPrometheusServer()
	defer promServer.Close()
	server.prometheusEndpoint = promServer.URL

	// 2. 创建请求和响应
	request := createTestRequestInStatus("GET", "/colocation-metrics/cluster")
	recorder, response := createTestResponseInStatus()

	// 3. 执行请求
	server.getColocationClusterMetrics(request, response)

	// 4. 验证响应
	assert.Equal(t, http.StatusOK, recorder.Code)

	var result struct {
		Data []PrometheusResponse `json:"data"`
	}
	err := json.Unmarshal(recorder.Body.Bytes(), &result)
	assert.NoError(t, err)

	// 验证返回了集群指标数据
	assert.NotEmpty(t, result.Data)
	for _, resp := range result.Data {
		assert.Equal(t, "success", resp.Status)
		assert.NotEmpty(t, resp.MetricName)
		assert.Len(t, resp.Data.Result, 1)
		assert.Equal(t, "cluster", resp.Data.Result[0].Metric.Instance)
	}
}

// 测试4: 节点指标数据过滤和补全
func TestFilterResourceByNodeNames(t *testing.T) {
	// 准备测试数据
	testData := PrometheusResponse{
		Data: struct {
			ResultType string             `json:"resultType"`
			Result     []PrometheusMetric `json:"result"`
		}{
			Result: []PrometheusMetric{
				{Metric: struct {
					Instance string `json:"instance"`
				}(struct{ Instance string }{"node1"}), Value: [][]interface{}{}},
				{Metric: struct {
					Instance string `json:"instance"`
				}(struct{ Instance string }{"node3"}), Value: [][]interface{}{}},
			},
		},
	}

	length := 10
	params := PrometheusQueryParams{
		Start: time.Now().Add(time.Duration(-length) * time.Minute),
		End:   time.Now(),
		Step:  "60s",
	}

	// 测试过滤和补全
	filtered := filterResourceByNodeNames([]string{"node1", "node2"}, testData, params)

	// 验证结果
	resultNodeCount := 2
	assert.Len(t, filtered.Data.Result, resultNodeCount)

	// 验证存在的节点
	found := false
	for _, metric := range filtered.Data.Result {
		if metric.Metric.Instance == "node1" {
			found = true
		}
	}
	assert.True(t, found)

	// 验证补全的节点
	found = false
	for _, metric := range filtered.Data.Result {
		if metric.Metric.Instance == "node2" {
			found = true
			assert.NotEmpty(t, metric.Value) // 补全的节点应该有默认值
		}
	}
	assert.True(t, found)
}

// 测试5: 工作负载类型识别
func TestGetWorkloadQosType(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		expected    colocationv1.QosType
	}{
		{"高优在线", map[string]string{colocationv1.QosWorkloadAnn: "HLS"}, colocationv1.QosHLS},
		{"普通在线", map[string]string{colocationv1.QosWorkloadAnn: "LS"}, colocationv1.QosLS},
		{"离线", map[string]string{colocationv1.QosWorkloadAnn: "BE"}, colocationv1.QosBE},
		{"无注解", map[string]string{}, colocationv1.QosNONE},
		{"未知类型", map[string]string{colocationv1.QosWorkloadAnn: "UNKNOWN"}, colocationv1.QosType("UNKNOWN")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta := metav1.ObjectMeta{Annotations: tt.annotations}
			result := getWorkloadQosType(meta)
			assert.Equal(t, tt.expected, result)
		})
	}
}
