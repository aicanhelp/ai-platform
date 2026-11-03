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
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/thoas/go-funk"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"

	colocationv1 "openfuyao.com/colocation-service/pkg/apis/colocation/v1"
	"openfuyao.com/colocation-service/pkg/utils"
)

func (server *Server) initColocationStatusRoute() {
	ws := server.ws
	// 提供混部统计的管理接口
	ws.Route(ws.GET("/colocation-status/").To(server.getColocationStatus).
		Doc("get colocation statistics"))
	ws.Route(ws.GET("/colocation-metrics/nodes/{node}").To(server.getColocationNodeMetrics).
		Doc("get colocation node metrics"))
	ws.Route(ws.GET("/colocation-metrics/cluster").To(server.getColocationClusterMetrics).
		Doc("get colocation cluster metrics"))
}

// ColocationStatus 混部节点以及pod的详情
type ColocationStatus struct {
	// TotalNodeCount 总节点数量
	TotalNodeCount int `json:"totalNodeCount"`
	// ColocationNodeCount 混部节点数量
	ColocationNodeCount int `json:"colocationNodeCount"`
	// NonColocationNodeCount 非混部节点数量
	NonColocationNodeCount int `json:"nonColocationNodeCount"`
	// PodStat pod数量统计
	PodStat WorkloadStatus `json:"podCount"`
}

// WorkloadStatus 工作负载统计信息
type WorkloadStatus struct {
	// NormalCount 普通工作负载的数量
	NormalCount int `json:"normalCount"`
	// HLSCount 高优在线工作负载的数量
	HLSCount int `json:"hlsCount"`
	// LSCount 普通在线工作负载的数量
	LSCount int `json:"lsCount"`
	// BECount 离线工作负载的数量
	BECount int `json:"beCount"`
}

// PrometheusResponse 普罗响应信息
type PrometheusResponse struct {
	Status     string `json:"status"`
	MetricName string `json:"metricName"`
	Data       struct {
		ResultType string             `json:"resultType"`
		Result     []PrometheusMetric `json:"result"`
	} `json:"data"`
}

// PrometheusMetric 普罗metric信息
type PrometheusMetric struct {
	Metric struct {
		Instance string `json:"instance"`
	} `json:"metric"`
	Value [][]interface{} `json:"values"`
}

// PrometheusQueryParams promql query params
type PrometheusQueryParams struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
	Step  string    `json:"step"`
}

// defaultQuerySpan the default query span for promql in minutes
const defaultStartSpan = 10

// NewDefaultPrometheusQueryParams init default PrometheusQueryParams
func NewDefaultPrometheusQueryParams() *PrometheusQueryParams {
	currentTime := time.Now()
	return &PrometheusQueryParams{
		Start: currentTime.Add(-defaultStartSpan * time.Minute),
		End:   currentTime,
		Step:  "60s",
	}
}

// getColocationStatus 查询混部统计数据：包含（混部节点数量，各级Qos节点数量）
func (server *Server) getColocationStatus(request *restful.Request, response *restful.Response) {
	colocationStat := ColocationStatus{}
	// 10.查询所有节点数据
	nodes, err := server.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.Error(err)
		response.WriteHeaderAndEntity(http.StatusInternalServerError, utils.NewResponseResultWithError(
			AppCode, FeatureCode, err))
		return
	}

	colocationStat.TotalNodeCount = len(nodes.Items)

	for _, node := range nodes.Items {
		// 如果存在混部标签则认为是混部节点
		if value, ok := node.Labels[colocationv1.ColocationNodeLabel]; ok && value == "true" {
			colocationStat.ColocationNodeCount++
		}
	}

	// 普通节点数量 = 总节点 - 混部节点
	colocationStat.NonColocationNodeCount = colocationStat.TotalNodeCount - colocationStat.ColocationNodeCount

	// 20.查询各种类型的工作负载的普通类型、混部在线与离线的Pod数量
	err = server.getPodStatus(&colocationStat.PodStat)
	if err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError, utils.NewResponseResultWithError(
			AppCode, FeatureCode, err))
		return
	}

	// 30.返回结果
	response.WriteHeaderAndEntity(http.StatusOK, utils.NewResponseResultOk(colocationStat))
}

func (server *Server) getPodStatus(stat *WorkloadStatus) error {
	pods, err := server.client.CoreV1().Pods("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		klog.Error(err)
		return err
	}

	for _, pod := range pods.Items {
		qosType := getWorkloadQosType(pod.ObjectMeta)
		workloadCount(qosType, stat)
	}

	return nil
}

func (server *Server) getColocationClusterMetrics(request *restful.Request, response *restful.Response) {
	clusterMetrics, err := server.queryClusterResource()
	if err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError, utils.NewResponseResultWithError(
			AppCode, FeatureCode, err))
		return
	}

	response.WriteHeaderAndEntity(http.StatusOK, utils.NewResponseResultOk(clusterMetrics))
}

func (server *Server) getColocationNodeMetrics(request *restful.Request, response *restful.Response) {
	// 获取nodes
	rawNodeParam := request.PathParameter("node") // eg: "node1|node2|node3"
	nodes := strings.Split(rawNodeParam, "|")
	nodeMetrics, err := server.queryNodeResource(nodes)
	if err != nil {
		response.WriteHeaderAndEntity(http.StatusInternalServerError, utils.NewResponseResultWithError(
			AppCode, FeatureCode, err))
		return
	}

	response.WriteHeaderAndEntity(http.StatusOK, utils.NewResponseResultOk(nodeMetrics))
}

func (server *Server) queryClusterResource() ([]PrometheusResponse, error) {
	var response []PrometheusResponse
	prometheusQueryParams := NewDefaultPrometheusQueryParams()
	for _, metricName := range clusterLevelMetrics {
		query := prometheusQLSet[metricName]
		resp, err := queryPrometheus(server.prometheusEndpoint, query, *prometheusQueryParams)
		if err != nil {
			klog.Error(err)
			return nil, err
		}
		resp.MetricName = metricName
		response = append(response, resp)

	}
	return response, nil
}

func (server *Server) queryNodeResource(nodes []string) ([]PrometheusResponse, error) {
	var response []PrometheusResponse
	prometheusQueryParams := NewDefaultPrometheusQueryParams()
	for _, metricName := range nodeLevelMetrics {
		query := prometheusQLSet[metricName]
		resp, err := queryPrometheus(server.prometheusEndpoint, query, *prometheusQueryParams)
		if err != nil {
			klog.Error(err)
			return nil, err
		}
		resp.MetricName = metricName
		resp = filterResourceByNodeNames(nodes, resp, *prometheusQueryParams)
		response = append(response, resp)

	}
	return response, nil
}

func getWorkloadQosType(objectMeta metav1.ObjectMeta) colocationv1.QosType {
	qos, ok := objectMeta.Annotations[colocationv1.QosWorkloadAnn]
	if !ok {
		return colocationv1.QosNONE
	}

	return colocationv1.QosType(qos)
}

func workloadCount(wlType colocationv1.QosType, stat *WorkloadStatus) {
	switch wlType {
	case colocationv1.QosHLS:
		stat.HLSCount++
	case colocationv1.QosLS:
		stat.LSCount++
	case colocationv1.QosBE:
		stat.BECount++
	default:
		stat.NormalCount++
	}
}

func filterResourceByNodeNames(nodeNames []string, resp PrometheusResponse,
	params PrometheusQueryParams) PrometheusResponse {
	filteredResp := resp
	filteredResp.Data.Result = make([]PrometheusMetric, 0)
	addedNodes := make([]string, 0)

	// filter unselected nodes
	for _, result := range resp.Data.Result {
		if funk.ContainsString(nodeNames, result.Metric.Instance) {
			addedNodes = append(addedNodes, result.Metric.Instance)
			filteredResp.Data.Result = append(filteredResp.Data.Result, result)
		}
	}

	// complete discarded nodes
	if len(nodeNames) != len(addedNodes) {
		for _, nodeName := range nodeNames {
			if funk.ContainsString(addedNodes, nodeName) {
				continue
			}
			emptyResult := makeDefaultEmptyMetric(nodeName, params)
			filteredResp.Data.Result = append(filteredResp.Data.Result, emptyResult)
		}
	}
	return filteredResp
}

func makeDefaultEmptyMetric(nodeName string, params PrometheusQueryParams) PrometheusMetric {
	emptyMetric := PrometheusMetric{
		Metric: struct {
			Instance string `json:"instance"`
		}{Instance: nodeName},
		Value: make([][]interface{}, 0),
	}

	timestamps, err := generateTimestamps(params)
	if err != nil {
		return emptyMetric
	}

	for _, timestamp := range timestamps {
		emptyPoint := []interface{}{timestamp, "0"}
		emptyMetric.Value = append(emptyMetric.Value, emptyPoint)
	}

	return emptyMetric
}

func generateTimestamps(params PrometheusQueryParams) ([]int64, error) {
	step, err := time.ParseDuration(params.Step)
	if err != nil {
		return nil, fmt.Errorf("invalid step duration: %w", err)
	}

	var result []int64
	for ts := params.Start; !ts.After(params.End); ts = ts.Add(step) {
		result = append(result, ts.Unix())
	}
	return result, nil
}

// buildPrometheusRequestURL 构建Prometheus查询URL
func buildPrometheusRequestURL(prometheusURL, query string, params PrometheusQueryParams) (string, error) {
	u, err := url.Parse(prometheusURL)
	if err != nil {
		return "", fmt.Errorf("failed to parse URL: %v", err)
	}

	q := u.Query()
	q.Set("query", query)
	q.Set("start", fmt.Sprintf("%d", params.Start.Unix()))
	q.Set("end", fmt.Sprintf("%d", params.End.Unix()))
	q.Set("step", params.Step)
	u.RawQuery = q.Encode()

	return u.String(), nil
}

// fillMissingTimestamps 填充缺失的时间戳数据点
// fillMissingTimestamps 填充缺失的时间戳数据点
func fillMissingTimestamps(metrics []PrometheusMetric, params PrometheusQueryParams) ([]PrometheusMetric, error) {
	expectedTimestamps, err := generateTimestamps(params)
	if err != nil {
		return nil, fmt.Errorf("failed to generate timestamps: %v", err)
	}

	for i := range metrics {
		valueMap := createValueMap(metrics[i].Value)
		metrics[i].Value = buildCompleteValues(expectedTimestamps, valueMap)
	}

	return metrics, nil
}

// createValueMap 从原始数据点创建时间戳到值的映射
func createValueMap(points [][]interface{}) map[int64]string {
	pointLen := 2
	valueMap := make(map[int64]string)
	for _, point := range points {
		if len(point) < pointLen {
			continue
		}
		if ts, ok := point[0].(float64); ok {
			valueMap[int64(ts)] = fmt.Sprintf("%v", point[1])
		}
	}
	return valueMap
}

// buildCompleteValues 构建完整的时间序列数据点，填充缺失值
func buildCompleteValues(timestamps []int64, valueMap map[int64]string) [][]interface{} {
	var newValues [][]interface{}
	for _, ts := range timestamps {
		newValues = append(newValues, createDataPoint(ts, valueMap))
	}
	return newValues
}

// createDataPoint 创建单个数据点，处理存在和不存在的情况
func createDataPoint(ts int64, valueMap map[int64]string) []interface{} {
	if val, exists := valueMap[ts]; exists {
		return []interface{}{float64(ts), val}
	}
	return []interface{}{float64(ts), ""}
}

// queryPrometheus 查询Prometheus并返回结果
func queryPrometheus(prometheusURL, query string, params PrometheusQueryParams) (PrometheusResponse, error) {
	var result PrometheusResponse

	// 1. 构建请求URL
	requestURL, err := buildPrometheusRequestURL(prometheusURL, query, params)
	if err != nil {
		return result, err
	}

	// 2. 发送HTTP请求
	resp, err := http.Get(requestURL)
	if err != nil {
		return result, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	// 3. 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return result, fmt.Errorf("failed to read response body: %v", err)
	}

	// 4. 解析JSON数据
	if err := json.Unmarshal(body, &result); err != nil {
		return result, fmt.Errorf("failed to unmarshal JSON: %v", err)
	}

	if result.Status != "success" {
		return result, fmt.Errorf("prometheus query failed: %s", result.Status)
	}

	// 5. 处理时间序列数据，填充缺失的时间点
	result.Data.Result, err = fillMissingTimestamps(result.Data.Result, params)
	if err != nil {
		return result, err
	}

	return result, nil
}
