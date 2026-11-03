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

// Package v1 provides the HTTP request handling and interaction logic for
// a logging server designed to work within Kubernetes environments. This package
// encapsulates functionalities for making and processing HTTP requests
// to Loki, a horizontally-scalable, highly-available, multi-tenant log aggregation
// system. It aims to facilitate logging data retrieval and analysis by
// providing structured access to logs stored in Loki.
//
// The package defines the APIServer and Handler types that manage the HTTP server
// and request routing respectively. The APIServer serves as the entry point for
// incoming HTTP requests, whereas Handler implements the logic for processing
// these requests, including querying Loki for log data based on various
// parameters such as filename, time range, and log level.
//
// Key Components:
//
//   - HTTPRequest: A struct that encapsulates details of an HTTP request, including
//     URL, method, headers, and body. It serves as a foundational element for
//     constructing requests to Loki.
//
//   - Handler: The core struct that handles incoming requests, routes them to
//     appropriate processing functions, and interacts with Loki to fetch or
//     manipulate log data. It maintains a map of predefined Loki query configurations
//     and a dictionary mapping human-readable time units to their respective
//     durations for flexible time range queries.
//
//   - LogData: A struct designed to unmarshal JSON responses from Loki, specifically
//     targeting the data structure returned by Loki's series and query_range APIs.
//
// Functionalities:
//
//   - Dynamic Loki Query Construction: Allows building and executing complex Loki
//     queries based on request parameters, enabling clients to perform detailed log
//     searches and analysis.
//
//   - Time Range Calculation: Offers utilities to calculate start and end times for
//     queries based on either specific timestamps or relative time units (e.g., "last 5 minutes").
//
//   - API Endpoints: Defines RESTful endpoints for interacting with the logging server,
//     including endpoints for listing available log sources and querying logs based
//     on various filters.
//
// Usage:
// To use this package, create an instance of the Handler struct via the New function,
// then call its methods to handle incoming requests or interact with Loki. The
// package's functionality is encapsulated in the Handler type, making it the primary
// interface for clients.
//
// Note:
// This package assumes that Loki is accessible via HTTP and that clients have
// necessary permissions to query Loki. It also relies on external packages such
// as "github.com/emicklei/go-restful/v3" for RESTful routing and
// "k8s.io/apimachinery/pkg/util/runtime" for Kubernetes-related utilities.
//
// The package is designed with extensibility in mind, allowing for future
// enhancements such as additional Loki query types, improved error handling,
// and richer response parsing.
package v1

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/emicklei/go-restful/v3"
	"github.com/pkg/errors"
	"gopkg.in/yaml.v3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"

	"openfuyao/logging-operator/pkg/server"
	"openfuyao/logging-operator/pkg/tools"
)

// New initializes and returns a new instance of Handler with predefined HTTPRequest
// configurations and time unit mappings.
func New() *Handler {
	podClient, configmapClient, scretsClient := newKubernetesAPI("loki")
	handler := &Handler{
		RequestsMap:     make(map[string]*HTTPRequest),
		timeUnitsDict:   initializeTimeUnits(),
		PodClient:       podClient,
		ConfigmapClient: configmapClient,
		SecretsClient:   scretsClient,
	}
	initializeRequests(handler)
	return handler
}

func newKubernetesAPI(namespace string) (corev1.PodInterface, corev1.ConfigMapInterface, corev1.SecretInterface) {
	config, err := rest.InClusterConfig()

	if err != nil {
		panic(err.Error())
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, nil, nil
	}

	podsClient := clientset.CoreV1().Pods(namespace)
	configmapClient := clientset.CoreV1().ConfigMaps(namespace)
	scretsClient := clientset.CoreV1().Secrets(namespace)
	return podsClient, configmapClient, scretsClient
}

func initializeRequests(h *Handler) {
	baseRequests := map[string]string{
		"getFileNameList":  "/loki/api/v1/label/filename/values",
		"filterRequest":    "/loki/api/v1/query_range",
		"getResourceList":  "/loki/api/v1/series",
		"getConfigmapList": "/api/kubernetes/api/v1/namespaces/loki/configmaps/loki-promtail",
		"getPodList":       "/api/kubernetes/api/v1/namespaces/loki/pods",
		"getPod":           "/api/kubernetes/api/v1/namespaces/loki/pods/",
		"patchConfigMap":   "/api/kubernetes/api/v1/namespaces/loki/configmaps/loki-promtail", // New entry for PATCH request
		"getAlertingRules": "/loki/api/v1/rules",
	}

	for key, path := range baseRequests {
		baseURL := os.Getenv("LOKI_HOSTS")
		if key == "patchConfigMap" {
			h.RequestsMap[key] = &HTTPRequest{
				URL:    baseURL + path,
				Method: "PATCH",
				Headers: map[string]string{
					ContentType: "application/merge-patch+json", // Specific content type for patch operations
				},
				Body: []byte(`{
					"data": {
						"promtail.yaml": "updated configuration here..."
					}
				}`),
			}
		} else {
			h.RequestsMap[key] = &HTTPRequest{
				URL:    baseURL + path,
				Method: "GET",
				Headers: map[string]string{
					ContentType: ContentTypeJSON,
				},
			}
		}
	}
}

func sendHTTPRequestResourceOnly(req *HTTPRequest, fullURL string) (map[string]interface{}, error) {
	httpReq, err := http.NewRequest(req.Method, fullURL, bytes.NewBuffer(req.Body))
	if err != nil {
		errorMessage := fmt.Sprintf("failed to create HTTP request: %v", err)
		tools.LogError(errorMessage)
		return nil, errors.New(errorMessage)
	}
	for key, value := range req.Headers {
		httpReq.Header.Set(key, value)
	}
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError sending request to Loki: %v", err)
		tools.LogError(errorMessage)
		return nil, errors.New(errorMessage)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError reading response body: %v", err)
		tools.LogError(errorMessage)
		return nil, errors.New(errorMessage)
	}
	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		errorMessage := fmt.Sprintf("LogError parsing JSON response: %v", err)
		tools.LogError(errorMessage)
		return nil, errors.New(errorMessage)
	}
	return result, nil
}

func (h *Handler) getResourceList(request *restful.Request, response *restful.Response) {
	req, exists := h.RequestsMap["getResourceList"]
	if !exists {
		errorMessage := fmt.Sprintf("Request configuration for 'getFileNameList' not found." +
			" Please ensure it is defined in the RequestsMap.")
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusNotFound, "Resource not found")
	}
	namespace := server.GetQueryParamOrDefault(request, server.Namespace, "")
	pod := server.GetQueryParamOrDefault(request, server.PodName, "")
	container := server.GetQueryParamOrDefault(request, server.Container, "")
	params := QueryResourcesParams{
		Namespace: namespace,
		Pod:       pod,
		Container: container,
	}
	start, end := calculateQueryTimesUnix("504h")
	queryParams := url.Values{}
	queryParams.Add("start", start)
	queryParams.Add("end", end)
	fullURL := fmt.Sprintf("%s?%s", req.URL, queryParams.Encode())
	result, err := sendHTTPRequestResourceOnly(req, fullURL)
	if err != nil {
		handleError(response, "Failed to send http request to loki: ", err)
	}
	ns, po, co := getList(result, params)
	resources := map[string][]string{
		"namespaces": ns,
		"pods":       po,
		"containers": co,
	}
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: resources,
	}
	err = response.WriteEntity(apiResponse)
	if err != nil {
		handleError(response, "Failed to write entity", err)
	}
}
func initializeTimeUnits() map[string]string {
	return map[string]string{
		"5min":   "5m",
		"15min":  "15m",
		"30min":  "30m",
		"1hour":  "1h",
		"2hour":  "2h",
		"6hour":  "6h",
		"12hour": "12h",
		"1day":   "24h",
		"2day":   "48h",
		"3day":   "72h",
		"7day":   "168h",
	}
}

func calculateQueryTimesUnix(duration string) (string, string) {
	var startTime, endTime time.Time
	d, err := time.ParseDuration(duration)
	if err != nil {
		errorMessage := fmt.Sprintf("Invalid duration format: %s."+
			" Please provide a valid duration string (e.g., '300ms', '1.5h' "+
			"or '2h45m'). LogError: %v", duration, err)
		tools.LogError(errorMessage)
		return "0", "0"
	}
	now := time.Now()
	endTime = now
	startTime = now.Add(-d)
	fmt.Println("start time: ", startTime.Unix(), "end time: ", endTime.Unix())
	return fmt.Sprintf("%d", startTime.Unix()), fmt.Sprintf("%d", endTime.Unix())
}

func (h *Handler) getLokiFileName(request *restful.Request, response *restful.Response) {
	req, exists := h.RequestsMap["getFileNameList"]
	if !exists {
		errorMessage := fmt.Sprintf("Request configuration for 'getFileNameList' not found." +
			" Please ensure it is defined in the RequestsMap.")
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusNotFound, "Resource not found")
		return
	}
	start, end := calculateQueryTimesUnix("500h")
	queryParams := url.Values{}
	queryParams.Add("start", start)
	queryParams.Add("end", end)
	fullURL := fmt.Sprintf("%s?%s", req.URL, queryParams.Encode())
	result, err := sendHTTPRequestResourceOnly(req, fullURL)
	if err != nil {
		handleError(response, "Failed to send http request to loki component", err)
	}
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: result,
	}
	fmt.Println(apiResponse)
	err = response.WriteEntity(apiResponse)
	fmt.Println(response.StatusCode())
	if err != nil {
		errorMessage := fmt.Sprintf("Request configuration for 'getFileNameList' not found." +
			" Please ensure it is defined in the RequestsMap.")
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusNotFound, errorMessage)
	}

}

func queryParamBuilder(startTime, endTime, sizes, query string) url.Values {
	queryParams := url.Values{}
	queryParams.Add("start", startTime)
	queryParams.Add("end", endTime)
	queryParams.Add("limit", sizes)
	queryParams.Add("query", query)
	return queryParams
}

func lokiQueryBuilder(params LokiQueryParams) string {
	// 初始化查询语句的基础部分
	var queryParts []string

	// 根据参数添加查询条件
	if params.Namespace != "" {
		queryParts = append(queryParts, fmt.Sprintf(`namespace="%s"`, params.Namespace))
	}
	if params.Pod != "" {
		queryParts = append(queryParts, fmt.Sprintf(`pod="%s"`, params.Pod))
	}
	if params.Container != "" {
		queryParts = append(queryParts, fmt.Sprintf(`container="%s"`, params.Container))
	}
	if params.Filename != "" {
		queryParts = append(queryParts, fmt.Sprintf(`filename="%s"`, params.Filename))
	}

	// 构建基础查询表达式
	baseQuery := "{" + strings.Join(queryParts, ", ") + "}"

	// 添加关键词过滤，如果有的话
	if params.Keyword != "" {
		baseQuery += fmt.Sprintf(` |= "%s"`, params.Keyword)
	}

	// 添加日志级别过滤，如果有的话
	// 输入的日志等级参数有五个可能的级别：warning，error， debug， info， critical
	if params.LogLevel != "" {
		regex := logLevelRegex[params.LogLevel]
		baseQuery += fmt.Sprintf(` |~ "%s"`, regex)
	}
	return baseQuery
}

func sendHTTPRequest(req *HTTPRequest, fullURL string) (Response, error) {
	httpReq, err := http.NewRequest(req.Method, fullURL, bytes.NewBuffer(req.Body))
	var data Response
	if err != nil {
		errorMessage := fmt.Sprintf("failed to create HTTP request: %v", err)
		tools.LogError(errorMessage)
		return data, errors.New(errorMessage)
	}
	for key, value := range req.Headers {
		httpReq.Header.Set(key, value)
	}
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError sending request to Loki:")
		tools.LogError(errorMessage)
		return data, errors.New(errorMessage)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError reading response body:")
		tools.LogError(errorMessage)
		return data, errors.New(errorMessage)
	}
	if err := json.Unmarshal(body, &data); err != nil {
		errorMessage := fmt.Sprintf("LogError parsing JSON response:")
		tools.LogError(errorMessage)
		return data, errors.New(errorMessage)
	}
	return data, nil
}

func (h *Handler) lokiFilterSearch(request *restful.Request, response *restful.Response) {
	filename := server.GetQueryParamOrDefault(request, server.FileName, "")
	keyword := server.GetQueryParamOrDefault(request, server.Keyword, "")
	loglevel := server.GetQueryParamOrDefault(request, server.Level, "")
	params := LokiQueryParams{
		Filename: filename,
		Keyword:  keyword,
		LogLevel: loglevel,
	}
	h.performLokiSearch(request, response, params)
}

func (h *Handler) lokiNamespaceSearch(request *restful.Request, response *restful.Response) {
	namespace := request.PathParameter("namespace")
	keyword := server.GetQueryParamOrDefault(request, server.Keyword, "")
	loglevel := server.GetQueryParamOrDefault(request, server.Level, "")
	params := LokiQueryParams{
		Namespace: namespace,
		Keyword:   keyword,
		LogLevel:  loglevel,
	}
	h.performLokiSearch(request, response, params)
}

func (h *Handler) lokiPodSearch(request *restful.Request, response *restful.Response) {
	namespace := request.PathParameter("namespace")
	pod := request.PathParameter("pod")
	keyword := server.GetQueryParamOrDefault(request, server.Keyword, "")
	loglevel := server.GetQueryParamOrDefault(request, server.Level, "")
	params := LokiQueryParams{
		Namespace: namespace,
		Pod:       pod,
		Keyword:   keyword,
		LogLevel:  loglevel,
	}
	h.performLokiSearch(request, response, params)
}

func (h *Handler) lokiContainerSearch(request *restful.Request, response *restful.Response) {
	namespace := request.PathParameter("namespace")
	pod := request.PathParameter("pod")
	container := request.PathParameter("container")
	keyword := server.GetQueryParamOrDefault(request, server.Keyword, "")
	loglevel := server.GetQueryParamOrDefault(request, server.Level, "")
	params := LokiQueryParams{
		Namespace: namespace,
		Pod:       pod,
		Container: container,
		Keyword:   keyword,
		LogLevel:  loglevel,
	}
	h.performLokiSearch(request, response, params)
}

func (h *Handler) performLokiSearch(request *restful.Request, response *restful.Response, params LokiQueryParams) {
	req, exists := h.RequestsMap["filterRequest"]
	if !exists {
		response.WriteErrorString(http.StatusNotFound, "Request configuration"+
			" for 'filterRequest' not found. Please ensure it is defined in the RequestsMap.")
		return
	}

	timePeriod := server.GetQueryParamOrDefault(request, server.TimePeriod, "")
	start := server.GetQueryParamOrDefault(request, server.StartTime, "1712460747")
	end := server.GetQueryParamOrDefault(request, server.EndTime, "1712461647")
	sizes := server.GetQueryParamOrDefault(request, server.Sizes, "10")
	if timePeriod != "" {
		duration := h.timeUnitsDict[timePeriod]
		start, end = calculateQueryTimesUnix(duration)
	}

	query := lokiQueryBuilder(params)
	queryParams := queryParamBuilder(start, end, sizes, query)
	tools.LogInfo("Query: ", query)
	fullURL := fmt.Sprintf("%s?%s", req.URL, queryParams.Encode())

	result, err := sendHTTPRequest(req, fullURL)
	if err != nil {
		handleError(response, "Failed to send http request to loki component", err)
		return
	}
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: result,
	}
	err = response.WriteEntity(apiResponse)
	if err != nil {
		handleError(response, "Failed to write entity: %v", err)
	}
}

func getList(result map[string]interface{}, params QueryResourcesParams) (namespaces, pods, containers []string) {
	nsSet := make(map[string]bool)
	podSet := make(map[string]bool)
	containerSet := make(map[string]bool)

	sources, exists := result["data"].([]interface{})
	if !exists {
		errorMessage := fmt.Sprintf("Request configuration for 'data' not found." +
			" Please ensure it is defined in the Json output.")
		tools.LogError(errorMessage)
	}
	for _, item := range sources {
		entry, ok := item.(map[string]interface{})
		if !ok {
			errorMessage := fmt.Sprintf("type assertion failed for item: expected map[string]interface{}, got %T", item)
			tools.LogError(errorMessage)
			continue
		}
		namespace, nOk := entry["namespace"].(string)
		pod, pOk := entry["pod"].(string)
		container, cOk := entry["container"].(string)
		if nOk && pOk && cOk {
			tools.LogInfo("resource exists")
		}
		nsCondition := (params.Namespace == "" || (nOk && namespace == params.Namespace))
		podCondition := (params.Pod == "" || (pOk && pod == params.Pod))
		containerCondition := (params.Container == "" || (cOk && container == params.Container))
		if nsCondition && podCondition && containerCondition {
			if nOk {
				nsSet[namespace] = true
			}
			if pOk {
				podSet[pod] = true
			}
			if cOk {
				containerSet[container] = true
			}
		}
	}
	for ns := range nsSet {
		namespaces = append(namespaces, ns)
	}
	for p := range podSet {
		pods = append(pods, p)
	}
	for c := range containerSet {
		containers = append(containers, c)
	}

	return namespaces, pods, containers
}

// findContainerByName
func findContainerByName(containers []Container, containerName string) (*Container, error) {
	for _, container := range containers {
		if container.Name == containerName {
			return &container, nil
		}
	}
	return nil, fmt.Errorf("container %s not found", containerName)
}

// extractReadOnlyMounts
func extractReadOnlyMounts(container *Container) []VolumeMount {
	var mounts []VolumeMount
	for _, mount := range container.VolumeMounts {
		if mount.ReadOnly {
			mounts = append(mounts, mount)
		}
	}
	return mounts
}

// parsePodDetail
func parsePodDetail(respBody []byte, containerName string) ([]VolumeMount, error) {
	var podDetail PodDetail
	if err := json.Unmarshal(respBody, &podDetail); err != nil {
		return nil, fmt.Errorf("error parsing JSON response: %v", err)
	}

	container, err := findContainerByName(podDetail.Spec.Containers, containerName)
	if err != nil {
		errorMessage := fmt.Sprintf("Fail to find Pod name" +
			" Please ensure it is defined in the Json output.")
		tools.LogError(errorMessage)
		return nil, err
	}

	return extractReadOnlyMounts(container), nil
}

func (h *Handler) sendHTTPRequest(reqKey, podName string) (*http.Response, error) {
	req, exists := h.RequestsMap[reqKey]
	if !exists {
		errorMessage := fmt.Sprintf("Fail to load requestMap" +
			" Please ensure it is defined in the Json output.")
		tools.LogError(errorMessage)
		return nil, fmt.Errorf("request configuration for '%s' not found", reqKey)
	}

	httpReq, err := http.NewRequest(req.Method, req.URL+podName, bytes.NewBuffer(req.Body))
	if err != nil {
		errorMessage := fmt.Sprintf("Fail to build new request" +
			" Please ensure it is defined in the Json output.")
		tools.LogError(errorMessage)
		return nil, err
	}

	for key, value := range req.Headers {
		httpReq.Header.Set(key, value)
	}

	client := &http.Client{}
	return client.Do(httpReq)
}

func (h *Handler) getAlertingRules(request *restful.Request, response *restful.Response) {
	req, exists := h.RequestsMap["getAlertingRules"]
	if !exists {
		errorMessage := fmt.Sprintf("Request configuration for 'getAlertingRules' not found." +
			" Please ensure it is defined in the RequestsMap.")
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusNotFound, "Resource not found")
		return
	}
	httpReq, err := http.NewRequest(req.Method, req.URL, bytes.NewBuffer(req.Body))
	if err != nil {
		errorMessage := fmt.Sprintf("LogError sending request to Loki:")
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusNotFound, errorMessage)
		return
	}
	for key, value := range req.Headers {
		httpReq.Header.Set(key, value)
	}
	client := &http.Client{}
	resp, err := client.Do(httpReq)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError sending request to Loki:")
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusBadRequest, errorMessage)
		return
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError reading response body:")
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusBadRequest, errorMessage)
		return
	}
	var ruleFile RuleFile
	err = yaml.Unmarshal(body, &ruleFile)
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: ruleFile,
	}
	err = response.WriteEntity(apiResponse)
	if err != nil {
		errorMessage := fmt.Sprintf("Failed to write entity: %v", err)
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusBadRequest, errorMessage)
		return
	}
}

func (h *Handler) displayconfigmap(request *restful.Request, response *restful.Response) {
	result, err := h.getReadOnlyVolumesByNode()
	if err != nil {
		fmt.Println(result)
	}
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: result,
	}
	err = response.WriteEntity(apiResponse)
	if err != nil {
		errorMessage := fmt.Sprintf("Failed to write entity: %v", err)
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusBadRequest, errorMessage)

	}
}
func (h *Handler) findPodByNamePrefix(prefix string) (string, error) {
	pods, err := h.PodClient.List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		errorMessage := fmt.Sprintf("Fail to find Pod name" +
			" Please ensure it is defined in the Json output.")
		tools.LogError(errorMessage)
		return "", err
	}
	for _, pod := range pods.Items {
		if strings.HasPrefix(pod.Name, prefix) {
			return pod.Name, nil
		}
	}
	return "", fmt.Errorf("no pod found with prefix %s", prefix)
}

func (h *Handler) getPodNode(podName string) (string, error) {
	pod, err := h.PodClient.Get(context.TODO(), podName, metav1.GetOptions{})
	if err != nil {
		errorMessage := fmt.Sprintf("Fail to build Pod client" +
			" Please ensure it is defined in the Json output.")
		tools.LogError(errorMessage)
		return "", fmt.Errorf("failed to get pod: %v", err)
	}

	return pod.Spec.NodeName, nil
}
func (h *Handler) getReadOnlyVolumesByNode() (map[string][]VolumeMount, error) {
	podName, err := h.findPodByNamePrefix("loki-promtail")
	if err != nil {
		errorMessage := fmt.Sprintf("Fail to find Pod name" +
			" Please ensure it is defined in the Json output.")
		tools.LogError(errorMessage)
		return nil, err
	}
	returnValues := make(map[string][]VolumeMount)

	pod, err := h.PodClient.Get(context.TODO(), podName, metav1.GetOptions{})
	if err != nil {
		fmt.Println("Failed to get pod:", err)
	}

	for _, container := range pod.Spec.Containers {
		fmt.Printf("Container Name: %s\n", container.Name)
		if container.Name != "promtail" {
			continue
		}
		for _, mount := range container.VolumeMounts {
			if mount.ReadOnly && mount.MountPath != "/var/run/secrets/kubernetes.io/serviceaccount" {
				key := fmt.Sprintf("%s/%s", pod.Name, container.Name)
				vol := VolumeMount{
					MountPath: mount.MountPath,
					Name:      mount.Name,
					ReadOnly:  mount.ReadOnly,
				}
				returnValues[key] = append(returnValues[key], vol)
				fmt.Printf("Mount Path: %s, Volume Name: %s, ReadOnly: %t\n",
					mount.MountPath, mount.Name, mount.ReadOnly)
			}
		}
	}

	return returnValues, nil
}

func (h *Handler) getConfigmapDetails(request *restful.Request, response *restful.Response) {
	cm, err := h.ConfigmapClient.Get(context.TODO(), "loki-promtail", metav1.GetOptions{})
	if err != nil {
		handleError(response, "Failed to get configmap", err)
		return
	}

	promtailYAML, ok := cm.Data["promtail.yaml"]
	if !ok {
		handleError(response, "promtail.yaml not found in configmap", nil)
		return
	}

	var promtailConfig PromtailConfig
	if err = yaml.Unmarshal([]byte(promtailYAML), &promtailConfig); err != nil {
		handleError(response, "Failed to unmarshal promtail configuration", err)
		return
	}

	jobPaths := getJobPaths(promtailConfig)
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: jobPaths,
	}

	if err = response.WriteEntity(apiResponse); err != nil {
		handleError(response, "Failed to write entity", err)
	}
}
func getJobPaths(promtailConfig PromtailConfig) map[string]string {
	jobPaths := make(map[string]string)
	for _, scrapeConfig := range promtailConfig.ScrapeConfigs {
		jobPaths[scrapeConfig.JobName] = determinePath(scrapeConfig)
	}
	return jobPaths
}

func determinePath(scrapeConfig ScrapeConfig) string {
	path := "" // Default path if not found
	if scrapeConfig.JobName == "kubernetes-pods" {
		path = "/var/log/pods/*/*/*.log"
	} else {
		for _, staticConfig := range scrapeConfig.StaticConfigs {
			if p, ok := staticConfig.Labels["__path__"]; ok {
				path = p
				break
			}
		}
	}
	return path
}

func handleError(response *restful.Response, message string, err error) {
	apiResponse := ApiResponse{
		Code: 400,
		Msg:  "Unexpected Error occurred",
		Data: nil,
	}
	errorMessage := message
	if err != nil {
		errorMessage = fmt.Sprintf("%s: %v", message, err)
	}
	tools.LogError(errorMessage)
	response.WriteErrorString(http.StatusBadRequest, errorMessage)
	err = response.WriteEntity(apiResponse)
	if err != nil {
		errorMessage := "Error Write Entity: "
		tools.LogError(errorMessage, err)
	}
}

func (h *Handler) getMountPaths(response *restful.Response) ([]string, error) {
	result, err := h.getReadOnlyVolumesByNode()
	var mountPaths []string
	if err != nil {
		return mountPaths, err
	}

	for _, mounts := range result {
		for _, mount := range mounts {
			mountPaths = append(mountPaths, mount.MountPath)
		}
	}
	return mountPaths, err
}

func validatePath(path string) bool {
	if path == "" {
		errorMessage := "Input Path should not be empty"
		tools.LogError(errorMessage)
		return false
	}
	maxLength := 300
	if len(path) > maxLength {
		errorMessage := "Input Path should less than 300 chars"
		tools.LogError(errorMessage)
		return false
	}

	if !strings.HasSuffix(path, ".log") {
		errorMessage := "Input Path should end with .log"
		tools.LogError(errorMessage)
		return false
	}

	pattern := `^[\w\-.*\/]+$`
	match, err := regexp.MatchString(pattern, path)
	if err != nil {
		fmt.Println("正则表达式错误:", err)
		return false
	}
	if !match {
		errorMessage := "Input path can only use - / . * for special characters"
		tools.LogError(errorMessage)
	}
	return match
}
func isValidUpdateRequest(req PromtailUpdateRequest, mountPaths []string) bool {

	isValidPrefix := false
	for _, mountPath := range mountPaths {
		if strings.HasPrefix(req.Path, mountPath) {
			isValidPrefix = true
			break
		}
	}
	if !isValidPrefix {
		errorMessage := "The path should follow with the agents collect source"
		tools.LogError(errorMessage)
	}
	isValidValues := false
	if validatePath(req.Path) && validateJobName(req.JobName) {
		isValidValues = true
	} else {
		errorMessage := "The patch body format is invalid"
		tools.LogError(errorMessage)
		isValidValues = false
	}
	isValidJobName := true
	requiredFields := []string{"jobName", "path"}

	// 验证字段名是否存在
	for _, field := range requiredFields {
		switch field {
		case "jobName":
			if req.JobName == "" {
				isValidJobName = false
				errorMessage := "The input body job name should as same as jobName"
				tools.LogError(errorMessage)

			}
		case "path":
			if req.Path == "" {
				isValidJobName = false
				errorMessage := "The input body path name should as same as path"
				tools.LogError(errorMessage)
			}
		default:
			isValidJobName = false
		}
	}
	isValidContent := isValidValues && isValidJobName
	return isValidContent && isValidPrefix
}
func (h *Handler) isValidUpdateRequest(updateReq PromtailUpdateRequest, response *restful.Response) bool {
	mountPaths, err := h.getMountPaths(response)
	if err != nil {
		handleError(response, "Failed to get mount paths", err)
		return false
	}
	if !isValidUpdateRequest(updateReq, mountPaths) {
		handleError(response, "Failed to validate update request", fmt.Errorf("the body format is invalid"))
		return false
	}
	return true
}
func (h *Handler) createScrapeConfig(updateReq PromtailUpdateRequest) ScrapeConfig {
	return ScrapeConfig{
		JobName: updateReq.JobName,
		StaticConfigs: []StaticConfig{{
			Targets: []string{"localhost"},
			Labels: map[string]string{
				"__path__": updateReq.Path,
				"job":      updateReq.JobName,
			},
		}},
	}
}
func validateJobName(jobName string) bool {
	maxLength := 30
	if len(jobName) < 1 || len(jobName) > maxLength {
		errorMessage := "Input job name should not be empty or combined with more than 30 chars"
		tools.LogError(errorMessage)
		return false
	}
	// 正则表达式，匹配只包含数字、字母、下划线和连字符的组合
	pattern := `^[a-zA-Z0-9_-]+$`
	match, err := regexp.MatchString(pattern, jobName)
	if err != nil {
		return false
	}
	if !match {
		errorMessage := "Input job name should only contains a-z A-Z 0-9 - _"
		tools.LogError(errorMessage)
	}
	return match
}
func (h *Handler) updatePromtailConfig(request *restful.Request, response *restful.Response) {
	cm, err := h.ConfigmapClient.Get(context.TODO(), "loki-promtail", metav1.GetOptions{})
	if err != nil {
		handleError(response, "Failed to get configmap client", err)
	}
	promtailYAML, ok := cm.Data["promtail.yaml"]
	if !ok {
		handleError(response, "Failed to find promtail.yaml file", err)
	}
	var promtailConfig PromtailConfig
	err = yaml.Unmarshal([]byte(promtailYAML), &promtailConfig)
	if err != nil {
		handleError(response, "Failed to unmarshal yaml file", err)
	}
	var updateReq PromtailUpdateRequest
	if err := request.ReadEntity(&updateReq); err != nil {
		response.WriteErrorString(http.StatusBadRequest, "Invalid request")
	}
	if !h.isValidUpdateRequest(updateReq, response) {
		response.WriteErrorString(http.StatusBadRequest, "Invalid request format")
		return
	}
	newConfig := h.createScrapeConfig(updateReq)
	promtailConfig.ScrapeConfigs = append(promtailConfig.ScrapeConfigs, newConfig)
	updatedYAML, err := yaml.Marshal(promtailConfig)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("failed to marshal updated promtail "+
			"config: %v", err))
	}
	cm.Data["promtail.yaml"] = string(updatedYAML)
	_, err = h.ConfigmapClient.Update(context.TODO(), cm, metav1.UpdateOptions{})
	if err != nil {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("failed to update ConfigMap: %v", err))
	}
	podName, err := h.findPodByNamePrefix("loki-promtail")
	if err != nil {
		handleError(response, "Failed to find pod name", err)
	}
	h.deletePod(podName)
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: nil,
	}
	err = response.WriteEntity(apiResponse)
}

func (h *Handler) deleteJob(request *restful.Request, response *restful.Response) {
	jobName := request.PathParameter("job-name")
	cm, err := h.ConfigmapClient.Get(context.TODO(), "loki-promtail", metav1.GetOptions{})
	if err != nil {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("failed to get configmap: %v", err))
		return
	}
	var promtailConfig PromtailConfig
	err = yaml.Unmarshal([]byte(cm.Data["promtail.yaml"]), &promtailConfig)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("failed to unmarshal promtail"+
			" config: %v", err))
		return
	}
	// Filter out the job to delete
	var updatedJobs []ScrapeConfig
	for _, job := range promtailConfig.ScrapeConfigs {
		if job.JobName != jobName {
			updatedJobs = append(updatedJobs, job)
		}
	}
	if len(updatedJobs) == len(promtailConfig.ScrapeConfigs) {
		response.WriteErrorString(http.StatusBadRequest, "Job name not found")
		return
	}
	promtailConfig.ScrapeConfigs = updatedJobs
	updatedYAML, err := yaml.Marshal(promtailConfig)
	if err != nil {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("failed to marshal updated promtail "+
			"config: %v", err))
		return
	}
	// Update the ConfigMap
	cm.Data["promtail.yaml"] = string(updatedYAML)
	_, err = h.ConfigmapClient.Update(context.TODO(), cm, metav1.UpdateOptions{})
	if err != nil {
		response.WriteError(http.StatusInternalServerError, fmt.Errorf("failed to update configmap: %v", err))
		return
	}
	podName, err := h.findPodByNamePrefix("loki-promtail")
	if err != nil {
		return
	}
	err = h.deletePod(podName)
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: nil}
	err = response.WriteEntity(apiResponse)
}

func (h *Handler) deletePod(podName string) error {
	deletePolicy := metav1.DeletePropagationForeground
	return h.PodClient.Delete(context.TODO(), podName, metav1.DeleteOptions{
		PropagationPolicy: &deletePolicy,
	})
}

func (h *Handler) getPodRunningStatus(request *restful.Request, response *restful.Response) {
	podName, err := h.findPodByNamePrefix("loki-promtail")
	pod, err := h.PodClient.Get(context.TODO(), podName, metav1.GetOptions{})
	if err != nil {
		return // 处理错误，可能是因为找不到 Pod
	}
	// 检查 Pod 的状态
	for _, cond := range pod.Status.Conditions {
		if cond.Type == "Ready" && cond.Status == "True" {
			apiResponse := ApiResponse{
				Code: 200,
				Msg:  "The promtail now is available",
				Data: nil,
			}
			err = response.WriteEntity(apiResponse)
			if err != nil {
				return // 处理错误，可能是因为找不到 Pod
			}
			return
		}
	}
	apiResponse := ApiResponse{
		Code: 503,
		Msg:  "The promtail now is unavailable",
		Data: nil,
	}
	err = response.WriteEntity(apiResponse)
	return
}

func calculateTimeRange(pivotTime string, interval string, isFirstHalf bool) (string, string) {
	base := 10
	bitSize := 64
	nanoSec, err := strconv.ParseInt(pivotTime, base, bitSize)
	if err != nil {
		tools.LogError(fmt.Sprintf("Error parsing timestamp: %v", err))
		return "0", "0"
	}
	pivot := time.Unix(0, nanoSec)
	duration, err := time.ParseDuration(interval)
	if err != nil {
		errorMessage := fmt.Sprintf("Invalid duration format: %s."+
			" Please provide a valid duration string (e.g., '300ms', '1.5h' "+
			"or '2h45m'). LogError: %v", duration, err)
		tools.LogError(errorMessage)
		return "0", "0"
	}
	if isFirstHalf {
		endTime := pivot
		startTime := endTime.Add(-duration)
		return fmt.Sprintf("%d", startTime.Unix()), fmt.Sprintf("%d", endTime.Unix())
	} else {
		startTime := pivot
		endTime := startTime.Add(duration)
		return fmt.Sprintf("%d", startTime.Unix()), fmt.Sprintf("%d", endTime.Unix())
	}
}
func (h *Handler) handleContextLogDetails(request *restful.Request, response *restful.Response) {
	if !h.checkRequestConfig(response) {
		return
	}

	params := h.buildQueryParams(request)
	query := lokiQueryBuilder(params)

	result1, err := h.executeLokiQuery(request, params, query, true)
	if err != nil {
		handleError(response, "Failed to send HTTP request", err)
		return
	}

	result2, err := h.executeLokiQuery(request, params, query, false)
	if err != nil {
		handleError(response, "Failed to send HTTP request", err)
		return
	}

	h.respondWithResults(response, result1, result2)
}

func (h *Handler) checkRequestConfig(response *restful.Response) bool {
	_, exists := h.RequestsMap["filterRequest"]
	if !exists {
		errorMessage := "Request configuration for 'filterRequest' not found. Please ensure it is defined " +
			"in the RequestsMap."
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusNotFound, errorMessage)
		return false
	}
	return true
}
func (h *Handler) buildQueryParams(request *restful.Request) LokiQueryParams {
	filename := server.GetQueryParamOrDefault(request, server.FileName, "")
	loglevel := server.GetQueryParamOrDefault(request, server.Level, "")
	keyword := server.GetQueryParamOrDefault(request, server.Keyword, "")
	return LokiQueryParams{
		Filename: filename,
		LogLevel: loglevel,
		Keyword:  keyword,
	}
}
func (h *Handler) executeLokiQuery(request *restful.Request, params LokiQueryParams, query string,
	isFirstQuery bool) (interface{}, error) {
	timestamp := server.GetQueryParamOrDefault(request, server.TimeStamp, "")
	contextInterval := server.GetQueryParamOrDefault(request, server.TimePeriod, "")
	duration := h.timeUnitsDict[contextInterval]
	startTime, endTime := calculateTimeRange(timestamp, duration, isFirstQuery)
	fmt.Println("start time: ", startTime, "end time: ", endTime)
	queryParams := queryParamBuilder(startTime, endTime, server.GetQueryParamOrDefault(request, server.Sizes,
		"10"), query)
	req, _ := h.RequestsMap["filterRequest"]
	fullURL := fmt.Sprintf("%s?%s", req.URL, queryParams.Encode())
	fmt.Println(fullURL)
	return sendHTTPRequest(req, fullURL)
}
func (h *Handler) respondWithResults(response *restful.Response, result1, result2 interface{}) {
	combinedResult := CombinedResult{Results: []interface{}{result1, result2}}
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: combinedResult,
	}
	if err := response.WriteEntity(apiResponse); err != nil {
		handleError(response, "Failed to write entity", err)
	}
}

func (h *Handler) getlogSecrets(request *restful.Request, response *restful.Response) {
	secret, err := h.SecretsClient.Get(context.TODO(), "loki", metav1.GetOptions{})
	if err != nil {
		errorMessage := "Fail to build k8s secret client"
		tools.LogError(errorMessage)
		return
	}

	// 解码 Secret 数据
	data, ok := secret.Data["loki.yaml"]
	if !ok {
		errorMessage := "Fail to get secret information"
		tools.LogError(errorMessage)
		return
	}

	var config SecretConfig
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		errorMessage := "Error parsing JSON: "
		tools.LogError(errorMessage, err)
		return
	}
	apiResponse := ApiResponse{
		Code: 200,
		Msg:  "Operation successful",
		Data: config,
	}
	err = response.WriteEntity(apiResponse)
	if err != nil {
		errorMessage := fmt.Sprintf("Failed to write entity: %v", err)
		tools.LogError(errorMessage)
		response.WriteErrorString(http.StatusBadRequest, errorMessage)
	}
}
