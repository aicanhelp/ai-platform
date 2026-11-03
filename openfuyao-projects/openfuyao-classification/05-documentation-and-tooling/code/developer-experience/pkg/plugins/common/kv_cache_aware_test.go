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

package common

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// TestNewKVCacheAware tests the NewKVCacheAware constructor.
const (
	testManagerIP   = "127.0.0.1"
	testManagerPort = 8080
	testManagerPath = "/kvcache"
	testTimeout     = 5 * time.Second
)

func TestNewKVCacheAware(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)

	if kvCacheAware == nil {
		t.Fatal("NewKVCacheAware returned nil")
	}
	if kvCacheAware.managerIP != testManagerIP {
		t.Errorf("expected managerIP %s, got %s", testManagerIP, kvCacheAware.managerIP)
	}
	if kvCacheAware.managerPort != testManagerPort {
		t.Errorf("expected managerPort %d, got %d", testManagerPort, kvCacheAware.managerPort)
	}
	if kvCacheAware.managerPath != testManagerPath {
		t.Errorf("expected managerPath %s, got %s", testManagerPath, kvCacheAware.managerPath)
	}
	if kvCacheAware.timeout != testTimeout {
		t.Errorf("expected timeout %v, got %v", testTimeout, kvCacheAware.timeout)
	}
}

// TestGetKVCacheHitRateEmptyPods tests GetKVCacheHitRate with empty pods list.
func TestGetKVCacheHitRateEmptyPods(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	ctx := context.Background()
	request := &types.LLMRequest{RequestId: "test-request"}

	_, err := kvCacheAware.GetKVCacheHitRate(ctx, []types.Pod{}, request)

	if err == nil {
		t.Fatal("expected error for empty pods list, got nil")
	}
}

// TestGetKVCacheHitRateSuccess tests successful GetKVCacheHitRate call.
const (
	testEndpointPath    = "/kvcache"
	testPod1Name        = "pod-1"
	testPod1IP          = "10.0.0.1"
	testPod2Name        = "pod-2"
	testPod2IP          = "10.0.0.2"
	testHitRate1        = 0.8
	testHitRate2        = 0.5
	testModelName       = "test-model"
	testPrompt          = "test prompt"
	testRequestID       = "test-request-id"
	testResponseStatus  = 200
	testResponseMessage = "success"
	testHTTPMethodPost  = "POST"
	expectedHitRate     = 2
	expectedServerIPs   = 2
)

func setupMockKVCacheServer(t *testing.T) (*httptest.Server, *[]string) {
	mux := http.NewServeMux()
	var capturedServerIPs []string
	mux.HandleFunc(testEndpointPath, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != testHTTPMethodPost {
			t.Fatalf("expected %s method, got %s", testHTTPMethodPost, r.Method)
		}
		var req struct {
			ServerIP []string        `json:"server_ip"`
			Body     json.RawMessage `json:"body"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		capturedServerIPs = req.ServerIP
		resp := KVCacheManagerResponse{
			ServerScoreList: []map[string]float64{
				{testPod1IP: testHitRate1},
				{testPod2IP: testHitRate2},
			},
			Status:  testResponseStatus,
			Message: testResponseMessage,
		}
		err := json.NewEncoder(w).Encode(resp)
		if err != nil {
			t.Fatalf("failed to encode response: %v", err)
		}
	})
	server := httptest.NewServer(mux)
	return server, &capturedServerIPs
}

func parseServerURL(t *testing.T, serverURL string) (string, int) {
	url, err := url.Parse(serverURL)
	if err != nil {
		t.Fatalf("failed to parse test server url: %v", err)
	}
	host, portStr, err := net.SplitHostPort(url.Host)
	if err != nil {
		t.Fatalf("failed to split host/port: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("failed to convert port: %v", err)
	}
	return host, port
}

func TestGetKVCacheHitRateSuccess(t *testing.T) {
	server, capturedServerIPs := setupMockKVCacheServer(t)
	defer server.Close()

	host, port := parseServerURL(t, server.URL)
	kvCacheAware := NewKVCacheAware(host, port, testEndpointPath, testTimeout)
	pod1 := &types.PodMetrics{
		Pod: &backend.Pod{PodName: testPod1Name, Address: testPod1IP},
	}
	pod2 := &types.PodMetrics{
		Pod: &backend.Pod{PodName: testPod2Name, Address: testPod2IP},
	}
	request := &types.LLMRequest{
		RequestId:   testRequestID,
		TargetModel: testModelName,
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{Prompt: testPrompt},
		},
	}

	hitRateMap, err := kvCacheAware.GetKVCacheHitRate(context.Background(), []types.Pod{pod1, pod2}, request)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(hitRateMap) != expectedHitRate {
		t.Fatalf("expected 2 hit rates, got %d", len(hitRateMap))
	}
	if hitRateMap[testPod1IP] != testHitRate1 {
		t.Errorf("expected hit rate %f for pod1, got %f", testHitRate1, hitRateMap[testPod1IP])
	}
	if hitRateMap[testPod2IP] != testHitRate2 {
		t.Errorf("expected hit rate %f for pod2, got %f", testHitRate2, hitRateMap[testPod2IP])
	}
	if len(*capturedServerIPs) != expectedServerIPs {
		t.Fatalf("expected 2 server IPs, got %d", len(*capturedServerIPs))
	}
}

// TestBuildRequestBodyNilRequest tests buildRequestBody with nil request.
func TestBuildRequestBodyNilRequest(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	pod := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: testPod1Name,
			Address: testPod1IP,
		},
	}

	_, err := kvCacheAware.buildRequestBody([]types.Pod{pod}, nil)
	if err == nil {
		t.Fatal("expected error for nil request, got nil")
	}
}

// TestBuildRequestBodyEmptyPods tests buildRequestBody with empty pods.
func TestBuildRequestBodyEmptyPods(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	request := &types.LLMRequest{RequestId: testRequestID}

	_, err := kvCacheAware.buildRequestBody([]types.Pod{}, request)
	if err == nil {
		t.Fatal("expected error for empty pods, got nil")
	}
}

// TestBuildRequestBodyNilPod tests buildRequestBody with pod that has nil GetPod().
func TestBuildRequestBodyNilPod(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	request := &types.LLMRequest{RequestId: testRequestID}
	pod := &types.PodMetrics{}

	_, err := kvCacheAware.buildRequestBody([]types.Pod{pod}, request)
	if err == nil {
		t.Fatal("expected error for nil pod, got nil")
	}
}

// TestBuildRequestBodyNilBody tests buildRequestBody with nil request body.
func TestBuildRequestBodyNilBody(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	request := &types.LLMRequest{RequestId: testRequestID}
	pod := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: testPod1Name,
			Address: testPod1IP,
		},
	}

	_, err := kvCacheAware.buildRequestBody([]types.Pod{pod}, request)
	if err == nil {
		t.Fatal("expected error for nil request body, got nil")
	}
}

// TestBuildRequestBodyCompletions tests buildRequestBody with Completions request.
func TestBuildRequestBodyCompletions(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	pod := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: testPod1Name,
			Address: testPod1IP,
		},
	}
	request := &types.LLMRequest{
		RequestId:   testRequestID,
		TargetModel: testModelName,
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: testPrompt,
			},
		},
	}

	requestData, err := kvCacheAware.buildRequestBody([]types.Pod{pod}, request)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	serverIPs, ok := requestData["server_ip"].([]string)
	if !ok || len(serverIPs) != 1 || serverIPs[0] != testPod1IP {
		t.Errorf("unexpected server IPs: %v", serverIPs)
	}
	body, ok := requestData["body"].(KVCacheManagerRequestBody)
	if !ok {
		t.Fatal("request body type assertion failed")
	}
	if body.Model != testModelName {
		t.Errorf("expected model %s, got %s", testModelName, body.Model)
	}
}

// TestBuildRequestBodyChatCompletions tests buildRequestBody with ChatCompletions request.
const (
	testChatMessageRole    = "user"
	testChatMessageContent = "hello"
)

func TestBuildRequestBodyChatCompletions(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	pod := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: testPod1Name,
			Address: testPod1IP,
		},
	}
	messages := []types.Message{
		{
			Role: testChatMessageRole,
			Content: types.Content{
				Raw: testChatMessageContent,
			},
		},
	}
	request := &types.LLMRequest{
		RequestId:   testRequestID,
		TargetModel: testModelName,
		Body: &types.LLMRequestBody{
			ChatCompletions: &types.ChatCompletionsRequest{
				Messages: messages,
			},
		},
	}

	requestData, err := kvCacheAware.buildRequestBody([]types.Pod{pod}, request)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, ok := requestData["body"].(KVCacheManagerRequestBody)
	if !ok {
		t.Fatal("request body type assertion failed")
	}
	if body.Model != testModelName {
		t.Errorf("expected model %s, got %s", testModelName, body.Model)
	}
}

// TestBuildRequestBodyUnsupportedType tests buildRequestBody with unsupported request type.
func TestBuildRequestBodyUnsupportedType(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	pod := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: testPod1Name,
			Address: testPod1IP,
		},
	}
	request := &types.LLMRequest{
		RequestId:   testRequestID,
		TargetModel: testModelName,
		Body:        &types.LLMRequestBody{},
	}

	_, err := kvCacheAware.buildRequestBody([]types.Pod{pod}, request)

	if err == nil {
		t.Fatal("expected error for unsupported request type, got nil")
	}
}

// TestSendRequestSuccess tests successful sendRequest call.
func TestSendRequestSuccess(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc(testEndpointPath, func(w http.ResponseWriter, r *http.Request) {
		resp := KVCacheManagerResponse{
			ServerScoreList: []map[string]float64{{testPod1IP: testHitRate1}},
			Status:          testResponseStatus,
			Message:         testResponseMessage,
		}
		err := json.NewEncoder(w).Encode(resp)
		if err != nil {
			t.Fatalf("failed to encode response: %v", err)
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("failed to encode response: %v", err)
		}
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	serverURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("failed to parse test server url: %v", err)
	}
	host, portStr, err := net.SplitHostPort(serverURL.Host)
	if err != nil {
		t.Fatalf("failed to split host/port: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("failed to convert port: %v", err)
	}
	kvCacheAware := NewKVCacheAware(host, port, testEndpointPath, testTimeout)
	requestData := map[string]interface{}{"server_ip": []string{testPod1IP},
		"body": KVCacheManagerRequestBody{Model: testModelName, Messages: testPrompt},
	}
	response, err := kvCacheAware.sendRequest(context.Background(), requestData)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if response.Status != testResponseStatus {
		t.Errorf("expected status %d, got %d", testResponseStatus, response.Status)
	}
	if len(response.ServerScoreList) != 1 {
		t.Errorf("expected 1 server score, got %d", len(response.ServerScoreList))
	}
}

// TestSendRequestNonOKStatus tests sendRequest with non-OK HTTP status.
const (
	testErrorStatus = 500
)

func TestSendRequestNonOKStatus(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc(testEndpointPath, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(testErrorStatus)
	})

	server := httptest.NewServer(mux)
	defer server.Close()

	serverURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("failed to parse test server url: %v", err)
	}
	host, portStr, err := net.SplitHostPort(serverURL.Host)
	if err != nil {
		t.Fatalf("failed to split host/port: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("failed to convert port: %v", err)
	}

	kvCacheAware := NewKVCacheAware(host, port, testEndpointPath, testTimeout)
	requestData := map[string]interface{}{
		"server_ip": []string{testPod1IP},
		"body":      map[string]interface{}{},
	}

	_, err = kvCacheAware.sendRequest(context.Background(), requestData)
	if err == nil {
		t.Fatal("expected error for non-OK status, got nil")
	}
}

// TestParseResponse tests parseResponse function.
func TestParseResponse(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	response := KVCacheManagerResponse{
		ServerScoreList: []map[string]float64{
			{testPod1IP: testHitRate1},
			{testPod2IP: testHitRate2},
		},
		Status:  testResponseStatus,
		Message: testResponseMessage,
	}

	hitRateMap := kvCacheAware.parseResponse(response)

	if len(hitRateMap) != expectedHitRate {
		t.Fatalf("expected 2 hit rates, got %d", len(hitRateMap))
	}
	if hitRateMap[testPod1IP] != testHitRate1 {
		t.Errorf("expected hit rate %f for pod1, got %f", testHitRate1, hitRateMap[testPod1IP])
	}
	if hitRateMap[testPod2IP] != testHitRate2 {
		t.Errorf("expected hit rate %f for pod2, got %f", testHitRate2, hitRateMap[testPod2IP])
	}
}

// TestParseResponseEmpty tests parseResponse with empty response.
func TestParseResponseEmpty(t *testing.T) {
	kvCacheAware := NewKVCacheAware(testManagerIP, testManagerPort, testManagerPath, testTimeout)
	response := KVCacheManagerResponse{
		ServerScoreList: []map[string]float64{},
	}

	hitRateMap := kvCacheAware.parseResponse(response)

	if len(hitRateMap) != 0 {
		t.Fatalf("expected 0 hit rates, got %d", len(hitRateMap))
	}
}

// TestCalculateMaxWaitingRequest tests CalculateMaxWaitingRequest function.
const (
	testWaitingQueueSize1 = 10
	testWaitingQueueSize2 = 20
	testWaitingQueueSize3 = 15
)

func TestCalculateMaxWaitingRequest(t *testing.T) {
	pod1 := &types.PodMetrics{
		MetricsState: &metrics.MetricsState{
			WaitingQueueSize: testWaitingQueueSize1,
		},
	}
	pod2 := &types.PodMetrics{
		MetricsState: &metrics.MetricsState{
			WaitingQueueSize: testWaitingQueueSize2,
		},
	}
	pod3 := &types.PodMetrics{
		MetricsState: &metrics.MetricsState{
			WaitingQueueSize: testWaitingQueueSize3,
		},
	}
	scoredPods := []types.ScoredPod{
		{Pod: pod1},
		{Pod: pod2},
		{Pod: pod3},
	}

	maxWaitingRequest := CalculateMaxWaitingRequest(scoredPods)

	if maxWaitingRequest != testWaitingQueueSize2 {
		t.Errorf("expected max waiting request %d, got %d", testWaitingQueueSize2, maxWaitingRequest)
	}
}

// TestCalculateMaxWaitingRequestWithNilMetrics tests CalculateMaxWaitingRequest with nil metrics.
func TestCalculateMaxWaitingRequestWithNilMetrics(t *testing.T) {
	pod1 := &types.PodMetrics{
		MetricsState: &metrics.MetricsState{
			WaitingQueueSize: testWaitingQueueSize1,
		},
	}
	pod2 := &types.PodMetrics{}
	scoredPods := []types.ScoredPod{
		{Pod: pod1},
		{Pod: pod2},
	}

	maxWaitingRequest := CalculateMaxWaitingRequest(scoredPods)

	if maxWaitingRequest != testWaitingQueueSize1 {
		t.Errorf("expected max waiting request %d, got %d", testWaitingQueueSize1, maxWaitingRequest)
	}
}

// TestCalculateMaxWaitingRequestEmpty tests CalculateMaxWaitingRequest with empty list.
func TestCalculateMaxWaitingRequestEmpty(t *testing.T) {
	var scoredPods []types.ScoredPod

	maxWaitingRequest := CalculateMaxWaitingRequest(scoredPods)

	if maxWaitingRequest != 0 {
		t.Errorf("expected max waiting request 0, got %d", maxWaitingRequest)
	}
}

// TestCalculateWaitingRequestWeight tests CalculateWaitingRequestWeight function.
const (
	testWeightGamma       = 1.0
	testWaitingQueueSize  = 10
	testMaxWaitingRequest = 20
	testExpectedWeight    = 0.5
	testWeightTolerance   = 1e-6
)

func TestCalculateWaitingRequestWeight(t *testing.T) {
	weight := CalculateWaitingRequestWeight(testWeightGamma, testWaitingQueueSize, testMaxWaitingRequest)

	expected := testExpectedWeight
	if weight < expected-testWeightTolerance || weight > expected+testWeightTolerance {
		t.Errorf("expected weight %f, got %f", expected, weight)
	}
}

// TestCalculateWaitingRequestWeightZeroMax tests CalculateWaitingRequestWeight with zero max.
const (
	testZeroMax = 0
)

func TestCalculateWaitingRequestWeightZeroMax(t *testing.T) {
	weight := CalculateWaitingRequestWeight(testWeightGamma, testWaitingQueueSize, testZeroMax)

	if weight != testZeroMax {
		t.Errorf("expected weight %d, got %f", testZeroMax, weight)
	}
}

// TestCalculatePodKVScore tests CalculatePodKVScore function.
const (
	testKVCacheHitNotRateWeight = 1.0
	testXPUCacheUsageWeight     = 2.0
	testWaitingRequestWeight    = 3.0
	testKVCacheUsagePercent     = 0.3
	testHitRate                 = 0.7
)

func TestCalculatePodKVScore(t *testing.T) {
	pod1 := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: testPod1Name,
			Address: testPod1IP,
		},
		MetricsState: &metrics.MetricsState{
			KVCacheUsagePercent: testKVCacheUsagePercent,
			WaitingQueueSize:    testWaitingQueueSize,
		},
	}
	scoredPods := []types.ScoredPod{
		{Pod: pod1, Score: 0.0},
	}
	hitRateMap := map[string]float64{
		testPod1IP: testHitRate,
	}

	CalculatePodKVScore(
		context.Background(),
		KVCacheScoreWeights{
			KVCacheNotHitRateWeight: testKVCacheHitNotRateWeight,
			XPUCacheUsageWeight:     testXPUCacheUsageWeight,
			WaitingRequestWeight:    testWaitingRequestWeight,
		},
		hitRateMap,
		scoredPods,
	)

	expectedScore := testKVCacheHitNotRateWeight*(1.0-testHitRate) +
		testXPUCacheUsageWeight*testKVCacheUsagePercent +
		testWaitingRequestWeight*(float64(testWaitingQueueSize)/float64(testWaitingQueueSize))

	if scoredPods[0].Score < expectedScore-testWeightTolerance ||
		scoredPods[0].Score > expectedScore+testWeightTolerance {
		t.Errorf("expected score %f, got %f", expectedScore, scoredPods[0].Score)
	}
}

// TestCalculatePodKVScoreNilPod tests CalculatePodKVScore with pod that has nil GetPod().
func TestCalculatePodKVScoreNilPod(t *testing.T) {
	pod := &types.PodMetrics{}
	scoredPods := []types.ScoredPod{
		{Pod: pod, Score: 0.0},
	}
	hitRateMap := map[string]float64{}

	CalculatePodKVScore(
		context.Background(),
		KVCacheScoreWeights{
			KVCacheNotHitRateWeight: testKVCacheHitNotRateWeight,
			XPUCacheUsageWeight:     testXPUCacheUsageWeight,
			WaitingRequestWeight:    testWaitingRequestWeight,
		},
		hitRateMap,
		scoredPods,
	)

	if scoredPods[0].Score != 0.0 {
		t.Errorf("expected score 0.0 for nil pod, got %f", scoredPods[0].Score)
	}
}

// TestCalculatePodKVScoreNilMetrics tests CalculatePodKVScore with nil metrics.
func TestCalculatePodKVScoreNilMetrics(t *testing.T) {
	pod := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: testPod1Name,
			Address: testPod1IP,
		},
	}
	scoredPods := []types.ScoredPod{
		{Pod: pod, Score: 0.0},
	}
	hitRateMap := map[string]float64{
		testPod1IP: testHitRate,
	}

	CalculatePodKVScore(
		context.Background(),
		KVCacheScoreWeights{
			KVCacheNotHitRateWeight: testKVCacheHitNotRateWeight,
			XPUCacheUsageWeight:     testXPUCacheUsageWeight,
			WaitingRequestWeight:    testWaitingRequestWeight,
		},
		hitRateMap,
		scoredPods,
	)

	expectedScore := testKVCacheHitNotRateWeight * (1.0 - testHitRate)
	if scoredPods[0].Score < expectedScore-testWeightTolerance ||
		scoredPods[0].Score > expectedScore+testWeightTolerance {
		t.Errorf("expected score %f, got %f", expectedScore, scoredPods[0].Score)
	}
}
