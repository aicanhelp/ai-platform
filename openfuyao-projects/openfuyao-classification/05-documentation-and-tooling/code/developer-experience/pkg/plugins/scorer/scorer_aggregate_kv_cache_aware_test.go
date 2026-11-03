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

package scorer

import (
	"context"
	"encoding/json"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	"hermes-router/pkg/plugins/common"
)

// TestAggKVCacheAwareFactory tests AggKVCacheAwareFactory function.
const (
	testFactoryName = "test-factory-scorer"
)

func TestAggKVCacheAwareFactory(t *testing.T) {
	rawParams := json.RawMessage(`{}`)
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := AggKVCacheAwareFactory(testFactoryName, rawParams, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin, got nil")
	}
	scorer, ok := plugin.(*AggKVCacheAware)
	if !ok {
		t.Fatal("expected *AggKVCacheAware type")
	}
	if scorer.typedName.Name != testFactoryName {
		t.Errorf("expected name %s, got %s", testFactoryName, scorer.typedName.Name)
	}
}

// TestAggKVCacheAwareFactoryWithParams tests AggKVCacheAwareFactory with parameters.
const (
	testFactoryWithParamsName = "test-factory-with-params"
	testParamKVCacheWeight    = 1.5
	testParamXPUWeight        = 2.5
	testParamWaitingWeight    = 3.5
)

func TestAggKVCacheAwareFactoryWithParams(t *testing.T) {
	rawParams := json.RawMessage(`{
		"kvCacheHitNotRateWeight": ` + strconv.FormatFloat(testParamKVCacheWeight, 'f', -1, 64) + `,
		"xpuCacheUsageWeight": ` + strconv.FormatFloat(testParamXPUWeight, 'f', -1, 64) + `,
		"waitingRequestWeight": ` + strconv.FormatFloat(testParamWaitingWeight, 'f', -1, 64) + `
	}`)
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := AggKVCacheAwareFactory(testFactoryWithParamsName, rawParams, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	scorer, ok := plugin.(*AggKVCacheAware)
	if !ok {
		t.Fatal("expected *AggKVCacheAware type, but got different type")
	}
	if scorer.params.KVCacheNotHitRateWeight != testParamKVCacheWeight {
		t.Errorf("expected kvCacheWeight %f, got %f",
			testParamKVCacheWeight, scorer.params.KVCacheNotHitRateWeight)
	}
}

// TestAggKVCacheAwareFactoryWithInvalidParams tests AggKVCacheAwareFactory with invalid parameters.
const (
	testFactoryInvalidParamsName = "test-factory-invalid-params"
	testInvalidJSON              = "{invalid json}"
)

func TestAggKVCacheAwareFactoryWithInvalidParams(t *testing.T) {
	rawParams := json.RawMessage(testInvalidJSON)
	handle := plugins.NewEppHandle(context.Background(), nil)

	_, err := AggKVCacheAwareFactory(testFactoryInvalidParamsName, rawParams, handle)

	if err == nil {
		t.Error("expected error for invalid parameters, got nil")
	}
}

// TestAggKVCacheAwareFactoryWithNilParams tests AggKVCacheAwareFactory with nil parameters.
func TestAggKVCacheAwareFactoryWithNilParams(t *testing.T) {
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := AggKVCacheAwareFactory(testFactoryName, nil, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin, got nil")
	}
}

// TestNewAggKVCacheAware tests NewAggKVCacheAware function.

func TestNewAggKVCacheAware(t *testing.T) {
	params := common.KVCacheAwareParameters{
		KVCacheNotHitRateWeight: kvCacheWeight,
		XPUCacheUsageWeight:     xpuUsageWeight,
		WaitingRequestWeight:    waitingWeight,
		KVCacheManagerIP:        testManagerHost,
		KVCacheManagerPort:      testManagerPort,
		KVCacheManagerPath:      kvCacheEndpointPath,
		KVCacheManagerTimeout:   common.DefaultKVCacheManagerTimeout,
	}

	scorer := NewAggKVCacheAware(context.Background(), params)

	if scorer.typedName.Type != AggKVCacheAwareType {
		t.Errorf("expected type %s, got %s", AggKVCacheAwareType, scorer.typedName.Type)
	}
	if scorer.params.KVCacheNotHitRateWeight != kvCacheWeight {
		t.Errorf("expected kvCacheWeight %f, got %f", kvCacheWeight, scorer.params.KVCacheNotHitRateWeight)
	}
	if scorer.kvCacheAware == nil {
		t.Error("expected non-nil kvCacheAware, got nil")
	}
}

// TestTypedName tests TypedName method.
const (
	testTypedNameName = "test-typed-name"
)

func TestTypedName(t *testing.T) {
	scorer := &AggKVCacheAware{
		typedName: plugins.TypedName{Type: AggKVCacheAwareType, Name: testTypedNameName},
	}

	typedName := scorer.TypedName()

	if typedName.Type != AggKVCacheAwareType {
		t.Errorf("expected type %s, got %s", AggKVCacheAwareType, typedName.Type)
	}
	if typedName.Name != testTypedNameName {
		t.Errorf("expected name %s, got %s", testTypedNameName, typedName.Name)
	}
}

// TestWithName tests WithName method.
const (
	testOriginalName = "original-name"
	testNewName      = "new-name"
)

func TestWithName(t *testing.T) {
	original := &AggKVCacheAware{
		typedName: plugins.TypedName{Type: AggKVCacheAwareType, Name: testOriginalName},
	}

	newScorer := original.WithName(testNewName)

	if newScorer.typedName.Name != testNewName {
		t.Errorf("expected name %s, got %s", testNewName, newScorer.typedName.Name)
	}
	if original.typedName.Name != testNewName {
		t.Error("expected original scorer to be modified")
	}
}

// TestScoreSuccess tests Score with successful KV cache manager response.
const (
	kvCacheEndpointPath      = "/kvcache"
	kvCacheWeight            = 1.0
	xpuUsageWeight           = 2.0
	waitingWeight            = 3.0
	unitScoreValue           = 1.0
	scoreDelta               = 1e-6
	primaryPodName           = "pod-1"
	primaryPodIP             = "10.0.0.1"
	primaryKVCacheHitRate    = 0.8
	primaryKVCacheUsage      = 0.1
	primaryWaitingRequests   = 10
	secondaryPodName         = "pod-2"
	secondaryPodIP           = "10.0.0.2"
	secondaryKVCacheHitRate  = 0.5
	secondaryKVCacheUsage    = 0.2
	secondaryWaitingRequests = 20
	primaryRequestID         = "request-primary"
	testModelName            = "test-model"
	testPrompt               = "test prompt"
	okResponseMessage        = "ok"
	testHTTPMethodPost       = "POST"
	testManagerHost          = "localhost"
	testManagerPort          = 8080
)

func TestScoreSuccess(t *testing.T) {
	server, host, port := setupMockKVCacheServer(t, kvCacheEndpointPath)
	defer server.Close()

	scorer := createTestScorer(host, port, kvCacheEndpointPath)
	pods := createTestPods()
	request := createTestRequest()

	scores := scorer.Score(context.Background(), &types.CycleState{}, request, pods)

	verifyScores(t, scores, pods)
}

// TestScoreKVManagerError tests Score when KV cache manager returns error.
const (
	errorPodIP             = "10.0.0.3"
	errorPodName           = "pod-error"
	errorRequestID         = "request-error"
	errorWaitingQueue      = 5
	errorResponseBody      = "error"
	testEmptyPath          = ""
	testExpectedZeroScores = 0
)

func TestScoreKVManagerError(t *testing.T) {
	errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, errorResponseBody, http.StatusInternalServerError)
	}))
	defer errorServer.Close()

	host, port := parseServerURL(t, errorServer.URL)
	scorer := createTestScorer(host, port, testEmptyPath)
	pod := createErrorPod()
	request := createErrorRequest()

	scores := scorer.Score(context.Background(), &types.CycleState{}, request, []types.Pod{pod})

	if len(scores) != testExpectedZeroScores {
		t.Errorf("expected %d scores due to kv manager error, got %d", testExpectedZeroScores, len(scores))
	}
}

// Helper functions

func setupMockKVCacheServer(t *testing.T, endpointPath string) (*httptest.Server, string, int) {
	mux := http.NewServeMux()
	mux.HandleFunc(endpointPath, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != testHTTPMethodPost {
			t.Fatalf("expected %s method, got %s", testHTTPMethodPost, r.Method)
		}
		var req struct {
			ServerIP []string        `json:"server_ip"`
			Body     json.RawMessage `json:"body"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}
		resp := common.KVCacheManagerResponse{
			ServerScoreList: []map[string]float64{
				{primaryPodIP: primaryKVCacheHitRate},
				{secondaryPodIP: secondaryKVCacheHitRate},
			},
			Status:  http.StatusOK,
			Message: okResponseMessage,
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("failed to encode response: %v", err)
		}
	})
	server := httptest.NewServer(mux)
	host, port := parseServerURL(t, server.URL)
	return server, host, port
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

func createTestScorer(host string, port int, path string) *AggKVCacheAware {
	return NewAggKVCacheAware(context.Background(), common.KVCacheAwareParameters{
		KVCacheNotHitRateWeight: kvCacheWeight,
		XPUCacheUsageWeight:     xpuUsageWeight,
		WaitingRequestWeight:    waitingWeight,
		KVCacheManagerIP:        host,
		KVCacheManagerPort:      port,
		KVCacheManagerPath:      path,
		KVCacheManagerTimeout:   common.DefaultKVCacheManagerTimeout,
	})
}

func createTestPods() []types.Pod {
	pod1 := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: primaryPodName,
			Address: primaryPodIP,
		},
		MetricsState: &metrics.MetricsState{
			WaitingQueueSize:    primaryWaitingRequests,
			KVCacheUsagePercent: primaryKVCacheUsage,
		},
	}
	pod2 := &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: secondaryPodName,
			Address: secondaryPodIP,
		},
		MetricsState: &metrics.MetricsState{
			WaitingQueueSize:    secondaryWaitingRequests,
			KVCacheUsagePercent: secondaryKVCacheUsage,
		},
	}
	return []types.Pod{pod1, pod2}
}

func createTestRequest() *types.LLMRequest {
	return &types.LLMRequest{
		RequestId:   primaryRequestID,
		TargetModel: testModelName,
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: testPrompt,
			},
		},
	}
}

func createErrorPod() types.Pod {
	return &types.PodMetrics{
		Pod: &backend.Pod{
			PodName: errorPodName,
			Address: errorPodIP,
		},
		MetricsState: &metrics.MetricsState{
			WaitingQueueSize:    errorWaitingQueue,
			KVCacheUsagePercent: primaryKVCacheUsage,
		},
	}
}

func createErrorRequest() *types.LLMRequest {
	return &types.LLMRequest{
		RequestId:   errorRequestID,
		TargetModel: testModelName,
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: testPrompt,
			},
		},
	}
}

func verifyScores(t *testing.T, scores map[types.Pod]float64, pods []types.Pod) {
	if len(scores) != len(pods) {
		t.Errorf("expected %d scores, got %d", len(pods), len(scores))
	}
	pod1, pod2 := pods[0], pods[1]
	expectedScores := map[types.Pod]float64{
		pod1: kvCacheWeight*(unitScoreValue-primaryKVCacheHitRate) +
			xpuUsageWeight*primaryKVCacheUsage +
			waitingWeight*(float64(primaryWaitingRequests)/float64(secondaryWaitingRequests)),
		pod2: kvCacheWeight*(unitScoreValue-secondaryKVCacheHitRate) +
			xpuUsageWeight*secondaryKVCacheUsage +
			waitingWeight*(float64(secondaryWaitingRequests)/float64(secondaryWaitingRequests)),
	}
	for pod, want := range expectedScores {
		got, ok := scores[pod]
		if !ok {
			t.Errorf("missing score for pod %v", pod)
			continue
		}
		if math.Abs(got-want) > scoreDelta {
			t.Errorf("unexpected score for pod %v: want %.4f, got %.4f", pod, want, got)
		}
	}
}
