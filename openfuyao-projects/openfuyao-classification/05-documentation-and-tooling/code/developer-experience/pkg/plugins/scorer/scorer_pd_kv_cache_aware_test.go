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
	"math/rand"
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

// TestPDKVCacheAwareFactory tests PDKVCacheAwareFactory function.
const (
	testPDKVCacheAwareFactoryName = "test-factory-pd-kv-cache-aware"
)

func TestPDKVCacheAwareFactory(t *testing.T) {
	rawParams := json.RawMessage(`{}`)
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := PDKVCacheAwareFactory(testPDKVCacheAwareFactoryName, rawParams, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin, got nil")
	}
	scorer, ok := plugin.(*PDKVCacheAware)
	if !ok {
		t.Fatal("expected *PDKVCacheAware type")
	}
	if scorer.typedName.Name != testPDKVCacheAwareFactoryName {
		t.Errorf("expected name %s, got %s", testPDKVCacheAwareFactoryName, scorer.typedName.Name)
	}
}

// TestPDKVCacheAwareFactoryWithParams tests PDKVCacheAwareFactory with parameters.
const (
	testPDKVCacheAwareFactoryWithParamsName = "test-factory-with-params"
	testPDParamKVCacheWeight                = 1.5
	testPDParamXPUWeight                    = 2.5
	testPDParamPrefillWaitingWeight         = 3.5
	testPDParamDecodeWaitingWeight          = 4.5
	testPDParamPrefillPodScoreWeight        = 0.7
	testPDParamDecodePodScoreWeight         = 0.3
)

func TestPDKVCacheAwareFactoryWithParams(t *testing.T) {
	rawParams := json.RawMessage(`{
		"kvCacheHitNotRateWeight": ` + strconv.FormatFloat(testPDParamKVCacheWeight, 'f', -1, 64) + `,
		"xpuCacheUsageWeight": ` + strconv.FormatFloat(testPDParamXPUWeight, 'f', -1, 64) + `,
		"prefillWaitingRequestWeight": ` + strconv.FormatFloat(testPDParamPrefillWaitingWeight, 'f', -1, 64) + `,
		"decodeWaitingRequestWeight": ` + strconv.FormatFloat(testPDParamDecodeWaitingWeight, 'f', -1, 64) + `,
		"prefillPodScoreWeight": ` + strconv.FormatFloat(testPDParamPrefillPodScoreWeight, 'f', -1, 64) + `,
		"decodePodScoreWeight": ` + strconv.FormatFloat(testPDParamDecodePodScoreWeight, 'f', -1, 64) + `
	}`)
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := PDKVCacheAwareFactory(testPDKVCacheAwareFactoryWithParamsName, rawParams, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	scorer, ok := plugin.(*PDKVCacheAware)
	if !ok {
		t.Fatal("expected *PDKVCacheAware type, but got different type")
	}
	if scorer.params.KVCacheNotHitRateWeight != testPDParamKVCacheWeight {
		t.Errorf("expected kvCacheWeight %f, got %f", testPDParamKVCacheWeight, scorer.params.KVCacheNotHitRateWeight)
	}
}

// TestPDKVCacheAwareFactoryWithInvalidParams tests PDKVCacheAwareFactory with invalid parameters.
const (
	testPDKVCacheAwareFactoryInvalidParamsName = "test-factory-invalid-params"
	testPDInvalidJSON                          = "{invalid json}"
)

func TestPDKVCacheAwareFactoryWithInvalidParams(t *testing.T) {
	rawParams := json.RawMessage(testPDInvalidJSON)
	handle := plugins.NewEppHandle(context.Background(), nil)

	_, err := PDKVCacheAwareFactory(testPDKVCacheAwareFactoryInvalidParamsName, rawParams, handle)

	if err == nil {
		t.Error("expected error for invalid parameters, got nil")
	}
}

// TestPDKVCacheAwareFactoryWithNilParams tests PDKVCacheAwareFactory with nil parameters.
func TestPDKVCacheAwareFactoryWithNilParams(t *testing.T) {
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := PDKVCacheAwareFactory(testPDKVCacheAwareFactoryName, nil, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin, got nil")
	}
}

// TestNewPDKVCacheAware tests NewPDKVCacheAware function.

func TestNewPDKVCacheAware(t *testing.T) {
	params := createPDKVCacheAwareParams(testPDManagerHost, testPDManagerPort, pdKVCacheEndpointPath)
	scorer := NewPDKVCacheAware(context.Background(), params)

	if scorer.typedName.Type != PDKVCacheAwareType {
		t.Errorf("expected type %s, got %s", PDKVCacheAwareType, scorer.typedName.Type)
	}
	if scorer.kvCacheAware == nil {
		t.Error("expected non-nil kvCacheAware, got nil")
	}
}

// TestPDKVCacheAwareTypedName tests TypedName method.
const (
	testPDKVCacheAwareTypedNameName = "test-typed-name"
)

func TestPDKVCacheAwareTypedName(t *testing.T) {
	scorer := &PDKVCacheAware{
		typedName: plugins.TypedName{Type: PDKVCacheAwareType, Name: testPDKVCacheAwareTypedNameName},
	}

	typedName := scorer.TypedName()

	if typedName.Type != PDKVCacheAwareType {
		t.Errorf("expected type %s, got %s", PDKVCacheAwareType, typedName.Type)
	}
	if typedName.Name != testPDKVCacheAwareTypedNameName {
		t.Errorf("expected name %s, got %s", testPDKVCacheAwareTypedNameName, typedName.Name)
	}
}

// TestPDKVCacheAwareWithName tests WithName method.
const (
	testPDKVCacheAwareOriginalName = "original-name"
	testPDKVCacheAwareNewName      = "new-name"
)

func TestPDKVCacheAwareWithName(t *testing.T) {
	original := &PDKVCacheAware{
		typedName: plugins.TypedName{Type: PDKVCacheAwareType, Name: testPDKVCacheAwareOriginalName},
	}

	newScorer := original.WithName(testPDKVCacheAwareNewName)

	if newScorer.typedName.Name != testPDKVCacheAwareNewName {
		t.Errorf("expected name %s, got %s", testPDKVCacheAwareNewName, newScorer.typedName.Name)
	}
	if original.typedName.Name != testPDKVCacheAwareNewName {
		t.Error("expected original scorer to be modified")
	}
}

// TestPDKVCacheAwareScoreSuccess tests Score with successful execution.
const (
	pdKVCacheEndpointPath     = "/match_sort"
	pdKVCacheWeight           = 1.0
	pdXpuUsageWeight          = 2.0
	pdPrefillWaitingWeight    = 3.0
	pdDecodeWaitingWeight     = 4.0
	pdPrefillPodScoreWeight   = 0.6
	pdDecodePodScoreWeight    = 0.4
	testPDSeed                = 42
	testPDGroup1ID            = "group-1"
	testPDGroup2ID            = "group-2"
	testPDGroup3ID            = "group-3"
	testPDGroup1PrefillCount  = 1
	testPDGroup1DecodeCount   = 3
	testPDGroup2PrefillCount  = 3
	testPDGroup2DecodeCount   = 1
	testPDGroup3PrefillCount  = 3
	testPDGroup3DecodeCount   = 3
	testPDLeaderIP            = "10.0.2.1"
	testPDRequestID           = "test-request"
	testPDModelName           = "test-model"
	testPDPrompt              = "test prompt"
	testPDOKMessage           = "ok"
	testPDHTTPMethodPost      = "POST"
	testPDExpectedEmptyScores = 0
	testPDZeroValue           = 0.0
	testPDZeroInt             = 0
	testPDMaxRandomValue      = 21
)

func TestPDKVCacheAwareScoreSuccess(t *testing.T) {
	server, host, port := setupPDMockKVCacheServer(t)
	defer server.Close()

	scorer := NewPDKVCacheAware(context.Background(), createPDKVCacheAwareParams(host, port, pdKVCacheEndpointPath))
	groups := createPDTestGroups()
	state := createPDStateWithGroups(groups)
	request := createPDTestRequest()
	pods := createPDPodsFromGroups(groups)

	scores := scorer.Score(context.Background(), state, request, pods)

	verifyPDScoreResults(t, scores, state, groups)
}

// TestPDKVCacheAwareScoreMissingState tests Score when state is missing.
func TestPDKVCacheAwareScoreMissingState(t *testing.T) {
	scorer := NewPDKVCacheAware(context.Background(), createPDKVCacheAwareParams(testPDManagerHost, testPDManagerPort, pdKVCacheEndpointPath))
	request := createPDTestRequest()

	scores := scorer.Score(context.Background(), &types.CycleState{}, request, []types.Pod{})

	if len(scores) != testPDExpectedEmptyScores {
		t.Errorf("expected %d scores, got %d", testPDExpectedEmptyScores, len(scores))
	}
}

// TestPDKVCacheAwareScoreEmptyGroupList tests Score with empty group list.
func TestPDKVCacheAwareScoreEmptyGroupList(t *testing.T) {
	state := &types.CycleState{}
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{Groups: []common.PDGroup{}})
	scorer := NewPDKVCacheAware(context.Background(), createPDKVCacheAwareParams(testPDManagerHost, testPDManagerPort, pdKVCacheEndpointPath))
	request := createPDTestRequest()

	scores := scorer.Score(context.Background(), state, request, []types.Pod{})

	if len(scores) != testPDExpectedEmptyScores {
		t.Errorf("expected %d scores, got %d", testPDExpectedEmptyScores, len(scores))
	}
}

// TestPDKVCacheAwareScoreKVManagerError tests Score when KV cache manager returns error.
const (
	testPDErrorResponseBody = "error"
	testPDEmptyPath         = ""
)

func TestPDKVCacheAwareScoreKVManagerError(t *testing.T) {
	errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, testPDErrorResponseBody, http.StatusInternalServerError)
	}))
	defer errorServer.Close()

	host, port := parsePDServerURL(t, errorServer.URL)
	scorer := NewPDKVCacheAware(context.Background(), createPDKVCacheAwareParams(host, port, testPDEmptyPath))
	groups := createPDTestGroups()
	state := createPDStateWithGroups(groups)
	request := createPDTestRequest()
	pods := createPDPodsFromGroups(groups)

	scores := scorer.Score(context.Background(), state, request, pods)

	if len(scores) != testPDExpectedEmptyScores {
		t.Errorf("expected %d scores due to kv manager error, got %d", testPDExpectedEmptyScores, len(scores))
	}
}

// TestPDKVCacheAwareScoreEmptyPrefillPods tests Score with empty prefill pods.
const (
	testPDEmptyGroupID = "empty-group"
)

func TestPDKVCacheAwareScoreEmptyPrefillPods(t *testing.T) {
	server, host, port := setupPDMockKVCacheServer(t)
	defer server.Close()

	scorer := NewPDKVCacheAware(context.Background(), createPDKVCacheAwareParams(host, port, pdKVCacheEndpointPath))
	groups := []common.PDGroup{
		{
			ID:          testPDEmptyGroupID,
			PrefillPods: []types.ScoredPod{},
			DecodePods:  createPDDecodePods(testPDEmptyGroupID, 1),
			LeaderPod:   createPDLeaderPod(testPDEmptyGroupID),
		},
	}
	state := createPDStateWithGroups(groups)
	request := createPDTestRequest()

	scores := scorer.Score(context.Background(), state, request, []types.Pod{})

	if len(scores) != testPDExpectedEmptyScores {
		t.Errorf("expected %d scores, got %d", testPDExpectedEmptyScores, len(scores))
	}
}

// TestPDKVCacheAwareScoreDecodePodNilMetrics tests Score with decode pod having nil metrics.
const (
	testPDNilMetricsGroupID = "nil-metrics-group"
)

func TestPDKVCacheAwareScoreDecodePodNilMetrics(t *testing.T) {
	server, host, port := setupPDMockKVCacheServer(t)
	defer server.Close()

	scorer := NewPDKVCacheAware(context.Background(), createPDKVCacheAwareParams(host, port, pdKVCacheEndpointPath))
	decodePod := types.ScoredPod{
		Pod: &types.PodMetrics{
			Pod:          &backend.Pod{PodName: "decode-nil-metrics", Address: "10.0.1.1"},
			MetricsState: nil,
		},
		Score: testPDZeroValue,
	}
	groups := []common.PDGroup{
		{
			ID:          testPDNilMetricsGroupID,
			PrefillPods: createPDPrefillPods(testPDNilMetricsGroupID, 1),
			DecodePods:  []types.ScoredPod{decodePod},
			LeaderPod:   createPDLeaderPod(testPDNilMetricsGroupID),
		},
	}
	state := createPDStateWithGroups(groups)
	request := createPDTestRequest()
	pods := createPDPodsFromGroups(groups)

	scores := scorer.Score(context.Background(), state, request, pods)

	if len(scores) != testPDExpectedEmptyScores {
		t.Errorf("expected %d scores, got %d", testPDExpectedEmptyScores, len(scores))
	}
}

// Helper functions

const (
	testPDManagerHost = "localhost"
	testPDManagerPort = 8080
)

func setupPDMockKVCacheServer(t *testing.T) (*httptest.Server, string, int) {
	mux := http.NewServeMux()
	mux.HandleFunc(pdKVCacheEndpointPath, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != testPDHTTPMethodPost {
			t.Fatalf("expected %s method, got %s", testPDHTTPMethodPost, r.Method)
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
				{"10.0.0.1": 0.8}, {"10.0.0.2": 0.5}, {"10.0.0.3": 0.6},
				{"10.0.0.4": 0.7}, {"10.0.0.5": 0.4}, {"10.0.0.6": 0.9}, {"10.0.0.7": 0.3},
			},
			Status:  http.StatusOK,
			Message: testPDOKMessage,
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("failed to encode response: %v", err)
		}
	})
	server := httptest.NewServer(mux)
	host, port := parsePDServerURL(t, server.URL)
	return server, host, port
}

func parsePDServerURL(t *testing.T, serverURL string) (string, int) {
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

func createPDKVCacheAwareParams(host string, port int, path string) common.KVCacheAwareParameters {
	return common.KVCacheAwareParameters{
		KVCacheNotHitRateWeight:     pdKVCacheWeight,
		XPUCacheUsageWeight:         pdXpuUsageWeight,
		PrefillWaitingRequestWeight: pdPrefillWaitingWeight,
		DecodeWaitingRequestWeight:  pdDecodeWaitingWeight,
		PrefillPodScoreWeight:       pdPrefillPodScoreWeight,
		DecodePodScoreWeight:        pdDecodePodScoreWeight,
		KVCacheManagerIP:            host,
		KVCacheManagerPort:          port,
		KVCacheManagerPath:          path,
		KVCacheManagerTimeout:       common.DefaultKVCacheManagerTimeout,
	}
}

func createPDTestGroups() []common.PDGroup {
	rng := rand.New(rand.NewSource(testPDSeed))
	groups := []common.PDGroup{
		createPDGroupWithCounts(rng, testPDGroup1ID, testPDGroup1PrefillCount, testPDGroup1DecodeCount),
		createPDGroupWithCounts(rng, testPDGroup2ID, testPDGroup2PrefillCount, testPDGroup2DecodeCount),
		createPDGroupWithCounts(rng, testPDGroup3ID, testPDGroup3PrefillCount, testPDGroup3DecodeCount),
	}
	return groups
}

const (
	testPDPrefillIPBase = "10.0.0."
	testPDDecodeIPBase  = "10.0.1."
	testPDPrefillSuffix = "-prefill-"
	testPDDecodeSuffix  = "-decode-"
	testPDLeaderSuffix  = "-leader"
)

func createPDGroupWithCounts(rng *rand.Rand, groupID string, prefillCount, decodeCount int) common.PDGroup {
	prefillPods := make([]types.ScoredPod, prefillCount)
	for i := 0; i < prefillCount; i++ {
		ip := testPDPrefillIPBase + strconv.Itoa(i+1)
		prefillPods[i] = newPDScoredPodWithMetrics(
			groupID+testPDPrefillSuffix+strconv.Itoa(i),
			ip,
			rng.Float64(),
			rng.Intn(testPDMaxRandomValue),
		)
	}
	decodePods := createPDDecodePods(groupID, decodeCount)
	leaderPod := createPDLeaderPod(groupID)
	return common.PDGroup{
		ID:          groupID,
		PrefillPods: prefillPods,
		DecodePods:  decodePods,
		LeaderPod:   leaderPod,
	}
}

func createPDPrefillPods(groupID string, count int) []types.ScoredPod {
	pods := make([]types.ScoredPod, count)
	for i := 0; i < count; i++ {
		ip := testPDPrefillIPBase + strconv.Itoa(i+1)
		pods[i] = newPDScoredPodWithMetrics(
			groupID+testPDPrefillSuffix+strconv.Itoa(i),
			ip,
			testPDZeroValue,
			testPDZeroInt,
		)
	}
	return pods
}

func createPDDecodePods(groupID string, count int) []types.ScoredPod {
	pods := make([]types.ScoredPod, count)
	for i := 0; i < count; i++ {
		ip := testPDDecodeIPBase + strconv.Itoa(i+1)
		pods[i] = newPDScoredPodWithMetrics(
			groupID+testPDDecodeSuffix+strconv.Itoa(i),
			ip,
			testPDZeroValue,
			testPDZeroInt,
		)
	}
	return pods
}

func createPDLeaderPod(groupID string) types.ScoredPod {
	return newPDScoredPodWithMetrics(
		groupID+testPDLeaderSuffix,
		testPDLeaderIP,
		testPDZeroValue,
		testPDZeroInt,
	)
}

func createPDStateWithGroups(groups []common.PDGroup) *types.CycleState {
	state := &types.CycleState{}
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{Groups: groups})
	return state
}

func createPDTestRequest() *types.LLMRequest {
	return &types.LLMRequest{
		RequestId:   testPDRequestID,
		TargetModel: testPDModelName,
		Body: &types.LLMRequestBody{
			Completions: &types.CompletionsRequest{
				Prompt: testPDPrompt,
			},
		},
	}
}

func createPDPodsFromGroups(groups []common.PDGroup) []types.Pod {
	pods := make([]types.Pod, 0)
	for _, group := range groups {
		for _, prefillPod := range group.PrefillPods {
			pods = append(pods, prefillPod.Pod)
		}
		for _, decodePod := range group.DecodePods {
			pods = append(pods, decodePod.Pod)
		}
		pods = append(pods, group.LeaderPod.Pod)
	}
	return pods
}

func verifyPDScoreResults(t *testing.T, scores map[types.Pod]float64, state *types.CycleState, groups []common.PDGroup) {
	if scores == nil {
		t.Fatal("scores should not be nil")
	}
	if len(scores) != testPDExpectedEmptyScores {
		t.Errorf("expected %d scores, got %d", testPDExpectedEmptyScores, len(scores))
	}
	updatedPDGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err != nil {
		t.Fatalf("failed to read pd group list from state: %v", err)
	}
	if len(updatedPDGroupList.Groups) != len(groups) {
		t.Errorf("expected %d groups, got %d", len(groups), len(updatedPDGroupList.Groups))
	}
	for i, group := range groups {
		pdGroup := updatedPDGroupList.Groups[i]
		if pdGroup.ID != group.ID {
			t.Errorf("expected group ID %s, got %s", group.ID, pdGroup.ID)
		}
		if pdGroup.SelectedPrefillPod.Pod == nil {
			t.Errorf("group %s: SelectedPrefillPod should be set", group.ID)
		}
		if pdGroup.SelectedDecodePod.Pod == nil {
			t.Errorf("group %s: SelectedDecodePod should be set", group.ID)
		}
		if pdGroup.LeaderPod.Score == testPDZeroValue {
			t.Errorf("group %s: LeaderPod.Score should be calculated", group.ID)
		}
	}
}

func newPDScoredPodWithMetrics(name, address string, kvCacheUsagePercent float64, waitingQueueSize int) types.ScoredPod {
	return types.ScoredPod{
		Pod: &types.PodMetrics{
			Pod: &backend.Pod{
				PodName: name,
				Address: address,
			},
			MetricsState: &metrics.MetricsState{
				KVCacheUsagePercent: kvCacheUsagePercent,
				WaitingQueueSize:    waitingQueueSize,
			},
		},
		Score: testPDZeroValue,
	}
}
