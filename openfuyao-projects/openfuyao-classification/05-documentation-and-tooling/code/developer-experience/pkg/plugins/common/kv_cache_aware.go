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

// Package common provides common utilities and structures for plugins.
package common

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// DefaultKVCacheNotHitRateWeight is the default weight for KV cache not hit rate.
	DefaultKVCacheNotHitRateWeight = 1.0
	// DefaultXPUCacheUsageWeight is the default weight for XPU cache usage metric.
	DefaultXPUCacheUsageWeight = 1.0
	// DefaultWaitingRequestWeight is the default weight for waiting request metric,
	// used for aggregate architecture inference backends.
	DefaultWaitingRequestWeight = 1.0
	// DefaultPrefillWaitingRequestWeight is the default weight for Prefill pod waiting request metric,
	// used for PD architecture inference backends.
	DefaultPrefillWaitingRequestWeight = 1.0
	// DefaultDecodeWaitingRequestWeight is the default weight for Decode pod waiting request metric,
	// used for PD architecture inference backends.
	DefaultDecodeWaitingRequestWeight = 1.0
	// DefaultPrefillPodScoreWeight is the default weight for Prefill pod final score,
	// used for PD architecture inference backends.
	DefaultPrefillPodScoreWeight = 1.0
	// DefaultDecodePodScoreWeight is the default weight for Decode pod final score,
	// used for PD architecture inference backends.
	DefaultDecodePodScoreWeight = 1.0
	// DefaultKVCacheManagerTimeout is the default timeout for KV cache manager requests.
	DefaultKVCacheManagerTimeout = 5 * time.Second
)

// KVCacheAwareParameters is the parameter struct for KV cache aware scorer.
type KVCacheAwareParameters struct {
	KVCacheNotHitRateWeight     float64 `json:"kvCacheHitNotRateWeight"`
	XPUCacheUsageWeight         float64 `json:"xpuCacheUsageWeight"`
	WaitingRequestWeight        float64 `json:"waitingRequestWeight"`
	PrefillWaitingRequestWeight float64 `json:"prefillWaitingRequestWeight"`
	DecodeWaitingRequestWeight  float64 `json:"decodeWaitingRequestWeight"`
	PrefillPodScoreWeight       float64 `json:"prefillPodScoreWeight"`
	DecodePodScoreWeight        float64 `json:"decodePodScoreWeight"`

	KVCacheManagerIP   string `json:"kvCacheManagerIP"`
	KVCacheManagerPort int    `json:"kvCacheManagerPort"`
	KVCacheManagerPath string `json:"kvCacheManagerPath"`

	// KVCacheManagerTimeout is the timeout for KV cache manager requests.
	KVCacheManagerTimeout time.Duration `json:"kvCacheManagerTimeout"`
}

// KVCacheAware is a common struct for getting KV cache hit rates,
// managing requests and responses to the KV cache manager.
type KVCacheAware struct {
	managerIP   string
	managerPort int
	managerPath string
	timeout     time.Duration
}

// NewKVCacheAware creates a new KVCacheAware instance.
func NewKVCacheAware(managerIP string, managerPort int, managerPath string, timeout time.Duration) *KVCacheAware {
	return &KVCacheAware{
		managerIP:   managerIP,
		managerPort: managerPort,
		managerPath: managerPath,
		timeout:     timeout,
	}
}

// GetKVCacheHitRate gets the KV cache hit rate for each pod in the pods list.
// the return map's key is the pod IP address and the value is the KV cache hit rate.
func (k *KVCacheAware) GetKVCacheHitRate(ctx context.Context, pods []types.Pod, request *types.LLMRequest) (
	map[string]float64, error) {
	if len(pods) == 0 {
		return nil, fmt.Errorf("pods list is empty")
	}

	requestData, err := k.buildRequestBody(pods, request)
	if err != nil {
		return nil, err
	}

	response, err := k.sendRequest(ctx, requestData)
	if err != nil {
		return nil, err
	}

	hitRateMap := k.parseResponse(response)
	return hitRateMap, nil
}

// KVCacheManagerRequestBody is the request body sent to the KV cache manager.
type KVCacheManagerRequestBody struct {
	Model string `json:"model"`
	// Messages contains the inference content from LLMRequest,
	// supporting both Completions and ChatCompletions request types.
	Messages interface{} `json:"messages"`
}

// KVCacheManagerResponse is the response body from the KV cache manager.
type KVCacheManagerResponse struct {
	// ServerScoreList is the (ip:hit rate) list for each pod returned by the KV cache manager.
	ServerScoreList []map[string]float64 `json:"server_score_list"`
	Message         string               `json:"message"`
	Status          int                  `json:"status"`
}

// buildRequestBody builds the request body KVCacheManagerRequestBody.
func (k *KVCacheAware) buildRequestBody(pods []types.Pod, request *types.LLMRequest) (
	map[string]interface{}, error) {
	if request == nil {
		return nil, fmt.Errorf("llm request is nil")
	}

	if len(pods) == 0 {
		return nil, fmt.Errorf("pods list is empty")
	}

	serverIPs := make([]string, 0, len(pods))
	for _, pod := range pods {
		k8sPod := pod.GetPod()
		if k8sPod == nil {
			return nil, fmt.Errorf("underlying kubernetes pod is nil")
		}

		serverIPs = append(serverIPs, k8sPod.Address)
	}

	if request.Body == nil {
		return nil, fmt.Errorf("request body is nil")
	}

	// Set Messages based on request type (Completions or ChatCompletions).
	var messages interface{}
	if request.Body.ChatCompletions != nil {
		messages = request.Body.ChatCompletions.Messages
	} else if request.Body.Completions != nil {
		messages = request.Body.Completions.Prompt
	} else {
		return nil, fmt.Errorf("unsupported request body type: neither Completions nor ChatCompletions")
	}

	requestBody := KVCacheManagerRequestBody{
		Model:    request.TargetModel,
		Messages: messages,
	}

	requestData := map[string]interface{}{
		"server_ip": serverIPs,
		"body":      requestBody,
	}

	return requestData, nil
}

// sendRequest sends the request to the KV cache manager and returns the response body.
func (k *KVCacheAware) sendRequest(ctx context.Context, requestData map[string]interface{}) (
	KVCacheManagerResponse, error) {
	requestBody, err := json.Marshal(requestData)
	if err != nil {
		return KVCacheManagerResponse{}, fmt.Errorf("failed to marshal request body: %w", err)
	}
	requestURL := fmt.Sprintf("http://%s:%d%s", k.managerIP, k.managerPort, k.managerPath)
	client := &http.Client{Timeout: k.timeout}
	// Send POST request and get response.
	response, err := client.Post(requestURL, "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		return KVCacheManagerResponse{}, fmt.Errorf("failed to send request, error is: %w", err)
	}

	if response.StatusCode != http.StatusOK {
		return KVCacheManagerResponse{}, fmt.Errorf("failed to send request, status is: %s", response.Status)
	}
	defer response.Body.Close()
	// Deserialize response body to KVCacheManagerResponse type.
	var responseBody KVCacheManagerResponse
	if err := json.NewDecoder(response.Body).Decode(&responseBody); err != nil {
		return KVCacheManagerResponse{}, fmt.Errorf("failed to decode response body: %w", err)
	}

	serverScoreListJSON, err := json.Marshal(responseBody.ServerScoreList)
	if err != nil {
		return KVCacheManagerResponse{}, fmt.Errorf("failed to marshal server score list: %w", err)
	}

	log.FromContext(ctx).V(logging.DEBUG).Info("Successfully received response from KV cache manager",
		"url", requestURL, "status", responseBody.Status,
		"message", responseBody.Message, "serverCount", len(responseBody.ServerScoreList),
		"serverScoreList", string(serverScoreListJSON))

	return responseBody, nil
}

// parseResponse parses the response body and returns the KV cache hit rate for each pod.
func (k *KVCacheAware) parseResponse(response KVCacheManagerResponse) map[string]float64 {
	scoreMap := make(map[string]float64, len(response.ServerScoreList))
	for _, serverScore := range response.ServerScoreList {
		for serverIP, hitRate := range serverScore {
			scoreMap[serverIP] = hitRate
		}
	}
	return scoreMap
}

// CalculateMaxWaitingRequest calculates the maximum waiting request count among all pods.
// Returns the maximum waiting request count among all pods.
func CalculateMaxWaitingRequest(scoredPods []types.ScoredPod) int {
	maxWaitingRequest := 0
	for _, scoredPod := range scoredPods {
		pod := scoredPod.Pod
		metrics := pod.GetMetrics()
		if metrics != nil && metrics.WaitingQueueSize > maxWaitingRequest {
			maxWaitingRequest = metrics.WaitingQueueSize
		}
	}
	return maxWaitingRequest
}

// CalculateWaitingRequestWeight calculates the waiting queue weight score.
// Returns the waiting queue weight score, returns 0 when maxWaitingRequest is 0.
func CalculateWaitingRequestWeight(gamma float64, waitingQueueSize int, maxWaitingRequest int) float64 {
	if maxWaitingRequest > 0 {
		return gamma * float64(waitingQueueSize) / float64(maxWaitingRequest)
	}
	return 0
}

// KVCacheScoreWeights contains weights for calculating KV cache scores.
type KVCacheScoreWeights struct {
	KVCacheNotHitRateWeight float64
	XPUCacheUsageWeight     float64
	WaitingRequestWeight    float64
}

// CalculatePodKVScore calculates KV cache scores for each pod based on metrics and weights.
func CalculatePodKVScore(ctx context.Context, weights KVCacheScoreWeights,
	hitRateMap map[string]float64, scoredPods []types.ScoredPod) {
	alpha := weights.KVCacheNotHitRateWeight
	beta := weights.XPUCacheUsageWeight
	gamma := weights.WaitingRequestWeight

	// Calculate maximum waiting request count.
	maxWaitingRequest := CalculateMaxWaitingRequest(scoredPods)

	// Calculate score for each pod.
	for i := range scoredPods {
		scoredPod := &scoredPods[i]
		pod := scoredPod.Pod
		backendPod := pod.GetPod()
		if backendPod == nil {
			continue
		}

		hitRate := hitRateMap[backendPod.Address]

		metrics := pod.GetMetrics()
		var kvCacheUsagePercent float64
		var waitingQueueSize int
		if metrics != nil {
			kvCacheUsagePercent = metrics.KVCacheUsagePercent
			waitingQueueSize = metrics.WaitingQueueSize
		}

		// Calculate waiting queue weight score.
		waitingRequestWeight := CalculateWaitingRequestWeight(gamma, waitingQueueSize, maxWaitingRequest)

		score := alpha*(1.0-hitRate) + beta*kvCacheUsagePercent + waitingRequestWeight
		scoredPod.Score = score

		log.FromContext(ctx).V(logging.DEBUG).Info("Calculated pod score",
			"podAddress", backendPod.Address, "score", score, "kvCacheHitRate", hitRate,
			"kvCacheUsagePercent", kvCacheUsagePercent, "waitingQueueSize", waitingQueueSize,
			"maxWaitingRequest", maxWaitingRequest)
	}
}
