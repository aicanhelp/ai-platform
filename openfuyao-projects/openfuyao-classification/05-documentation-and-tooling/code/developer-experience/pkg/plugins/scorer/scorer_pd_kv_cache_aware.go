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

// Package scorer provides scoring plugins for various routing strategies.
package scorer

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	"hermes-router/pkg/plugins/common"
)

const (
	PDKVCacheAwareType = "scorer-pd-kv-cache-aware"
)

// PDKVCacheAwareFactory creates a PD KV cache aware scorer instance from configuration parameters.
func PDKVCacheAwareFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	params := common.KVCacheAwareParameters{
		KVCacheNotHitRateWeight:     common.DefaultKVCacheNotHitRateWeight,
		XPUCacheUsageWeight:         common.DefaultXPUCacheUsageWeight,
		PrefillWaitingRequestWeight: common.DefaultPrefillWaitingRequestWeight,
		DecodeWaitingRequestWeight:  common.DefaultDecodeWaitingRequestWeight,
		KVCacheManagerTimeout:       common.DefaultKVCacheManagerTimeout,
		PrefillPodScoreWeight:       common.DefaultPrefillPodScoreWeight,
		DecodePodScoreWeight:        common.DefaultDecodePodScoreWeight,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", PDKVCacheAwareType, err)
		}
	}
	return NewPDKVCacheAware(handle.Context(), params).WithName(name), nil
}

var _ framework.Scorer = &PDKVCacheAware{}

// NewPDKVCacheAware creates a PD KV cache aware scorer instance with the given parameters.
func NewPDKVCacheAware(ctx context.Context, params common.KVCacheAwareParameters) *PDKVCacheAware {
	kvCacheAware := common.NewKVCacheAware(
		params.KVCacheManagerIP,
		params.KVCacheManagerPort,
		params.KVCacheManagerPath,
		params.KVCacheManagerTimeout,
	)

	return &PDKVCacheAware{
		typedName:    plugins.TypedName{Type: PDKVCacheAwareType},
		params:       params,
		kvCacheAware: kvCacheAware,
	}
}

// PDKVCacheAware implements a scoring plugin based on PD KV cache hit rate and other factors.
type PDKVCacheAware struct {
	typedName    plugins.TypedName
	params       common.KVCacheAwareParameters
	kvCacheAware *common.KVCacheAware
}

// TypedName returns the plugin's type and name identifier.
func (s *PDKVCacheAware) TypedName() plugins.TypedName {
	return s.typedName
}

// WithName sets the plugin's name identifier.
func (s *PDKVCacheAware) WithName(name string) *PDKVCacheAware {
	s.typedName.Name = name
	return s
}

// Score calculates scores for each pod based on PD KV cache hit rate, XPU cache usage, and waiting request queue size.
func (s *PDKVCacheAware) Score(ctx context.Context, state *types.CycleState, request *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {

	logger := log.FromContext(ctx)

	// Results are stored in state, not returned in podScoreMap.
	podScoreMap := make(map[types.Pod]float64, len(pods))

	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](state, common.PDGroupsCycleStateKey)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to read pd group list from state")
		return podScoreMap
	}

	totalPrefillPods := 0
	for _, group := range pdGroupList.Groups {
		totalPrefillPods += len(group.PrefillPods)
	}

	prefillPodList := make([]types.Pod, 0, totalPrefillPods)
	for _, group := range pdGroupList.Groups {
		for _, scoredPod := range group.PrefillPods {
			prefillPodList = append(prefillPodList, scoredPod.Pod)
		}
	}

	// Send prefillPodList to KV cache manager and get response.
	hitRateMap, err := s.kvCacheAware.GetKVCacheHitRate(ctx, prefillPodList, request)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to get kv cache hit rate for prefill pods")
		return podScoreMap
	}

	// Calculate score for each PodGroup based on metrics and scoring function.
	for i := range pdGroupList.Groups {
		if err := s.calculatePodGroupScore(ctx, &pdGroupList.Groups[i], hitRateMap); err != nil {
			logger.V(logging.DEFAULT).Error(err, "failed to calculate pod group score")
			return podScoreMap
		}
	}

	// Write updated pdGroupList back to state for Picker to use.
	state.Write(common.PDGroupsCycleStateKey, pdGroupList)

	return podScoreMap
}

// calculatePodGroupScore calculates the score for a pod group based on the hit rate map.
func (s *PDKVCacheAware) calculatePodGroupScore(ctx context.Context, pdGroup *common.PDGroup, hitRateMap map[string]float64) error {
	logger := log.FromContext(ctx)

	// Calculate score for each prefill pod and record the one with the lowest score to podGroup.SelectedPrefillPod.
	common.CalculatePodKVScore(ctx, common.KVCacheScoreWeights{
		KVCacheNotHitRateWeight: s.params.KVCacheNotHitRateWeight,
		XPUCacheUsageWeight:     s.params.XPUCacheUsageWeight,
		WaitingRequestWeight:    s.params.PrefillWaitingRequestWeight,
	}, hitRateMap, pdGroup.PrefillPods)

	bestPrefill := pdGroup.PrefillPods[0]
	for _, scoredPod := range pdGroup.PrefillPods {
		if scoredPod.Score < bestPrefill.Score {
			bestPrefill = scoredPod
		}
	}
	pdGroup.SelectedPrefillPod = bestPrefill

	// Calculate score for each decode pod and record the one with the lowest score to podGroup.SelectedDecodePod.
	if err := s.calculateDecodeScore(ctx, pdGroup.DecodePods); err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to calculate decode score")
		return err
	}
	bestDecode := pdGroup.DecodePods[0]
	for _, scoredPod := range pdGroup.DecodePods {
		if scoredPod.Score < bestDecode.Score {
			bestDecode = scoredPod
		}
	}
	pdGroup.SelectedDecodePod = bestDecode

	// Calculate PD group total score based on prefill and decode scores, and record it to LeaderPod.Score.
	pdGroup.LeaderPod.Score = s.params.PrefillPodScoreWeight*pdGroup.SelectedPrefillPod.Score +
		s.params.DecodePodScoreWeight*pdGroup.SelectedDecodePod.Score

	return nil
}

// calculateDecodeScore calculates the score for a decode pod based on the waiting request queue size.
func (s *PDKVCacheAware) calculateDecodeScore(ctx context.Context, scoredPods []types.ScoredPod) error {
	logger := log.FromContext(ctx)

	maxWaitingRequest := common.CalculateMaxWaitingRequest(scoredPods)
	for i := range scoredPods {
		metrics := scoredPods[i].Pod.GetMetrics()
		if metrics == nil {
			continue
		}
		scoredPods[i].Score = common.CalculateWaitingRequestWeight(
			s.params.DecodeWaitingRequestWeight,
			metrics.WaitingQueueSize,
			maxWaitingRequest,
		)
		backendPod := scoredPods[i].Pod.GetPod()
		if backendPod != nil {
			logger.V(logging.DEBUG).Info("Calculated decode pod score",
				"podAddress", backendPod.Address,
				"score", scoredPods[i].Score,
				"waitingQueueSize", metrics.WaitingQueueSize,
				"maxWaitingRequest", maxWaitingRequest)
		}
	}
	return nil
}
