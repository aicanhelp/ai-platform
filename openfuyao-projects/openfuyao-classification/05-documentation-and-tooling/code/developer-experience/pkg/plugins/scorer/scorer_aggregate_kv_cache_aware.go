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
	AggKVCacheAwareType = "scorer-aggregate-kv-cache-aware"
)

// AggKVCacheAware implements a scoring plugin based on KV cache hit rate and other factors.
type AggKVCacheAware struct {
	typedName    plugins.TypedName
	params       common.KVCacheAwareParameters
	kvCacheAware *common.KVCacheAware
}

// TypedName returns the plugin's type and name identifier.
func (a *AggKVCacheAware) TypedName() plugins.TypedName {
	return a.typedName
}

// WithName sets the plugin's name identifier.
func (a *AggKVCacheAware) WithName(n string) *AggKVCacheAware {
	a.typedName.Name = n
	return a
}

// Score calculates scores for each pod based on KV cache hit rate, XPU cache usage, and waiting request queue size.
func (s *AggKVCacheAware) Score(ctx context.Context, state *types.CycleState, request *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	logger := log.FromContext(ctx)

	podScoreMap := make(map[types.Pod]float64, len(pods))
	scoredPods := make([]types.ScoredPod, len(pods))
	for i, pod := range pods {
		scoredPods[i] = types.ScoredPod{
			Pod:   pod,
			Score: 0.0,
		}
	}

	// Get KV cache hit rate for each pod from the KV cache manager.
	hitRateMap, err := s.kvCacheAware.GetKVCacheHitRate(ctx, pods, request)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to get kv cache hit rate")
		return podScoreMap
	}

	// Calculate pod scores based on metrics and weights, outputting to scoredPods.
	common.CalculatePodKVScore(ctx, common.KVCacheScoreWeights{
		KVCacheNotHitRateWeight: s.params.KVCacheNotHitRateWeight,
		XPUCacheUsageWeight:     s.params.XPUCacheUsageWeight,
		WaitingRequestWeight:    s.params.WaitingRequestWeight,
	}, hitRateMap, scoredPods)

	for _, scoredPod := range scoredPods {
		podScoreMap[scoredPod.Pod] = scoredPod.Score
	}

	return podScoreMap
}

// Compile-time interface verification.
var _ framework.Scorer = &AggKVCacheAware{}

// NewAggKVCacheAware creates a KV cache aware scorer instance with the given parameters.
func NewAggKVCacheAware(ctx context.Context, params common.KVCacheAwareParameters) *AggKVCacheAware {
	kvCacheAware := common.NewKVCacheAware(
		params.KVCacheManagerIP,
		params.KVCacheManagerPort,
		params.KVCacheManagerPath,
		params.KVCacheManagerTimeout,
	)

	return &AggKVCacheAware{
		typedName:    plugins.TypedName{Type: AggKVCacheAwareType},
		params:       params,
		kvCacheAware: kvCacheAware,
	}
}

// AggKVCacheAwareFactory creates a KV cache aware scorer instance from configuration parameters.
func AggKVCacheAwareFactory(n string, jsonParam json.RawMessage, h plugins.Handle) (plugins.Plugin, error) {
	params := common.KVCacheAwareParameters{
		KVCacheNotHitRateWeight: common.DefaultKVCacheNotHitRateWeight,
		XPUCacheUsageWeight:     common.DefaultXPUCacheUsageWeight,
		WaitingRequestWeight:    common.DefaultWaitingRequestWeight,
		KVCacheManagerTimeout:   common.DefaultKVCacheManagerTimeout,
	}
	if len(jsonParam) == 0 {
		return NewAggKVCacheAware(h.Context(), params).WithName(n), nil
	}
	if err := json.Unmarshal(jsonParam, &params); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", AggKVCacheAwareType, err)
	}

	return NewAggKVCacheAware(h.Context(), params).WithName(n), nil
}
