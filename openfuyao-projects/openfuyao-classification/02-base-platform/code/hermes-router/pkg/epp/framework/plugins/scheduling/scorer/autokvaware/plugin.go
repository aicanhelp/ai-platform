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

// Package autokvaware provides the automatic KV-cache-aware scorer plugin.
package autokvaware

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"time"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/score"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PluginType is the registered plugin type identifier.
	PluginType                     = "auto-kvcache-aware-scorer"
	defaultKVCacheNotHitRateWeight = 1.0
	defaultXPUCacheUsageWeight     = 1.0
	defaultWaitingRequestWeight    = 1.0
	defaultInflightWeight          = 1.0
	defaultWeight                  = 1.0
	defaultGroupWeight             = 0.5
)

type pluginParameters struct {
	KVCacheNotHitRateWeight *float64 `json:"kvCacheNotHitRateWeight,omitempty"`
	XPUCacheUsageWeight     *float64 `json:"xpuCacheUsageWeight,omitempty"`
	WaitingRequestWeight    *float64 `json:"waitingRequestWeight,omitempty"`
	InflightWeight          *float64 `json:"inflightWeight,omitempty"`
	PrefillKVUsageWeight    *float64 `json:"prefillKVUsageWeight,omitempty"`
	PrefillPrefixWeight     *float64 `json:"prefillPrefixWeight,omitempty"`
	PrefillQueueWeight      *float64 `json:"prefillQueueWeight,omitempty"`
	PrefillInflightWeight   *float64 `json:"prefillInflightWeight,omitempty"`
	DecodeRunningWeight     *float64 `json:"decodeRunningWeight,omitempty"`
	DecodeKVUsageWeight     *float64 `json:"decodeKVUsageWeight,omitempty"`
	DecodeQueueWeight       *float64 `json:"decodeQueueWeight,omitempty"`
	DecodeInflightWeight    *float64 `json:"decodeInflightWeight,omitempty"`
	PrefillScoreWeight      *float64 `json:"prefillScoreWeight,omitempty"`
	DecodeScoreWeight       *float64 `json:"decodeScoreWeight,omitempty"`

	DeprecatedPrefillWaitingRequestWeight *float64 `json:"prefillWaitingRequestWeight,omitempty"`
	DeprecatedDecodeWaitingRequestWeight  *float64 `json:"decodeWaitingRequestWeight,omitempty"`
	DeprecatedPrefillPodScoreWeight       *float64 `json:"prefillPodScoreWeight,omitempty"`
	DeprecatedDecodePodScoreWeight        *float64 `json:"decodePodScoreWeight,omitempty"`
	DeprecatedKVCacheManagerTimeout       *float64 `json:"kvCacheManagerTimeout,omitempty"`
	DeprecatedKVCacheManagerIP            string   `json:"kvCacheManagerIP,omitempty"`
	DeprecatedKVCacheManagerPath          string   `json:"kvCacheManagerPath,omitempty"`
	DeprecatedKVCacheManagerPort          int      `json:"kvCacheManagerPort,omitempty"`
}

// Config holds aggregate and PD scoring weights after JSON parameters are mapped.
type Config struct {
	Aggregate score.AggregateWeights
	Prefill   score.PrefillWeights
	Decode    score.DecodeWeights
	Group     score.GroupWeights
}

// AutoKVCacheAwareScorer describes AutoKVCacheAwareScorer data used by Hermes Router.
type AutoKVCacheAwareScorer struct {
	typedName fwkplugin.TypedName
	config    Config
	rng       *rand.Rand
	randMu    sync.Mutex
}

var _ fwkscheduling.Scorer = (*AutoKVCacheAwareScorer)(nil)

// Factory constructs a plugin instance from raw parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("auto kv-aware scorer %q: %w", name, err)
	}
	return New(name, config), nil
}

// New constructs a plugin instance with validated configuration.
func New(name string, config Config) *AutoKVCacheAwareScorer {
	return &AutoKVCacheAwareScorer{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		config:    resolveConfig(config),
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// TypedName returns the framework plugin type and instance name.
func (s *AutoKVCacheAwareScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the scheduling scorer category.
func (s *AutoKVCacheAwareScorer) Category() fwkscheduling.ScorerCategory {
	return fwkscheduling.Balance
}

// Score assigns scheduling scores to candidate endpoints.
func (s *AutoKVCacheAwareScorer) Score(
	ctx context.Context,
	state *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) map[fwkscheduling.Endpoint]float64 {
	_ = ctx
	_ = state
	_ = request
	scores := make(map[fwkscheduling.Endpoint]float64, len(pods))
	aggregatePods := aggregateEndpoints(pods)
	maxAggregateQueue := score.MaxWaitingQueue(aggregatePods)
	maxAggregateInflight := score.MaxInflightTokens(aggregatePods)
	pdScorer := score.PDScorer{
		Prefill:      s.config.Prefill,
		Decode:       s.config.Decode,
		Group:        s.config.Group,
		Limits:       score.CollectPDLimits(pods),
		NextRandIntn: s.nextRandIntn,
	}

	for _, endpoint := range pods {
		if info, ok := pdgroup.GetDirect(endpoint); ok {
			groupScore, updated, ok := score.PDGroup(info, pdScorer)
			if !ok {
				continue
			}
			pdgroup.SetOwned(endpoint, updated)
			if endpoint != nil {
				scores[endpoint] = groupScore
			}
			continue
		}
		if endpoint == nil {
			continue
		}
		scores[endpoint] = score.Aggregate(endpoint, maxAggregateQueue, maxAggregateInflight, s.config.Aggregate)
	}

	return scores
}

func (s *AutoKVCacheAwareScorer) nextRandIntn(count int) int {
	if count <= 1 {
		return 0
	}
	s.randMu.Lock()
	defer s.randMu.Unlock()
	if s.rng == nil {
		s.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	return s.rng.Intn(count)
}

func parseAndValidateParameters(raw json.RawMessage) (Config, error) {
	params := pluginParameters{}
	if err := utils.DecodePluginParameters(raw, &params); err != nil {
		return Config{}, err
	}
	config := resolveConfig(Config{})
	assignWeight(&config.Aggregate.KVCacheNotHitRateWeight, params.KVCacheNotHitRateWeight)
	assignWeight(&config.Aggregate.XPUCacheUsageWeight, params.XPUCacheUsageWeight)
	assignWeight(&config.Aggregate.WaitingRequestWeight, params.WaitingRequestWeight)
	assignWeight(&config.Aggregate.InflightWeight, params.InflightWeight)
	assignWeight(&config.Prefill.KVWeight, params.PrefillKVUsageWeight)
	assignWeight(&config.Prefill.PrefixWeight, params.PrefillPrefixWeight)
	assignWeight(&config.Prefill.QueueWeight, params.PrefillQueueWeight)
	assignWeight(&config.Prefill.InflightWeight, params.PrefillInflightWeight)
	assignWeight(&config.Decode.RunningWeight, params.DecodeRunningWeight)
	assignWeight(&config.Decode.KVWeight, params.DecodeKVUsageWeight)
	assignWeight(&config.Decode.QueueWeight, params.DecodeQueueWeight)
	assignWeight(&config.Decode.InflightWeight, params.DecodeInflightWeight)
	assignWeight(&config.Group.PrefillScoreWeight, params.PrefillScoreWeight)
	assignWeight(&config.Group.DecodeScoreWeight, params.DecodeScoreWeight)
	if !validWeights(config) {
		return Config{}, fmt.Errorf("weights must be non-negative and each scoring path must have a positive total weight")
	}
	return config, nil
}

func assignWeight(dst *float64, src *float64) bool {
	if src == nil {
		return false
	}
	*dst = *src
	return true
}

func resolveConfig(config Config) Config {
	resolved := config
	if resolved.Aggregate.Sum() <= 0 {
		resolved.Aggregate = score.AggregateWeights{
			KVCacheNotHitRateWeight: defaultKVCacheNotHitRateWeight,
			XPUCacheUsageWeight:     defaultXPUCacheUsageWeight,
			WaitingRequestWeight:    defaultWaitingRequestWeight,
			InflightWeight:          defaultInflightWeight,
		}
	}
	score.ResolvePDWeightDefaults(
		&resolved.Prefill,
		&resolved.Decode,
		&resolved.Group,
		defaultWeight,
		defaultGroupWeight,
	)
	return resolved
}

func validWeights(config Config) bool {
	if config.Aggregate.HasNegative() ||
		config.Prefill.HasNegative() ||
		config.Decode.HasNegative() ||
		config.Group.HasNegative() {
		return false
	}
	return config.Aggregate.Sum() > 0 &&
		config.Prefill.Sum() > 0 &&
		config.Decode.Sum() > 0 &&
		config.Group.Sum() > 0
}

func aggregateEndpoints(pods []fwkscheduling.Endpoint) []fwkscheduling.Endpoint {
	aggregate := make([]fwkscheduling.Endpoint, 0, len(pods))
	for _, endpoint := range pods {
		if _, ok := pdgroup.GetDirect(endpoint); ok {
			continue
		}
		aggregate = append(aggregate, endpoint)
	}
	return aggregate
}
