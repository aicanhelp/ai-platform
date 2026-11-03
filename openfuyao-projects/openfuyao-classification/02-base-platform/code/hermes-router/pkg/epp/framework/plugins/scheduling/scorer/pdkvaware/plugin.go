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

// Package pdkvaware provides the PD KV-cache-aware scorer plugin.
package pdkvaware

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
	PluginType         = "pd-kvcache-aware-scorer"
	defaultWeight      = 1.0
	defaultGroupWeight = 0.5
)

type pluginParameters struct {
	PrefillKVUsageWeight  *float64 `json:"prefillKVUsageWeight,omitempty"`
	PrefillPrefixWeight   *float64 `json:"prefillPrefixWeight,omitempty"`
	PrefillQueueWeight    *float64 `json:"prefillQueueWeight,omitempty"`
	PrefillInflightWeight *float64 `json:"prefillInflightWeight,omitempty"`
	DecodeRunningWeight   *float64 `json:"decodeRunningWeight,omitempty"`
	DecodeKVUsageWeight   *float64 `json:"decodeKVUsageWeight,omitempty"`
	DecodeQueueWeight     *float64 `json:"decodeQueueWeight,omitempty"`
	DecodeInflightWeight  *float64 `json:"decodeInflightWeight,omitempty"`
	PrefillScoreWeight    *float64 `json:"prefillScoreWeight,omitempty"`
	DecodeScoreWeight     *float64 `json:"decodeScoreWeight,omitempty"`
}

// Config holds PD scoring weights after JSON parameters are mapped.
type Config struct {
	Prefill score.PrefillWeights
	Decode  score.DecodeWeights
	Group   score.GroupWeights
}

// PDKVCacheAwareScorer describes PDKVCacheAwareScorer data used by Hermes Router.
type PDKVCacheAwareScorer struct {
	typedName fwkplugin.TypedName
	config    Config
	rng       *rand.Rand
	randMu    sync.Mutex
}

var _ fwkscheduling.Scorer = (*PDKVCacheAwareScorer)(nil)

// Factory constructs a plugin instance from raw parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("pd kv-aware scorer %q: %w", name, err)
	}
	return New(name, config), nil
}

// New constructs a plugin instance with validated configuration.
func New(name string, config Config) *PDKVCacheAwareScorer {
	return &PDKVCacheAwareScorer{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		config:    resolveConfig(config),
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// TypedName returns the framework plugin type and instance name.
func (s *PDKVCacheAwareScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the scheduling scorer category.
func (s *PDKVCacheAwareScorer) Category() fwkscheduling.ScorerCategory {
	return fwkscheduling.Balance
}

// Score assigns scheduling scores to candidate endpoints.
func (s *PDKVCacheAwareScorer) Score(
	ctx context.Context,
	state *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) map[fwkscheduling.Endpoint]float64 {
	_ = ctx
	_ = state
	_ = request
	scores := make(map[fwkscheduling.Endpoint]float64, len(pods))
	pdScorer := score.PDScorer{
		Prefill:      s.config.Prefill,
		Decode:       s.config.Decode,
		Group:        s.config.Group,
		Limits:       score.CollectPDLimits(pods),
		NextRandIntn: s.nextRandIntn,
	}

	for _, endpoint := range pods {
		info, ok := pdgroup.GetDirect(endpoint)
		if !ok {
			continue
		}
		groupScore, updated, ok := score.PDGroup(info, pdScorer)
		if !ok {
			continue
		}
		pdgroup.SetOwned(endpoint, updated)
		if endpoint != nil {
			scores[endpoint] = groupScore
		}
	}
	return scores
}

func (s *PDKVCacheAwareScorer) nextRandIntn(count int) int {
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
	config := Config{}
	hasPrefillWeights := applyPrefillWeights(params, &config)
	hasDecodeWeights := applyDecodeWeights(params, &config)
	hasGroupWeights := applyGroupWeights(params, &config)
	applyDefaultWeights(&config, hasPrefillWeights, hasDecodeWeights, hasGroupWeights)
	if !validWeights(config) {
		return Config{}, fmt.Errorf("weights must be non-negative and subgroup sums must be greater than 0")
	}
	return config, nil
}

func applyPrefillWeights(params pluginParameters, config *Config) bool {
	applied := false
	if assignWeight(&config.Prefill.KVWeight, params.PrefillKVUsageWeight) {
		applied = true
	}
	if assignWeight(&config.Prefill.PrefixWeight, params.PrefillPrefixWeight) {
		applied = true
	}
	if assignWeight(&config.Prefill.QueueWeight, params.PrefillQueueWeight) {
		applied = true
	}
	if assignWeight(&config.Prefill.InflightWeight, params.PrefillInflightWeight) {
		applied = true
	}
	return applied
}

func applyDecodeWeights(params pluginParameters, config *Config) bool {
	applied := false
	if assignWeight(&config.Decode.RunningWeight, params.DecodeRunningWeight) {
		applied = true
	}
	if assignWeight(&config.Decode.KVWeight, params.DecodeKVUsageWeight) {
		applied = true
	}
	if assignWeight(&config.Decode.QueueWeight, params.DecodeQueueWeight) {
		applied = true
	}
	if assignWeight(&config.Decode.InflightWeight, params.DecodeInflightWeight) {
		applied = true
	}
	return applied
}

func applyGroupWeights(params pluginParameters, config *Config) bool {
	applied := false
	if assignWeight(&config.Group.PrefillScoreWeight, params.PrefillScoreWeight) {
		applied = true
	}
	if assignWeight(&config.Group.DecodeScoreWeight, params.DecodeScoreWeight) {
		applied = true
	}
	return applied
}

func assignWeight(dst *float64, src *float64) bool {
	if src == nil {
		return false
	}
	*dst = *src
	return true
}

func applyDefaultWeights(config *Config, hasPrefillWeights, hasDecodeWeights, hasGroupWeights bool) {
	if !hasPrefillWeights {
		config.Prefill.FillDefaults(defaultWeight)
	}
	if !hasDecodeWeights {
		config.Decode.FillDefaults(defaultWeight)
	}
	if !hasGroupWeights {
		config.Group.FillDefaults(defaultGroupWeight)
	}
}

func resolveConfig(config Config) Config {
	resolved := config
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
	if config.Prefill.HasNegative() || config.Decode.HasNegative() || config.Group.HasNegative() {
		return false
	}
	return config.Prefill.Sum() > 0 && config.Decode.Sum() > 0 && config.Group.Sum() > 0
}
