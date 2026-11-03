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

// Package decodekvaware provides the decode-first KV-cache-aware scorer plugin.
package decodekvaware

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
	PluginType    = "decode-kvcache-aware-scorer"
	defaultWeight = 1.0
)

type pluginParameters struct {
	PrefixWeight   *float64 `json:"prefixWeight,omitempty"`
	RunningWeight  *float64 `json:"runningWeight,omitempty"`
	KVUsageWeight  *float64 `json:"kvUsageWeight,omitempty"`
	QueueWeight    *float64 `json:"queueWeight,omitempty"`
	InflightWeight *float64 `json:"inflightWeight,omitempty"`
}

// Config holds decode scoring weights after JSON parameters are mapped.
type Config struct {
	Decode score.DecodeWeights
}

// DecodeKVCacheAwareScorer scores PD leaders by decode pod selection only.
type DecodeKVCacheAwareScorer struct {
	typedName fwkplugin.TypedName
	config    Config
	rng       *rand.Rand
	randMu    sync.Mutex
}

var _ fwkscheduling.Scorer = (*DecodeKVCacheAwareScorer)(nil)

// Factory constructs a plugin instance from raw parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("decode kv-aware scorer %q: %w", name, err)
	}
	return New(name, config), nil
}

// New constructs a plugin instance with validated configuration.
func New(name string, config Config) *DecodeKVCacheAwareScorer {
	return &DecodeKVCacheAwareScorer{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		config:    resolveConfig(config),
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// TypedName returns the framework plugin type and instance name.
func (s *DecodeKVCacheAwareScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the scheduling scorer category.
func (s *DecodeKVCacheAwareScorer) Category() fwkscheduling.ScorerCategory {
	return fwkscheduling.Balance
}

// Score assigns scheduling scores to candidate endpoints.
func (s *DecodeKVCacheAwareScorer) Score(
	ctx context.Context,
	state *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) map[fwkscheduling.Endpoint]float64 {
	_ = ctx
	_ = state
	_ = request
	scores := make(map[fwkscheduling.Endpoint]float64, len(pods))
	decodeScorer := score.DecodeScorer{
		Decode:       s.config.Decode,
		Limits:       score.CollectPDLimits(pods).Decode,
		NextRandIntn: s.nextRandIntn,
	}

	for _, endpoint := range pods {
		info, ok := pdgroup.GetDirect(endpoint)
		if !ok {
			continue
		}
		groupScore, updated, ok := score.DecodeGroup(info, decodeScorer)
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

func (s *DecodeKVCacheAwareScorer) nextRandIntn(count int) int {
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
	hasDecodeWeights := applyDecodeWeights(params, &config)
	applyDefaultWeights(&config, hasDecodeWeights)
	if !validWeights(config) {
		return Config{}, fmt.Errorf("weights must be non-negative and sum to greater than 0")
	}
	return config, nil
}

func applyDecodeWeights(params pluginParameters, config *Config) bool {
	applied := false
	if assignWeight(&config.Decode.PrefixWeight, params.PrefixWeight) {
		applied = true
	}
	if assignWeight(&config.Decode.RunningWeight, params.RunningWeight) {
		applied = true
	}
	if assignWeight(&config.Decode.KVWeight, params.KVUsageWeight) {
		applied = true
	}
	if assignWeight(&config.Decode.QueueWeight, params.QueueWeight) {
		applied = true
	}
	if assignWeight(&config.Decode.InflightWeight, params.InflightWeight) {
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

func applyDefaultWeights(config *Config, hasDecodeWeights bool) {
	if !hasDecodeWeights {
		config.Decode.FillDefaults(defaultWeight)
	}
}

func resolveConfig(config Config) Config {
	resolved := config
	resolved.Decode.FillDefaults(defaultWeight)
	return resolved
}

func validWeights(config Config) bool {
	if config.Decode.HasNegative() {
		return false
	}
	return config.Decode.Sum() > 0
}
