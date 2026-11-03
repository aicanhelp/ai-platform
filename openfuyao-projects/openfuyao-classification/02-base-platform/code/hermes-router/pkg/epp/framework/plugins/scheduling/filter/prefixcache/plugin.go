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

// Package prefixcache provides prefix-cache request data and scheduling filters.
package prefixcache

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"time"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/score"
)

const (
	// PluginType is the registered plugin type identifier.
	PluginType              = "prefix-cache-filter"
	defaultThreshold        = 0.5
	defaultPFilter          = 0.8
	defaultMaxTTFTPenaltyMs = 5000.0
)

type pluginParameters struct {
	Threshold        *float64 `json:"threshold,omitempty"`
	PFilter          *float64 `json:"pFilter,omitempty"`
	MaxTTFTPenaltyMs *float64 `json:"maxTTFTPenaltyMs,omitempty"`
}

// Config configures probabilistic stickiness and the TTFT gate.
type Config struct {
	// Threshold stores the Threshold value.
	Threshold float64
	// PFilter stores the PFilter value.
	PFilter float64
	// MaxTTFTPenaltyMs stores the MaxTTFTPenaltyMs value.
	MaxTTFTPenaltyMs float64
}

// Filter preserves sticky endpoints while allowing load-driven escape hatches.
type Filter struct {
	typedName        fwkplugin.TypedName
	threshold        float64
	pFilter          float64
	maxTTFTPenaltyMs float64
	rng              *rand.Rand
	randMu           sync.Mutex
	randFloat        func() float64
}

var _ fwkscheduling.Filter = (*Filter)(nil)

// Factory constructs the filter from EPP plugin parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("prefix cache filter %q: %w", name, err)
	}
	return New(name, config), nil
}

// New creates a filter using resolved config defaults.
func New(name string, config Config) *Filter {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	return &Filter{
		typedName:        fwkplugin.TypedName{Type: PluginType, Name: name},
		threshold:        config.Threshold,
		pFilter:          config.PFilter,
		maxTTFTPenaltyMs: config.MaxTTFTPenaltyMs,
		rng:              rng,
	}
}

// TypedName returns the framework plugin type and instance name.
func (f *Filter) TypedName() fwkplugin.TypedName {
	return f.typedName
}

// Filter keeps sticky endpoints, probabilistically removes cold endpoints, and
// optionally breaks stickiness when TTFT is too high.
func (f *Filter) Filter(
	ctx context.Context,
	state *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) []fwkscheduling.Endpoint {
	_ = ctx
	_ = state
	_ = request
	if len(pods) <= 1 {
		return pods
	}

	stickyPods, nonStickyPods := f.partitionPods(pods)
	if f.shouldBreakStickiness(stickyPods, nonStickyPods) {
		return pods
	}

	filtered := make([]fwkscheduling.Endpoint, 0, len(pods))
	filtered = append(filtered, stickyPods...)
	for _, endpoint := range nonStickyPods {
		if f.shouldKeepNonSticky() {
			filtered = append(filtered, endpoint)
		}
	}
	if len(filtered) == 0 {
		return pods
	}
	return filtered
}

func (f *Filter) partitionPods(pods []fwkscheduling.Endpoint) ([]fwkscheduling.Endpoint, []fwkscheduling.Endpoint) {
	stickyPods := make([]fwkscheduling.Endpoint, 0, len(pods))
	nonStickyPods := make([]fwkscheduling.Endpoint, 0, len(pods))
	for _, endpoint := range pods {
		if effectivePrefixScore(endpoint) >= f.threshold {
			stickyPods = append(stickyPods, endpoint)
			continue
		}
		nonStickyPods = append(nonStickyPods, endpoint)
	}
	return stickyPods, nonStickyPods
}

func (f *Filter) shouldKeepNonSticky() bool {
	return f.nextRandFloat() >= f.pFilter
}

func (f *Filter) nextRandFloat() float64 {
	if f.randFloat != nil {
		return f.randFloat()
	}
	f.randMu.Lock()
	defer f.randMu.Unlock()
	if f.rng == nil {
		f.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	return f.rng.Float64()
}

func (f *Filter) shouldBreakStickiness(stickyPods, nonStickyPods []fwkscheduling.Endpoint) bool {
	if f.maxTTFTPenaltyMs <= 0 || len(stickyPods) == 0 || len(nonStickyPods) == 0 {
		return false
	}
	bestSticky, stickyOK := bestPredictedTTFT(stickyPods)
	bestNonSticky, nonStickyOK := bestPredictedTTFT(nonStickyPods)
	if !stickyOK || !nonStickyOK {
		return false
	}
	return bestSticky-bestNonSticky > f.maxTTFTPenaltyMs
}

func bestPredictedTTFT(pods []fwkscheduling.Endpoint) (float64, bool) {
	best := 0.0
	found := false
	for _, endpoint := range pods {
		ttft, ok := effectivePredictedTTFT(endpoint)
		if !ok {
			continue
		}
		if !found || ttft < best {
			best = ttft
			found = true
		}
	}
	return best, found
}

func effectivePredictedTTFT(endpoint fwkscheduling.Endpoint) (float64, bool) {
	if info, ok := pdgroup.GetDirect(endpoint); ok && pdgroup.Validate(info) == nil {
		return bestPrefillPredictedTTFT(info.PrefillPods)
	}
	prediction, ok := latencyPrediction(endpoint)
	if !ok || !prediction.TTFTValid() {
		return 0, false
	}
	return prediction.TTFT(), true
}

func bestPrefillPredictedTTFT(pods []fwkscheduling.ScoredEndpoint) (float64, bool) {
	best := 0.0
	found := false
	for _, pod := range pods {
		prediction, ok := latencyPrediction(pod.Endpoint)
		if !ok || !prediction.TTFTValid() {
			continue
		}
		if !found || prediction.TTFT() < best {
			best = prediction.TTFT()
			found = true
		}
	}
	return best, found
}

func latencyPrediction(endpoint fwkscheduling.Endpoint) (*latencyattr.LatencyPredictionInfo, bool) {
	if endpoint == nil {
		return nil, false
	}
	value, ok := endpoint.Get(latencyattr.LatencyPredictionInfoKey)
	if !ok {
		return nil, false
	}
	prediction, ok := value.(*latencyattr.LatencyPredictionInfo)
	if !ok || prediction == nil {
		return nil, false
	}
	return prediction, true
}

func parseAndValidateParameters(raw json.RawMessage) (Config, error) {
	params := pluginParameters{}
	if len(raw) > 0 {
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&params); err != nil {
			return Config{}, fmt.Errorf("invalid parameters: %w", err)
		}
	}
	config := Config{
		Threshold:        defaultThreshold,
		PFilter:          defaultPFilter,
		MaxTTFTPenaltyMs: defaultMaxTTFTPenaltyMs,
	}
	if params.Threshold != nil {
		config.Threshold = *params.Threshold
	}
	if params.PFilter != nil {
		config.PFilter = *params.PFilter
	}
	if params.MaxTTFTPenaltyMs != nil {
		config.MaxTTFTPenaltyMs = *params.MaxTTFTPenaltyMs
	}
	if config.Threshold < 0 || config.Threshold > 1 {
		return Config{}, fmt.Errorf("threshold must be between 0 and 1")
	}
	if config.PFilter < 0 || config.PFilter > 1 {
		return Config{}, fmt.Errorf("pFilter must be between 0 and 1")
	}
	if config.MaxTTFTPenaltyMs < 0 {
		return Config{}, fmt.Errorf("maxTTFTPenaltyMs must be greater than or equal to 0")
	}
	return config, nil
}

func effectivePrefixScore(endpoint fwkscheduling.Endpoint) float64 {
	if info, ok := pdgroup.GetDirect(endpoint); ok && pdgroup.Validate(info) == nil {
		return maxPrefillPrefixScore(info.PrefillPods)
	}
	return score.PrefixCacheScoreFromEndpoint(endpoint)
}

func maxPrefillPrefixScore(pods []fwkscheduling.ScoredEndpoint) float64 {
	maxScore := 0.0
	for _, pod := range pods {
		prefixScore := score.PrefixCacheScoreFromEndpoint(pod.Endpoint)
		if prefixScore > maxScore {
			maxScore = prefixScore
		}
	}
	return maxScore
}
