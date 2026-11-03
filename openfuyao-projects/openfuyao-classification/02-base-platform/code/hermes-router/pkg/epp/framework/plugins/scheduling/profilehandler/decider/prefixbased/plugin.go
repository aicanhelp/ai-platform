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

// Package prefixbased decides Prefill from cache-indexer PrefixCacheMatchInfo.
package prefixbased

import (
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	"hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider"
	"hermes-router/pkg/epp/internal/score"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PluginType decides Prefill from cache-indexer PrefixCacheMatchInfo.
	PluginType = "prefix-based-prefill-decider"

	// DefaultMinHitRatio is the default MatchBlocks/TotalBlocks threshold for Prefill bypass.
	DefaultMinHitRatio = 0.8
	// DefaultMaxUncachedTokens is the default max residual prompt tokens for Prefill bypass.
	DefaultMaxUncachedTokens = 128
)

type pluginParameters struct {
	MinHitRatio       *float64 `json:"minHitRatio,omitempty"`
	MaxUncachedTokens *int     `json:"maxUncachedTokens,omitempty"`
}

// Config configures dual-threshold Prefill bypass.
type Config struct {
	MinHitRatio       float64
	MaxUncachedTokens int
}

// Decider reads PrefixCacheMatchInfo published by PrefixCacheProducer
// (cache-indexer) on the selected Decode endpoint and returns
// shouldRunPrefill=false only when dual thresholds pass.
type Decider struct {
	typedName fwkplugin.TypedName
	config    Config
}

var _ decider.Strategy = (*Decider)(nil)

// Factory constructs a PrefixBased Decider.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params pluginParameters
	if err := utils.DecodePluginParameters(raw, &params); err != nil {
		return nil, fmt.Errorf("prefix-based prefill decider %q: %w", name, err)
	}
	config, err := validateParameters(params)
	if err != nil {
		return nil, fmt.Errorf("prefix-based prefill decider %q: %w", name, err)
	}
	return New(name, config), nil
}

// New constructs a PrefixBased Decider.
func New(name string, config Config) *Decider {
	return &Decider{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		config:    config,
	}
}

// TypedName returns the plugin type and instance name.
func (d *Decider) TypedName() fwkplugin.TypedName {
	return d.typedName
}

// Decide returns false (d path / skip Prefill) only when PrefixCacheMatchInfo
// satisfies the dual thresholds. Missing cache-indexer data yields the pd path.
func (d *Decider) Decide(
	_ context.Context,
	request *fwkscheduling.LLMRequest,
	selectedDecode fwkscheduling.Endpoint,
) (bool, error) {
	if selectedDecode == nil {
		return true, fmt.Errorf("selected decode endpoint is nil")
	}
	info, ok := prefixCacheMatchInfoOf(selectedDecode)
	if !ok {
		return true, nil
	}
	totalBlocks := info.TotalBlocks()
	matchBlocks := info.MatchBlocks()
	blockSize := info.BlockSizeTokens()
	if totalBlocks <= 0 || blockSize <= 0 {
		return true, nil
	}
	hitRatio := score.ComputePrefixCacheScore(matchBlocks, totalBlocks)
	if hitRatio <= d.config.MinHitRatio {
		return true, nil
	}
	requestTokens, ok := requestTokenCount(request)
	if !ok {
		return true, nil
	}
	uncachedTokens := requestTokens - matchBlocks*blockSize
	if uncachedTokens < 0 {
		uncachedTokens = 0
	}
	if uncachedTokens >= d.config.MaxUncachedTokens {
		return true, nil
	}
	return false, nil
}

func requestTokenCount(request *fwkscheduling.LLMRequest) (int, bool) {
	if request == nil || request.TokenizedPrompt == nil || len(request.TokenizedPrompt.TokenIDs) == 0 {
		return 0, false
	}
	return len(request.TokenizedPrompt.TokenIDs), true
}

func prefixCacheMatchInfoOf(endpoint fwkscheduling.Endpoint) (*prefixattr.PrefixCacheMatchInfo, bool) {
	if endpoint == nil {
		return nil, false
	}
	value, ok := endpoint.Get(prefixattr.PrefixCacheMatchInfoKey)
	if !ok {
		return nil, false
	}
	info, ok := value.(*prefixattr.PrefixCacheMatchInfo)
	if !ok || info == nil {
		return nil, false
	}
	return info, true
}

func validateParameters(params pluginParameters) (Config, error) {
	config := Config{
		MinHitRatio:       DefaultMinHitRatio,
		MaxUncachedTokens: DefaultMaxUncachedTokens,
	}
	if params.MinHitRatio != nil {
		if *params.MinHitRatio < 0 || *params.MinHitRatio > 1 {
			return Config{}, fmt.Errorf("minHitRatio must be in [0,1]")
		}
		config.MinHitRatio = *params.MinHitRatio
	}
	if params.MaxUncachedTokens != nil {
		if *params.MaxUncachedTokens < 0 {
			return Config{}, fmt.Errorf("maxUncachedTokens must be >= 0")
		}
		config.MaxUncachedTokens = *params.MaxUncachedTokens
	}
	return config, nil
}
