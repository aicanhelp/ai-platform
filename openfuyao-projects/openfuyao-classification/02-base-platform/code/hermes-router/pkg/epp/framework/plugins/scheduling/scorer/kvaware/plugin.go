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

// Package kvaware provides aggregate-only KV-aware scoring plugins.
package kvaware

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/score"
)

const (
	// PluginType is the registered plugin type identifier.
	PluginType                     = "kvcache-aware-scorer"
	defaultKVCacheNotHitRateWeight = 1.0
	defaultXPUCacheUsageWeight     = 1.0
	defaultWaitingRequestWeight    = 1.0
	defaultInflightWeight          = 1.0
)

type pluginParameters struct {
	KVCacheNotHitRateWeight *float64 `json:"kvCacheNotHitRateWeight,omitempty"`
	XPUCacheUsageWeight     *float64 `json:"xpuCacheUsageWeight,omitempty"`
	WaitingRequestWeight    *float64 `json:"waitingRequestWeight,omitempty"`
	InflightWeight          *float64 `json:"inflightWeight,omitempty"`

	DeprecatedPrefillWaitingRequestWeight *float64 `json:"prefillWaitingRequestWeight,omitempty"`
	DeprecatedDecodeWaitingRequestWeight  *float64 `json:"decodeWaitingRequestWeight,omitempty"`
	DeprecatedPrefillPodScoreWeight       *float64 `json:"prefillPodScoreWeight,omitempty"`
	DeprecatedDecodePodScoreWeight        *float64 `json:"decodePodScoreWeight,omitempty"`
	DeprecatedKVCacheManagerTimeout       *float64 `json:"kvCacheManagerTimeout,omitempty"`
	DeprecatedKVCacheManagerIP            string   `json:"kvCacheManagerIP,omitempty"`
	DeprecatedKVCacheManagerPath          string   `json:"kvCacheManagerPath,omitempty"`
	DeprecatedKVCacheManagerPort          int      `json:"kvCacheManagerPort,omitempty"`
}

// Parameters describes Parameters data used by Hermes Router.
type Parameters struct {
	// KVCacheNotHitRateWeight stores the KVCacheNotHitRateWeight value.
	KVCacheNotHitRateWeight float64
	// XPUCacheUsageWeight stores the XPUCacheUsageWeight value.
	XPUCacheUsageWeight float64
	// WaitingRequestWeight stores the WaitingRequestWeight value.
	WaitingRequestWeight float64
	// InflightWeight stores the InflightWeight value.
	InflightWeight float64
}

// KVCacheAwareScorer describes KVCacheAwareScorer data used by Hermes Router.
type KVCacheAwareScorer struct {
	typedName fwkplugin.TypedName
	params    Parameters
}

var _ fwkscheduling.Scorer = (*KVCacheAwareScorer)(nil)

// Factory constructs a plugin instance from raw parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, err
	}
	return New(name, params), nil
}

// New constructs a plugin instance with validated configuration.
func New(name string, params Parameters) *KVCacheAwareScorer {
	return &KVCacheAwareScorer{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		params:    params,
	}
}

// TypedName returns the framework plugin type and instance name.
func (s *KVCacheAwareScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// Category returns the scheduling scorer category.
func (s *KVCacheAwareScorer) Category() fwkscheduling.ScorerCategory {
	return fwkscheduling.Affinity
}

// Score assigns scheduling scores to candidate endpoints.
func (s *KVCacheAwareScorer) Score(
	ctx context.Context,
	state *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	pods []fwkscheduling.Endpoint,
) map[fwkscheduling.Endpoint]float64 {
	_ = ctx
	_ = state
	_ = request
	aggregatePods := aggregateEndpoints(pods)
	maxQueue := score.MaxWaitingQueue(aggregatePods)
	maxInflight := score.MaxInflightTokens(aggregatePods)
	scores := make(map[fwkscheduling.Endpoint]float64, len(aggregatePods))

	for _, endpoint := range aggregatePods {
		if endpoint == nil {
			continue
		}
		scores[endpoint] = score.Aggregate(endpoint, maxQueue, maxInflight, s.aggregateWeights())
	}

	return scores
}

func (s *KVCacheAwareScorer) aggregateWeights() score.AggregateWeights {
	return score.AggregateWeights{
		KVCacheNotHitRateWeight: s.params.KVCacheNotHitRateWeight,
		XPUCacheUsageWeight:     s.params.XPUCacheUsageWeight,
		WaitingRequestWeight:    s.params.WaitingRequestWeight,
		InflightWeight:          s.params.InflightWeight,
	}
}

func parseAndValidateParameters(raw json.RawMessage) (Parameters, error) {
	params := defaultParameters()
	if len(raw) == 0 {
		return params, nil
	}

	decoded := pluginParameters{}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return Parameters{}, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", PluginType, err)
	}

	if decoded.KVCacheNotHitRateWeight != nil {
		params.KVCacheNotHitRateWeight = *decoded.KVCacheNotHitRateWeight
	}
	if decoded.XPUCacheUsageWeight != nil {
		params.XPUCacheUsageWeight = *decoded.XPUCacheUsageWeight
	}
	if decoded.WaitingRequestWeight != nil {
		params.WaitingRequestWeight = *decoded.WaitingRequestWeight
	}
	if decoded.InflightWeight != nil {
		params.InflightWeight = *decoded.InflightWeight
	}
	if err := validateParameters(params); err != nil {
		return Parameters{}, fmt.Errorf("invalid parameters for the '%s' scorer - %w", PluginType, err)
	}
	return params, nil
}

func validateParameters(params Parameters) error {
	if params.KVCacheNotHitRateWeight < 0 || params.XPUCacheUsageWeight < 0 ||
		params.WaitingRequestWeight < 0 || params.InflightWeight < 0 {
		return fmt.Errorf("weights must be greater than or equal to 0")
	}
	if params.KVCacheNotHitRateWeight+params.XPUCacheUsageWeight+params.WaitingRequestWeight+params.InflightWeight <= 0 {
		return fmt.Errorf("at least one weight must be greater than 0")
	}
	return nil
}

func defaultParameters() Parameters {
	return Parameters{
		KVCacheNotHitRateWeight: defaultKVCacheNotHitRateWeight,
		XPUCacheUsageWeight:     defaultXPUCacheUsageWeight,
		WaitingRequestWeight:    defaultWaitingRequestWeight,
		InflightWeight:          defaultInflightWeight,
	}
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
