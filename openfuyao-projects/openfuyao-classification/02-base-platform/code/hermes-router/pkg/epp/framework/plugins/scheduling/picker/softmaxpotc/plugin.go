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

// Package softmaxpotc provides Hermes Router support.
package softmaxpotc

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/pdgroup"
)

const (
	// PluginType is the registered plugin type identifier.
	PluginType         = "softmax-potc-picker"
	defaultTemperature = 1.0
)

type pluginParameters struct {
	Temperature *float64 `json:"temperature,omitempty"`
}

// Config configures softmax temperature.
type Config struct {
	// Temperature stores the Temperature value.
	Temperature float64
}

// Picker samples two endpoints and returns the higher-scored one.
type Picker struct {
	typedName   fwkplugin.TypedName
	temperature float64
	rng         *rand.Rand
	randMu      sync.Mutex
	randFloat   func() float64
}

var _ fwkscheduling.Picker = (*Picker)(nil)

// Factory constructs a picker from user parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	config, err := parseAndValidateParameters(raw)
	if err != nil {
		return nil, fmt.Errorf("softmax POTC picker %q: %w", name, err)
	}
	return New(name, config), nil
}

// New creates a picker with a seeded RNG.
func New(name string, config Config) *Picker {
	resolved := resolveConfig(config)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	return &Picker{
		typedName:   fwkplugin.TypedName{Type: PluginType, Name: name},
		temperature: resolved.Temperature,
		rng:         rng,
	}
}

// TypedName returns the framework plugin type and instance name.
func (p *Picker) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Pick selects target endpoints from scored candidates.
func (p *Picker) Pick(
	ctx context.Context,
	cycleState *fwkscheduling.CycleState,
	scoredPods []*fwkscheduling.ScoredEndpoint,
) *fwkscheduling.ProfileRunResult {
	_ = ctx
	_ = cycleState
	candidates := scoredPods
	if pdgroup.IsScoredPDMode(scoredPods) {
		candidates = filterReadyLeaders(scoredPods)
	}
	if len(candidates) == 0 {
		return &fwkscheduling.ProfileRunResult{}
	}
	if len(candidates) == 1 {
		return buildResult(candidates[0])
	}
	first, second := p.sampleTwo(candidates)
	winner := first
	if sanitizedScore(second) > sanitizedScore(first) {
		winner = second
	}
	return buildResult(winner)
}

func (p *Picker) sampleTwo(
	scoredPods []*fwkscheduling.ScoredEndpoint,
) (*fwkscheduling.ScoredEndpoint, *fwkscheduling.ScoredEndpoint) {
	firstIndex := p.sampleIndex(scoredPods)
	secondIndex := p.sampleIndexFromN(len(scoredPods) - 1)
	if secondIndex >= firstIndex {
		secondIndex++
	}
	return scoredPods[firstIndex], scoredPods[secondIndex]
}

func (p *Picker) sampleIndex(scoredPods []*fwkscheduling.ScoredEndpoint) int {
	weights := softmaxWeights(scoredPods, p.temperature)
	target := p.nextRandFloat()
	cumulative := 0.0
	for index, weight := range weights {
		cumulative += weight
		if target <= cumulative {
			return index
		}
	}
	return len(scoredPods) - 1
}

func (p *Picker) sampleIndexFromN(count int) int {
	target := p.nextRandFloat()
	step := 1.0 / float64(count)
	cumulative := 0.0
	for index := 0; index < count; index++ {
		cumulative += step
		if target <= cumulative {
			return index
		}
	}
	return count - 1
}

func (p *Picker) nextRandFloat() float64 {
	if p.randFloat != nil {
		return p.randFloat()
	}
	p.randMu.Lock()
	defer p.randMu.Unlock()
	if p.rng == nil {
		p.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
	return p.rng.Float64()
}

func buildResult(selected *fwkscheduling.ScoredEndpoint) *fwkscheduling.ProfileRunResult {
	if selected == nil || selected.Endpoint == nil {
		return &fwkscheduling.ProfileRunResult{}
	}
	return &fwkscheduling.ProfileRunResult{TargetEndpoints: []fwkscheduling.Endpoint{selected.Endpoint}}
}

func softmaxWeights(scoredPods []*fwkscheduling.ScoredEndpoint, temperature float64) []float64 {
	weights := make([]float64, len(scoredPods))
	maxScore := sanitizedScore(scoredPods[0])
	for _, endpoint := range scoredPods[1:] {
		if score := sanitizedScore(endpoint); score > maxScore {
			maxScore = score
		}
	}
	sum := 0.0
	for index, endpoint := range scoredPods {
		weight := math.Exp((sanitizedScore(endpoint) - maxScore) / temperature)
		weights[index] = weight
		sum += weight
	}
	for index := range weights {
		weights[index] /= sum
	}
	return weights
}

func sanitizedScore(endpoint *fwkscheduling.ScoredEndpoint) float64 {
	if endpoint == nil || math.IsNaN(endpoint.Score) || endpoint.Score < 0 {
		return 0
	}
	return endpoint.Score
}

func filterReadyLeaders(scoredPods []*fwkscheduling.ScoredEndpoint) []*fwkscheduling.ScoredEndpoint {
	leaders := make([]*fwkscheduling.ScoredEndpoint, 0, len(scoredPods))
	for _, scoredPod := range scoredPods {
		if scoredPod == nil || scoredPod.Endpoint == nil {
			continue
		}
		info, ok := pdgroup.GetDirect(scoredPod.Endpoint)
		if !ok || !selectedMembersReady(info) {
			continue
		}
		leaders = append(leaders, scoredPod)
	}
	return leaders
}

func selectedMembersReady(info *pdgroup.Info) bool {
	if pdgroup.Validate(info) != nil {
		return false
	}
	return info.SelectedPrefillPod != nil && info.SelectedPrefillPod.Endpoint != nil &&
		info.SelectedDecodePod != nil && info.SelectedDecodePod.Endpoint != nil
}

func parseAndValidateParameters(raw json.RawMessage) (Config, error) {
	params := pluginParameters{}
	trimmed := bytes.TrimSpace(raw)
	if bytes.Equal(trimmed, []byte("null")) {
		return Config{}, fmt.Errorf("invalid parameters: explicit null is not allowed")
	}
	if len(trimmed) > 0 {
		decoder := json.NewDecoder(bytes.NewReader(trimmed))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&params); err != nil {
			return Config{}, fmt.Errorf("invalid parameters: %w", err)
		}
	}
	config := resolveConfig(Config{})
	if params.Temperature != nil {
		config.Temperature = *params.Temperature
	}
	if config.Temperature <= 0 {
		return Config{}, fmt.Errorf("temperature must be greater than 0")
	}
	return config, nil
}

func resolveConfig(config Config) Config {
	resolved := config
	if resolved.Temperature <= 0 {
		resolved.Temperature = defaultTemperature
	}
	return resolved
}
