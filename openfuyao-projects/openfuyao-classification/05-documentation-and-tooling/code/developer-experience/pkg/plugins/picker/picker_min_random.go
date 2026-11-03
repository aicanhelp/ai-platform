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

// Package picker provides various picker plugins.
package picker

import (
	"context"
	"encoding/json"
	"math/rand"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	PickerMinRandomType = "picker-min-random"
)

// PickerMinRandom is the plugin struct that implements the framework.Picker interface.
type PickerMinRandom struct {
	typedName plugins.TypedName
	rng       *rand.Rand
}

var _ framework.Picker = &PickerMinRandom{}

// PickerMinRandomFactory is a factory function that creates a plugin instance based on parameters specified by users in EPP configuration files.
func PickerMinRandomFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	return NewPickerMinRandom(name, handle.Context()), nil
}

// NewPickerMinRandom creates a plugin instance based on parameters specified by users in EPP configuration files.
func NewPickerMinRandom(name string, ctx context.Context) *PickerMinRandom {
	return &PickerMinRandom{
		typedName: plugins.TypedName{Type: PickerMinRandomType, Name: name},
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// TypedName returns the plugin's type name.
func (p *PickerMinRandom) TypedName() plugins.TypedName {
	return p.typedName
}

// WithName sets the plugin's name.
func (p *PickerMinRandom) WithName(name string) *PickerMinRandom {
	p.typedName.Name = name
	return p
}

// Pick selects the pod with the lowest score from the list of scored pods.
func (p *PickerMinRandom) Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	if len(scoredPods) == 0 {
		return &types.ProfileRunResult{}
	}

	if p.rng == nil {
		p.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	minScore := scoredPods[0].Score
	candidates := []*types.ScoredPod{scoredPods[0]}

	for _, sp := range scoredPods[1:] {
		switch {
		case sp.Score < minScore:
			minScore = sp.Score
			candidates = []*types.ScoredPod{sp}
		case sp.Score == minScore:
			candidates = append(candidates, sp)
		}
	}

	selected := candidates[p.rng.Intn(len(candidates))]

	return &types.ProfileRunResult{
		TargetPods: []types.Pod{selected.Pod},
	}
}
