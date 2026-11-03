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

// Package maxscore provides the max-score endpoint picker plugin.
package maxscore

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"time"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/utils"
)

// PluginType is the registered plugin type identifier.
const PluginType = "maximum-score-picker"

// Picker describes Picker data used by Hermes Router.
type Picker struct {
	typedName fwkplugin.TypedName
	rng       *rand.Rand
}

var _ fwkscheduling.Picker = (*Picker)(nil)

// Factory constructs a plugin instance from raw parameters.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if err := parseParameters(raw); err != nil {
		return nil, fmt.Errorf("max-score picker %q: %w", name, err)
	}
	return New(name), nil
}

// New constructs a plugin instance with validated configuration.
func New(name string) *Picker {
	return &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
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
	p.ensureRNG()
	return utils.SelectExtremeScoreResult(scoredPods, p.rng, func(candidate, current float64) bool {
		return candidate > current
	})
}

func (p *Picker) ensureRNG() {
	if p.rng == nil {
		p.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
}

func parseParameters(raw json.RawMessage) error {
	return utils.ValidateEmptyPluginParameters(raw)
}
