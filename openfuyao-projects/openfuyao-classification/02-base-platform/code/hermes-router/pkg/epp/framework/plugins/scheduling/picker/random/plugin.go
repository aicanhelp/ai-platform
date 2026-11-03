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

// Package random provides random picker plugins.
package random

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	endpointmeta "hermes-router/pkg/epp/internal/endpointmeta"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PluginType is the config-facing plugin identifier.
	PluginType = "uniform-random-picker"
)

// Picker picks uniformly among the prepared candidates.
type Picker struct {
	typedName fwkplugin.TypedName
	rng       *rand.Rand
}

var _ fwkscheduling.Picker = &Picker{}

// Factory constructs a plugin instance from raw parameters.
func Factory(name string, raw json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if err := parseParameters(raw); err != nil {
		return nil, fmt.Errorf("random picker %q: %w", name, err)
	}
	_ = handle
	return New(name), nil
}

// New creates a random picker with a fresh RNG.
func New(name string) *Picker {
	return &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		rng:       newRNG(),
	}
}

// TypedName returns the plugin's type name.
func (p *Picker) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// WithName sets the plugin's name.
func (p *Picker) WithName(name string) *Picker {
	p.typedName.Name = name
	return p
}

// Pick selects one candidate uniformly. In PD mode, leaders are prepared first.
func (p *Picker) Pick(
	ctx context.Context,
	cycleState *fwkscheduling.CycleState,
	scoredPods []*fwkscheduling.ScoredEndpoint,
) *fwkscheduling.ProfileRunResult {
	_ = cycleState
	logger := log.FromContext(ctx).WithName("Picker.Pick")

	if len(scoredPods) == 0 {
		return &fwkscheduling.ProfileRunResult{}
	}

	p.ensureRNG()
	candidates := pdgroup.PreparePickerCandidates(scoredPods, p.rng)
	if len(candidates) == 0 {
		return &fwkscheduling.ProfileRunResult{}
	}

	logCandidatePodsSummary(ctx, "Picker.Pick", "Random picker candidate pods summary", candidates)

	selected := candidates[p.rng.Intn(len(candidates))]

	var selectedName, selectedIP string
	if selected != nil && selected.Endpoint != nil {
		selectedName = endpointmeta.EndpointPodName(selected.Endpoint)
		selectedIP = endpointmeta.EndpointAddress(selected.Endpoint)
	}
	logger.V(logutil.DEBUG).Info("Random picker final selection detail",
		"podCount", len(candidates),
		"selectedName", selectedName,
		"selectedIP", selectedIP,
		"score", selected.Score,
	)

	return &fwkscheduling.ProfileRunResult{TargetEndpoints: []fwkscheduling.Endpoint{selected.Endpoint}}
}

func (p *Picker) ensureRNG() {
	if p.rng == nil {
		p.rng = newRNG()
	}
}

func newRNG() *rand.Rand {
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}

func parseParameters(raw json.RawMessage) error {
	return utils.ValidateEmptyPluginParameters(raw)
}

func logCandidatePodsSummary(
	ctx context.Context,
	loggerName string,
	msg string,
	scoredPods []*fwkscheduling.ScoredEndpoint,
) {
	if len(scoredPods) == 0 {
		return
	}

	logger := log.FromContext(ctx)
	if loggerName != "" {
		logger = logger.WithName(loggerName)
	}

	logger.V(logutil.DEBUG).Info(msg,
		"podCount", len(scoredPods),
		"pods", func() []map[string]any {
			podsInfo := make([]map[string]any, 0, len(scoredPods))
			for _, scoredPod := range scoredPods {
				podInfo := map[string]any{}
				if scoredPod != nil {
					podInfo["score"] = scoredPod.Score
				}
				if scoredPod != nil && scoredPod.Endpoint != nil {
					podInfo["name"] = endpointmeta.EndpointPodName(scoredPod.Endpoint)
					podInfo["ip"] = endpointmeta.EndpointAddress(scoredPod.Endpoint)
				}
				podsInfo = append(podsInfo, podInfo)
			}
			return podsInfo
		}(),
	)
}
