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

// Package disagg provides the Decode-first PD ProfileHandler that can skip
// PD profile scheduling on the d path.
package disagg

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/framework/plugins/scheduling/profilehandler/decider"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	// PluginType is the registered ProfileHandler type.
	PluginType = "disagg-profile-handler"

	defaultDecodeProfileName = "decode"
	defaultPDProfileName     = "pd"
)

type pluginParameters struct {
	DecodeProfileName       string `json:"decodeProfileName,omitempty"`
	PDProfileName           string `json:"pdProfileName,omitempty"`
	PrefillDeciderPluginRef string `json:"prefillDeciderPluginRef"`
}

// Config configures profile names and the prefill Decider plugin reference.
type Config struct {
	DecodeProfileName       string
	PDProfileName           string
	PrefillDeciderPluginRef string
}

// Handler runs Decode first, then optionally the PD profile. When the Decider
// chooses the d path, Pick returns no further profiles so PD is neither
// scheduled nor invoked.
type Handler struct {
	typedName   fwkplugin.TypedName
	config      Config
	handle      fwkplugin.Handle
	decider     decider.Strategy
	deciderOnce sync.Once
	deciderErr  error
}

var _ fwkscheduling.ProfileHandler = (*Handler)(nil)

// Factory constructs a Disagg ProfileHandler.
func Factory(name string, raw json.RawMessage, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var params pluginParameters
	if err := utils.DecodePluginParameters(raw, &params); err != nil {
		return nil, fmt.Errorf("disagg profile handler %q: %w", name, err)
	}
	config, err := validateParameters(params)
	if err != nil {
		return nil, fmt.Errorf("disagg profile handler %q: %w", name, err)
	}
	return New(name, config, handle), nil
}

// New constructs a Disagg ProfileHandler.
func New(name string, config Config, handle fwkplugin.Handle) *Handler {
	return &Handler{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
		config:    resolveConfig(config),
		handle:    handle,
	}
}

// TypedName returns the plugin type and instance name.
func (h *Handler) TypedName() fwkplugin.TypedName {
	return h.typedName
}

// Pick selects Decode first. After Decode completes, it asks the Decider
// whether the PD profile must run. The d path returns an empty map so PD is not
// scheduled.
func (h *Handler) Pick(
	ctx context.Context,
	_ *fwkscheduling.CycleState,
	request *fwkscheduling.LLMRequest,
	profiles map[string]fwkscheduling.SchedulerProfile,
	profileResults map[string]*fwkscheduling.ProfileRunResult,
) map[string]fwkscheduling.SchedulerProfile {
	logger := log.FromContext(ctx).V(logutil.DEBUG)

	if _, ok := profileResults[h.config.DecodeProfileName]; !ok {
		decodeProfile, ok := profiles[h.config.DecodeProfileName]
		if !ok {
			logger.Info("disagg profile handler: decode profile missing",
				"decodeProfile", h.config.DecodeProfileName)
			return map[string]fwkscheduling.SchedulerProfile{}
		}
		return map[string]fwkscheduling.SchedulerProfile{
			h.config.DecodeProfileName: decodeProfile,
		}
	}

	if _, ok := profileResults[h.config.PDProfileName]; ok {
		return map[string]fwkscheduling.SchedulerProfile{}
	}

	selectedDecode, err := selectedDecodeEndpoint(profileResults[h.config.DecodeProfileName])
	if err != nil {
		logger.Info("disagg profile handler: missing selected decode, stopping",
			"error", err.Error())
		return map[string]fwkscheduling.SchedulerProfile{}
	}

	shouldRunPrefill, decideErr := h.decidePrefill(ctx, request, selectedDecode)
	if decideErr != nil {
		logger.Info("disagg profile handler: prefill decider error, forcing pd path",
			"error", decideErr.Error())
		shouldRunPrefill = true
	}

	if !shouldRunPrefill {
		logger.Info("disagg profile handler: d path skips pd profile")
		return map[string]fwkscheduling.SchedulerProfile{}
	}

	return h.pickPD(profiles)
}

// ProcessResults builds the final SchedulingResult.
func (h *Handler) ProcessResults(
	_ context.Context,
	_ *fwkscheduling.CycleState,
	_ *fwkscheduling.LLMRequest,
	profileResults map[string]*fwkscheduling.ProfileRunResult,
) (*fwkscheduling.SchedulingResult, error) {
	pdResult, pdRan := profileResults[h.config.PDProfileName]
	if pdRan {
		if pdResult == nil {
			return nil, fmt.Errorf(
				"disagg profile handler: pd profile %q failed",
				h.config.PDProfileName,
			)
		}
		return &fwkscheduling.SchedulingResult{
			PrimaryProfileName: h.config.PDProfileName,
			ProfileResults: map[string]*fwkscheduling.ProfileRunResult{
				h.config.PDProfileName: pdResult,
			},
		}, nil
	}

	decodeResult := profileResults[h.config.DecodeProfileName]
	if _, err := selectedDecodeEndpoint(decodeResult); err != nil {
		return nil, fmt.Errorf(
			"disagg profile handler: decode profile %q is incomplete: %w",
			h.config.DecodeProfileName,
			err,
		)
	}
	return &fwkscheduling.SchedulingResult{
		PrimaryProfileName: h.config.DecodeProfileName,
		ProfileResults: map[string]*fwkscheduling.ProfileRunResult{
			h.config.DecodeProfileName: decodeResult,
		},
	}, nil
}

func (h *Handler) pickPD(
	profiles map[string]fwkscheduling.SchedulerProfile,
) map[string]fwkscheduling.SchedulerProfile {
	pdProfile, ok := profiles[h.config.PDProfileName]
	if !ok {
		return map[string]fwkscheduling.SchedulerProfile{}
	}
	return map[string]fwkscheduling.SchedulerProfile{
		h.config.PDProfileName: pdProfile,
	}
}

func (h *Handler) decidePrefill(
	ctx context.Context,
	request *fwkscheduling.LLMRequest,
	selectedDecode fwkscheduling.Endpoint,
) (bool, error) {
	resolved, err := h.resolveDecider()
	if err != nil {
		return true, err
	}
	return resolved.Decide(ctx, request, selectedDecode)
}

func (h *Handler) resolveDecider() (decider.Strategy, error) {
	h.deciderOnce.Do(func() {
		if h.decider != nil {
			return
		}
		if h.handle == nil {
			h.deciderErr = fmt.Errorf("plugin handle is nil")
			return
		}
		resolved, err := fwkplugin.PluginByType[decider.Strategy](
			h.handle,
			h.config.PrefillDeciderPluginRef,
		)
		if err != nil {
			h.deciderErr = err
			return
		}
		h.decider = resolved
	})
	if h.deciderErr != nil {
		return nil, h.deciderErr
	}
	if h.decider == nil {
		return nil, fmt.Errorf(
			"prefill decider %q is unavailable",
			h.config.PrefillDeciderPluginRef,
		)
	}
	return h.decider, nil
}

func selectedDecodeEndpoint(result *fwkscheduling.ProfileRunResult) (fwkscheduling.Endpoint, error) {
	if result == nil || len(result.TargetEndpoints) == 0 || result.TargetEndpoints[0] == nil {
		return nil, fmt.Errorf("decode profile result is missing")
	}
	info, ok := pdgroup.GetDirect(result.TargetEndpoints[0])
	if !ok || info == nil {
		return nil, fmt.Errorf("decode leader is missing PD group metadata")
	}
	if info.SelectedDecodePod == nil || info.SelectedDecodePod.Endpoint == nil {
		return nil, fmt.Errorf("selected decode pod is missing")
	}
	return info.SelectedDecodePod.Endpoint, nil
}

func validateParameters(params pluginParameters) (Config, error) {
	if params.PrefillDeciderPluginRef == "" {
		return Config{}, fmt.Errorf("prefillDeciderPluginRef is required")
	}
	return Config{
		DecodeProfileName:       params.DecodeProfileName,
		PDProfileName:           params.PDProfileName,
		PrefillDeciderPluginRef: params.PrefillDeciderPluginRef,
	}, nil
}

func resolveConfig(config Config) Config {
	if config.DecodeProfileName == "" {
		config.DecodeProfileName = defaultDecodeProfileName
	}
	if config.PDProfileName == "" {
		config.PDProfileName = defaultPDProfileName
	}
	return config
}
