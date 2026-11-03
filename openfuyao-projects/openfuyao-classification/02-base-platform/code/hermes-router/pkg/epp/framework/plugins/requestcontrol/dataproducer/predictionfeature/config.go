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

package predictionfeature

import (
	"encoding/json"
	"fmt"
	"strings"

	"hermes-router/pkg/epp/internal/endpointmeta"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	defaultPrefillValue = "prefill"
	defaultDecodeValue  = "decode"
)

// pluginParameters is the JSON shape accepted from the EndpointPickerConfig
// YAML. Pointer-typed PrefillValue/DecodeValue let callers distinguish "field
// omitted, use default" from "field set to empty string" — the latter is
// rejected during validation because a blank canonical role makes the
// classifier unable to distinguish prefill from decode.
type pluginParameters struct {
	PDLabelName        string                            `json:"pdLabelName,omitempty"`
	PrefillValue       *string                           `json:"prefillValue,omitempty"`
	DecodeValue        *string                           `json:"decodeValue,omitempty"`
	TargetPodSelector  *endpointmeta.LabelSelectorConfig `json:"targetPodSelector,omitempty"`
	TargetPodRoleLabel string                            `json:"targetPodRoleLabel,omitempty"`
	TargetPodRoles     []string                          `json:"targetPodRoles,omitempty"`
}

// Config is the resolved feature extractor configuration ready for runtime
// consumption. Constructed by parseAndValidateParameters or supplied directly
// by callers using New.
type Config struct {
	PDLabelName        string
	PrefillValue       string
	DecodeValue        string
	TargetPodSelector  *endpointmeta.LabelSelectorConfig
	TargetPodRoleLabel string
	TargetPodRoles     []string
}

func parseAndValidateParameters(raw json.RawMessage) (Config, error) {
	params := pluginParameters{}
	if err := utils.DecodePluginParameters(raw, &params); err != nil {
		return Config{}, err
	}
	config := Config{
		PDLabelName:        params.PDLabelName,
		PrefillValue:       defaultPrefillValue,
		DecodeValue:        defaultDecodeValue,
		TargetPodSelector:  params.TargetPodSelector,
		TargetPodRoleLabel: strings.TrimSpace(params.TargetPodRoleLabel),
		TargetPodRoles:     endpointmeta.NormalizeLabelValues(params.TargetPodRoles),
	}
	if params.PrefillValue != nil {
		config.PrefillValue = *params.PrefillValue
	}
	if params.DecodeValue != nil {
		config.DecodeValue = *params.DecodeValue
	}
	if config.PDLabelName != "" &&
		(strings.TrimSpace(config.PrefillValue) == "" || strings.TrimSpace(config.DecodeValue) == "") {
		return Config{}, fmt.Errorf("prefillValue and decodeValue must be non-empty when pdLabelName is set")
	}
	if _, _, err := endpointmeta.CompileProducerTargetSelector(
		config.TargetPodSelector,
		config.TargetPodRoleLabel,
		config.TargetPodRoles,
	); err != nil {
		return Config{}, err
	}
	return resolveConfig(config), nil
}

func resolveConfig(config Config) Config {
	resolved := config
	if resolved.PrefillValue == "" {
		resolved.PrefillValue = defaultPrefillValue
	}
	if resolved.DecodeValue == "" {
		resolved.DecodeValue = defaultDecodeValue
	}
	resolved.TargetPodRoleLabel = strings.TrimSpace(resolved.TargetPodRoleLabel)
	resolved.TargetPodRoles = endpointmeta.NormalizeLabelValues(resolved.TargetPodRoles)
	return resolved
}
