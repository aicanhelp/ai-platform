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

package endpointmeta

import (
	"fmt"
	"strings"

	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// LabelSelectorConfig describes lightweight endpoint label matching rules.
type LabelSelectorConfig struct {
	Exists []string            `json:"exists,omitempty"`
	In     map[string][]string `json:"in,omitempty"`
}

// LabelSelector stores normalized exists and in-clause label rules.
type LabelSelector struct {
	exists []string
	in     map[string]map[string]struct{}
}

// CompileLabelSelector normalizes selector rules and reports whether any remain.
func CompileLabelSelector(cfg *LabelSelectorConfig) (LabelSelector, bool, error) {
	if cfg == nil {
		return LabelSelector{}, false, nil
	}

	exists, err := compileExistsRules(cfg.Exists)
	if err != nil {
		return LabelSelector{}, false, err
	}
	inRules, err := compileInRules(cfg.In)
	if err != nil {
		return LabelSelector{}, false, err
	}
	if len(exists) == 0 && len(inRules) == 0 {
		return LabelSelector{}, false, nil
	}

	return LabelSelector{exists: exists, in: inRules}, true, nil
}

// CompileLegacyRoleSelector converts the legacy label-plus-roles configuration into an in-clause selector.
func CompileLegacyRoleSelector(label string, roles []string) (LabelSelector, bool, error) {
	trimmedLabel := strings.TrimSpace(label)
	if trimmedLabel == "" && len(roles) == 0 {
		return LabelSelector{}, false, nil
	}
	if len(roles) == 0 {
		return LabelSelector{}, false, nil
	}
	if trimmedLabel == "" {
		return LabelSelector{}, false, fmt.Errorf("legacy role selector requires a non-empty label when roles are provided")
	}

	normalizedRoles := NormalizeLabelValues(roles)
	if len(normalizedRoles) == 0 {
		return LabelSelector{}, false, fmt.Errorf(
			"legacy role selector for label %q must include at least one non-empty role",
			trimmedLabel,
		)
	}

	return CompileLabelSelector(&LabelSelectorConfig{
		In: map[string][]string{
			trimmedLabel: normalizedRoles,
		},
	})
}

// CompileProducerTargetSelector applies the producer-facing compatibility
// contract for explicit and legacy target pod selectors.
func CompileProducerTargetSelector(
	selectorConfig *LabelSelectorConfig,
	legacyLabel string,
	legacyRoles []string,
) (LabelSelector, bool, error) {
	trimmedLegacyLabel := strings.TrimSpace(legacyLabel)
	normalizedLegacyRoles := NormalizeLabelValues(legacyRoles)
	if selectorConfig != nil && (trimmedLegacyLabel != "" || len(normalizedLegacyRoles) > 0) {
		return LabelSelector{}, false, fmt.Errorf(
			"targetPodSelector cannot be combined with targetPodRoleLabel/targetPodRoles",
		)
	}
	if selectorConfig != nil {
		return CompileLabelSelector(selectorConfig)
	}
	if trimmedLegacyLabel != "" && len(normalizedLegacyRoles) == 0 {
		return LabelSelector{}, false, fmt.Errorf("targetPodRoleLabel requires at least one non-empty targetPodRoles value")
	}
	return CompileLegacyRoleSelector(trimmedLegacyLabel, normalizedLegacyRoles)
}

// Matches reports whether an endpoint satisfies the compiled selector.
func (s LabelSelector) Matches(endpoint fwkscheduling.Endpoint) bool {
	for _, key := range s.exists {
		value, ok := EndpointLabelValue(endpoint, key)
		if !ok || strings.TrimSpace(value) == "" {
			return false
		}
	}
	for key, allowed := range s.in {
		if _, ok := EndpointLabelValueAllowed(endpoint, key, allowed); !ok {
			return false
		}
	}
	return true
}

func compileExistsRules(labels []string) ([]string, error) {
	if len(labels) == 0 {
		return nil, nil
	}

	normalized := make([]string, 0, len(labels))
	seen := make(map[string]struct{}, len(labels))
	for _, label := range labels {
		trimmedLabel := strings.TrimSpace(label)
		if trimmedLabel == "" {
			return nil, fmt.Errorf("exists label name cannot be empty")
		}
		if err := validateSelectorLabelName(trimmedLabel, "exists"); err != nil {
			return nil, err
		}
		if _, ok := seen[trimmedLabel]; ok {
			continue
		}
		seen[trimmedLabel] = struct{}{}
		normalized = append(normalized, trimmedLabel)
	}
	if len(normalized) == 0 {
		return nil, nil
	}
	return normalized, nil
}

func compileInRules(in map[string][]string) (map[string]map[string]struct{}, error) {
	if len(in) == 0 {
		return nil, nil
	}

	normalized := make(map[string]map[string]struct{}, len(in))
	for label, values := range in {
		trimmedLabel := strings.TrimSpace(label)
		if trimmedLabel == "" {
			return nil, fmt.Errorf("in-clause label name cannot be empty")
		}
		if err := validateSelectorLabelName(trimmedLabel, "in-clause"); err != nil {
			return nil, err
		}

		normalizedValues := NormalizeLabelValues(values)
		if len(normalizedValues) == 0 {
			return nil, fmt.Errorf("in-clause for label %q must include at least one non-empty value", trimmedLabel)
		}

		allowed := normalized[trimmedLabel]
		if len(allowed) == 0 {
			allowed = make(map[string]struct{}, len(normalizedValues))
			normalized[trimmedLabel] = allowed
		}
		for _, value := range normalizedValues {
			allowed[value] = struct{}{}
		}
	}
	if len(normalized) == 0 {
		return nil, nil
	}
	return normalized, nil
}

func validateSelectorLabelName(label string, context string) error {
	problems := k8svalidation.IsQualifiedName(label)
	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("%s label name %q is invalid: %s", context, label, strings.Join(problems, "; "))
}
