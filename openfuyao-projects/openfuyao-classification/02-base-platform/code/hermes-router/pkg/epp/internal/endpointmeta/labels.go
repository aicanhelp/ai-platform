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
	"strings"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// NormalizeLabelValues trims values, drops empties, and removes duplicates while preserving order.
func NormalizeLabelValues(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	normalized := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		normalized = append(normalized, trimmed)
	}
	if len(normalized) == 0 {
		return nil
	}
	return normalized
}

// LabelValueSet builds a set from normalized label values.
func LabelValueSet(values []string) map[string]struct{} {
	if len(values) == 0 {
		return nil
	}
	set := make(map[string]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	return set
}

// EndpointLabelValueAllowed returns the endpoint label value when it exists and is allowed.
// An empty allowed set means any existing label value is accepted.
func EndpointLabelValueAllowed(
	endpoint fwkscheduling.Endpoint,
	key string,
	allowed map[string]struct{},
) (string, bool) {
	value, ok := EndpointLabelValue(endpoint, key)
	if !ok {
		return "", false
	}
	if len(allowed) == 0 {
		return value, true
	}
	_, ok = allowed[value]
	if !ok {
		return "", false
	}
	return value, true
}
