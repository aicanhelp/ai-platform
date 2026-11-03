/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package adapterutil contains pure helpers shared by the OpenFuyao host
// adapters. It has no collector, network, filesystem or product side effects.
package adapterutil

import (
	"strconv"
	"strings"

	"gopkg.openfuyao.cn/preflight/api"
)

// HumanizeSourceID turns a source identifier into an operator-facing name.
func HumanizeSourceID(value string) string {
	parts := strings.FieldsFunc(value, func(r rune) bool { return r == '-' || r == '_' || r == '/' })
	for i, part := range parts {
		if part != "" {
			parts[i] = strings.ToUpper(part[:1]) + part[1:]
		}
	}
	if len(parts) == 0 {
		return "Installation source"
	}
	return strings.Join(parts, " ")
}

// SourceUnreachableFix describes how to repair access to an installation source.
func SourceUnreachableFix(name, endpoint string) string {
	target := name
	if value := strings.TrimSpace(endpoint); value != "" {
		target += " (" + value + ")"
	}
	return "unable to access " + target +
		"; check network connectivity, the endpoint address, proxy/NO_PROXY, and firewall settings, " +
		"then run preflight again"
}

// SourceCheckName adds an endpoint locator to an operator-facing source name.
func SourceCheckName(name, endpoint string) string {
	if value := strings.TrimSpace(endpoint); value != "" {
		return name + " is accessible (" + value + ")"
	}
	return name + " is accessible"
}

// UniqueRules returns the first RuleTemplate for each stable rule ID while
// preserving input order.
func UniqueRules[T any](items []T, ruleFor func(T) api.RuleTemplate) []api.RuleTemplate {
	seen := make(map[string]struct{}, len(items))
	out := make([]api.RuleTemplate, 0, len(items))
	for _, item := range items {
		rule := ruleFor(item)
		if _, exists := seen[rule.ID]; exists {
			continue
		}
		seen[rule.ID] = struct{}{}
		out = append(out, rule)
	}
	return out
}

// PlanInt returns a supported integer plan value, or zero when the key is
// absent or has another type.
func PlanInt(plan api.OperationPlan, key string) int {
	value, ok := plan.Values[key]
	if !ok {
		return 0
	}
	switch typed := value.Value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
}

// FormatGiB renders a byte count in GiB. Whole GiB values stay integers so
// thresholds never render as e.g. 16.000000000000001 GiB.
func FormatGiB(bytes int64) string {
	const gib = int64(1024 * 1024 * 1024)
	if bytes%gib == 0 {
		return strconv.FormatInt(bytes/gib, 10) + " GiB"
	}
	return strconv.FormatFloat(float64(bytes)/float64(gib), 'f', -1, 64) + " GiB"
}
