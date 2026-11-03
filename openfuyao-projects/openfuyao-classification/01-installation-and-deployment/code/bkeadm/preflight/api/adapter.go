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

package api

import "time"

// ResolveResult is the complete side-effect-free output of an Adapter.
// Issues represent expected run-level problems; a Go error is reserved for an
// adapter/framework fault.
type ResolveResult struct {
	Plan     OperationPlan
	Issues   []RunIssue
	Bindings []RuleBinding
	Facts    []Fact
}

// RuleBinding binds a trusted Bundle rule template to one concrete subject
// and fact key. It cannot override the rule's expected value, severity or
// unknown policy. Applicable=false means the rule is not instantiated and is
// omitted from public results.
type RuleBinding struct {
	TemplateID string
	InstanceID string
	// Name overrides the generic Bundle template name for a dynamic binding.
	// Adapters should include the concrete source or target here when the rule
	// has one, so the same name is visible in both text and JSON reports.
	Name string
	// Fix overrides the generic Bundle remediation for a dynamic binding. It
	// must remain operator-facing and must not contain credentials.
	Fix        string
	Subject    Subject
	Fact       string
	Applicable bool
}

// RuntimeOptions controls execution; it does not change rules, levels,
// thresholds or unknown policies.
type RuntimeOptions struct {
	OverallTimeout time.Duration
	FactTimeout    time.Duration
	MaxConcurrency int
}
