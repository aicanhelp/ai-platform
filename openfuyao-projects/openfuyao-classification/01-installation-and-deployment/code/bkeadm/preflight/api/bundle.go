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

import "reflect"

const (
	rangeCardinality = 2
	bytesWordLength  = 5
)

// Op is a rule evaluation operator. Built-in operators are translated to CEL
// by the evaluator. OpCEL is reserved for validated custom checks.
type Op string

// Supported rule operators.
const (
	OpEq        Op = "eq"        // OpEq compares normalized equality.
	OpIn        Op = "in"        // OpIn checks membership in an expected set.
	OpGte       Op = "gte"       // OpGte compares a minimum threshold.
	OpLte       Op = "lte"       // OpLte compares a maximum threshold.
	OpRange     Op = "range"     // OpRange checks an inclusive range.
	OpExists    Op = "exists"    // OpExists checks presence.
	OpFree      Op = "free"      // OpFree checks that a port is available.
	OpReachable Op = "reachable" // OpReachable checks connectivity.
	OpRelation  Op = "relation"  // OpRelation evaluates a plan relation.
	OpCEL       Op = "cel"       // OpCEL evaluates the rule's CEL expression.
)

// Valid reports whether o is a supported operator.
func (o Op) Valid() bool {
	switch o {
	case OpEq, OpIn, OpGte, OpLte, OpRange, OpExists, OpFree, OpReachable, OpRelation, OpCEL:
		return true
	default:
		return false
	}
}

// RuleTemplate is the declarative rule carried in a PreflightBundle. It is
// data, not a Go plugin. A rule answers exactly one independently-remediable
// question; node/port/path expansion produces separate instances.
type RuleTemplate struct {
	// ID is the globally stable check ID, e.g. INIT-PORT-36443.
	ID string `json:"id"                  yaml:"id"`
	// Name is the human-facing check name. It is optional for legacy/static
	// catalog entries; renderers fall back to a friendly name derived from ID.
	Name string `json:"name,omitempty"      yaml:"name,omitempty"`
	// Operation is init or cluster-create.
	Operation Operation `json:"operation"           yaml:"operation"`
	// Fact is the fact key to collect, e.g. port.6443.free.
	Fact string `json:"fact"                yaml:"fact"`
	// Op is the evaluation operator.
	Op Op `json:"op"                  yaml:"op"`
	// Expected is a single value, set, range or relation expression. It must
	// be a concrete, evaluable value — never "meets requirements".
	Expected any `json:"expected"            yaml:"expected"`
	// ExpectedText is an optional operator-facing expectation. Custom checks
	// use it instead of exposing collector inputs or CEL implementation details.
	ExpectedText string `json:"expectedText,omitempty" yaml:"expectedText,omitempty"`
	// Expression is required only for OpCEL. It is compiled during config and
	// Bundle validation and evaluated with the single variable "actual".
	Expression string `json:"expression,omitempty"   yaml:"expression,omitempty"`
	// Level is the failure severity when the expectation is not met.
	Level FailureSeverity `json:"level"               yaml:"level"`
	// Unknown is the policy when the fact cannot be obtained.
	Unknown UnknownPolicy `json:"unknown"             yaml:"unknown"`
	// Fix is the user-facing remediation.
	Fix string `json:"fix"                 yaml:"fix"`
	// DependsOn lists rule IDs this rule depends on; a failed dependency
	// yields SKIP referencing the root cause. Cycles are rejected at load.
	DependsOn []string `json:"dependsOn,omitempty" yaml:"dependsOn,omitempty"`
}

// Validate checks the rule template's static contract: required fields,
// recognized operator, valid level and unknown policy, and non-empty fix.
// A rule that fails this MUST NOT enter a PolicySnapshot (BUNDLE_ERROR).
func (r RuleTemplate) Validate() *RuleContractError {
	if r.ID == "" {
		return &RuleContractError{Rule: r.ID, Field: "id", Reason: "missing"}
	}
	if !IsSupportedOperation(r.Operation) {
		return &RuleContractError{Rule: r.ID, Field: "operation", Reason: "unsupported operation"}
	}
	if r.Fact == "" {
		return &RuleContractError{Rule: r.ID, Field: "fact", Reason: "missing"}
	}
	if !r.Op.Valid() {
		return &RuleContractError{Rule: r.ID, Field: "op", Reason: "unsupported operator"}
	}
	if r.Expected == nil {
		return &RuleContractError{Rule: r.ID, Field: "expected", Reason: "missing"}
	}
	if reason := validateExpected(r.Op, r.Expected); reason != "" {
		return &RuleContractError{Rule: r.ID, Field: "expected", Reason: reason}
	}
	if r.Op == OpCEL && r.Expression == "" {
		return &RuleContractError{Rule: r.ID, Field: "expression", Reason: "missing"}
	}
	if r.Op != OpCEL && r.Expression != "" {
		return &RuleContractError{Rule: r.ID, Field: "expression", Reason: "only valid for cel operator"}
	}
	if !r.Level.Valid() {
		return &RuleContractError{Rule: r.ID, Field: "level", Reason: "invalid severity"}
	}
	if !r.Unknown.Valid() {
		return &RuleContractError{Rule: r.ID, Field: "unknown", Reason: "invalid unknown policy"}
	}
	if r.Fix == "" {
		return &RuleContractError{Rule: r.ID, Field: "fix", Reason: "missing remediation"}
	}
	return nil
}

func validateExpected(op Op, expected any) string {
	switch op {
	case OpIn:
		kind := reflect.TypeOf(expected).Kind()
		if kind != reflect.Slice && kind != reflect.Array {
			return "must be a set"
		}
	case OpGte, OpLte:
		if _, ok := numericValue(expected); !ok {
			return "must be numeric"
		}
	case OpRange:
		if !validRangeValue(expected) {
			return "must be a numeric two-element range"
		}
	case OpExists, OpFree, OpReachable, OpRelation:
		if _, ok := expected.(bool); !ok {
			return "must be boolean"
		}
	case OpCEL:
		if _, ok := expected.(bool); !ok {
			return "must be boolean"
		}
	default:
		return ""
	}
	return ""
}

func validRangeValue(value any) bool {
	rv := reflect.ValueOf(value)
	if rv.IsValid() && (rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array) && rv.Len() == rangeCardinality {
		_, first := numericValue(rv.Index(0).Interface())
		_, second := numericValue(rv.Index(1).Interface())
		return first && second
	}
	if values, ok := value.(map[string]any); ok {
		_, first := numericValue(values["minInclusive"])
		_, second := numericValue(values["maxInclusive"])
		return first && second
	}
	return false
}

func numericValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case int:
		return float64(typed), true
	case int32:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case uint:
		return float64(typed), true
	case uint32:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	case float32:
		return float64(typed), true
	case float64:
		return typed, true
	default:
		return 0, false
	}
}

// ValueType returns the report type implied by a validated rule expectation.
// Fact.Type, when present, remains authoritative for the observed value.
func (r RuleTemplate) ValueType() ValueType {
	if r.Op == OpCEL {
		return ValueTypeBool
	}
	if r.Op == OpIn {
		return ValueTypeSet
	}
	if r.Op == OpRange {
		return ValueTypeRange
	}
	if r.Op == OpRelation {
		return ValueTypeRelation
	}
	if r.Op == OpExists || r.Op == OpFree || r.Op == OpReachable {
		return ValueTypeBool
	}
	if containsBytesFact(r.Fact) {
		return ValueTypeBytes
	}
	return InferValueType(r.Expected)
}

func containsBytesFact(fact string) bool {
	for i := 0; i+bytesWordLength <= len(fact); i++ {
		fragment := fact[i : i+bytesWordLength]
		if fragment == "Bytes" || fragment == "bytes" {
			return true
		}
	}
	return false
}

// RuleContractError describes a rule that violates the static contract. It is
// returned by RuleTemplate.Validate and surfaces as BUNDLE_ERROR at load time.
type RuleContractError struct {
	Rule   string
	Field  string
	Reason string
}

func (e *RuleContractError) Error() string {
	return "rule contract error: " + e.Rule + "." + e.Field + " " + e.Reason
}

// PreflightBundle is the neutral SDK input. The host compiles it from the
// human-authored policy in the same process; the SDK only reads and validates
// it. It carries no node IPs, passwords, tokens, private keys or full
// kubeconfig.
type PreflightBundle struct {
	Operations        []Operation        `json:"operations"                  yaml:"operations"`
	Rules             []RuleTemplate     `json:"rules,omitempty"             yaml:"rules,omitempty"`
	VerifiedPlatforms []VerifiedPlatform `json:"verifiedPlatforms,omitempty" yaml:"verifiedPlatforms,omitempty"`
}

// VerifiedPlatform is a policy-derived compatibility row. KernelPrefixes
// contain major.minor series such as "5.10.".
type VerifiedPlatform struct {
	Distribution   string   `json:"distribution"   yaml:"distribution"`
	Version        string   `json:"version"        yaml:"version"`
	Architectures  []string `json:"architectures"  yaml:"architectures"`
	KernelPrefixes []string `json:"kernelPrefixes" yaml:"kernelPrefixes"`
}

// RuleIDs returns the IDs of all rule templates, in declaration order.
func (b *PreflightBundle) RuleIDs() []string {
	out := make([]string, 0, len(b.Rules))
	for _, r := range b.Rules {
		out = append(out, r.ID)
	}
	return out
}

// HasDuplicateRuleIDs reports whether any rule ID appears more than once and
// returns the first duplicate. Duplicate IDs are a BUNDLE_ERROR.
func (b *PreflightBundle) HasDuplicateRuleIDs() (string, bool) {
	seen := make(map[string]struct{}, len(b.Rules))
	for _, r := range b.Rules {
		if _, ok := seen[r.ID]; ok {
			return r.ID, true
		}
		seen[r.ID] = struct{}{}
	}
	return "", false
}
