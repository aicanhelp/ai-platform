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

// Package engine evaluates rule templates against collected facts.
//
// The evaluator is the single place that maps (fact, operator, expected) to a
// PASS/FAIL/UNKNOWN/SKIPPED evaluation result. It performs a type check and
// unit normalization before comparison; a type mismatch is a contract error,
// not a WARN. When a fact is unavailable (FactError), the result is UNKNOWN
// and the rule's UnknownPolicy decides the public status.
//
// Built-in operators and custom expressions are all handled by CEL. Built-in
// operators use fixed expressions; only custom rules carry an expression.
package engine

import (
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/celcheck"
)

// EvalState is the internal evaluation result, before normalization to a
// public Status. UNKNOWN means the fact was unavailable; it is the only state
// routed through UnknownPolicy.
type EvalState string

// Internal evaluator states.
const (
	EvalPass    EvalState = "PASS"    // EvalPass means the fact satisfies the rule.
	EvalFail    EvalState = "FAIL"    // EvalFail means the fact violates the rule.
	EvalUnknown EvalState = "UNKNOWN" // EvalUnknown means the fact is unavailable.
	EvalSkipped EvalState = "SKIPPED" // EvalSkipped means evaluation did not apply.
)

// EvalResult is the evaluator's output for one rule instance.
type EvalResult struct {
	State     EvalState
	Expected  any
	Actual    any
	CauseCode string
}

// Evaluate applies a rule's operator and expected value to a collected fact.
// It returns the internal EvalState; the caller normalizes to a public Status
// using the rule's FailureSeverity and UnknownPolicy.
//
// If the fact is unavailable (Available=false, non-nil Error), the result is
// UNKNOWN regardless of operator/expected — the UnknownPolicy handles it.
// If the fact is available, the operator is applied; a type mismatch between
// actual and expected is a contract failure with a distinct cause code; Run
// promotes it to INTERNAL_ERROR so it cannot look like an environment FAIL.
func Evaluate(rule api.RuleTemplate, fact api.Fact) EvalResult {
	// Unavailable fact -> UNKNOWN. The caller applies UnknownPolicy.
	if !fact.Available {
		cause := api.CauseUnknownFact
		if fact.Error != nil && fact.Error.Code != "" {
			cause = fact.Error.Code
		}
		return EvalResult{State: EvalUnknown, Expected: rule.Expected, CauseCode: cause}
	}
	if !factTypeCompatible(rule, fact) {
		return EvalResult{State: EvalFail, Expected: rule.Expected, Actual: fact.Value, CauseCode: "FACT_TYPE_MISMATCH"}
	}
	matched, cause := evaluateCEL(rule, fact.Value)
	if cause != "" {
		return EvalResult{State: EvalFail, Expected: rule.Expected, Actual: fact.Value, CauseCode: cause}
	}
	if matched {
		return EvalResult{State: EvalPass, Expected: rule.Expected, Actual: fact.Value}
	}
	return EvalResult{State: EvalFail, Expected: rule.Expected, Actual: fact.Value}
}

func evaluateCEL(rule api.RuleTemplate, actual any) (bool, string) {
	if rule.Op == api.OpCEL {
		program, err := celcheck.Compile(rule.Expression)
		if err != nil {
			return false, "CEL_COMPILE_ERROR"
		}
		matched, err := program.Evaluate(actual)
		if err != nil {
			return false, "CEL_EVALUATION_ERROR"
		}
		return matched, ""
	}
	program, err := celcheck.CompileBuiltIn(string(rule.Op))
	if err != nil {
		return false, "UNSUPPORTED_OPERATOR"
	}
	matched, err := program.Evaluate(actual, rule.Expected)
	if err != nil {
		return false, "CEL_EVALUATION_ERROR"
	}
	return matched, ""
}

func factTypeCompatible(rule api.RuleTemplate, fact api.Fact) bool {
	if rule.Op == api.OpCEL {
		return customFactTypeCompatible(fact)
	}
	expected := rule.ValueType()
	if fact.Type != "" {
		return declaredFactTypeCompatible(expected, fact)
	}
	return inferredFactTypeCompatible(expected, api.InferValueType(fact.Value))
}

func customFactTypeCompatible(fact api.Fact) bool {
	if fact.Type == "" {
		return api.InferValueType(fact.Value).Valid()
	}
	return fact.Type.Valid() && ValueMatchesDeclaredType(fact.Type, fact.Value)
}

func declaredFactTypeCompatible(expected api.ValueType, fact api.Fact) bool {
	if !fact.Type.Valid() || !ValueMatchesDeclaredType(fact.Type, fact.Value) {
		return false
	}
	switch expected {
	case api.ValueTypeRelation:
		return fact.Type == api.ValueTypeRelation || fact.Type == api.ValueTypeBool
	case api.ValueTypeSet:
		// in compares a scalar or tuple against the expected set.
		return true
	default:
		return fact.Type == expected
	}
}

func inferredFactTypeCompatible(expected, actual api.ValueType) bool {
	switch expected {
	case api.ValueTypeBytes, api.ValueTypeDuration, api.ValueTypeInteger:
		return actual == api.ValueTypeInteger || actual == api.ValueTypeBytes || actual == api.ValueTypeDuration
	case api.ValueTypeBool, api.ValueTypeRelation:
		return actual == api.ValueTypeBool || actual == api.ValueTypeRelation
	case api.ValueTypeString, api.ValueTypeEnum, api.ValueTypeAddress:
		return actual == api.ValueTypeString || actual == api.ValueTypeEnum || actual == api.ValueTypeAddress
	case api.ValueTypeSet:
		// in compares one value against a set, so the observed value may be a
		// scalar or a tuple represented as a slice.
		return actual.Valid()
	case api.ValueTypeRange:
		return actual == api.ValueTypeInteger || actual == api.ValueTypeBytes || actual == api.ValueTypeDuration
	default:
		return false
	}
}

// NormalizeStatus maps an EvalResult + rule to the public Status. This is the
// single sanctioned normalization; callers MUST NOT recompute.
func NormalizeStatus(rule api.RuleTemplate, res EvalResult) api.Status {
	switch res.State {
	case EvalPass:
		return api.StatusPass
	case EvalFail:
		// INFO rules are observation-only: a failed evaluation is never a
		// blocking FAIL; it degrades to SKIP so observations never block.
		if rule.Level == api.SeverityInfo {
			return api.StatusSkip
		}
		return api.NormalizeFailure(rule.Level)
	case EvalUnknown:
		return api.NormalizeUnknown(rule.Unknown)
	case EvalSkipped:
		return api.StatusSkip
	default:
		return api.StatusFail
	}
}
