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

package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/preflight/api"
)

// mkRule builds a rule with the given op/expected and ERROR/BLOCK defaults.
func mkRule(op api.Op, expected any) api.RuleTemplate {
	return api.RuleTemplate{
		ID: "T", Operation: api.OperationInit, Fact: "f", Op: op, Expected: expected,
		Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: "f",
	}
}

// mkFact builds an available fact with the given value.
func mkFact(v any) api.Fact {
	return api.Fact{Key: "f", Value: v, Available: true}
}

// unavailableFact builds a fact that could not be collected.
func unavailableFact(code string) api.Fact {
	return api.Fact{Key: "f", Available: false, Error: &api.FactError{Code: code}}
}

func TestEvaluateCustomCEL(t *testing.T) {
	rule := api.RuleTemplate{
		Op: api.OpCEL, Expected: true,
		Expression: `actual.matches("(?m)^\\s*(127\\.0\\.0\\.1|::1)(\\s|$)")`,
	}
	result := Evaluate(rule, mkFact("127.0.0.1 localhost\n"))
	assert.Equal(t, EvalPass, result.State)
	result = Evaluate(rule, mkFact("10.0.0.1 host\n"))
	assert.Equal(t, EvalFail, result.State)
}

func TestEvaluateCustomCELContractErrors(t *testing.T) {
	rule := api.RuleTemplate{Op: api.OpCEL, Expected: true, Expression: "actual.missing"}
	result := Evaluate(rule, mkFact(map[string]any{"healthy": true}))
	assert.Equal(t, "CEL_EVALUATION_ERROR", result.CauseCode)
	rule.Expression = "actual +"
	result = Evaluate(rule, mkFact(true))
	assert.Equal(t, "CEL_COMPILE_ERROR", result.CauseCode)
}

// TestEqTruthTable freezes the eq operator across types.
func TestEqTruthTable(t *testing.T) {
	r := mkRule(api.OpEq, true)
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(true)).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact(false)).State)

	r = mkRule(api.OpEq, "linux")
	assert.Equal(t, EvalPass, Evaluate(r, mkFact("linux")).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact("darwin")).State)

	// Numeric cross-kind equality: int vs float64.
	r = mkRule(api.OpEq, 4)
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(float64(4))).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact(float64(5))).State)
}

// TestInTruthTable freezes the in operator.
func TestInTruthTable(t *testing.T) {
	r := mkRule(api.OpIn, []any{"openeuler", "ubuntu", "uos"})
	assert.Equal(t, EvalPass, Evaluate(r, mkFact("ubuntu")).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact("centos")).State)
	// Set as a tuple (slice of slices) also works via DeepEqual on items.
	r = mkRule(api.OpIn, []any{[]any{"openeuler", "22.03"}, []any{"ubuntu", "22.04"}})
	assert.Equal(t, EvalPass, Evaluate(r, mkFact([]any{"ubuntu", "22.04"})).State)
	// Collectors may return []string while YAML decoding produces []any.
	assert.Equal(t, EvalPass, Evaluate(r, mkFact([]string{"ubuntu", "22.04"})).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact([]any{"centos", "8"})).State)
}

// TestGteTruthTable freezes the gte operator and boundary.
func TestGteTruthTable(t *testing.T) {
	r := mkRule(api.OpGte, 2)
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(4)).State)
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(2)).State) // boundary inclusive
	assert.Equal(t, EvalFail, Evaluate(r, mkFact(1)).State)
	// Non-numeric actual is a contract FAIL with a cause code.
	res := Evaluate(r, mkFact("not-a-number"))
	assert.Equal(t, EvalFail, res.State)
	assert.Equal(t, "FACT_TYPE_MISMATCH", res.CauseCode)
}

// TestLteTruthTable freezes the lte operator and boundary.
func TestLteTruthTable(t *testing.T) {
	r := mkRule(api.OpLte, 10) // e.g. clock skew <= 10s
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(5)).State)
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(10)).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact(11)).State)
}

// TestRangeTruthTable freezes the range operator (map and slice forms).
func TestRangeTruthTable(t *testing.T) {
	r := mkRule(api.OpRange, map[string]any{"minInclusive": 4.0, "maxInclusive": 6.0})
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(5)).State)
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(4)).State)
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(6)).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact(3)).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact(7)).State)

	// Two-element slice form.
	r = mkRule(api.OpRange, []any{4, 6})
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(5)).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact(7)).State)
}

// TestExistsFreeReachableTruthTable freezes the bool operators.
func TestExistsFreeReachableTruthTable(t *testing.T) {
	for _, op := range []api.Op{api.OpExists, api.OpFree, api.OpReachable} {
		r := mkRule(op, true)
		assert.Equal(t, EvalPass, Evaluate(r, mkFact(true)).State, "op %s true/true", op)
		assert.Equal(t, EvalFail, Evaluate(r, mkFact(false)).State, "op %s true/false", op)
		// Non-bool actual is a contract FAIL.
		assert.Equal(t, EvalFail, Evaluate(r, mkFact("x")).State, "op %s non-bool", op)
	}
}

// TestRelationTruthTable freezes the relation operator (fact carries bool).
func TestRelationTruthTable(t *testing.T) {
	r := mkRule(api.OpRelation, false) // expected: relation false (e.g. not colocated)
	assert.Equal(t, EvalPass, Evaluate(r, mkFact(false)).State)
	assert.Equal(t, EvalFail, Evaluate(r, mkFact(true)).State)
}

// TestUnsupportedOperatorIsFail freezes that an unknown op is never silent.
func TestUnsupportedOperatorIsFail(t *testing.T) {
	r := mkRule(api.Op("bogus"), true)
	res := Evaluate(r, mkFact(true))
	assert.Equal(t, EvalFail, res.State)
	assert.Equal(t, "UNSUPPORTED_OPERATOR", res.CauseCode)
}

// TestUnavailableFactIsUnknown freezes the core unknown contract: an
// unavailable fact yields UNKNOWN regardless of operator/expected, and the
// cause code carries the fact error code.
func TestUnavailableFactIsUnknown(t *testing.T) {
	for _, op := range []api.Op{api.OpEq, api.OpGte, api.OpIn, api.OpExists} {
		r := mkRule(op, true)
		res := Evaluate(r, unavailableFact(api.FactErrTimeout))
		assert.Equal(t, EvalUnknown, res.State, "op %s", op)
		assert.Equal(t, api.FactErrTimeout, res.CauseCode)
	}
}

// TestNormalizeStatusMap freezes the eval-state -> public-status mapping,
// including the unknown-policy routing (BLOCK->FAIL, WARN->WARN, IGNORE->SKIP).
func TestNormalizeStatusMap(t *testing.T) {
	r := mkRule(api.OpEq, true)
	// Pass.
	assert.Equal(t, api.StatusPass, NormalizeStatus(r, EvalResult{State: EvalPass}))
	// Fail + ERROR -> FAIL.
	assert.Equal(t, api.StatusFail, NormalizeStatus(r, EvalResult{State: EvalFail}))
	// Unknown + BLOCK -> FAIL.
	assert.Equal(t, api.StatusFail, NormalizeStatus(r, EvalResult{State: EvalUnknown}))
	// Unknown + WARN -> WARN.
	r.Unknown = api.UnknownWarn
	assert.Equal(t, api.StatusWarn, NormalizeStatus(r, EvalResult{State: EvalUnknown}))
	// Unknown + IGNORE -> SKIP.
	r.Unknown = api.UnknownIgnore
	assert.Equal(t, api.StatusSkip, NormalizeStatus(r, EvalResult{State: EvalUnknown}))
	// Skipped -> SKIP.
	assert.Equal(t, api.StatusSkip, NormalizeStatus(r, EvalResult{State: EvalSkipped}))

	// Fail + WARNING -> WARN.
	r.Level = api.SeverityWarning
	assert.Equal(t, api.StatusWarn, NormalizeStatus(r, EvalResult{State: EvalFail}))
}

// TestUnknownNeverPass freezes the invariant end-to-end: no combination of an
// unavailable fact yields a PASS through NormalizeStatus.
func TestUnknownNeverPass(t *testing.T) {
	for _, unk := range []api.UnknownPolicy{api.UnknownBlock, api.UnknownWarn, api.UnknownIgnore} {
		r := mkRule(api.OpEq, true)
		r.Unknown = unk
		res := Evaluate(r, unavailableFact(api.FactErrUnreachable))
		st := NormalizeStatus(r, res)
		assert.NotEqual(t, api.StatusPass, st, "unknown=%s must not yield PASS", unk)
	}
}
