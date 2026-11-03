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

	"gopkg.openfuyao.cn/preflight/api"
)

// FuzzEvaluatorNeverPanics checks that Evaluate never panics on arbitrary
// (op, expectedBool, actualBool) combinations, and that an unavailable fact
// always yields UNKNOWN (never PASS) regardless of inputs.
//
// Fuzz arguments are restricted to primitives, so this exercises the bool
// operators (eq/exists/free/reachable/relation) which form the bulk of the
// truth table; the numeric and set operators are covered by table tests.
func FuzzEvaluatorNeverPanics(f *testing.F) {
	f.Add("eq", true, true)
	f.Add("exists", true, false)
	f.Add("free", false, true)
	f.Add("bogus", true, true)

	f.Fuzz(func(t *testing.T, opStr string, expected, actual bool) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Evaluate panicked: %v", r)
			}
		}()
		op := api.Op(opStr)
		rule := api.RuleTemplate{
			ID: "F", Operation: api.OperationInit, Fact: "f", Op: op, Expected: expected,
			Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: "f",
		}
		// Available fact: must not panic; status must be a recognized value.
		res := Evaluate(rule, api.Fact{Key: "f", Value: actual, Available: true})
		st := NormalizeStatus(rule, res)
		if !st.Valid() {
			t.Fatalf("invalid status %q for op=%s", st, opStr)
		}
		// Unavailable fact: must be UNKNOWN, never PASS.
		res2 := Evaluate(rule, api.Fact{Key: "f", Available: false,
			Error: &api.FactError{Code: api.FactErrTimeout}})
		if res2.State != EvalUnknown {
			t.Fatalf("unavailable fact yielded %s, want UNKNOWN", res2.State)
		}
		st2 := NormalizeStatus(rule, res2)
		if st2 == api.StatusPass {
			t.Fatalf("unavailable fact yielded PASS (forbidden)")
		}
	})
}
