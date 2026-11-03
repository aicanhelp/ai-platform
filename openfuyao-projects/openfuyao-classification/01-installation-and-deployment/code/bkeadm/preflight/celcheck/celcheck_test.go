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

package celcheck

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompileAndEvaluate(t *testing.T) {
	program, err := Compile(`actual.matches("(?m)^\\s*(127\\.0\\.0\\.1|::1)(\\s|$)")`)
	require.NoError(t, err)
	matched, err := program.Evaluate("127.0.0.1 localhost\n")
	require.NoError(t, err)
	assert.True(t, matched)
	matched, err = program.Evaluate("10.0.0.1 host\n")
	require.NoError(t, err)
	assert.False(t, matched)
}

func TestObjectExpression(t *testing.T) {
	program, err := Compile(`actual.healthy && actual.runtime in ["containerd", "isula"]`)
	require.NoError(t, err)
	matched, err := program.Evaluate(map[string]any{"healthy": true, "runtime": "containerd"})
	require.NoError(t, err)
	assert.True(t, matched)
}

func TestCompileRejectsInvalidContracts(t *testing.T) {
	_, err := Compile("")
	assert.ErrorContains(t, err, "required")
	_, err = Compile("actual +")
	assert.ErrorContains(t, err, "compile")
	_, err = Compile("'not bool'")
	assert.ErrorContains(t, err, "must return bool")
	_, err = Compile(strings.Repeat("x", MaxExpressionBytes+1))
	assert.ErrorContains(t, err, "exceeds")
	_, err = (*Program)(nil).Evaluate(true)
	assert.ErrorContains(t, err, "nil")
	program, err := Compile("actual")
	require.NoError(t, err)
	_, err = program.Evaluate("not bool")
	assert.ErrorContains(t, err, "instead of bool")
}

func TestBuiltInPrograms(t *testing.T) {
	tests := []struct {
		op       string
		actual   any
		expected any
		matched  bool
	}{
		{"eq", int64(2), 2, true},
		{"in", "arm64", []any{"amd64", "arm64"}, true},
		{"gte", int64(4), 2, true},
		{"lte", 3, int64(2), false},
		{"range", 4, map[string]any{"minInclusive": 2, "maxInclusive": 5}, true},
		{"exists", true, true, true},
	}
	for _, test := range tests {
		t.Run(test.op, func(t *testing.T) {
			program, err := CompileBuiltIn(test.op)
			require.NoError(t, err)
			matched, err := program.Evaluate(test.actual, test.expected)
			require.NoError(t, err)
			assert.Equal(t, test.matched, matched)
		})
	}
	_, err := CompileBuiltIn("unknown")
	assert.ErrorContains(t, err, "unsupported")
	_, err = (*BuiltInProgram)(nil).Evaluate(true, true)
	assert.ErrorContains(t, err, "nil")
}

func TestNormalizeCELValues(t *testing.T) {
	assert.Equal(t, int64(1), normalize(int(1)))
	assert.Equal(t, int64(2), normalize(int32(2)))
	assert.Equal(t, uint64(3), normalize(uint(3)))
	assert.Equal(t, uint64(4), normalize(uint32(4)))
	assert.Equal(t, float64(1.5), normalize(float32(1.5)))
	assert.Equal(t, []any{"a"}, normalize([]string{"a"}))
	assert.Equal(t, []any{int64(1)}, normalize([]any{int(1)}))
	assert.Equal(t, map[string]any{"n": int64(1)}, normalize(map[string]any{"n": int(1)}))
}

// TestBuiltInExpressionTable freezes the operator-to-expression mapping.
func TestBuiltInExpressionTable(t *testing.T) {
	for _, operator := range []string{"eq", "exists", "free", "reachable", "relation", "in", "gte", "lte", "range"} {
		expression, ok := BuiltInExpression(operator)
		assert.True(t, ok, "%s should map to an expression", operator)
		assert.NotEmpty(t, expression)
	}
	_, ok := BuiltInExpression("matches")
	assert.False(t, ok)
}

// TestBuiltInEvaluateErrorPaths freezes the runtime error classification:
// type mismatches surface as evaluation errors, not environment results.
func TestBuiltInEvaluateErrorPaths(t *testing.T) {
	program, err := CompileBuiltIn("gte")
	require.NoError(t, err)
	_, err = program.Evaluate("not-a-number", 2)
	assert.ErrorContains(t, err, "evaluate built-in")

	program, err = CompileBuiltIn("in")
	require.NoError(t, err)
	_, err = program.Evaluate("x", "not-a-list")
	assert.ErrorContains(t, err, "evaluate built-in")
}

// TestNormalizeExpectedRangeShape freezes the map-form range conversion.
func TestNormalizeExpectedRangeShape(t *testing.T) {
	assert.Equal(t,
		[]any{int64(2), int64(5)},
		normalizeExpected(map[string]any{"minInclusive": int(2), "maxInclusive": int(5)}),
	)
	assert.Equal(t,
		map[string]any{"other": int64(1)},
		normalizeExpected(map[string]any{"other": int(1)}),
	)
	assert.Equal(t, int64(7), normalizeExpected(int(7)))
}

// TestCompileRejectsOverCostExpression freezes the cost-limit guard.
func TestCompileRejectsOverCostExpression(t *testing.T) {
	_, err := Compile(strings.Repeat("actual && ", 2000) + "actual")
	assert.Error(t, err)
}
