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

// Package celcheck compiles and evaluates the bounded CEL subset used by
// preflight rules. It registers no host functions and exposes only "actual".
package celcheck

import (
	"fmt"
	"strings"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
)

const (
	// MaxExpressionBytes bounds user-authored expressions before compilation.
	MaxExpressionBytes = 2048
	maxEvaluationCost  = 10000
)

// Program is an immutable, compiled CEL boolean expression.
type Program struct {
	program cel.Program
}

// Compile validates expression syntax and its boolean result type.
func Compile(expression string) (*Program, error) {
	expression = strings.TrimSpace(expression)
	if expression == "" {
		return nil, fmt.Errorf("CEL expression is required")
	}
	if len(expression) > MaxExpressionBytes {
		return nil, fmt.Errorf("CEL expression exceeds %d bytes", MaxExpressionBytes)
	}
	environment, err := cel.NewEnv(cel.Variable("actual", cel.DynType))
	if err != nil {
		return nil, fmt.Errorf("create CEL environment: %w", err)
	}
	ast, issues := environment.Compile(expression)
	if issues != nil && issues.Err() != nil {
		return nil, fmt.Errorf("compile CEL expression: %w", issues.Err())
	}
	if ast.OutputType() != cel.BoolType && ast.OutputType() != cel.DynType {
		return nil, fmt.Errorf("CEL expression must return bool, got %s", ast.OutputType())
	}
	program, err := environment.Program(ast, cel.CostLimit(maxEvaluationCost))
	if err != nil {
		return nil, fmt.Errorf("create CEL program: %w", err)
	}
	return &Program{program: program}, nil
}

// Evaluate executes a compiled program and requires a concrete bool result.
func (p *Program) Evaluate(actual any) (bool, error) {
	if p == nil || p.program == nil {
		return false, fmt.Errorf("CEL program is nil")
	}
	value, _, err := p.program.Eval(map[string]any{"actual": normalize(actual)})
	if err != nil {
		return false, fmt.Errorf("evaluate CEL expression: %w", err)
	}
	result, ok := value.(types.Bool)
	if !ok {
		return false, fmt.Errorf("CEL expression returned %s instead of bool", value.Type())
	}
	return bool(result), nil
}

// BuiltInExpression returns the fixed CEL expression for a built-in operator.
func BuiltInExpression(operator string) (string, bool) {
	switch operator {
	case "eq", "exists", "free", "reachable", "relation":
		return "actual == expected", true
	case "in":
		return "actual in expected", true
	case "gte":
		return "actual >= expected", true
	case "lte":
		return "actual <= expected", true
	case "range":
		return "actual >= expected[0] && actual <= expected[1]", true
	default:
		return "", false
	}
}

// CompileBuiltIn creates the fixed CEL program for a built-in operator.
func CompileBuiltIn(operator string) (*BuiltInProgram, error) {
	expression, ok := BuiltInExpression(operator)
	if !ok {
		return nil, fmt.Errorf("unsupported operator %q", operator)
	}
	environment, err := cel.NewEnv(
		cel.Variable("actual", cel.DynType),
		cel.Variable("expected", cel.DynType),
	)
	if err != nil {
		return nil, fmt.Errorf("create CEL environment: %w", err)
	}
	ast, issues := environment.Compile(expression)
	if issues != nil && issues.Err() != nil {
		return nil, fmt.Errorf("compile built-in CEL expression: %w", issues.Err())
	}
	program, err := environment.Program(ast, cel.CostLimit(maxEvaluationCost))
	if err != nil {
		return nil, fmt.Errorf("create built-in CEL program: %w", err)
	}
	return &BuiltInProgram{program: program}, nil
}

// BuiltInProgram is a fixed operator program with actual and expected inputs.
type BuiltInProgram struct {
	program cel.Program
}

// Evaluate executes a built-in comparison.
func (p *BuiltInProgram) Evaluate(actual, expected any) (bool, error) {
	if p == nil || p.program == nil {
		return false, fmt.Errorf("built-in CEL program is nil")
	}
	value, _, err := p.program.Eval(map[string]any{
		"actual": normalize(actual), "expected": normalizeExpected(expected),
	})
	if err != nil {
		return false, fmt.Errorf("evaluate built-in CEL expression: %w", err)
	}
	result, ok := value.(types.Bool)
	if !ok {
		return false, fmt.Errorf("built-in CEL expression did not return bool")
	}
	return bool(result), nil
}

func normalizeExpected(value any) any {
	if values, ok := value.(map[string]any); ok {
		minimum, hasMinimum := values["minInclusive"]
		maximum, hasMaximum := values["maxInclusive"]
		if hasMinimum && hasMaximum {
			return []any{normalize(minimum), normalize(maximum)}
		}
	}
	return normalize(value)
}

func normalize(value any) any {
	switch typed := value.(type) {
	case int:
		return int64(typed)
	case int32:
		return int64(typed)
	case uint:
		return uint64(typed)
	case uint32:
		return uint64(typed)
	case float32:
		return float64(typed)
	case []string:
		out := make([]any, 0, len(typed))
		for _, item := range typed {
			out = append(out, item)
		}
		return out
	case []any:
		out := make([]any, 0, len(typed))
		for _, item := range typed {
			out = append(out, normalize(item))
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(typed))
		for key, item := range typed {
			out[key] = normalize(item)
		}
		return out
	default:
		return value
	}
}
