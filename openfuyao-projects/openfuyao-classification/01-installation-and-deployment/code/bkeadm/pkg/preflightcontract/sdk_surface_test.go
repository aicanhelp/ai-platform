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

// Package preflightcontract_test exercises the SDK public surface from the
// root module so the coverage pipeline (which instruments root-module tests)
// also accounts for the preflight submodule behavior.
package preflightcontract_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/bundle"
	"gopkg.openfuyao.cn/preflight/celcheck"
	"gopkg.openfuyao.cn/preflight/collectors/custom"
)

// --- api: status / overall / bundle contracts ---

func TestStatusContracts(t *testing.T) {
	for _, status := range []api.Status{api.StatusPass, api.StatusFail, api.StatusWarn, api.StatusSkip} {
		assert.True(t, status.Valid(), status)
		assert.Equal(t, string(status), status.String())
	}
	assert.False(t, api.Status("BOGUS").Valid())
	assert.Equal(t, -1, api.SeverityRank(api.Status("BOGUS")))
	assert.Greater(t, api.SeverityRank(api.StatusFail), api.SeverityRank(api.StatusWarn))
	assert.Greater(t, api.SeverityRank(api.StatusWarn), api.SeverityRank(api.StatusPass))

	for _, severity := range []api.FailureSeverity{api.SeverityError, api.SeverityWarning, api.SeverityInfo} {
		assert.True(t, severity.Valid(), severity)
	}
	assert.False(t, api.FailureSeverity("FATAL").Valid())

	for _, policy := range []api.UnknownPolicy{api.UnknownBlock, api.UnknownWarn, api.UnknownIgnore} {
		assert.True(t, policy.Valid(), policy)
	}
	assert.False(t, api.UnknownPolicy("RETRY").Valid())

	assert.Equal(t, api.StatusFail, api.NormalizeUnknown(api.UnknownBlock))
	assert.Equal(t, api.StatusWarn, api.NormalizeUnknown(api.UnknownWarn))
	assert.Equal(t, api.StatusSkip, api.NormalizeUnknown(api.UnknownIgnore))
	assert.Equal(t, api.StatusFail, api.NormalizeUnknown(api.UnknownPolicy("bad")))

	assert.Equal(t, api.StatusFail, api.NormalizeFailure(api.SeverityError))
	assert.Equal(t, api.StatusWarn, api.NormalizeFailure(api.SeverityWarning))
	assert.Equal(t, api.StatusFail, api.NormalizeFailure(api.SeverityInfo))
	assert.Equal(t, api.StatusFail, api.NormalizeFailure(api.FailureSeverity("bad")))
}

func TestOverallExitCodeAndAggregation(t *testing.T) {
	assert.Equal(t, 0, api.ExitCode(api.OverallPass))
	assert.Equal(t, 2, api.ExitCode(api.OverallWarn))
	assert.Equal(t, 2, api.ExitCode(api.OverallSkip))
	assert.Equal(t, 3, api.ExitCode(api.OverallFail))
	assert.Equal(t, 4, api.ExitCode(api.OverallInputError))
	assert.Equal(t, 4, api.ExitCode(api.OverallBundleError))
	assert.Equal(t, 5, api.ExitCode(api.OverallInternalError))
	assert.Equal(t, 124, api.ExitCode(api.OverallTimedOut))
	assert.Equal(t, 130, api.ExitCode(api.OverallCanceled))
	assert.Equal(t, 5, api.ExitCode(api.Overall("BOGUS")))

	assert.Equal(t, api.OverallPass, api.AggregateOverall([]api.Status{api.StatusPass, api.StatusSkip}))
	assert.Equal(t, api.OverallWarn, api.AggregateOverall([]api.Status{api.StatusPass, api.StatusWarn}))
	assert.Equal(t, api.OverallFail, api.AggregateOverall([]api.Status{api.StatusWarn, api.StatusFail}))
	assert.Equal(t, api.OverallSkip, api.AggregateOverall([]api.Status{api.StatusSkip}))
	assert.Equal(t, api.OverallSkip, api.AggregateOverall(nil))

	summary := api.Summarize([]api.Status{api.StatusPass, api.StatusFail, api.StatusWarn, api.StatusSkip, api.StatusPass})
	assert.Equal(t, 5, summary.Total)
	assert.Equal(t, 2, summary.Pass)
	assert.Equal(t, 1, summary.Fail)
	assert.Equal(t, 1, summary.Warn)
	assert.Equal(t, 1, summary.Skip)

	assert.Equal(t, []api.Status{api.StatusFail, api.StatusWarn},
		api.FailedStatuses([]api.Status{api.StatusPass, api.StatusFail, api.StatusWarn, api.StatusFail}))
	assert.Empty(t, api.FailedStatuses([]api.Status{api.StatusPass, api.StatusSkip}))

	assert.NoError(t, api.AssertExitCode(api.OverallPass, 0))
	assert.ErrorContains(t, api.AssertExitCode(api.OverallPass, 3), "mismatch")
}

func validRule() api.RuleTemplate {
	return api.RuleTemplate{
		ID: "INIT-PORT-36443", Operation: api.OperationInit, Fact: "port.36443.free",
		Op: api.OpEq, Expected: true, Level: api.SeverityError,
		Unknown: api.UnknownBlock, Fix: "stop the process using port 36443",
	}
}

func TestRuleTemplateContracts(t *testing.T) {
	assert.Nil(t, validRule().Validate())

	// ValueType mapping per operator and the bytes-fact heuristic.
	rule := validRule()
	assert.Equal(t, api.ValueTypeBool, rule.ValueType())
	rule.Op = api.OpIn
	rule.Expected = []any{"a"}
	assert.Equal(t, api.ValueTypeSet, rule.ValueType())
	rule.Op = api.OpRange
	rule.Expected = []any{1, 2}
	assert.Equal(t, api.ValueTypeRange, rule.ValueType())
	rule.Op = api.OpRelation
	rule.Expected = true
	assert.Equal(t, api.ValueTypeRelation, rule.ValueType())
	rule.Op = api.OpGte
	rule.Expected = 1
	rule.Fact = "resources.memory.totalBytes"
	assert.Equal(t, api.ValueTypeBytes, rule.ValueType())
	rule.Fact = "resources.cpu.logical"
	assert.Equal(t, api.ValueTypeInteger, rule.ValueType())

	// RuleContractError message.
	err := &api.RuleContractError{Rule: "INIT-X", Field: "fact", Reason: "is required"}
	assert.Equal(t, "rule contract error: INIT-X.fact is required", err.Error())

	// Bundle rule enumeration and duplicate detection.
	b := &api.PreflightBundle{Rules: []api.RuleTemplate{validRule()}}
	assert.Equal(t, []string{"INIT-PORT-36443"}, b.RuleIDs())
	dup, ok := (&api.PreflightBundle{Rules: []api.RuleTemplate{validRule(), validRule()}}).HasDuplicateRuleIDs()
	assert.True(t, ok)
	assert.Equal(t, "INIT-PORT-36443", dup)
	_, ok = b.HasDuplicateRuleIDs()
	assert.False(t, ok)
}

// --- bundle: load contracts ---

func validBundleYAML(t *testing.T) []byte {
	t.Helper()
	rule := validRule()
	raw :=
		"operations: [init, cluster-create]\n" +
			"rules:\n" +
			"  - id: " + rule.ID + "\n" +
			"    operation: " + string(rule.Operation) + "\n" +
			"    fact: " + rule.Fact + "\n" +
			"    op: " + string(rule.Op) + "\n" +
			"    expected: true\n" +
			"    level: " + string(rule.Level) + "\n" +
			"    unknown: " + string(rule.Unknown) + "\n" +
			"    fix: " + rule.Fix + "\n"
	return []byte(raw)
}

func TestBundleLoadContract(t *testing.T) {
	loaded, err := bundle.Load(validBundleYAML(t))
	require.NoError(t, err)
	assert.Len(t, loaded.Rules, 1)

	// Empty and malformed inputs are LoadErrors.
	_, err = bundle.Load(nil)
	assert.True(t, bundle.IsLoadError(err))
	_, err = bundle.Load([]byte("::: not yaml"))
	assert.True(t, bundle.IsLoadError(err))
}

// --- celcheck: compile / evaluate contracts ---

func TestCELCheckContracts(t *testing.T) {
	program, err := celcheck.Compile("actual > 2")
	require.NoError(t, err)
	matched, err := program.Evaluate(3)
	require.NoError(t, err)
	assert.True(t, matched)
	matched, err = program.Evaluate(1)
	require.NoError(t, err)
	assert.False(t, matched)

	_, err = celcheck.Compile("")
	assert.ErrorContains(t, err, "required")
	_, err = celcheck.Compile("actual +")
	assert.ErrorContains(t, err, "compile")
	_, err = celcheck.Compile("'not bool'")
	assert.ErrorContains(t, err, "must return bool")
	_, err = celcheck.Compile(strings.Repeat("x", celcheck.MaxExpressionBytes+1))
	assert.ErrorContains(t, err, "exceeds")
	_, err = (*celcheck.Program)(nil).Evaluate(true)
	assert.ErrorContains(t, err, "nil")

	// Built-in operator programs.
	for _, operator := range []string{"eq", "in", "gte", "lte", "range", "exists", "free", "reachable", "relation"} {
		expression, ok := celcheck.BuiltInExpression(operator)
		assert.True(t, ok, operator)
		assert.NotEmpty(t, expression)
	}
	_, ok := celcheck.BuiltInExpression("matches")
	assert.False(t, ok)

	builtin, err := celcheck.CompileBuiltIn("gte")
	require.NoError(t, err)
	matched, err = builtin.Evaluate(4, 2)
	require.NoError(t, err)
	assert.True(t, matched)
	_, err = builtin.Evaluate("not-a-number", 2)
	assert.ErrorContains(t, err, "evaluate built-in")

	_, err = celcheck.CompileBuiltIn("unknown")
	assert.ErrorContains(t, err, "unsupported")
	_, err = (*celcheck.BuiltInProgram)(nil).Evaluate(true, true)
	assert.ErrorContains(t, err, "nil")
}

// --- collectors/custom: fact decoding and redaction contracts ---

type fakeCustomExecutor struct {
	fileOutput custom.Output
	runOutput  custom.Output
	err        error
}

func (f *fakeCustomExecutor) ReadFile(
	_ context.Context, _ api.Subject, _ string, _ int64,
) (custom.Output, error) {
	return f.fileOutput, f.err
}

func (f *fakeCustomExecutor) Run(
	_ context.Context, _ api.Subject, _ string, _ []string, _ int64,
) (custom.Output, error) {
	return f.runOutput, f.err
}

func collectCustom(t *testing.T, collector *custom.Collector, key string) api.Fact {
	t.Helper()
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{key}, Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	return facts[0]
}

func TestCustomCollectorContracts(t *testing.T) {
	assert.Equal(t, "custom.init.hosts", custom.FactKey(api.OperationInit, "hosts"))

	// fileContent: value is evaluable but the display is redacted.
	executor := &fakeCustomExecutor{fileOutput: custom.Output{
		Data: []byte("127.0.0.1 localhost\n"), Digest: "sha256:observed",
	}}
	collector, err := custom.New(executor, []custom.Spec{
		{FactKey: "custom.init.hosts", Type: "fileContent", Path: "/etc/hosts"},
	})
	require.NoError(t, err)
	assert.Equal(t, "custom", collector.ID())
	fact := collectCustom(t, collector, "custom.init.hosts")
	assert.True(t, fact.Available)
	assert.Equal(t, "127.0.0.1 localhost\n", fact.Value)
	assert.Equal(t, "已读取（内容不展示）", fact.Display.Value)

	// exec object: only public fields reach the display.
	executor = &fakeCustomExecutor{runOutput: custom.Output{Digest: "sha256:script", Data: []byte(`{
		"apiVersion":"preflight.openfuyao.io/fact/v1",
		"available":true,
		"type":"object",
		"value":{"runtime":"containerd","healthy":true,"private":"hidden"}
	}`)}}
	collector, err = custom.New(executor, []custom.Spec{{
		FactKey: "custom.cluster.runtime", Type: "exec", Path: "/opt/check-runtime",
		ResultType: api.ValueTypeObject, ResultKind: "object",
		PublicFields: []string{"runtime", "healthy"},
	}})
	require.NoError(t, err)
	fact = collectCustom(t, collector, "custom.cluster.runtime")
	assert.True(t, fact.Available)
	assert.Equal(t, map[string]any{"runtime": "containerd", "healthy": true}, fact.Display.Value)
	assert.Equal(t, "hidden", fact.Value.(map[string]any)["private"])

	// Constructor and request contract guards.
	_, err = custom.New(nil, []custom.Spec{{}})
	assert.ErrorContains(t, err, "required")
	_, err = custom.New(nil, []custom.Spec{{FactKey: "x"}, {FactKey: "x"}})
	assert.ErrorContains(t, err, "duplicate")
	_, err = collector.Collect(context.Background(), api.FactRequest{})
	assert.ErrorContains(t, err, "one subject")

	// Unknown spec keys and executor failures map to unavailable facts.
	guarded, err := custom.New(nil, []custom.Spec{{FactKey: "x", Type: "fileContent", Path: "/x"}})
	require.NoError(t, err)
	fact = collectCustom(t, guarded, "missing")
	assert.Equal(t, api.FactErrUnsupported, fact.Error.Code)
	fact = collectCustom(t, guarded, "x")
	assert.Equal(t, api.FactErrUnsupported, fact.Error.Code)

	failing, err := custom.New(&fakeCustomExecutor{err: errors.New("denied")},
		[]custom.Spec{{FactKey: "x", Type: "fileContent", Path: "/x"}})
	require.NoError(t, err)
	fact = collectCustom(t, failing, "x")
	assert.Equal(t, api.FactErrUnreachable, fact.Error.Code)

	// Malformed script output is a parse error, not an environment failure.
	malformed, err := custom.New(&fakeCustomExecutor{runOutput: custom.Output{Data: []byte("not-json")}},
		[]custom.Spec{{FactKey: "x", Type: "exec", Path: "/opt/check"}})
	require.NoError(t, err)
	fact = collectCustom(t, malformed, "x")
	assert.Equal(t, api.FactErrParseError, fact.Error.Code)
}
