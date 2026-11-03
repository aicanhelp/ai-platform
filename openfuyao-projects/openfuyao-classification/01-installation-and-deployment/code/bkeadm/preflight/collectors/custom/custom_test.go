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

package custom

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

type fakeExecutor struct {
	fileOutput Output
	runOutput  Output
	err        error
	path       string
	args       []string
}

func (f *fakeExecutor) ReadFile(
	_ context.Context,
	_ api.Subject,
	path string,
	_ int64,
) (Output, error) {
	f.path = path
	return f.fileOutput, f.err
}

func (f *fakeExecutor) Run(
	_ context.Context,
	_ api.Subject,
	path string,
	args []string,
	_ int64,
) (Output, error) {
	f.path, f.args = path, append([]string(nil), args...)
	return f.runOutput, f.err
}

func collectOne(t *testing.T, collector *Collector, key string) api.Fact {
	t.Helper()
	facts, err := collector.Collect(context.Background(), api.FactRequest{
		Keys: []string{key}, Subjects: []api.Subject{{ID: "node-1"}},
	})
	require.NoError(t, err)
	require.Len(t, facts, 1)
	return facts[0]
}

func TestFileContentIsEvaluableButRedacted(t *testing.T) {
	executor := &fakeExecutor{fileOutput: Output{
		Data: []byte("127.0.0.1 localhost\n"), Digest: "sha256:observed",
	}}
	collector, err := New(executor, []Spec{{FactKey: "custom.init.hosts", Type: "fileContent", Path: "/etc/hosts"}})
	require.NoError(t, err)
	fact := collectOne(t, collector, "custom.init.hosts")
	assert.True(t, fact.Available)
	assert.Equal(t, "127.0.0.1 localhost\n", fact.Value)
	assert.Equal(t, "已读取（内容不展示）", fact.Display.Value)
	assert.NotContains(t, fact.Evidence[0].Observed, "localhost")
	assert.Equal(t, "/etc/hosts", executor.path)
}

func TestExecObjectUsesOnlyPublicFieldsForDisplay(t *testing.T) {
	executor := &fakeExecutor{runOutput: Output{Digest: "sha256:script", Data: []byte(`{
  "apiVersion":"preflight.openfuyao.io/fact/v1",
  "available":true,
  "type":"object",
  "value":{"runtime":"containerd","healthy":true,"private":"hidden"}
}`)}}
	collector, err := New(executor, []Spec{{
		FactKey: "custom.cluster.runtime", Type: "exec", Path: "/opt/check-runtime",
		Args: []string{"--json"}, ResultType: api.ValueTypeObject, ResultKind: "object",
		PublicFields: []string{"runtime", "healthy"},
	}})
	require.NoError(t, err)
	fact := collectOne(t, collector, "custom.cluster.runtime")
	assert.True(t, fact.Available)
	assert.Equal(t, map[string]any{"runtime": "containerd", "healthy": true}, fact.Display.Value)
	assert.Equal(t, "hidden", fact.Value.(map[string]any)["private"])
	assert.Equal(t, []string{"--json"}, executor.args)
}

func TestExecUnavailableFactIsPreserved(t *testing.T) {
	executor := &fakeExecutor{runOutput: Output{Data: []byte(`{
  "apiVersion":"preflight.openfuyao.io/fact/v1",
  "available":false,
  "error":{"code":"PERMISSION_DENIED","message":"script is not executable"}
}`)}}
	collector, err := New(executor, []Spec{{
		FactKey: "custom.init.script", Type: "exec", Path: "/opt/check",
		ResultType: api.ValueTypeBool, ResultKind: "bool",
	}})
	require.NoError(t, err)
	fact := collectOne(t, collector, "custom.init.script")
	assert.False(t, fact.Available)
	assert.Equal(t, api.FactErrPermissionDenied, fact.Error.Code)
}

func TestCollectorFailureModes(t *testing.T) {
	assert.Equal(t, "custom.init.health", FactKey(api.OperationInit, "health"))
	_, err := New(nil, []Spec{{}, {FactKey: "x"}})
	assert.ErrorContains(t, err, "required")
	_, err = New(nil, []Spec{{FactKey: "x"}, {FactKey: "x"}})
	assert.ErrorContains(t, err, "duplicate")

	collector, err := New(nil, []Spec{{FactKey: "x", Type: "fileContent", Path: "/x"}})
	require.NoError(t, err)
	_, err = collector.Collect(context.Background(), api.FactRequest{})
	assert.ErrorContains(t, err, "one subject")
	fact := collectOne(t, collector, "missing")
	assert.Equal(t, api.FactErrUnsupported, fact.Error.Code)
	fact = collectOne(t, collector, "x")
	assert.Equal(t, api.FactErrUnsupported, fact.Error.Code)

	executor := &fakeExecutor{err: errors.New("denied")}
	collector, err = New(executor, []Spec{{FactKey: "x", Type: "fileContent", Path: "/x"}})
	require.NoError(t, err)
	fact = collectOne(t, collector, "x")
	assert.Equal(t, api.FactErrUnreachable, fact.Error.Code)

	collector, err = New(&fakeExecutor{}, []Spec{{FactKey: "x", Type: "unknown"}})
	require.NoError(t, err)
	fact = collectOne(t, collector, "x")
	assert.Equal(t, api.FactErrUnsupported, fact.Error.Code)
	assert.Equal(t, "custom", collector.ID())
}

func TestDecodeScriptFactRejectsInvalidOutput(t *testing.T) {
	spec := Spec{ResultType: api.ValueTypeInteger, ResultKind: "integer"}
	tests := [][]byte{
		nil,
		[]byte(`{"apiVersion":"wrong","available":true,"type":"integer","value":1}`),
		[]byte(`{"apiVersion":"preflight.openfuyao.io/fact/v1","available":true,"type":"string","value":"x"}`),
		[]byte(`{"apiVersion":"preflight.openfuyao.io/fact/v1","available":true,"type":"integer","value":1.2}`),
		[]byte(`{"apiVersion":"preflight.openfuyao.io/fact/v1","available":true,"type":"integer","value":1} {}`),
		[]byte(`{"apiVersion":"preflight.openfuyao.io/fact/v1","available":true,"type":"integer","value":1,"extra":true}`),
	}
	for _, raw := range tests {
		_, err := decodeScriptFact(raw, spec)
		assert.Error(t, err, string(raw))
	}
}

func TestNormalizeObjectRejectsNestedValues(t *testing.T) {
	_, err := normalizeObject(map[string]any{"nested": map[string]any{"x": true}})
	assert.ErrorContains(t, err, "scalar")
	_, err = normalizeList([]any{map[string]any{"x": true}}, api.ValueTypeSet, "")
	assert.ErrorContains(t, err, "scalar")
}

func TestNormalizeScriptValueTypes(t *testing.T) {
	tests := []struct {
		valueType api.ValueType
		kind      string
		value     any
		expected  any
	}{
		{api.ValueTypeBool, "bool", true, true},
		{api.ValueTypeInteger, "integer", json.Number("7"), int64(7)},
		{api.ValueTypeString, "string", "ready", "ready"},
		{api.ValueTypeSet, "stringList", []any{"a", "b"}, []any{"a", "b"}},
		{api.ValueTypeSet, "integerList", []any{json.Number("1")}, []any{int64(1)}},
	}
	for _, test := range tests {
		actual, err := normalizeScriptValue(test.valueType, test.value, test.kind)
		require.NoError(t, err)
		assert.Equal(t, test.expected, actual)
	}
	_, err := normalizeScriptValue(api.ValueTypeBool, "true", "bool")
	assert.ErrorContains(t, err, "does not match")
	_, err = normalizeList([]any{"x"}, api.ValueTypeSet, "integerList")
	assert.ErrorContains(t, err, "integer")
	_, err = normalizeList([]any{json.Number("1")}, api.ValueTypeSet, "stringList")
	assert.ErrorContains(t, err, "string")
}

func TestObjectRequiresEveryPublicField(t *testing.T) {
	_, err := publicObject(map[string]any{"healthy": true}, []string{"healthy", "runtime"})
	assert.ErrorContains(t, err, "runtime")
}

// TestNormalizeObjectAcceptsScalarsAndScalarLists freezes the object shape
// contract: string/bool/integer scalars plus scalar lists are accepted.
func TestNormalizeObjectAcceptsScalarsAndScalarLists(t *testing.T) {
	value, err := normalizeObject(map[string]any{
		"name": "node-1", "healthy": true, "count": json.Number("3"),
		"labels": []any{"a", "b"}, "ids": []any{json.Number("1"), json.Number("2")},
	})
	require.NoError(t, err)
	assert.Equal(t, map[string]any{
		"name": "node-1", "healthy": true, "count": int64(3),
		"labels": []any{"a", "b"}, "ids": []any{int64(1), int64(2)},
	}, value)
}

// TestNormalizeObjectRejectsInvalidShapes freezes every rejection branch.
func TestNormalizeObjectRejectsInvalidShapes(t *testing.T) {
	_, err := normalizeObject("not-an-object")
	assert.ErrorContains(t, err, "at most")

	tooMany := map[string]any{}
	for i := 0; i < maxObjectFields+1; i++ {
		tooMany[fmt.Sprintf("f%d", i)] = true
	}
	_, err = normalizeObject(tooMany)
	assert.ErrorContains(t, err, "at most")

	_, err = normalizeObject(map[string]any{"n": json.Number("not-int")})
	assert.ErrorContains(t, err, "integer")

	_, err = normalizeObject(map[string]any{"list": []any{map[string]any{"x": true}}})
	assert.ErrorContains(t, err, "list")

	_, err = normalizeObject(map[string]any{"nested": []any{true, "mixed"}})
	assert.ErrorContains(t, err, "scalar")
}

// TestExecFactErrorPaths freezes the exec error classification: executor
// failure maps to UNREACHABLE (or TIMEOUT under a canceled context) and
// malformed script output maps to PARSE_ERROR.
func TestExecFactErrorPaths(t *testing.T) {
	executor := &fakeExecutor{err: errors.New("script not found")}
	collector, err := New(executor, []Spec{{FactKey: "x", Type: "exec", Path: "/opt/check"}})
	require.NoError(t, err)
	fact := collectOne(t, collector, "x")
	assert.Equal(t, api.FactErrUnreachable, fact.Error.Code)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	facts, err := collector.Collect(ctx, api.FactRequest{Keys: []string{"x"}, Subjects: []api.Subject{{ID: "n"}}})
	require.NoError(t, err)
	assert.Equal(t, api.FactErrTimeout, facts[0].Error.Code)

	collector, err = New(&fakeExecutor{runOutput: Output{Data: []byte("not-json")}}, []Spec{{FactKey: "x", Type: "exec", Path: "/opt/check"}})
	require.NoError(t, err)
	fact = collectOne(t, collector, "x")
	assert.Equal(t, api.FactErrParseError, fact.Error.Code)
}

// TestDecodeScriptFactRejectsUnavailableWithoutError freezes the consistency
// contract between available and error fields.
func TestDecodeScriptFactRejectsUnavailableWithoutError(t *testing.T) {
	spec := Spec{ResultType: api.ValueTypeBool, ResultKind: "bool"}
	_, err := decodeScriptFact([]byte(`{"apiVersion":"preflight.openfuyao.io/fact/v1","available":false}`), spec)
	assert.ErrorContains(t, err, "inconsistent")
	_, err = decodeScriptFact([]byte(`{"apiVersion":"preflight.openfuyao.io/fact/v1","available":false,"error":{"code":"","message":"m"}}`), spec)
	assert.ErrorContains(t, err, "error code")
	_, err = decodeScriptFact([]byte(`{"apiVersion":"preflight.openfuyao.io/fact/v1","available":false,"error":{"code":"X","message":"  "}}`), spec)
	assert.ErrorContains(t, err, "error code")
}

// TestNormalizeScriptValueRejectsUnsupportedType freezes the default branch.
func TestNormalizeScriptValueRejectsUnsupportedType(t *testing.T) {
	_, err := normalizeScriptValue(api.ValueTypeDuration, "10s", "duration")
	assert.ErrorContains(t, err, "unsupported")
	_, err = normalizeScriptValue(api.ValueTypeInteger, "not-number", "integer")
	assert.ErrorContains(t, err, "does not match")
	_, err = normalizeScriptValue(api.ValueTypeSet, "not-list", "stringList")
	assert.ErrorContains(t, err, "array")
	_, err = normalizeList([]any{json.Number("1.5")}, api.ValueTypeSet, "integerList")
	assert.ErrorContains(t, err, "integer")
}

// TestRequireJSONEOFRejectsTrailingGarbage freezes the single-document rule:
// trailing malformed JSON is a decode error, a second valid document is
// rejected as "one JSON document".
func TestRequireJSONEOFRejectsTrailingGarbage(t *testing.T) {
	decoder := json.NewDecoder(bytes.NewReader([]byte(`{"a":1} {"b":2}`)))
	var first map[string]any
	require.NoError(t, decoder.Decode(&first))
	err := requireJSONEOF(decoder)
	assert.ErrorContains(t, err, "one JSON document")

	decoder = json.NewDecoder(bytes.NewReader([]byte(`{"a":1} garbage`)))
	require.NoError(t, decoder.Decode(&first))
	err = requireJSONEOF(decoder)
	assert.ErrorContains(t, err, "decode trailing")
}
