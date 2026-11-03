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

// Package custom collects operation-scoped facts from a bounded file reader
// or an external executable. It never evaluates rules or returns PASS/FAIL.
package custom

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"gopkg.openfuyao.cn/preflight/api"
)

const (
	// MaxOutputBytes is the hard bound for file and script output.
	MaxOutputBytes  = 64 << 10
	maxObjectFields = 32
	factAPIVersion  = "preflight.openfuyao.io/fact/v1"
)

// Spec is one validated collector definition indexed by its fact key.
type Spec struct {
	FactKey      string
	Type         string
	Path         string
	Args         []string
	ResultType   api.ValueType
	ResultKind   string
	PublicFields []string
}

// FactKey returns the stable fact key for an operation-scoped custom check.
func FactKey(operation api.Operation, key string) string {
	return "custom." + string(operation) + "." + key
}

// Output is bounded collector output plus the framework-computed SHA-256 of
// the observed file or executable.
type Output struct {
	Data   []byte
	Digest api.Digest
}

// Executor is implemented by the host so the SDK does not own OS or SSH
// credentials. Implementations must enforce the same output bound.
type Executor interface {
	ReadFile(ctx context.Context, subject api.Subject, path string, limit int64) (Output, error)
	Run(ctx context.Context, subject api.Subject, path string, args []string, limit int64) (Output, error)
}

// Collector dispatches validated custom specs to the injected host executor.
type Collector struct {
	executor Executor
	specs    map[string]Spec
}

// New constructs a collector. Duplicate or empty keys are rejected.
func New(executor Executor, specs []Spec) (*Collector, error) {
	indexed := make(map[string]Spec, len(specs))
	for _, spec := range specs {
		if spec.FactKey == "" {
			return nil, fmt.Errorf("custom collector fact key is required")
		}
		if _, exists := indexed[spec.FactKey]; exists {
			return nil, fmt.Errorf("duplicate custom collector fact key %q", spec.FactKey)
		}
		indexed[spec.FactKey] = spec
	}
	return &Collector{executor: executor, specs: indexed}, nil
}

// ID returns the stable collector identifier.
func (c *Collector) ID() string { return "custom" }

// Collect collects exactly one fact, as required by the SDK router contract.
func (c *Collector) Collect(ctx context.Context, request api.FactRequest) ([]api.Fact, error) {
	if len(request.Keys) != 1 || len(request.Subjects) != 1 {
		return nil, fmt.Errorf("custom collector requires one subject and one fact key")
	}
	key, subject := request.Keys[0], request.Subjects[0]
	spec, found := c.specs[key]
	if !found {
		return []api.Fact{unavailable(key, subject.ID, api.FactErrUnsupported, "custom collector spec not found")}, nil
	}
	if c.executor == nil {
		return []api.Fact{unavailable(key, subject.ID, api.FactErrUnsupported, "custom executor is not configured")}, nil
	}
	fact := c.collect(ctx, subject, spec)
	return []api.Fact{fact}, nil
}

func (c *Collector) collect(ctx context.Context, subject api.Subject, spec Spec) api.Fact {
	switch spec.Type {
	case "fileContent":
		return c.fileFact(ctx, subject, spec)
	case "exec":
		return c.execFact(ctx, subject, spec)
	default:
		return unavailable(spec.FactKey, subject.ID, api.FactErrUnsupported, "unsupported custom collector type")
	}
}

func (c *Collector) fileFact(ctx context.Context, subject api.Subject, spec Spec) api.Fact {
	output, err := c.executor.ReadFile(ctx, subject, spec.Path, MaxOutputBytes)
	if err != nil {
		return executionError(ctx, spec.FactKey, subject.ID, err)
	}
	return api.Fact{
		Key: spec.FactKey, Subject: subject.ID, Available: true,
		Type: api.ValueTypeString, Value: string(output.Data),
		Display:  &api.DisplayValue{Type: api.ValueTypeString, Value: "已读取（内容不展示）"},
		Evidence: []api.Evidence{{Kind: "file", Locator: spec.Path, Observed: output.Digest.String(), Source: "custom"}},
	}
}

func (c *Collector) execFact(ctx context.Context, subject api.Subject, spec Spec) api.Fact {
	output, err := c.executor.Run(ctx, subject, spec.Path, spec.Args, MaxOutputBytes)
	if err != nil {
		return executionError(ctx, spec.FactKey, subject.ID, err)
	}
	fact, err := decodeScriptFact(output.Data, spec)
	if err != nil {
		return unavailable(spec.FactKey, subject.ID, api.FactErrParseError, err.Error())
	}
	fact.Key, fact.Subject = spec.FactKey, subject.ID
	fact.Evidence = []api.Evidence{{
		Kind: "command", Locator: spec.Path, Observed: output.Digest.String(), Source: "custom",
	}}
	return fact
}

type scriptFact struct {
	APIVersion string         `json:"apiVersion"`
	Available  bool           `json:"available"`
	Type       string         `json:"type,omitempty"`
	Value      any            `json:"value,omitempty"`
	Error      *api.FactError `json:"error,omitempty"`
}

func decodeScriptFact(raw []byte, spec Spec) (api.Fact, error) {
	if len(raw) == 0 || len(raw) > MaxOutputBytes {
		return api.Fact{}, fmt.Errorf("script output must be 1..%d bytes", MaxOutputBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	decoder.UseNumber()
	var decoded scriptFact
	if err := decoder.Decode(&decoded); err != nil {
		return api.Fact{}, fmt.Errorf("decode Fact JSON: %w", err)
	}
	if err := requireJSONEOF(decoder); err != nil {
		return api.Fact{}, err
	}
	if decoded.APIVersion != factAPIVersion {
		return api.Fact{}, fmt.Errorf("apiVersion must be %q", factAPIVersion)
	}
	if decoded.Available == (decoded.Error != nil) {
		return api.Fact{}, fmt.Errorf("available and error state are inconsistent")
	}
	if !decoded.Available {
		if decoded.Error.Code == "" || strings.TrimSpace(decoded.Error.Message) == "" {
			return api.Fact{}, fmt.Errorf("unavailable Fact requires error code and message")
		}
		return api.Fact{Available: false, Error: decoded.Error}, nil
	}
	if decoded.Type != spec.ResultKind {
		return api.Fact{}, fmt.Errorf("fact type %q does not match resultType %q", decoded.Type, spec.ResultKind)
	}
	value, err := normalizeScriptValue(spec.ResultType, decoded.Value, spec.ResultKind)
	if err != nil {
		return api.Fact{}, err
	}
	display := api.DisplayValue{Type: spec.ResultType, Value: value}
	if spec.ResultType == api.ValueTypeObject {
		public, err := publicObject(value.(map[string]any), spec.PublicFields)
		if err != nil {
			return api.Fact{}, err
		}
		display.Value = public
	}
	return api.Fact{Available: true, Type: spec.ResultType, Value: value, Display: &display}, nil
}

func requireJSONEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err == io.EOF {
		return nil
	} else if err != nil {
		return fmt.Errorf("decode trailing Fact JSON: %w", err)
	}
	return fmt.Errorf("script output must contain one JSON document")
}

func normalizeScriptValue(valueType api.ValueType, value any, resultKind string) (any, error) {
	switch valueType {
	case api.ValueTypeBool:
		if typed, ok := value.(bool); ok {
			return typed, nil
		}
	case api.ValueTypeInteger:
		if typed, ok := value.(json.Number); ok {
			integer, err := typed.Int64()
			if err == nil {
				return integer, nil
			}
		}
	case api.ValueTypeString:
		if typed, ok := value.(string); ok {
			return typed, nil
		}
	case api.ValueTypeSet:
		return normalizeList(value, valueType, resultKind)
	case api.ValueTypeObject:
		return normalizeObject(value)
	default:
		return nil, fmt.Errorf("fact value has unsupported type %q", valueType)
	}
	return nil, fmt.Errorf("fact value does not match type %q", valueType)
}

func normalizeList(value any, _ api.ValueType, resultKind string) (any, error) {
	items, ok := value.([]any)
	if !ok {
		return nil, fmt.Errorf("fact list value must be an array")
	}
	out := make([]any, len(items))
	for i, item := range items {
		switch typed := item.(type) {
		case string:
			if resultKind == "integerList" {
				return nil, fmt.Errorf("fact list item %d must be an integer", i)
			}
			out[i] = typed
		case json.Number:
			if resultKind == "stringList" {
				return nil, fmt.Errorf("fact list item %d must be a string", i)
			}
			integer, err := typed.Int64()
			if err != nil {
				return nil, fmt.Errorf("fact list item %d must be an integer", i)
			}
			out[i] = integer
		default:
			return nil, fmt.Errorf("fact list item %d must be scalar", i)
		}
	}
	return out, nil
}

func normalizeObject(value any) (any, error) {
	object, ok := value.(map[string]any)
	if !ok || len(object) > maxObjectFields {
		return nil, fmt.Errorf("fact object must contain at most %d fields", maxObjectFields)
	}
	out := make(map[string]any, len(object))
	for key, item := range object {
		switch typed := item.(type) {
		case string, bool:
			out[key] = typed
		case json.Number:
			integer, err := typed.Int64()
			if err != nil {
				return nil, fmt.Errorf("fact object field %s must be an integer", key)
			}
			out[key] = integer
		case []any:
			list, err := normalizeList(typed, api.ValueTypeSet, "")
			if err != nil {
				return nil, fmt.Errorf("fact object field %s: %w", key, err)
			}
			out[key] = list
		default:
			return nil, fmt.Errorf("fact object field %s must be scalar or a scalar list", key)
		}
	}
	return out, nil
}

func publicObject(value map[string]any, fields []string) (map[string]any, error) {
	public := make(map[string]any, len(fields))
	for _, field := range fields {
		item, exists := value[field]
		if !exists {
			return nil, fmt.Errorf("fact object is missing public field %q", field)
		}
		public[field] = item
	}
	return public, nil
}

func executionError(ctx context.Context, key, subject string, err error) api.Fact {
	code := api.FactErrUnreachable
	if ctx.Err() != nil {
		code = api.FactErrTimeout
	}
	return unavailable(key, subject, code, err.Error())
}

func unavailable(key, subject, code, message string) api.Fact {
	return api.Fact{
		Key: key, Subject: subject, Available: false,
		Error: &api.FactError{Code: code, Message: message},
	}
}

var _ api.Collector = (*Collector)(nil)
