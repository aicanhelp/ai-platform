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

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

// validRule is a contract-conforming rule used as the baseline for the
// validation table.
func validRule() RuleTemplate {
	return RuleTemplate{
		ID:        "INIT-PORT-36443",
		Operation: OperationInit,
		Fact:      "port.36443.free",
		Op:        OpEq,
		Expected:  true,
		Level:     SeverityError,
		Unknown:   UnknownBlock,
		Fix:       "stop the process using port 36443",
	}
}

// TestRuleValidateAcceptsValid freezes that a well-formed rule passes.
func TestRuleValidateAcceptsValid(t *testing.T) {
	assert.Nil(t, validRule().Validate())
}

// TestRuleValidateRejectsMissingFields freezes the static contract: a rule
// missing any required field or carrying an invalid enum is a BUNDLE_ERROR.
func TestRuleValidateRejectsMissingFields(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(r *RuleTemplate)
		field  string
	}{
		{"missing id", func(r *RuleTemplate) { r.ID = "" }, "id"},
		{"unsupported operation", func(r *RuleTemplate) { r.Operation = "build" }, "operation"},
		{"missing fact", func(r *RuleTemplate) { r.Fact = "" }, "fact"},
		{"unsupported op", func(r *RuleTemplate) { r.Op = "matches" }, "op"},
		{"missing expected", func(r *RuleTemplate) { r.Expected = nil }, "expected"},
		{"invalid level", func(r *RuleTemplate) { r.Level = "FATAL" }, "level"},
		{"invalid unknown", func(r *RuleTemplate) { r.Unknown = "RETRY" }, "unknown"},
		{"missing fix", func(r *RuleTemplate) { r.Fix = "" }, "fix"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := validRule()
			tt.mutate(&r)
			err := r.Validate()
			assert.NotNil(t, err, "%s should fail validation", tt.name)
			assert.Equal(t, tt.field, err.Field)
		})
	}
}

func TestRuleValidateRejectsExpectedTypeMismatch(t *testing.T) {
	tests := []struct {
		op       Op
		expected any
	}{
		{OpIn, "not-a-set"},
		{OpGte, "not-a-number"},
		{OpLte, true},
		{OpRange, []any{"low", "high"}},
		{OpExists, "yes"},
		{OpFree, 1},
		{OpReachable, "true"},
		{OpRelation, "same"},
	}
	for _, tt := range tests {
		t.Run(string(tt.op), func(t *testing.T) {
			r := validRule()
			r.Op, r.Expected = tt.op, tt.expected
			err := r.Validate()
			assert.NotNil(t, err)
			assert.Equal(t, "expected", err.Field)
		})
	}
}

// TestOpValid freezes the complete operator set: nine built-ins plus CEL.
func TestOpValid(t *testing.T) {
	valid := []Op{OpEq, OpIn, OpGte, OpLte, OpRange, OpExists, OpFree, OpReachable, OpRelation, OpCEL}
	for _, o := range valid {
		assert.True(t, o.Valid(), "%s should be valid", o)
	}
	assert.False(t, Op("matches").Valid())
	assert.False(t, Op("").Valid())
}

func TestRuleValidateCELExpressionContract(t *testing.T) {
	rule := validRule()
	rule.Op, rule.Expected = OpCEL, true
	assert.Equal(t, "expression", rule.Validate().Field)
	rule.Expression = "actual == true"
	assert.Nil(t, rule.Validate())
	rule.Op = OpEq
	assert.Equal(t, "expression", rule.Validate().Field)
}

// TestBundleDuplicateRuleIDs freezes that duplicate rule IDs are detected
// (a BUNDLE_ERROR at load time).
func TestBundleDuplicateRuleIDs(t *testing.T) {
	b := &PreflightBundle{
		Rules: []RuleTemplate{
			validRule(),
			{ID: "INIT-PORT-36443", Operation: OperationInit, Fact: "x", Op: OpEq, Expected: true, Level: SeverityError, Unknown: UnknownBlock, Fix: "f"},
		},
	}
	dup, ok := b.HasDuplicateRuleIDs()
	assert.True(t, ok)
	assert.Equal(t, "INIT-PORT-36443", dup)

	b2 := &PreflightBundle{Rules: []RuleTemplate{validRule()}}
	_, ok2 := b2.HasDuplicateRuleIDs()
	assert.False(t, ok2)
}

// TestBundleRuleIDs freezes rule-ID enumeration order.
func TestBundleRuleIDs(t *testing.T) {
	b := &PreflightBundle{
		Rules: []RuleTemplate{
			{ID: "A", Operation: OperationInit, Fact: "f", Op: OpEq, Expected: true, Level: SeverityError, Unknown: UnknownBlock, Fix: "x"},
			{ID: "B", Operation: OperationClusterCreate, Fact: "f", Op: OpExists, Expected: true, Level: SeverityError, Unknown: UnknownBlock, Fix: "x"},
		},
	}
	assert.Equal(t, []string{"A", "B"}, b.RuleIDs())
}

// TestSupportedOperations freezes the enabled operation set for this release.
func TestSupportedOperations(t *testing.T) {
	assert.True(t, IsSupportedOperation(OperationInit))
	assert.True(t, IsSupportedOperation(OperationClusterCreate))
	assert.False(t, IsSupportedOperation(Operation("build")))
	assert.False(t, IsSupportedOperation(Operation("")))
}

// TestDigest freezes the digest helpers.
func TestDigest(t *testing.T) {
	var d Digest
	assert.True(t, d.IsZero())
	d = Digest("sha256:abc")
	assert.False(t, d.IsZero())
	assert.Equal(t, "sha256:abc", d.String())
}

// TestRuleValueType freezes the report type implied by each operator and the
// bytes-fact heuristic.
func TestRuleValueType(t *testing.T) {
	tests := []struct {
		name     string
		op       Op
		fact     string
		expected ValueType
	}{
		{"cel is bool", OpCEL, "custom.init.x", ValueTypeBool},
		{"in is set", OpIn, "platform.distribution", ValueTypeSet},
		{"range is range", OpRange, "resources.cpu", ValueTypeRange},
		{"relation is relation", OpRelation, "cluster.topology", ValueTypeRelation},
		{"exists is bool", OpExists, "path.workspace", ValueTypeBool},
		{"free is bool", OpFree, "port.36443.free", ValueTypeBool},
		{"reachable is bool", OpReachable, "endpoint.tcp.x", ValueTypeBool},
		{"bytes fact", OpGte, "resources.memory.totalBytes", ValueTypeBytes},
		{"lowercase bytes fact", OpGte, "node.systemDisk.totalbytes", ValueTypeBytes},
		{"integer fallback", OpGte, "resources.cpu.logical", ValueTypeInteger},
		{"string fallback", OpEq, "platform.os", ValueTypeString},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rule := validRule()
			rule.Op, rule.Fact = tt.op, tt.fact
			switch tt.op {
			case OpGte, OpLte:
				rule.Expected = 1
			case OpIn:
				rule.Expected = []any{"a"}
			case OpRange:
				rule.Expected = []any{1, 2}
			case OpRelation:
				rule.Expected = true
			case OpEq:
				if tt.name == "string fallback" {
					rule.Expected = "linux"
				}
			}
			assert.Equal(t, tt.expected, rule.ValueType())
		})
	}
}

// TestContainsBytesFact freezes the substring scan for bytes-typed facts.
func TestContainsBytesFact(t *testing.T) {
	assert.True(t, containsBytesFact("resources.memory.totalBytes"))
	assert.True(t, containsBytesFact("node.systemDisk.totalbytes"))
	assert.False(t, containsBytesFact("port.36443.free"))
	assert.False(t, containsBytesFact("Byt"))
}

// TestNumericValue freezes the numeric coercion used by range validation.
func TestNumericValue(t *testing.T) {
	accepted := []any{int(1), int32(1), int64(1), uint(1), uint32(1), uint64(1), float32(1.5), float64(1.5)}
	for _, value := range accepted {
		_, ok := numericValue(value)
		assert.True(t, ok, "%T should be numeric", value)
	}
	_, ok := numericValue("1")
	assert.False(t, ok)
	_, ok = numericValue(nil)
	assert.False(t, ok)
}

// TestValidRangeValue freezes both accepted range shapes.
func TestValidRangeValue(t *testing.T) {
	assert.True(t, validRangeValue([]any{1, 2}))
	assert.True(t, validRangeValue([]any{int64(1), float64(2)}))
	assert.True(t, validRangeValue(map[string]any{"minInclusive": 1, "maxInclusive": 2}))
	assert.False(t, validRangeValue([]any{1}))
	assert.False(t, validRangeValue([]any{"a", "b"}))
	assert.False(t, validRangeValue(map[string]any{"minInclusive": 1}))
	assert.False(t, validRangeValue("range"))
}

// TestRuleContractError freezes the error message format.
func TestRuleContractError(t *testing.T) {
	err := &RuleContractError{Rule: "INIT-X", Field: "fact", Reason: "is required"}
	assert.Equal(t, "rule contract error: INIT-X.fact is required", err.Error())
}

// TestValueTypeValid freezes the supported value-type set.
func TestValueTypeValid(t *testing.T) {
	valid := []ValueType{
		ValueTypeBool, ValueTypeInteger, ValueTypeBytes, ValueTypeDuration,
		ValueTypeString, ValueTypeEnum, ValueTypeAddress, ValueTypeSet,
		ValueTypeRange, ValueTypeRelation, ValueTypeObject,
	}
	for _, vt := range valid {
		assert.True(t, vt.Valid(), "%s should be valid", vt)
	}
	assert.False(t, ValueType("").Valid())
	assert.False(t, ValueType("blob").Valid())
}

// TestFrameworkError freezes both message shapes and unwrapping.
func TestFrameworkError(t *testing.T) {
	cause := errors.New("dial failed")
	withCause := &FrameworkError{Code: FrameworkErrInternal, Message: "boom", Cause: cause}
	assert.Equal(t, "INTERNAL: boom: dial failed", withCause.Error())
	assert.ErrorIs(t, withCause, cause)

	withoutCause := &FrameworkError{Code: FrameworkErrBundleLoad, Message: "bad bundle"}
	assert.Equal(t, "BUNDLE_LOAD: bad bundle", withoutCause.Error())
	assert.Nil(t, withoutCause.Unwrap())
}
