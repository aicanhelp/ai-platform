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
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

// mkRuleDeps builds a rule with dependencies.
func mkRuleDeps(id string, deps ...string) api.RuleTemplate {
	return api.RuleTemplate{
		ID: id, Operation: api.OperationInit, Fact: "f", Op: api.OpEq, Expected: true,
		Level: api.SeverityError, Unknown: api.UnknownBlock, Fix: "f",
		DependsOn: deps,
	}
}

// TestTopoSortNoDeps freezes that dependency-free rules sort in declaration order.
func TestTopoSortNoDeps(t *testing.T) {
	rules := []api.RuleTemplate{mkRuleDeps("A"), mkRuleDeps("B"), mkRuleDeps("C")}
	order, err := TopoSort(rules)
	require.NoError(t, err)
	assert.Equal(t, []string{"A", "B", "C"}, order)
}

// TestTopoSortDependencyOrder freezes that a dependency is emitted before its
// dependent.
func TestTopoSortDependencyOrder(t *testing.T) {
	// B depends on A; C depends on B. Order must have A before B before C.
	rules := []api.RuleTemplate{
		mkRuleDeps("C", "B"),
		mkRuleDeps("B", "A"),
		mkRuleDeps("A"),
	}
	order, err := TopoSort(rules)
	require.NoError(t, err)
	idx := make(map[string]int)
	for i, id := range order {
		idx[id] = i
	}
	assert.Less(t, idx["A"], idx["B"])
	assert.Less(t, idx["B"], idx["C"])
}

// TestTopoSortDetectsCycle freezes that a cycle is a CycleError.
func TestTopoSortDetectsCycle(t *testing.T) {
	// A -> B -> A.
	rules := []api.RuleTemplate{
		mkRuleDeps("A", "B"),
		mkRuleDeps("B", "A"),
	}
	_, err := TopoSort(rules)
	require.Error(t, err)
	ce, ok := err.(*CycleError)
	require.True(t, ok)
	assert.Contains(t, ce.Cycle, "A")
	assert.Contains(t, ce.Cycle, "B")
}

// TestTopoSortSelfCycle freezes that a self-dependence is a cycle.
func TestTopoSortSelfCycle(t *testing.T) {
	rules := []api.RuleTemplate{mkRuleDeps("A", "A")}
	_, err := TopoSort(rules)
	require.Error(t, err)
	_, ok := err.(*CycleError)
	assert.True(t, ok)
}

// TestTopoSortDanglingDepTolerated freezes that a DependsOn referencing a
// non-existent rule is tolerated (treated as satisfied), so a rule whose
// optional dependency was not instantiated does not block evaluation.
func TestTopoSortDanglingDepTolerated(t *testing.T) {
	rules := []api.RuleTemplate{mkRuleDeps("A", "GONE")}
	order, err := TopoSort(rules)
	require.NoError(t, err)
	assert.Equal(t, []string{"A"}, order)
}

// TestTopoSortDiamond freezes a diamond dependency resolves in valid order.
func TestTopoSortDiamond(t *testing.T) {
	// A -> B, A -> C, B -> D, C -> D  (D depends on B and C; B,C depend on A).
	rules := []api.RuleTemplate{
		mkRuleDeps("D", "B", "C"),
		mkRuleDeps("B", "A"),
		mkRuleDeps("C", "A"),
		mkRuleDeps("A"),
	}
	order, err := TopoSort(rules)
	require.NoError(t, err)
	idx := make(map[string]int)
	for i, id := range order {
		idx[id] = i
	}
	assert.Less(t, idx["A"], idx["B"])
	assert.Less(t, idx["A"], idx["C"])
	assert.Less(t, idx["B"], idx["D"])
	assert.Less(t, idx["C"], idx["D"])
}

func TestCycleErrorMessage(t *testing.T) {
err := &CycleError{Cycle: []string{"A", "B", "A"}}
assert.Contains(t, err.Error(), "rule dependency cycle")
assert.Contains(t, err.Error(), "A")
}
