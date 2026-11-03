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
	"fmt"

	"gopkg.openfuyao.cn/preflight/api"
)

// CycleError is returned when a rule dependency graph contains a cycle. A
// cycle is a BUNDLE_ERROR: the rules cannot be evaluated in a stable order.
type CycleError struct {
	Cycle []string
}

func (e *CycleError) Error() string {
	return "rule dependency cycle: " + fmt.Sprint(e.Cycle)
}

// TopoSort performs a topological sort of rules by their DependsOn edges.
// It returns the rule IDs in dependency order (dependencies first) or a
// *CycleError if the graph contains a cycle. Rules with no DependsOn come
// first. Missing dependencies (a DependsOn referencing a non-existent rule
// ID) are tolerated and treated as already-satisfied; the loader is the
// stricter gate for dangling references.
//
// DependsOn may only reference other rule IDs; self-dependence is a cycle.
func TopoSort(rules []api.RuleTemplate) ([]string, error) {
	ids := make(map[string]struct{}, len(rules))
	order := make([]string, 0, len(rules))
	for _, r := range rules {
		ids[r.ID] = struct{}{}
	}
	// Kahn's algorithm. In-degree counts how many rules depend ON each id
	// (reverse edges), so we emit a rule only after its dependencies.
	indeg := make(map[string]int, len(ids))
	for id := range ids {
		indeg[id] = 0
	}
	// Build reverse: who depends on me.
	dependents := make(map[string][]string, len(ids))
	for _, r := range rules {
		for _, dep := range r.DependsOn {
			if _, ok := ids[dep]; ok {
				dependents[dep] = append(dependents[dep], r.ID)
				indeg[r.ID]++
			}
		}
	}

	// Seed with in-degree 0.
	var queue []string
	for _, r := range rules {
		if indeg[r.ID] == 0 {
			queue = append(queue, r.ID)
		}
	}
	for len(queue) > 0 {
		current := queue
		next := make([]string, len(ids))
		nextCount := 0
		for _, id := range current {
			order = append(order, id)
			released := releasedDependents(indeg, dependents[id])
			nextCount += copy(next[nextCount:], released)
		}
		queue = next[:nextCount]
	}
	if len(order) != len(ids) {
		// Cycle: find the remaining nodes.
		var cycle []string
		for _, r := range rules {
			if indeg[r.ID] > 0 {
				cycle = append(cycle, r.ID)
			}
		}
		return nil, &CycleError{Cycle: cycle}
	}
	return order, nil
}

func releasedDependents(indeg map[string]int, dependents []string) []string {
	released := make([]string, len(dependents))
	count := 0
	for _, dependent := range dependents {
		if decrementIndegree(indeg, dependent) == 0 {
			released[count] = dependent
			count++
		}
	}
	return released[:count]
}

func decrementIndegree(indeg map[string]int, id string) int {
	if indeg == nil {
		return -1
	}
	current, exists := indeg[id]
	if !exists {
		return -1
	}
	remaining := current - 1
	indeg[id] = remaining
	return remaining
}
