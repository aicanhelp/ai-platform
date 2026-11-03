/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package topology

import (
	"fmt"
	"sort"
)

// FailurePolicy controls how the scheduler reacts to component errors.
type FailurePolicy string

const (
	FailurePolicyFailFast FailurePolicy = "FailFast"
	FailurePolicyContinue FailurePolicy = "Continue"
	FailurePolicyRollback FailurePolicy = "Rollback"
)

// InlineRef points to a ComponentFactory handler registration.
type InlineRef struct {
	Handler string
	Version string
}

// ComponentNode is one vertex in the upgrade DAG.
type ComponentNode struct {
	Name          string
	Version       string
	Inline        *InlineRef
	FailurePolicy FailurePolicy
	Dependencies  []string
}

// UpgradeDAG is the upgrade dependency graph with component metadata.
type UpgradeDAG struct {
	graph *Graph
	nodes map[string]*ComponentNode
}

// NewUpgradeDAG creates an empty upgrade DAG.
func NewUpgradeDAG() *UpgradeDAG {
	return &UpgradeDAG{
		graph: NewGraph(),
		nodes: make(map[string]*ComponentNode),
	}
}

// AddNode registers a component node.
func (d *UpgradeDAG) AddNode(node *ComponentNode) error {
	if node == nil || node.Name == "" {
		return fmt.Errorf("component node name is required")
	}
	if _, exists := d.nodes[node.Name]; exists {
		return fmt.Errorf("duplicate component %q", node.Name)
	}
	cp := *node
	d.nodes[node.Name] = &cp
	d.graph.AddNode(node.Name)
	return nil
}

// AddDependency records that prerequisite must complete before dependent.
func (d *UpgradeDAG) AddDependency(prerequisite, dependent string) error {
	if _, ok := d.nodes[dependent]; !ok {
		return fmt.Errorf("dependent component %q not in DAG", dependent)
	}
	if prerequisite != "" {
		if _, ok := d.nodes[prerequisite]; !ok {
			return fmt.Errorf("prerequisite component %q not in DAG", prerequisite)
		}
		return d.graph.AddEdge(prerequisite, dependent)
	}
	return nil
}

// GetNode returns the component node by name.
func (d *UpgradeDAG) GetNode(name string) (*ComponentNode, bool) {
	node, ok := d.nodes[name]
	return node, ok
}

// TopologicalBatches returns sorted execution batches.
func (d *UpgradeDAG) TopologicalBatches() ([][]string, error) {
	return d.graph.TopologicalBatches()
}

// NodeNames returns all component names in stable order.
func (d *UpgradeDAG) NodeNames() []string {
	names := make([]string, 0, len(d.nodes))
	for name := range d.nodes {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
