/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package graph

import (
	"strconv"
	"strings"
)

// TopologyProvider defines interface for providing topology information
type TopologyProvider interface {
	Topology() string
}

// TopologyGraph represents a 2D adjacency matrix for topology relationships
type TopologyGraph [][]int

// NewTopologyGraph creates a new topology graph with specified capacity
func NewTopologyGraph(capability int) TopologyGraph {
	if capability < 0 {
		panic("capacity must not be negative")
	}
	graph := make(TopologyGraph, capability)
	for i := range graph {
		graph[i] = make([]int, capability)
	}
	return graph
}

// GetTopologyGraph returns the serialized string representation of the topology
func (graph TopologyGraph) GetTopologyGraph() string {
	return graph.Serialize()
}

// Serialize converts the topology graph to a string representation
func (graph TopologyGraph) Serialize() string {
	result := make([]string, 0, len(graph))
	for _, row := range graph {
		tokens := make([]string, 0, len(row))
		for _, v := range row {
			tokens = append(tokens, strconv.Itoa(v))
		}
		result = append(result, strings.Join(tokens, ","))
	}
	return strings.Join(result, ";")
}
