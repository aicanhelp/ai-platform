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

import "testing"

func TestGraph_TopologicalBatches_Linear(t *testing.T) {
	g := NewGraph()
	for _, n := range []string{"a", "b", "c"} {
		g.AddNode(n)
	}
	if err := g.AddEdge("a", "b"); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge("b", "c"); err != nil {
		t.Fatal(err)
	}
	batches, err := g.TopologicalBatches()
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) != 3 {
		t.Fatalf("expected 3 batches, got %v", batches)
	}
	if batches[0][0] != "a" || batches[2][0] != "c" {
		t.Fatalf("unexpected order: %v", batches)
	}
}

func TestGraph_AddEdge_Cycle(t *testing.T) {
	g := NewGraph()
	g.AddNode("a")
	g.AddNode("b")
	_ = g.AddEdge("a", "b")
	if err := g.AddEdge("b", "a"); err == nil {
		t.Fatal("expected cycle error")
	}
}

func TestGraph_ParallelBatch(t *testing.T) {
	g := NewGraph()
	for _, n := range []string{"root", "x", "y", "z"} {
		g.AddNode(n)
	}
	_ = g.AddEdge("root", "x")
	_ = g.AddEdge("root", "y")
	_ = g.AddEdge("x", "z")
	_ = g.AddEdge("y", "z")

	batches, err := g.TopologicalBatches()
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) != 3 {
		t.Fatalf("expected 3 batches, got %v", batches)
	}
	if len(batches[1]) != 2 {
		t.Fatalf("expected parallel batch of 2, got %v", batches[1])
	}
}
