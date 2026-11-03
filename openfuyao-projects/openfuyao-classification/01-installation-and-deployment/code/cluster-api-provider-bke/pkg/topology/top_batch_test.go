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
	"testing"
)

// batchTestCase describes a dependency graph and the expected topological batches.
// These cases document batch partitioning for parallel upgrade design (bin/dag-batch-parallel-upgrade-design.md).
type batchTestCase struct {
	name    string
	nodes   []string
	edges   [][2]string // [prerequisite, dependent]
	want    [][]string
	wantErr bool
}

func TestTopologicalBatches_WithDependencies(t *testing.T) {
	cases := []batchTestCase{
		{
			name:  "T1-linear-chain",
			nodes: []string{"A", "B", "C"},
			edges: [][2]string{{"A", "B"}, {"B", "C"}},
			want:  [][]string{{"A"}, {"B"}, {"C"}},
		},
		{
			name:  "T2-diamond",
			nodes: []string{"root", "X", "Y", "Z"},
			edges: [][2]string{{"root", "X"}, {"root", "Y"}, {"X", "Z"}, {"Y", "Z"}},
			want:  [][]string{{"root"}, {"X", "Y"}, {"Z"}},
		},
		{
			name:  "T3-fanout",
			nodes: []string{"root", "A", "B", "C"},
			edges: [][2]string{{"root", "A"}, {"root", "B"}, {"root", "C"}},
			want:  [][]string{{"root"}, {"A", "B", "C"}},
		},
		{
			name:  "T4-two-roots-merge",
			nodes: []string{"A", "B", "C"},
			edges: [][2]string{{"A", "C"}, {"B", "C"}},
			want:  [][]string{{"A", "B"}, {"C"}},
		},
		{
			name:  "T5-wide-parallel-single-batch",
			nodes: []string{"coredns", "kube-proxy", "metrics", "ingress", "dashboard"},
			edges: nil,
			want:  [][]string{{"coredns", "dashboard", "ingress", "kube-proxy", "metrics"}},
		},
		{
			name: "T6-declarative-upgrade-style-chain",
			nodes: []string{
				"pre-upgrade-resources",
				"provider",
				"bke-agent",
				"etcd",
				"k8s-master",
				"k8s-worker",
			},
			edges: [][2]string{
				{"pre-upgrade-resources", "provider"},
				{"pre-upgrade-resources", "bke-agent"},
				{"provider", "etcd"},
				{"bke-agent", "etcd"},
				{"etcd", "k8s-master"},
				{"k8s-master", "k8s-worker"},
			},
			want: [][]string{
				{"pre-upgrade-resources"},
				{"bke-agent", "provider"},
				{"etcd"},
				{"k8s-master"},
				{"k8s-worker"},
			}, // batch[1] is the parallel layer for parallel-upgrade design
		},
		{
			name:  "T7-layered-base-mid-leaf",
			nodes: []string{"base", "mid", "extra", "leaf"},
			edges: [][2]string{{"base", "mid"}, {"base", "extra"}, {"mid", "leaf"}, {"extra", "leaf"}},
			want:  [][]string{{"base"}, {"extra", "mid"}, {"leaf"}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := NewGraph()
			for _, n := range tc.nodes {
				g.AddNode(n)
			}
			for _, e := range tc.edges {
				if err := g.AddEdge(e[0], e[1]); err != nil {
					t.Fatalf("add edge %s->%s: %v", e[0], e[1], err)
				}
			}

			batches, err := g.TopologicalBatches()
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}

			assertBatchesEqual(t, tc.name, batches, tc.want)
			assertBatchDependencyOrder(t, tc.edges, batches)
			assertIntraBatchNoDirectEdge(t, tc.edges, batches)
		})
	}
}

func TestTopologicalBatches_CycleRejectedAtBuildTime(t *testing.T) {
	// T7-cycle: cycles are rejected when adding edges; TopologicalBatches never sees a cyclic graph.
	g := NewGraph()
	g.AddNode("A")
	g.AddNode("B")
	if err := g.AddEdge("A", "B"); err != nil {
		t.Fatal(err)
	}
	if err := g.AddEdge("B", "A"); err == nil {
		t.Fatal("expected cycle error on AddEdge")
	}
	batches, err := g.TopologicalBatches()
	if err != nil {
		t.Fatal(err)
	}
	if len(batches) != 2 || !sameStringSet(batches[0], []string{"A"}) || !sameStringSet(batches[1], []string{"B"}) {
		t.Fatalf("acyclic remainder expected [[A],[B]], got %v", batches)
	}
}

func TestUpgradeDAG_TopologicalBatches_WithResolverDeps(t *testing.T) {
	components := []struct {
		name string
		deps []string
	}{
		{name: "pre-upgrade-resources"},
		{name: "provider", deps: []string{"pre-upgrade-resources"}},
		{name: "etcd", deps: []string{"provider"}},
		{name: "coredns", deps: []string{"etcd"}},
		{name: "kube-proxy", deps: []string{"etcd"}},
	}

	dag := NewUpgradeDAG()
	for _, c := range components {
		if err := dag.AddNode(&ComponentNode{Name: c.name, Version: "v1.0.0"}); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range components {
		for _, dep := range c.deps {
			if err := dag.AddDependency(dep, c.name); err != nil {
				t.Fatal(err)
			}
		}
	}

	batches, err := dag.TopologicalBatches()
	if err != nil {
		t.Fatal(err)
	}
	want := [][]string{
		{"pre-upgrade-resources"},
		{"provider"},
		{"etcd"},
		{"coredns", "kube-proxy"},
	}
	assertBatchesEqual(t, "resolver-deps", batches, want)

	var edges [][2]string
	for _, c := range components {
		for _, dep := range c.deps {
			edges = append(edges, [2]string{dep, c.name})
		}
	}
	assertBatchDependencyOrder(t, edges, batches)
}

func assertBatchesEqual(t *testing.T, name string, got, want [][]string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: batch count got %d want %d\ngot=%v\nwant=%v", name, len(got), len(want), got, want)
	}
	for i := range want {
		if !sameStringSet(got[i], want[i]) {
			t.Fatalf("%s: batch[%d] got %v want %v (all=%v)", name, i, got[i], want[i], got)
		}
	}
}

// assertBatchDependencyOrder checks dep runs in an earlier batch than dependent for every edge.
func assertBatchDependencyOrder(t *testing.T, edges [][2]string, batches [][]string) {
	t.Helper()
	index := batchIndexByNode(batches)
	for _, e := range edges {
		dep, comp := e[0], e[1]
		di, okD := index[dep]
		ci, okC := index[comp]
		if !okD || !okC {
			t.Fatalf("edge %s->%s: node missing in batches (dep=%v comp=%v)", dep, comp, okD, okC)
		}
		if di >= ci {
			t.Fatalf("dependency violated %s->%s: batch(%s)=%d batch(%s)=%d batches=%v",
				dep, comp, dep, di, comp, ci, batches)
		}
	}
}

// assertIntraBatchNoDirectEdge ensures no edge connects two nodes in the same batch (parallel-safe layer).
func assertIntraBatchNoDirectEdge(t *testing.T, edges [][2]string, batches [][]string) {
	t.Helper()
	index := batchIndexByNode(batches)
	for _, e := range edges {
		dep, comp := e[0], e[1]
		if index[dep] == index[comp] {
			t.Fatalf("same-batch dependency %s->%s at batch %d: %v", dep, comp, index[dep], batches)
		}
	}
}

func batchIndexByNode(batches [][]string) map[string]int {
	out := make(map[string]int, 32)
	for i, batch := range batches {
		for _, name := range batch {
			out[name] = i
		}
	}
	return out
}

func sameStringSet(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	seen := make(map[string]struct{}, len(got))
	for _, s := range got {
		seen[s] = struct{}{}
	}
	for _, s := range want {
		if _, ok := seen[s]; !ok {
			return false
		}
	}
	return true
}
