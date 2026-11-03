/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package all

import (
	"testing"

	"k8s.io/client-go/kubernetes/fake"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// ---------------------------------------------------------------------------
// filterPassedNodes
// ---------------------------------------------------------------------------

func TestFilterPassedNodes_AllPassed(t *testing.T) {
	nodes := []types.NodeInfo{
		{Name: "node-1", IP: "10.0.0.1"},
		{Name: "node-2", IP: "10.0.0.2"},
	}
	hwResult := &types.HardwareResult{
		Nodes: []types.NodeResult{
			{Name: "node-1", Status: types.StatusPassed},
			{Name: "node-2", Status: types.StatusPassed},
		},
	}
	passedNodes, passedNames := filterPassedNodes(nodes, hwResult)
	if len(passedNodes) != 2 {
		t.Errorf("expected 2 passed nodes, got %d", len(passedNodes))
	}
	if len(passedNames) != 2 {
		t.Errorf("expected 2 passed names, got %d", len(passedNames))
	}
	if passedNames[0] != "node-1" || passedNames[1] != "node-2" {
		t.Errorf("unexpected names: %v", passedNames)
	}
}

func TestFilterPassedNodes_NonePassed(t *testing.T) {
	nodes := []types.NodeInfo{
		{Name: "node-1", IP: "10.0.0.1"},
	}
	hwResult := &types.HardwareResult{
		Nodes: []types.NodeResult{
			{Name: "node-1", Status: types.StatusFailed},
		},
	}
	passedNodes, passedNames := filterPassedNodes(nodes, hwResult)
	if len(passedNodes) != 0 {
		t.Errorf("expected 0 passed nodes, got %d", len(passedNodes))
	}
	if len(passedNames) != 0 {
		t.Errorf("expected 0 passed names, got %d", len(passedNames))
	}
}

func TestFilterPassedNodes_PartialPassed(t *testing.T) {
	nodes := []types.NodeInfo{
		{Name: "node-1", IP: "10.0.0.1"},
		{Name: "node-2", IP: "10.0.0.2"},
		{Name: "node-3", IP: "10.0.0.3"},
	}
	hwResult := &types.HardwareResult{
		Nodes: []types.NodeResult{
			{Name: "node-1", Status: types.StatusPassed},
			{Name: "node-2", Status: types.StatusFailed},
			{Name: "node-3", Status: types.StatusPassed},
		},
	}
	passedNodes, passedNames := filterPassedNodes(nodes, hwResult)
	if len(passedNodes) != 2 {
		t.Errorf("expected 2 passed nodes, got %d", len(passedNodes))
	}
	if len(passedNames) != 2 {
		t.Errorf("expected 2 passed names, got %d", len(passedNames))
	}
	// verify correct nodes were selected
	nameSet := map[string]bool{}
	for _, n := range passedNames {
		nameSet[n] = true
	}
	if !nameSet["node-1"] {
		t.Error("expected node-1 in passed names")
	}
	if !nameSet["node-3"] {
		t.Error("expected node-3 in passed names")
	}
	if nameSet["node-2"] {
		t.Error("expected node-2 NOT in passed names")
	}
}

func TestFilterPassedNodes_EmptyNodes(t *testing.T) {
	hwResult := &types.HardwareResult{
		Nodes: []types.NodeResult{},
	}
	passedNodes, passedNames := filterPassedNodes([]types.NodeInfo{}, hwResult)
	if len(passedNodes) != 0 || len(passedNames) != 0 {
		t.Errorf("expected empty results for empty input, got nodes=%v names=%v", passedNodes, passedNames)
	}
}

func TestFilterPassedNodes_NodeResultNameMismatch(t *testing.T) {
	// hwResult contains a node name that doesn't exist in the nodes slice
	nodes := []types.NodeInfo{
		{Name: "node-1", IP: "10.0.0.1"},
	}
	hwResult := &types.HardwareResult{
		Nodes: []types.NodeResult{
			{Name: "node-1", Status: types.StatusPassed},
			{Name: "ghost-node", Status: types.StatusPassed}, // not in nodes slice
		},
	}
	passedNodes, passedNames := filterPassedNodes(nodes, hwResult)
	// ghost-node is passed in hwResult but not in nodes slice, so only node-1 returned
	if len(passedNodes) != 1 {
		t.Errorf("expected 1 passed node, got %d", len(passedNodes))
	}
	if len(passedNames) != 1 || passedNames[0] != "node-1" {
		t.Errorf("expected [node-1], got %v", passedNames)
	}
}

func TestFilterPassedNodes_NodeInfoPreserved(t *testing.T) {
	nodes := []types.NodeInfo{
		{Name: "node-1", IP: "192.168.1.100", Port: 2222, User: "admin", Password: "secret"},
	}
	hwResult := &types.HardwareResult{
		Nodes: []types.NodeResult{
			{Name: "node-1", Status: types.StatusPassed},
		},
	}
	passedNodes, _ := filterPassedNodes(nodes, hwResult)
	if len(passedNodes) != 1 {
		t.Fatalf("expected 1 passed node")
	}
	n := passedNodes[0]
	if n.IP != "192.168.1.100" || n.Port != 2222 || n.User != "admin" || n.Password != "secret" {
		t.Errorf("node info not preserved: %+v", n)
	}
}

// ---------------------------------------------------------------------------
// CheckAll
// ---------------------------------------------------------------------------

// TestCheckAll_HardwareTerminated verifies that when all hardware checks fail
// (SSH will fail for fake nodes), CheckAll sets HardwareTerminated=true and
// returns early without running K8s or ConfigEnv checks.
func TestCheckAll_HardwareTerminated(t *testing.T) {
	// Use nodes that will definitely fail SSH (unreachable IP, invalid port)
	nodes := []types.NodeInfo{
		{Name: "fake-node", IP: "192.0.2.1", Port: 1, User: "root", Password: "pass"},
	}
	values := map[string]interface{}{}
	k8sClient := executor.NewK8sClientForTest(fake.NewSimpleClientset())

	report := CheckAll(nodes, values, k8sClient, "", true)

	if report == nil {
		t.Fatal("expected non-nil report")
	}
	if report.Hardware == nil {
		t.Fatal("expected Hardware result to be set")
	}
	if !report.HardwareTerminated {
		t.Error("expected HardwareTerminated=true when no nodes pass hardware check")
	}
	if report.K8s != nil {
		t.Error("expected K8s to be nil when hardware terminated early")
	}
	if report.ConfigEnv != nil {
		t.Error("expected ConfigEnv to be nil when hardware terminated early")
	}
}

// TestCheckAll_EmptyNodes verifies behavior when no nodes are provided.
func TestCheckAll_EmptyNodes(t *testing.T) {
	nodes := []types.NodeInfo{}
	values := map[string]interface{}{}
	k8sClient := executor.NewK8sClientForTest(fake.NewSimpleClientset())

	report := CheckAll(nodes, values, k8sClient, "", true)

	if report == nil {
		t.Fatal("expected non-nil report")
	}
	if report.Hardware == nil {
		t.Fatal("expected Hardware result to be set")
	}
	// No nodes → no nodes passed hardware → terminated
	if !report.HardwareTerminated {
		t.Error("expected HardwareTerminated=true for empty node list")
	}
}

// TestCheckAll_ReportStructure verifies the report structure is always initialized.
func TestCheckAll_ReportStructure(t *testing.T) {
	nodes := []types.NodeInfo{
		{Name: "fake-node", IP: "192.0.2.1", Port: 1, User: "root", Password: "pass"},
	}
	k8sClient := executor.NewK8sClientForTest(fake.NewSimpleClientset())

	report := CheckAll(nodes, map[string]interface{}{}, k8sClient, "", true)

	if report.Hardware == nil {
		t.Error("Hardware field should never be nil")
	}
}
