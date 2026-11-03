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

package hardware

import (
	"errors"
	"strings"
	"testing"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// ---- crossNodeIPUniqueness ----

func TestCrossNodeIPUniqueness_Unique(t *testing.T) {
	ctxs := []*NodeCheckContext{
		{Node: types.NodeInfo{Name: "node1"}, HccnIPs: map[string]string{"0": "192.168.1.1"}},
		{Node: types.NodeInfo{Name: "node2"}, HccnIPs: map[string]string{"0": "192.168.1.2"}},
	}
	r := crossNodeIPUniqueness(ctxs)
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
}

func TestCrossNodeIPUniqueness_Duplicate(t *testing.T) {
	ctxs := []*NodeCheckContext{
		{Node: types.NodeInfo{Name: "node1"}, HccnIPs: map[string]string{"0": "192.168.1.1"}},
		{Node: types.NodeInfo{Name: "node2"}, HccnIPs: map[string]string{"0": "192.168.1.1"}},
	}
	r := crossNodeIPUniqueness(ctxs)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

// ---- crossNodeTLSConsistency ----

func TestCrossNodeTLSConsistency_Consistent(t *testing.T) {
	ctxs := []*NodeCheckContext{
		{Node: types.NodeInfo{Name: "node1"}, TLSStates: map[string]string{"0": "enable"}},
		{Node: types.NodeInfo{Name: "node2"}, TLSStates: map[string]string{"0": "enable"}},
	}
	r := crossNodeTLSConsistency(ctxs)
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
}

func TestCrossNodeTLSConsistency_Inconsistent(t *testing.T) {
	ctxs := []*NodeCheckContext{
		{Node: types.NodeInfo{Name: "node1"}, TLSStates: map[string]string{"0": "enable"}},
		{Node: types.NodeInfo{Name: "node2"}, TLSStates: map[string]string{"0": "disable"}},
	}
	r := crossNodeTLSConsistency(ctxs)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCrossNodeTLSConsistency_NoCertVsDisable(t *testing.T) {
	ctxs := []*NodeCheckContext{
		{Node: types.NodeInfo{Name: "node1"}, TLSStates: map[string]string{"0": "disable"}},
		{Node: types.NodeInfo{Name: "node2"}, TLSStates: map[string]string{"0": "no_cert"}},
	}
	r := crossNodeTLSConsistency(ctxs)
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed (no_cert==disable), got %v: %s", r.Status, r.Message)
	}
}

// ---- checkH10 (H-10) ----

func TestCheckCrossNodes_NotEnoughNodes(t *testing.T) {
	ctxs := []*NodeCheckContext{
		{Node: types.NodeInfo{Name: "node1"}, HccnIPs: map[string]string{},
			TLSStates: map[string]string{}},
	}
	results := checkH10(ctxs, true)
	if len(results) != 1 || results[0].Status != types.StatusSkipped {
		t.Errorf("expected single skipped result, got %v", results)
	}
}

func TestCheckCrossNodes_TwoNodes_IPConflict(t *testing.T) {
	// Two nodes with same IP -> should fail at step 1
	ctxs := []*NodeCheckContext{
		{Node: types.NodeInfo{Name: "node1"}, HccnIPs: map[string]string{"0": "192.168.1.1"},
			TLSStates: map[string]string{"0": "enable"}},
		{Node: types.NodeInfo{Name: "node2"}, HccnIPs: map[string]string{"0": "192.168.1.1"},
			TLSStates: map[string]string{"0": "enable"}},
	}
	results := checkH10(ctxs, true)
	if len(results) != 1 {
		t.Errorf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != types.StatusFailed {
		t.Errorf("expected step1 failed, got %v", results[0].Status)
	}
}

// ---- crossNodePingOne via CheckCrossNodes integration ----

func TestCheckCrossNodes_TwoNodes_Pass(t *testing.T) {
	origFactory := newSSHRunner
	defer func() { newSSHRunner = origFactory }()

	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		m := &mockRunner{
			nodeName: node.Name,
			responses: map[string]mockResponse{
				"hccn_tool": {stdout: "0.00% packet loss"},
			},
		}
		return m, func() {}, nil
	}

	ctxs := []*NodeCheckContext{
		{
			Node:      types.NodeInfo{Name: "node1"},
			PhyIDs:    []int{0},
			HccnIPs:   map[string]string{"0": "192.168.1.1"},
			TLSStates: map[string]string{"0": "enable"},
		},
		{
			Node:      types.NodeInfo{Name: "node2"},
			PhyIDs:    []int{0},
			HccnIPs:   map[string]string{"0": "192.168.1.2"},
			TLSStates: map[string]string{"0": "enable"},
		},
	}
	results := checkH10(ctxs, true)
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}
	for _, r := range results {
		if r.Status == types.StatusFailed {
			t.Errorf("unexpected failure: %s - %s", r.ID, r.Message)
		}
	}
}

func TestCheckCrossNodes_SSHFail(t *testing.T) {
	origFactory := newSSHRunner
	defer func() { newSSHRunner = origFactory }()

	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		return nil, func() {}, errors.New("ssh refused")
	}

	ctxs := []*NodeCheckContext{
		{
			Node:      types.NodeInfo{Name: "node1"},
			PhyIDs:    []int{0},
			HccnIPs:   map[string]string{"0": "192.168.1.1"},
			TLSStates: map[string]string{"0": "enable"},
		},
		{
			Node:      types.NodeInfo{Name: "node2"},
			PhyIDs:    []int{0},
			HccnIPs:   map[string]string{"0": "192.168.1.2"},
			TLSStates: map[string]string{"0": "enable"},
		},
	}
	results := checkH10(ctxs, true)
	// step 3 connectivity should pass vacuously (no failed pings collected since SSH failed)
	// but steps 1 and 2 should still run
	if len(results) != 3 {
		t.Errorf("expected 3 results, got %d", len(results))
	}
}

// TestCheckH10_ConnectivityDisabled tests that Step 3 is skipped when enableConnectivityCheck=false
func TestCheckH10_ConnectivityDisabled(t *testing.T) {
	ctxs := []*NodeCheckContext{
		{
			Node:      types.NodeInfo{Name: "node1"},
			PhyIDs:    []int{0},
			HccnIPs:   map[string]string{"0": "192.168.1.1"},
			TLSStates: map[string]string{"0": "enable"},
		},
		{
			Node:      types.NodeInfo{Name: "node2"},
			PhyIDs:    []int{0},
			HccnIPs:   map[string]string{"0": "192.168.1.2"},
			TLSStates: map[string]string{"0": "enable"},
		},
	}
	results := checkH10(ctxs, false)
	if len(results) != 3 {
		t.Errorf("expected 3 results (step 1, 2, 3), got %d", len(results))
	}
	// Step 1 and 2 should pass
	if results[0].Status != types.StatusPassed {
		t.Errorf("expected Step 1 passed, got %v", results[0].Status)
	}
	if results[1].Status != types.StatusPassed {
		t.Errorf("expected Step 2 passed, got %v", results[1].Status)
	}
	// Step 3 should be skipped
	if results[2].Status != types.StatusSkipped {
		t.Errorf("expected Step 3 skipped, got %v", results[2].Status)
	}
	if !strings.Contains(results[2].Message, "skipped") {
		t.Errorf("expected skip message, got: %s", results[2].Message)
	}
}
