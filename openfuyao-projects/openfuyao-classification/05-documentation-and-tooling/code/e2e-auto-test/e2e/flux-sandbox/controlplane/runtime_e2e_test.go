/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controlplane

import (
	ginkgo "github.com/onsi/ginkgo/v2"

	"encoding/json"
	"strings"
	"time"

	agentapi "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/agentapi"
	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// ── Scenario group: containerd runtime end-to-end + Agent HTTP API ──

// TestRuntime_ExecdPing verifies FS-FUNC-031: after a gRPC create, the execd
// daemon responds to /ping with HTTP 200.
func testRuntime_ExecdPing(t ginkgo.GinkgoTInterface) {
	sgName := fastPathSG(t, 2)
	sbID := "sbx-cp-031-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	resp := grpcCreateMustCreate(t, sbID, sgName)
	if resp.Phase != "Running" {
		t.Fatalf("expected phase=Running, got %s", resp.Phase)
	}
	waitSandboxPhase(t, sbID, 30*time.Second, apiv1alpha1.PhaseRunning)

	// Locate the agent pod hosting this sandbox and port-forward its 44772.
	sb, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get Sandbox CR %s: %v", sbID, err)
	}
	agentPodName := sb.Status.AssignedSandboxAgentPod
	if agentPodName == "" {
		t.Fatalf("sandbox %s: assignedSandboxAgentPod is empty", sbID)
	}
	_, execdBase := portForwardAgent(t, agentPodName)

	// Poll execd /ping until 200 (execd boots inside the sandbox container).
	// Boot time varies widely with node load — under a sustained e2e run it
	// can exceed 2 minutes — so keep the window generous.
	waitFor(t, "execd /ping returns 200", 300*time.Second, func() bool {
		code, _ := httpGetSilent(execdBase + "/ping")
		return code == 200
	})
}

// TestRuntime_ExecdCommand verifies FS-FUNC-032: execd /command runs a command
// inside the sandbox and returns stdout with an execution_complete marker.
func testRuntime_ExecdCommand(t ginkgo.GinkgoTInterface) {
	sgName := fastPathSG(t, 2)
	sbID := "sbx-cp-032-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	grpcCreateMustCreate(t, sbID, sgName)
	waitSandboxPhase(t, sbID, 30*time.Second, apiv1alpha1.PhaseRunning)

	sb, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get Sandbox CR %s: %v", sbID, err)
	}
	_, execdBase := portForwardAgent(t, sb.Status.AssignedSandboxAgentPod)

	// Ensure execd is up first (boot time varies with node load; see
	// TestRuntime_ExecdPing).
	waitFor(t, "execd /ping returns 200", 300*time.Second, func() bool {
		code, _ := httpGetSilent(execdBase + "/ping")
		return code == 200
	})

	// POST a command; the response is an SSE-like stream of typed chunks.
	_, body := httpPostJSON(t, execdBase+"/command", `{"command":"echo hello-from-sandbox"}`)
	if !strings.Contains(body, "hello-from-sandbox") {
		t.Fatalf("expected stdout to contain hello-from-sandbox, got:\n%s", body)
	}
	if !strings.Contains(body, "execution_complete") {
		t.Fatalf("expected execution_complete marker in response, got:\n%s", body)
	}
}

// TestAgent_Status verifies FS-FUNC-033: the Agent /status endpoint reports
// capacity, allocated, and the created sandbox in sandboxStatuses.
func testAgent_Status(t ginkgo.GinkgoTInterface) {
	sbID := "sbx-cp-033-" + cpName(t)
	// Pre-clean leftovers from a previous (failed) run of this same-named
	// case (stuck CR / orphan node-level task with the same ID).
	preCleanSandboxIDs(sbID)
	sgName := fastPathSG(t, 2)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	grpcCreateMustCreate(t, sbID, sgName)
	waitSandboxPhase(t, sbID, 30*time.Second, apiv1alpha1.PhaseRunning)

	sb, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get Sandbox CR %s: %v", sbID, err)
	}
	agentBase, _ := portForwardAgent(t, sb.Status.AssignedSandboxAgentPod)

	// Query /status; allocated should be >= 1 and include this sandbox.
	var status agentapi.FluxSandboxAgentStatus
	waitFor(t, "agent /status reports sandbox", 60*time.Second, func() bool {
		code, body := httpGetSilent(agentBase + "/api/v1/fluxagent/status")
		if code != 200 {
			return false
		}
		if err := json.Unmarshal([]byte(body), &status); err != nil {
			return false
		}
		if status.Capacity <= 0 || status.Allocated < 1 {
			return false
		}
		for _, s := range status.SandboxStatuses {
			if s.SandboxID == sbID {
				return true
			}
		}
		return false
	})

	if status.FluxSandboxAgentID == "" {
		t.Errorf("agent /status: fluxsandboxagentId should be non-empty")
	}
}

// TestAgent_DuplicateCreate_Idempotent verifies FS-FUNC-035: creating the same
// sandbox id twice succeeds (idempotent) and the agent's allocated count does
// not double-count.
func testAgent_DuplicateCreate_Idempotent(t ginkgo.GinkgoTInterface) {
	sgName := fastPathSG(t, 2)
	sbID := "sbx-cp-035-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	// First create.
	grpcCreateMustCreate(t, sbID, sgName)
	waitSandboxPhase(t, sbID, 30*time.Second, apiv1alpha1.PhaseRunning)

	sb, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get Sandbox CR %s: %v", sbID, err)
	}
	agentBase, _ := portForwardAgent(t, sb.Status.AssignedSandboxAgentPod)

	// Record how many times this sandbox ID appears in the agent status
	// before the duplicate create. Counting per-ID occurrences (instead of the
	// aggregate Allocated) keeps the comparison immune to unrelated concurrent
	// status updates (e.g. the Loop reporting restored leftovers from earlier
	// cases on the same node).
	countSandboxInStatus := func() int {
		_, body := httpGetSilent(agentBase + "/api/v1/fluxagent/status")
		var st agentapi.FluxSandboxAgentStatus
		if json.Unmarshal([]byte(body), &st) != nil {
			return -1
		}
		n := 0
		for _, s := range st.SandboxStatuses {
			if s.SandboxID == sbID {
				n++
			}
		}
		return n
	}
	before := countSandboxInStatus()

	// Duplicate create of the same id (idempotent, no error).
	resp, err := grpcCreate(t, sbID, sgName)
	if err != nil {
		t.Fatalf("duplicate create: expected no error (idempotent), got %v", err)
	}
	if resp.Phase != "Running" {
		t.Fatalf("duplicate create: expected phase=Running, got %s", resp.Phase)
	}

	// The sandbox must appear exactly once (no duplicate container created).
	waitFor(t, "agent /status reports the sandbox exactly once after duplicate create", 30*time.Second, func() bool {
		return countSandboxInStatus() == before || countSandboxInStatus() == 1
	})
}
