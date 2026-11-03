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

package controlplane

import (
	"context"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"

	"sigs.k8s.io/controller-runtime/pkg/client"

	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// ── Scenario group: FastPath gRPC sandbox lifecycle ──

// fastPathSG is a shared helper that builds a ready SandboxGroup with a single
// agent pod and capacity=maxPerPod, then returns its name. The caller's
// t.Cleanup will delete the SandboxGroup (via createSandboxGroup).
func fastPathSG(t ginkgo.GinkgoTInterface, maxPerPod int) string {
	t.Helper()
	name := "sg-fp-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, maxPerPod, withBuffer(0, 0))
	createSandboxGroup(t, sg)
	waitFor(t, "FastPath SandboxGroup Ready", 120*time.Second, func() bool {
		return getSandboxGroup(t, name).Status.Phase == apiv1alpha1.PhaseReady
	})
	waitAgentPodsReady(t, name, 1)
	// Wait until the scheduler has observed the agent AND the full capacity
	// is idle. A Ready pod only means containers are up; the scheduler still
	// needs one poll cycle to register the agent. Waiting for the full idle
	// count also guards against the cross-case cleanup race: a same-named sg
	// from an earlier case leaves stale sandbox containers on the node, and a
	// freshly rebuilt agent restores them (ListSandbox), consuming capacity
	// before this case even starts.
	waitFor(t, "FastPath agent ready with full idle", 300*time.Second, func() bool {
		return ssgIdle(t, name+"-0") >= maxPerPod
	})
	return name
}

// ssgIdle returns status.idle of the given SubSandboxGroup (0 when absent).
func ssgIdle(t ginkgo.GinkgoTInterface, ssgName string) int {
	t.Helper()
	var ssg apiv1alpha1.SubSandboxGroup
	if err := cpEnv.k8sClient.Get(context.Background(),
		client.ObjectKey{Namespace: cpNamespace, Name: ssgName}, &ssg); err != nil {
		return 0
	}
	return ssg.Status.Idle
}

// TestFastPath_Create verifies FS-FUNC-015: a gRPC CreateSandbox returns
// phase=Running and a non-empty agent pod endpoint.
func testFastPath_Create(t ginkgo.GinkgoTInterface) {
	sgName := fastPathSG(t, 2)
	sbID := "sbx-cp-015-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	resp := grpcCreateMustCreate(t, sbID, sgName)
	if resp.Phase != "Running" {
		t.Fatalf("expected phase=Running, got %s", resp.Phase)
	}
	if resp.SandboxId != sbID {
		t.Fatalf("expected sandbox_id=%s, got %s", sbID, resp.SandboxId)
	}
	if resp.AgentPodEndpoint == "" {
		t.Fatalf("expected non-empty agent_pod_endpoint, got %+v", resp)
	}
}

// TestFastPath_CRAsyncWrite verifies FS-FUNC-016: after a FastPath create, the
// Sandbox CR is asynchronously written with Running phase and the expected spec
// and status fields.
func testFastPath_CRAsyncWrite(t ginkgo.GinkgoTInterface) {
	sgName := fastPathSG(t, 2)
	sbID := "sbx-cp-016-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	grpcCreateMustCreate(t, sbID, sgName)

	// Wait for the CR worker to write the Sandbox CR (async). The CR creation
	// and its status writeback are two separate async steps: the phase reaches
	// Running only after the status worker observes the running runtime, so
	// wait for the phase rather than mere CR existence.
	waitSandboxPhase(t, sbID, 60*time.Second, apiv1alpha1.PhaseRunning)

	sb, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get Sandbox CR %s: %v", sbID, err)
	}
	if sb.Status.Phase != apiv1alpha1.PhaseRunning {
		t.Fatalf("expected Sandbox CR phase=Running, got %s", sb.Status.Phase)
	}
	if sb.Spec.SandboxGroup != sgName {
		t.Errorf("spec.sandboxGroup: expected %s, got %s", sgName, sb.Spec.SandboxGroup)
	}
	if sb.Spec.Image == "" {
		t.Errorf("spec.image should be non-empty")
	}
	if sb.Status.SandboxID == "" {
		t.Errorf("status.sandboxID should be non-empty")
	}
	if sb.Status.AssignedSandboxAgentPod == "" {
		t.Errorf("status.assignedSandboxAgentPod should be non-empty")
	}
	if sb.Status.SandboxAgentPodEndpoint == "" {
		t.Errorf("status.sandboxAgentPodEndpoint should be non-empty")
	}
}

// TestFastPath_Delete_Idempotent verifies FS-FUNC-018: deleting a sandbox
// returns deleted=true, and a repeated delete is idempotent (deleted=true, no
// NotFound error). The Sandbox CR is gone after deletion.
func testFastPath_Delete_Idempotent(t ginkgo.GinkgoTInterface) {
	sgName := fastPathSG(t, 2)
	sbID := "sbx-cp-018-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	grpcCreateMustCreate(t, sbID, sgName)

	// First delete.
	resp1, err := grpcDelete(t, sbID)
	if err != nil {
		t.Fatalf("first delete: expected no error, got %v", err)
	}
	if !resp1.Deleted {
		t.Fatalf("first delete: expected deleted=true, got %+v", resp1)
	}

	// Repeated delete (idempotent).
	resp2, err := grpcDelete(t, sbID)
	if err != nil {
		t.Fatalf("repeated delete: expected no error (idempotent), got %v", err)
	}
	if !resp2.Deleted {
		t.Fatalf("repeated delete: expected deleted=true (idempotent), got %+v", resp2)
	}

	// Sandbox CR should be gone (eventually, after async cleanup).
	waitFor(t, "Sandbox CR deleted", 30*time.Second, func() bool {
		_, err := getSandboxCR(sbID)
		return err != nil // not found
	})
}

// TestFastPath_CapacityFull verifies FS-FUNC-020: once capacity is full, a new
// CreateSandbox is rejected (the scheduler returns 500 / capacity exceeded).
func testFastPath_CapacityFull(t ginkgo.GinkgoTInterface) {
	const maxPerPod = 2
	// Pre-clean leftovers from a previous (failed) run of this same-named
	// case: a stuck Sandbox CR or node-level task with the same ID makes the
	// fill-create below fail intermittently.
	base := "sbx-cp-020-" + cpName(t)
	preCleanSandboxIDs(base+"-a", base+"-b", base+"-c", base+"-overflow")
	sgName := fastPathSG(t, maxPerPod)

	// Fill capacity: create maxPerPod sandboxes.
	var sbIDs []string
	t.Cleanup(func() {
		for _, id := range sbIDs {
			_, _ = grpcDelete(t, id)
			deleteSandboxCR(id)
		}
	})
	for i := 0; i < maxPerPod; i++ {
		id := "sbx-cp-020-" + cpName(t) + "-" + string(rune('a'+i))
		sbIDs = append(sbIDs, id)
		grpcCreateMustCreate(t, id, sgName)
	}

	// The (maxPerPod+1)-th create should fail.
	overflowID := "sbx-cp-020-" + cpName(t) + "-overflow"
	t.Cleanup(func() { deleteSandboxCR(overflowID) })
	_, err := grpcCreate(t, overflowID, sgName)
	if err == nil {
		t.Fatalf("expected CreateSandbox to fail at full capacity, but got nil error")
	}
	// The rejection surfaces as either a 500 (capacity exceeded) or a 503
	// (no available agent) from the scheduler — both mean capacity is full.
	if e := err.Error(); !strings.Contains(e, "500") && !strings.Contains(e, "503") &&
		!strings.Contains(strings.ToLower(e), "capacity") {
		t.Errorf("expected error to mention capacity/500/503, got: %s", e)
	}
}
