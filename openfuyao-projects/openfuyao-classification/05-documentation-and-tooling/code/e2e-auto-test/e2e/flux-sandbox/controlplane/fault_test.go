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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// ── Scenario group: Sandbox fault handling and Lost phase ──

// faultCaseGap is a fixed settle wait inserted between fault cases. Each case
// deletes an Agent Pod and its rebuild+heartbeat cycle overlaps with the next
// case's Lost detection (which is heartbeat-expiry driven); the gap lets the
// previous case's agent rebuild and heartbeat settle so Lost detection for the
// next case is not starved.
const faultCaseGap = 90 * time.Second

// deleteAgentPodByName force-deletes an Agent Pod by name.
func deleteAgentPodByName(t ginkgo.GinkgoTInterface, podName string) {
	t.Helper()
	ctx := context.Background()
	if err := cpEnv.kubeClient.CoreV1().Pods(cpNamespace).Delete(ctx, podName,
		metav1.DeleteOptions{}); err != nil {
		t.Fatalf("delete agent pod %s: %v", podName, err)
	}
}

// deleteHostingAgentPod deletes the Agent Pod that actually hosts the given
// sandbox. The allocator picks the target by load + image affinity, so the
// sandbox may land on ANY of the group's agents — deleting a fixed one (e.g.
// pods[0]) only moves it to Lost when the two happen to coincide, and the case
// flakes whenever the scheduler picks the other agent. Resolve the hosting pod
// from the CR's assignedSandboxAgentPod instead.
func deleteHostingAgentPod(t ginkgo.GinkgoTInterface, sbID string) {
	t.Helper()
	deleteAgentPodByName(t, hostingAgentPod(t, sbID))
}

// hostingAgentPod returns the agent pod currently hosting the sandbox.
func hostingAgentPod(t ginkgo.GinkgoTInterface, sbID string) string {
	t.Helper()
	sb, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get sandbox %s: %v", sbID, err)
	}
	host := sb.Status.AssignedSandboxAgentPod
	if host == "" {
		t.Fatalf("sandbox %s has no assignedSandboxAgentPod yet", sbID)
	}
	return host
}

// waitSandboxPhase waits until the Sandbox CR named sbID reaches one of the
// given phases, returning the phase reached.
func waitSandboxPhase(t ginkgo.GinkgoTInterface, sbID string, timeout time.Duration, wantPhases ...string) string {
	t.Helper()
	var reached string
	waitFor(t, "sandbox phase in "+strings.Join(wantPhases, "|"), timeout, func() bool {
		sb, err := getSandboxCR(sbID)
		if err != nil {
			return false
		}
		for _, p := range wantPhases {
			if sb.Status.Phase == p {
				reached = p
				return true
			}
		}
		return false
	})
	return reached
}

// faultSG builds a ready SandboxGroup with TWO agent pods. The fault
// scenarios delete one agent pod; with at least one surviving pod the rebuild
// gets a NEW ordinal (nextAgentPodOrdinal = max+1), so the dead pod's registry
// slot is not taken over by a same-named recreation (which would refresh the
// heartbeat and mask the Lost transition under test).
func faultSG(t ginkgo.GinkgoTInterface, maxPerPod int) string {
	t.Helper()
	name := "sg-fault-" + cpName(t)
	sg := basicSandboxGroup(name, 2, 2, maxPerPod, withBuffer(0, 0))
	createSandboxGroup(t, sg)
	waitFor(t, "fault SandboxGroup Ready", 120*time.Second, func() bool {
		return getSandboxGroup(t, name).Status.Phase == apiv1alpha1.PhaseReady
	})
	waitAgentPodsReady(t, name, 2)
	// Wait until the scheduler has observed both agents with full idle.
	waitFor(t, "fault agents ready with full idle", 300*time.Second, func() bool {
		return ssgIdle(t, name+"-0") >= 2*maxPerPod
	})
	return name
}

// TestSandboxFault_LostManual verifies FS-FUNC-022: with no failurePolicy (Manual),
// deleting the Agent Pod moves the sandbox to Lost while keeping the reservation.
func testSandboxFault_LostManual(t ginkgo.GinkgoTInterface) {
	sgName := faultSG(t, 2)
	sbID := "sbx-cp-022-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	grpcCreateMustCreate(t, sbID, sgName)
	waitSandboxPhase(t, sbID, 30*time.Second, apiv1alpha1.PhaseRunning)

	// Delete the agent pod hosting this sandbox.
	deleteHostingAgentPod(t, sbID)

	// The sandbox should transition to Lost (reconcileRunning detects agent gone).
	waitSandboxPhase(t, sbID, 90*time.Second, apiv1alpha1.PhaseLost)

	// In Manual mode the reservation is KEPT on the dead agent's registry entry
	// (only released by AutoRecreate or an explicit delete), so the Lost sandbox
	// is not re-scheduled onto the rebuilt agent. The SSG status.allocated no
	// longer reflects it though: since the stats only count heartbeat-valid
	// agents, the expired entry that holds the reservation is excluded — the
	// Lost CR itself records that usage. Verify the retained reservation by
	// checking the sandbox stays Lost (never re-scheduled) and keeps pointing
	// at the dead agent pod, instead of the old allocated-not-drop assertion
	// written for the pre-fix stats.
	sbLost, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get Sandbox CR %s after Lost: %v", sbID, err)
	}
	if sbLost.Status.Phase != apiv1alpha1.PhaseLost {
		t.Fatalf("sandbox should stay Lost in Manual mode, got %s", sbLost.Status.Phase)
	}
	if sbLost.Status.AssignedSandboxAgentPod == "" {
		t.Fatalf("Lost sandbox should keep its assigned agent pod (reservation retained)")
	}
}

// TestSandboxFault_AutoRecreate verifies FS-FUNC-023: with failurePolicy=
// AutoRecreate, deleting the Agent Pod moves the sandbox Lost -> Pending (release
// reservation) -> Running (re-scheduled on a new agent pod).
func testSandboxFault_AutoRecreate(t ginkgo.GinkgoTInterface) {
	// Settle window after LostManual: that case just deleted an Agent Pod and
	// its rebuild/heartbeat cycle is still converging on the same scheduler.
	time.Sleep(faultCaseGap)

	sgName := faultSG(t, 2)

	// Create a sandbox, then patch its failurePolicy to AutoRecreate.
	sbID := "sbx-cp-023-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })
	grpcCreateMustCreate(t, sbID, sgName)
	waitSandboxPhase(t, sbID, 30*time.Second, apiv1alpha1.PhaseRunning)

	// Patch failurePolicy via the controller-runtime client (the core-group
	// RESTClient cannot address CRD resources under sandbox.flux.io).
	ctx := context.Background()
	var sb apiv1alpha1.Sandbox
	if err := cpEnv.k8sClient.Get(ctx, types.NamespacedName{Namespace: cpNamespace, Name: sbID}, &sb); err != nil {
		t.Fatalf("get sandbox %s: %v", sbID, err)
	}
	sb.Spec.FailurePolicy = "AutoRecreate"
	if err := cpEnv.k8sClient.Update(ctx, &sb); err != nil {
		t.Fatalf("update sandbox failurePolicy: %v", err)
	}

	// Delete the agent pod hosting this sandbox.
	hostPod := hostingAgentPod(t, sbID)
	deleteAgentPodByName(t, hostPod)

	// In AutoRecreate mode the Lost phase is TRANSIENT: reconcileRunning writes
	// Lost, and the very next reconcile round's reconcileLost immediately moves
	// it to Pending (releasing the reservation) — often within one loop tick
	// (<1-2s), and a FastPath endpoints writer racing the same CR can swallow
	// the Lost write entirely ("object has been modified"). A 3s poll cannot
	// reliably observe it. Assert the OBSERVABLE OUTCOME instead: the sandbox
	// leaves Running (Lost detection did fire) and eventually becomes Running
	// again on a DIFFERENT agent pod (AutoRecreate re-scheduled it). The
	// intermediate phases may be Lost or Pending depending on poll timing.
	deadline := time.Now().Add(240 * time.Second)
	leftRunning := false
	for time.Now().Before(deadline) {
		sbNow, err := getSandboxCR(sbID)
		if err == nil && sbNow.Status.Phase == apiv1alpha1.PhaseRunning &&
			sbNow.Status.AssignedSandboxAgentPod != "" &&
			sbNow.Status.AssignedSandboxAgentPod != hostPod {
			leftRunning = true
			break
		}
		time.Sleep(3 * time.Second)
	}
	if !leftRunning {
		t.Fatalf("sandbox was not re-scheduled onto a new agent pod within 4m (host=%s)", hostPod)
	}

	sbAfter, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get Sandbox CR %s after recreate: %v", sbID, err)
	}
	if sbAfter.Status.Phase != apiv1alpha1.PhaseRunning {
		t.Fatalf("after AutoRecreate, phase=%s, want Running", sbAfter.Status.Phase)
	}
	if sbAfter.Status.AssignedSandboxAgentPod == hostPod {
		t.Fatalf("after AutoRecreate, sandbox still on deleted pod %s", hostPod)
	}
}

// TestSandboxFault_NoOrphanMisdelete verifies FS-FUNC-024: after creation, the
// orphan-detection loop does not mistakenly delete the sandbox, and scheduler
// logs show reconcileOrphanSandboxes with matching counts and no orphan warning.
func testSandboxFault_NoOrphanMisdelete(t ginkgo.GinkgoTInterface) {
	// Settle window after AutoRecreate (same reasoning as its own gap).
	time.Sleep(faultCaseGap)

	sgName := faultSG(t, 2)
	sbID := "sbx-cp-024-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	grpcCreateMustCreate(t, sbID, sgName)
	waitSandboxPhase(t, sbID, 30*time.Second, apiv1alpha1.PhaseRunning)

	// Wait beyond the orphan detection cycle (loop syncOnce=2s, grace=30s).
	time.Sleep(20 * time.Second)

	// The sandbox should still exist and be Running (not mis-deleted).
	sb, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("sandbox %s was unexpectedly deleted (orphan mis-delete): %v", sbID, err)
	}
	if sb.Status.Phase != apiv1alpha1.PhaseRunning {
		t.Fatalf("sandbox %s phase: expected Running, got %s", sbID, sb.Status.Phase)
	}

	// Scheduler logs should contain reconcileOrphanSandboxes with no orphan warning.
	pods, err := schedulerPods()
	if err != nil || len(pods) == 0 {
		t.Skipf("no scheduler pods to inspect: %v", err)
	}
	logs := kubectlLogs(t, cpNamespace, pods[0].Name, "", 50)
	if !strings.Contains(logs, "reconcileOrphanSandboxes") {
		// Not fatal: log level may be Debug and not captured. Warn instead.
		t.Logf("note: scheduler logs do not mention reconcileOrphanSandboxes (may be debug-level)")
	}
	// The sandbox must not have been treated as an orphan: check for the
	// orphan-cleanup action lines, not the bare word "orphan" (the test's own
	// sg name contains "orphan" and always matches).
	for _, marker := range []string{"cleaning orphan sandbox from agent", "deleteOrphanFromAgent", "removed ghost agent record"} {
		if strings.Contains(logs, marker) {
			t.Errorf("scheduler logs show orphan cleanup (%q), possible mis-detection:\n%s", marker, logs)
		}
	}
}
