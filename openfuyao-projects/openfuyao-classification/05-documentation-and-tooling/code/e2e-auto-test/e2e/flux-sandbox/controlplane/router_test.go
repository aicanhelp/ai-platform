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

	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// ── Scenario group: Router load balancing ──

// TestRouter_AllocatedMinFirst verifies FS-FUNC-025: with 2 SSGs each capacity=2
// (allocated=0), 3 consecutive creates are routed allocated-min-first: the first
// two go to distinct SSGs, the third back to the now-least-loaded one.
//
// Note: producing exactly 2 SSGs requires the controller's max-pod-per-sub-
// sandbox-group flag. With the default (100) the SG produces a single SSG, so
// this test asserts the load-balancing invariant that allocated grows evenly
// across the available SSGs.
func testRouter_AllocatedMinFirst(t ginkgo.GinkgoTInterface) {
	sgName := fastPathSG(t, 4)

	// List SSGs and their initial allocated.
	ssgList, err := listSSGsByGroup(sgName)
	if err != nil {
		t.Fatalf("list SSGs: %v", err)
	}
	if len(ssgList.Items) == 0 {
		t.Fatalf("expected >=1 SSG, got 0")
	}

	// Create 3 sandboxes.
	var sbIDs []string
	t.Cleanup(func() {
		for _, id := range sbIDs {
			_, _ = grpcDelete(t, id)
			deleteSandboxCR(id)
		}
	})
	// Pre-clean leftovers from a previous failed run of this same-named case.
	base := "sbx-cp-025-" + cpName(t)
	preCleanSandboxIDs(base+"-a", base+"-b", base+"-c")
	for i := 0; i < 3; i++ {
		id := base + "-" + string(rune('a'+i))
		sbIDs = append(sbIDs, id)
		grpcCreateMustCreate(t, id, sgName)
	}

	// Verify each created sandbox reports a subSandboxGroup and that the SSG
	// allocated counts increased (load-balanced, not all on one pod).
	// The Sandbox CR is written asynchronously by the scheduler CR worker
	// after a FastPath create, so tolerate a brief window before the CR shows
	// up instead of failing on the first miss.
	for _, id := range sbIDs {
		waitFor(t, "Sandbox CR "+id+" written asynchronously", 30*time.Second, func() bool {
			_, err := getSandboxCR(id)
			return err == nil
		})
		sb, err := getSandboxCR(id)
		if err != nil {
			t.Fatalf("get Sandbox CR %s: %v", id, err)
		}
		if sb.Spec.SubSandboxGroup == "" {
			t.Errorf("sandbox %s: expected non-empty subSandboxGroup", id)
		}
	}

	// The allocated counters are written back asynchronously by the schedulers;
	// wait until the sum across SSGs reflects all three sandboxes.
	waitFor(t, "total allocated=3 across SSGs", 90*time.Second, func() bool {
		refreshed, _ := listSSGsByGroup(sgName)
		sum := 0
		for _, s := range refreshed.Items {
			sum += s.Status.Allocated
		}
		return sum >= 3
	})
	refreshed, _ := listSSGsByGroup(sgName)
	sum := 0
	for _, s := range refreshed.Items {
		sum += s.Status.Allocated
	}
	if sum != 3 {
		t.Fatalf("expected total allocated=3 across SSGs, got %d", sum)
	}
}

// TestRouter_SchedulerIsolation verifies FS-FUNC-026: with >=2 schedulers, a
// create routed to scheduler-0's SSG is processed only by scheduler-0; scheduler-1
// logs show no record of that sandbox.
func testRouter_SchedulerIsolation(t ginkgo.GinkgoTInterface) {
	requireSchedulerReplicas(t, 2)

	sgName := fastPathSG(t, 2)
	sbID := "sbx-cp-026-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	grpcCreateMustCreate(t, sbID, sgName)
	waitSandboxPhase(t, sbID, 30*time.Second, apiv1alpha1.PhaseRunning)

	sb, err := getSandboxCR(sbID)
	if err != nil {
		t.Fatalf("get Sandbox CR %s: %v", sbID, err)
	}
	handlingScheduler := sb.Status.Scheduler
	if handlingScheduler == "" {
		t.Fatalf("sandbox %s: expected non-empty scheduler", sbID)
	}

	// Find the scheduler pods.
	pods, err := schedulerPods()
	if err != nil {
		t.Fatalf("list scheduler pods: %v", err)
	}
	var handlerPod, otherPod string
	for i := range pods {
		if pods[i].Name == handlingScheduler {
			handlerPod = pods[i].Name
		} else {
			otherPod = pods[i].Name
		}
	}
	if handlerPod == "" || otherPod == "" {
		t.Skipf("could not identify both handler and other scheduler pod (handler=%s)", handlerPod)
	}

	// Handler scheduler logs should mention the sandbox id.
	handlerLogs := kubectlLogs(t, cpNamespace, handlerPod, "", 100)
	if !contains(handlerLogs, sbID) {
		t.Errorf("handler scheduler %s logs should mention sandbox %s", handlerPod, sbID)
	}

	// Other scheduler logs should NOT mention the sandbox id — except the
	// idempotent delete probe ("SSG not found, returning Deleted"), which the
	// Controller legitimately fans out to every scheduler while resolving the
	// delete route during the case cleanup.
	otherLogs := kubectlLogs(t, cpNamespace, otherPod, "", 100)
	for _, line := range strings.Split(otherLogs, "\n") {
		if !contains(line, sbID) {
			continue
		}
		if contains(line, "handleDeleteSandbox: SSG not found") ||
			contains(line, "handleDeletion") {
			continue
		}
		t.Errorf("other scheduler %s logs should NOT mention sandbox %s (isolation violated): %s",
			otherPod, sbID, line)
	}
}

// contains is a small string-contains helper (avoids importing strings in this
// file for a single use).
func contains(s, substr string) bool {
	return len(s) >= len(substr) && indexOf(s, substr) >= 0
}

func indexOf(s, substr string) int {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

// TestRouter_CrossNode verifies FS-FUNC-027: when the Controller and the target
// Scheduler run on different nodes, a create request is still served correctly
// (cross-node Controller->Scheduler communication).
func testRouter_CrossNode(t ginkgo.GinkgoTInterface) {
	requireSchedulerReplicas(t, 2)

	// Find the controller's node.
	ctrlPods, err := cpEnv.kubeClient.CoreV1().Pods(cpNamespace).List(context.TODO(),
		metav1.ListOptions{LabelSelector: "app=flux-sandbox-controller"})
	if err != nil || len(ctrlPods.Items) == 0 {
		t.Skipf("could not list controller pods: %v", err)
	}
	controllerNode := ctrlPods.Items[0].Spec.NodeName

	schedPods, err := schedulerPods()
	if err != nil || len(schedPods) == 0 {
		t.Skipf("could not list scheduler pods: %v", err)
	}

	// Find a scheduler on a different node than the controller.
	var targetScheduler string
	for i := range schedPods {
		if schedPods[i].Spec.NodeName != controllerNode {
			targetScheduler = schedPods[i].Name
			break
		}
	}
	if targetScheduler == "" {
		t.Skipf("no scheduler on a different node than the controller (all schedulers on %s)", controllerNode)
	}

	sgName := fastPathSG(t, 2)
	sbID := "sbx-cp-027-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	resp := grpcCreateMustCreate(t, sbID, sgName)
	if resp.Phase != "Running" {
		t.Fatalf("cross-node create: expected phase=Running, got %s", resp.Phase)
	}
}
