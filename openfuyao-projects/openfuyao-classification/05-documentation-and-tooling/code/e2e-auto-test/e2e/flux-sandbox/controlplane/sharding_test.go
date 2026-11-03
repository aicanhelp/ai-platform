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
	"context"
	"strconv"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// ── Scenario group: Large-scale sharding ──

// TestSharding_LargeScale verifies FS-FUNC-039: with max-pod-per-sub-sandbox-
// group=2 (controller flag) and agentPodMax=8, the SandboxGroup splits into 4
// SSGs and 8 Agent Pods, and 4 sandboxes distribute across 4 different SSGs with
// no duplicate-volume errors in scheduler logs.
//
// This case is gated behind E2E_BIG_SHARDING because it provisions 8 agent pods
// and requires the controller be started with --max-pod-per-sub-sandbox-group=2.
func testSharding_LargeScale(t ginkgo.GinkgoTInterface) {
	requireBigSharding(t)

	name := "sg-cp-039-" + cpName(t)
	// agentPodMax=8; with max-pod-per-sub-sandbox-group=2 the controller should
	// produce ceil(8/2)=4 SSGs.
	sg := basicSandboxGroup(name, 8, 8, 2, withBuffer(0, 0))
	createSandboxGroup(t, sg)

	waitFor(t, "large-scale SandboxGroup Ready", 300*time.Second, func() bool {
		return getSandboxGroup(t, name).Status.Phase == apiv1alpha1.PhaseReady
	})
	waitFor(t, "8 agent pods ready", 300*time.Second, func() bool {
		pods, err := listAgentPods(name)
		if err != nil {
			return false
		}
		ready := 0
		for i := range pods {
			if isAgentPodReady(&pods[i]) {
				ready++
			}
		}
		return ready >= 8
	})

	// The SG flips Ready once the agent Pods are Ready, but the agents only
	// become allocatable after registering with their scheduler; creating in
	// that window surfaces a 503 "no available agent". Wait until every SSG
	// reports idle capacity (the same precondition fastPathSG/faultSG wait
	// for) before creating sandboxes.
	waitFor(t, "all 4 SSGs report idle agents", 120*time.Second, func() bool {
		for i := 0; i < 4; i++ {
			if ssgIdle(t, name+"-"+strconv.Itoa(i)) < 1 {
				return false
			}
		}
		return true
	})

	cur := getSandboxGroup(t, name)
	if cur.Status.SubSandboxGroupsCount != 4 {
		t.Fatalf("expected 4 SSGs, got %d", cur.Status.SubSandboxGroupsCount)
	}
	if cur.Status.TotalAgentPods != 8 {
		t.Fatalf("expected 8 agent pods, got %d", cur.Status.TotalAgentPods)
	}
	if cur.Status.TotalCapacity != 16 { // 8 pods * 2 per pod
		t.Fatalf("expected TotalCapacity=16, got %d", cur.Status.TotalCapacity)
	}

	// Create 4 sandboxes and verify they spread across 4 different SSGs.
	var sbIDs []string
	t.Cleanup(func() {
		for _, id := range sbIDs {
			_, _ = grpcDelete(t, id)
			deleteSandboxCR(id)
		}
	})
	base := "sbx-cp-039-" + cpName(t)
	preCleanSandboxIDs(base+"-a", base+"-b", base+"-c", base+"-d")
	for i := 0; i < 4; i++ {
		id := base + "-" + string(rune('a'+i))
		sbIDs = append(sbIDs, id)
		grpcCreateMustCreate(t, id, name)
	}

	// Collect the subSandboxGroup of each created sandbox.
	ssgSet := map[string]bool{}
	for _, id := range sbIDs {
		sb, err := getSandboxCR(id)
		if err != nil {
			t.Fatalf("get Sandbox CR %s: %v", id, err)
		}
		if sb.Spec.SubSandboxGroup == "" {
			t.Errorf("sandbox %s: empty subSandboxGroup", id)
		} else {
			ssgSet[sb.Spec.SubSandboxGroup] = true
		}
	}
	if len(ssgSet) != 4 {
		t.Fatalf("expected 4 sandboxes across 4 distinct SSGs, got %d (%v)", len(ssgSet), ssgSet)
	}

	// Scheduler logs should contain no "must be unique" (duplicate volume) errors.
	pods, err := schedulerPods()
	if err == nil {
		for i := range pods {
			logs := kubectlLogs(t, cpNamespace, pods[i].Name, "", 100)
			if strings.Contains(logs, "must be unique") {
				t.Errorf("scheduler %s logs contain 'must be unique' (duplicate volume): %s",
					pods[i].Name, logs)
			}
		}
	}
}

// keep context imported (used for scale-down cleanup scenarios in the future).
var _ = context.Background
var _ = metav1.ListOptions{}
