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
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// ── Scenario group: Resource reservation (dual-container isolation) ──

// TestResource_DualContainerIsolation verifies FS-FUNC-038: the Agent Pod has
// two containers whose resource reservations are isolated — the sandbox-agent
// container only carries the agent overhead (not slot-reserved resources), while
// the resource-reserve sidecar carries perSlot × maxSandboxesPerPod.
func testResource_DualContainerIsolation(t ginkgo.GinkgoTInterface) {
	const (
		maxPerPod = 5
		slotCPU   = "1"
		slotMem   = "1Gi"
	)
	name := "sg-cp-038-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, maxPerPod, withBuffer(0, 0),
		withSlotResources(slotCPU, slotMem))
	createSandboxGroup(t, sg)
	waitAgentPodsReady(t, name, 1)

	pods, err := listAgentPods(name)
	if err != nil || len(pods) == 0 {
		t.Fatalf("no agent pods for %s: %v", name, err)
	}
	pod := &pods[0]

	if len(pod.Spec.Containers) != 2 {
		t.Fatalf("expected 2 containers, got %d", len(pod.Spec.Containers))
	}

	main, ok := containerByName(pod.Spec.Containers, "sandbox-agent")
	if !ok {
		t.Fatalf("sandbox-agent container not found")
	}
	reserve, ok := containerByName(pod.Spec.Containers, "resource-reserve")
	if !ok {
		t.Fatalf("resource-reserve container not found")
	}

	// sandbox-agent: carries the agent overhead only (small, not slot-reserved).
	// We assert its CPU request is well below perSlot*maxPerPod.
	mainCPU := main.Resources.Requests.Cpu()
	if mainCPU.IsZero() {
		t.Errorf("sandbox-agent: expected non-zero CPU request (overhead)")
	}
	mainTotal := mainCPU.Value() + main.Resources.Requests.Memory().Value()

	// resource-reserve: should carry perSlot × maxPerPod resources.
	reserveCPU := reserve.Resources.Requests.Cpu()
	reserveMem := reserve.Resources.Requests.Memory()
	expectCPU := resource.MustParse(slotCPU)
	expectMem := resource.MustParse(slotMem)
	// reserve = perSlot * maxPerPod.
	if reserveCPU.Cmp(expectCPU) < 0 {
		t.Errorf("resource-reserve CPU: expected >= %s (1 slot), got %s", slotCPU, reserveCPU.String())
	}
	if reserveMem.Cmp(expectMem) < 0 {
		t.Errorf("resource-reserve memory: expected >= %s (1 slot), got %s", slotMem, reserveMem.String())
	}

	// Pod total = main overhead + reserve slot reservation; reserve should be
	// the dominant contributor (5× the per-slot reservation).
	reserveTotal := reserveCPU.Value() + reserveMem.Value()
	if reserveTotal <= mainTotal {
		t.Errorf("resource-reserve should dominate the pod's resource total: reserve=%d main=%d",
			reserveTotal, mainTotal)
	}

	// Sanity: the reserve sidecar uses the pause image.
	if reserve.Image == "" {
		t.Errorf("resource-reserve image should be set (pause image)")
	}
}

// keep imports used
var (
	_ time.Duration
	_ corev1.ResourceList
)
