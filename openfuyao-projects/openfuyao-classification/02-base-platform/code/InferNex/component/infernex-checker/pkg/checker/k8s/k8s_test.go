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

package k8s

import (
	"context"
	"os"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// TestMain sets a short DNS check timeout so tests that hit the timeout path
// complete in ~1 second instead of the default 60 seconds.
func TestMain(m *testing.M) {
	dnsCheckTimeout = 3 * time.Second
	dnsCheckInterval = 500 * time.Millisecond
	os.Exit(m.Run())
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func makePod(name, namespace string, phase corev1.PodPhase, labels map[string]string) corev1.Pod {
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    labels,
		},
		Status: corev1.PodStatus{Phase: phase},
	}
}

func makeNode(name string, ready bool, taints []corev1.Taint, cpu, mem string) corev1.Node {
	condStatus := corev1.ConditionFalse
	if ready {
		condStatus = corev1.ConditionTrue
	}
	allocatable := corev1.ResourceList{}
	if cpu != "" {
		allocatable[corev1.ResourceCPU] = resource.MustParse(cpu)
	}
	if mem != "" {
		allocatable[corev1.ResourceMemory] = resource.MustParse(mem)
	}
	return corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       corev1.NodeSpec{Taints: taints},
		Status: corev1.NodeStatus{
			Conditions:  []corev1.NodeCondition{{Type: corev1.NodeReady, Status: condStatus}},
			Allocatable: allocatable,
		},
	}
}

func newClient(objects ...runtime.Object) *executor.K8sClient {
	cs := fake.NewSimpleClientset(objects...)
	return executor.NewK8sClientForTest(cs)
}

// ---------------------------------------------------------------------------
// checkK01
// ---------------------------------------------------------------------------

func TestCheckK01_NoPods(t *testing.T) {
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset())
	result := checkK01(context.Background(), client, defaultDNSCheckImage, []string{"node-1"})
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when no CoreDNS pods, got %q", result.Status)
	}
	if result.ID != "K-01" {
		t.Errorf("expected ID K-01, got %q", result.ID)
	}
}

func TestCheckK01_AllPodsNotRunning(t *testing.T) {
	pod := makePod("coredns-1", "kube-system", corev1.PodPending,
		map[string]string{"k8s-app": "kube-dns"})
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&pod))
	result := checkK01(context.Background(), client, defaultDNSCheckImage, []string{"node-1"})
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when all CoreDNS pods not running, got %q", result.Status)
	}
}

func TestCheckK01_SomePodsNotRunning(t *testing.T) {
	pod1 := makePod("coredns-1", "kube-system", corev1.PodRunning,
		map[string]string{"k8s-app": "kube-dns"})
	pod2 := makePod("coredns-2", "kube-system", corev1.PodPending,
		map[string]string{"k8s-app": "kube-dns"})
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&pod1, &pod2))
	result := checkK01(context.Background(), client, defaultDNSCheckImage, []string{"node-1"})
	if result.Status != types.StatusWarning {
		t.Errorf("expected warning when some CoreDNS pods not running, got %q", result.Status)
	}
}

func TestCheckK01_TerminatingPodCountsAsNotRunning(t *testing.T) {
	now := metav1.Now()
	pod := makePod("coredns-1", "kube-system", corev1.PodRunning,
		map[string]string{"k8s-app": "kube-dns"})
	pod.DeletionTimestamp = &now
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&pod))
	result := checkK01(context.Background(), client, defaultDNSCheckImage, []string{"node-1"})
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when all CoreDNS pods are terminating, got %q", result.Status)
	}
}

func TestCheckK01_TerminatingPodPartial(t *testing.T) {
	now := metav1.Now()
	pod1 := makePod("coredns-1", "kube-system", corev1.PodRunning,
		map[string]string{"k8s-app": "kube-dns"})
	pod2 := makePod("coredns-2", "kube-system", corev1.PodRunning,
		map[string]string{"k8s-app": "kube-dns"})
	pod2.DeletionTimestamp = &now
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&pod1, &pod2))
	result := checkK01(context.Background(), client, defaultDNSCheckImage, []string{"node-1"})
	if result.Status != types.StatusWarning {
		t.Errorf("expected warning when one CoreDNS pod is terminating, got %q", result.Status)
	}
}

func TestCheckK01_PodRunning_DNSCheckPerNode(t *testing.T) {
	// CoreDNS pod is Running. Per-node DNS pods will be created but stay Pending
	// in the fake client → timeout → all nodes fail → StatusFailed.
	pod := makePod("coredns-1", "kube-system", corev1.PodRunning,
		map[string]string{"k8s-app": "kube-dns"})
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&pod))

	result := checkK01(context.Background(), client, defaultDNSCheckImage, []string{"node-1"})
	if result.ID != "K-01" {
		t.Errorf("expected ID K-01, got %q", result.ID)
	}
	// With fake client the DNS pod never transitions, so it times out → failed
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed (DNS pod timeout with fake client), got %q", result.Status)
	}
}

// ---------------------------------------------------------------------------
// checkK02
// ---------------------------------------------------------------------------

func TestCheckK02_NodeReady(t *testing.T) {
	node := makeNode("node-1", true, nil, "8", "16Gi")
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&node))
	results, readyNodes := checkK02(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != types.StatusPassed {
		t.Errorf("expected passed, got %q: %s", results[0].Status, results[0].Message)
	}
	if len(readyNodes) != 1 || readyNodes[0] != "node-1" {
		t.Errorf("expected readyNodes=[node-1], got %v", readyNodes)
	}
}

func TestCheckK02_NodeNotReady(t *testing.T) {
	node := makeNode("node-1", false, nil, "8", "16Gi")
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&node))
	results, readyNodes := checkK02(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != types.StatusFailed {
		t.Errorf("expected failed, got %q", results[0].Status)
	}
	if len(readyNodes) != 0 {
		t.Errorf("expected no ready nodes, got %v", readyNodes)
	}
}

func TestCheckK02_NodeNotFound(t *testing.T) {
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset())
	results, readyNodes := checkK02(context.Background(), client, []string{"missing-node"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != types.StatusFailed {
		t.Errorf("expected failed for missing node, got %q", results[0].Status)
	}
	if len(readyNodes) != 0 {
		t.Errorf("expected no ready nodes, got %v", readyNodes)
	}
}

func TestCheckK02_MultipleNodes(t *testing.T) {
	node1 := makeNode("node-1", true, nil, "8", "16Gi")
	node2 := makeNode("node-2", false, nil, "8", "16Gi")
	node3 := makeNode("node-3", true, nil, "8", "16Gi")
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&node1, &node2, &node3))
	results, readyNodes := checkK02(context.Background(), client, []string{"node-1", "node-2", "node-3"})
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	if len(readyNodes) != 2 {
		t.Errorf("expected 2 ready nodes, got %v", readyNodes)
	}
}

// ---------------------------------------------------------------------------
// checkK03
// ---------------------------------------------------------------------------

func TestCheckK03_NoTaints(t *testing.T) {
	node := makeNode("node-1", true, nil, "8", "16Gi")
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&node))
	results := checkK03(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != types.StatusPassed {
		t.Errorf("expected passed for no taints, got %q", results[0].Status)
	}
}

func TestCheckK03_WithTaints(t *testing.T) {
	taints := []corev1.Taint{
		{Key: "nvidia.com/gpu", Value: "present", Effect: corev1.TaintEffectNoSchedule},
	}
	node := makeNode("node-1", true, taints, "8", "16Gi")
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&node))
	results := checkK03(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != types.StatusInfo {
		t.Errorf("expected info for tainted node, got %q", results[0].Status)
	}
	if results[0].Message == "" {
		t.Error("expected non-empty message for taint")
	}
}

func TestCheckK03_MixedNodes(t *testing.T) {
	taints := []corev1.Taint{
		{Key: "key1", Value: "val1", Effect: corev1.TaintEffectNoSchedule},
	}
	nodeClean := makeNode("node-clean", true, nil, "8", "16Gi")
	nodeTainted := makeNode("node-tainted", true, taints, "8", "16Gi")
	client := newClient(&nodeClean, &nodeTainted)
	results := checkK03(context.Background(), client, []string{"node-clean", "node-tainted"})
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Status != types.StatusPassed {
		t.Errorf("expected node-clean (results[0]) to be passed, got %q", results[0].Status)
	}
	if results[1].Status != types.StatusInfo {
		t.Errorf("expected node-tainted (results[1]) to be info, got %q", results[1].Status)
	}
}

func TestCheckK03_NodeNotFound(t *testing.T) {
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset())
	results := checkK03(context.Background(), client, []string{"ghost-node"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result for missing node, got %d", len(results))
	}
	if results[0].Status != types.StatusFailed {
		t.Errorf("expected failed for missing node, got %q", results[0].Status)
	}
}

// ---------------------------------------------------------------------------
// checkK04
// ---------------------------------------------------------------------------

func TestCheckK04_ReturnsInfo(t *testing.T) {
	node := makeNode("node-1", true, nil, "32", "128Gi")
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&node))
	results := checkK04(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != types.StatusInfo {
		t.Errorf("expected info status, got %q", results[0].Status)
	}
	if results[0].ID != "K-04" {
		t.Errorf("expected ID K-04, got %q", results[0].ID)
	}
	if results[0].Detail == nil {
		t.Error("expected non-nil Detail")
	}
	if _, ok := results[0].Detail["cpu_allocatable"]; !ok {
		t.Error("expected cpu_allocatable in Detail")
	}
	if _, ok := results[0].Detail["mem_allocatable"]; !ok {
		t.Error("expected mem_allocatable in Detail")
	}
}

func TestCheckK04_MultipleNodes(t *testing.T) {
	node1 := makeNode("node-1", true, nil, "8", "32Gi")
	node2 := makeNode("node-2", true, nil, "16", "64Gi")
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&node1, &node2))
	results := checkK04(context.Background(), client, []string{"node-1", "node-2"})
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for _, r := range results {
		if r.Status != types.StatusInfo {
			t.Errorf("expected info status, got %q: %s", r.Status, r.Message)
		}
	}
}

func TestCheckK04_NodeWithNoResources(t *testing.T) {
	// Node with empty CPU and memory allocatable — should still return StatusInfo.
	node := makeNode("node-empty", true, nil, "", "")
	client := newClient(&node)
	results := checkK04(context.Background(), client, []string{"node-empty"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Status != types.StatusInfo {
		t.Errorf("expected info status, got %q", results[0].Status)
	}
}

func TestCheckK04_DetailNodeField(t *testing.T) {
	node := makeNode("node-1", true, nil, "4", "8Gi")
	client := newClient(&node)
	results := checkK04(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	detail := results[0].Detail
	if detail["node"] != "node-1" {
		t.Errorf("expected detail[node]=node-1, got %v", detail["node"])
	}
}

func TestCheckK04_NodeNotFound(t *testing.T) {
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset())
	results := checkK04(context.Background(), client, []string{"ghost-node"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result for missing node, got %d", len(results))
	}
	if results[0].Status != types.StatusFailed {
		t.Errorf("expected failed for missing node, got %q", results[0].Status)
	}
}

func TestCheckK04_WithRunningPods(t *testing.T) {
	// Node with 8 CPU, 32Gi memory allocatable
	node := makeNode("node-1", true, nil, "8", "32Gi")
	// Pod requesting 2 CPU and 4Gi memory
	pod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "workload-pod",
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			NodeName: "node-1",
			Containers: []corev1.Container{
				{
					Name: "app",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
	client := newClient(&node, &pod)
	results := checkK04(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	detail := results[0].Detail
	if detail["cpu_requested"] != "2" {
		t.Errorf("expected cpu_requested=2, got %v", detail["cpu_requested"])
	}
	if detail["mem_requested"] != "4Gi" {
		t.Errorf("expected mem_requested=4Gi, got %v", detail["mem_requested"])
	}
	if detail["cpu_remaining"] != "6" {
		t.Errorf("expected cpu_remaining=6, got %v", detail["cpu_remaining"])
	}
	if detail["mem_remaining"] != "28Gi" {
		t.Errorf("expected mem_remaining=28Gi, got %v", detail["mem_remaining"])
	}
}

func TestCheckK04_WithMultiplePods(t *testing.T) {
	node := makeNode("node-1", true, nil, "16", "64Gi")
	pod1 := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default"},
		Spec: corev1.PodSpec{
			NodeName: "node-1",
			Containers: []corev1.Container{
				{
					Name: "c1",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("4"),
							corev1.ResourceMemory: resource.MustParse("8Gi"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
	pod2 := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: "kube-system"},
		Spec: corev1.PodSpec{
			NodeName: "node-1",
			Containers: []corev1.Container{
				{
					Name: "c2",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning},
	}
	client := newClient(&node, &pod1, &pod2)
	results := checkK04(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	detail := results[0].Detail
	// 4+2=6 CPU requested, 8+4=12Gi memory requested
	if detail["cpu_requested"] != "6" {
		t.Errorf("expected cpu_requested=6, got %v", detail["cpu_requested"])
	}
	if detail["mem_requested"] != "12Gi" {
		t.Errorf("expected mem_requested=12Gi, got %v", detail["mem_requested"])
	}
	if detail["cpu_remaining"] != "10" {
		t.Errorf("expected cpu_remaining=10, got %v", detail["cpu_remaining"])
	}
	if detail["mem_remaining"] != "52Gi" {
		t.Errorf("expected mem_remaining=52Gi, got %v", detail["mem_remaining"])
	}
}

func TestCheckK04_TerminatedPodsNotCounted(t *testing.T) {
	node := makeNode("node-1", true, nil, "8", "32Gi")
	// Succeeded pod should not be counted
	pod := corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "done-pod", Namespace: "default"},
		Spec: corev1.PodSpec{
			NodeName: "node-1",
			Containers: []corev1.Container{
				{
					Name: "c1",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("4"),
							corev1.ResourceMemory: resource.MustParse("16Gi"),
						},
					},
				},
			},
		},
		Status: corev1.PodStatus{Phase: corev1.PodSucceeded},
	}
	client := newClient(&node, &pod)
	results := checkK04(context.Background(), client, []string{"node-1"})
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	detail := results[0].Detail
	// Succeeded pod should be filtered out by GetAllPodsOnNode
	if detail["cpu_requested"] != "0" {
		t.Errorf("expected cpu_requested=0 (terminated pod excluded), got %v", detail["cpu_requested"])
	}
	if detail["cpu_remaining"] != "8" {
		t.Errorf("expected cpu_remaining=8, got %v", detail["cpu_remaining"])
	}
}

// ---------------------------------------------------------------------------
// calculatePodRequest - InitContainers and Overhead
// ---------------------------------------------------------------------------

func TestCalculatePodRequest_ContainersOnly(t *testing.T) {
	pod := &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					},
				},
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("1"),
							corev1.ResourceMemory: resource.MustParse("2Gi"),
						},
					},
				},
			},
		},
	}
	cpu, mem := calculatePodRequest(pod)
	// sum(containers) = 2+1=3, 4+2=6
	if cpu.String() != "3" {
		t.Errorf("expected CPU=3, got %s", cpu.String())
	}
	if mem.String() != "6Gi" {
		t.Errorf("expected Memory=6Gi, got %s", mem.String())
	}
}

func TestCalculatePodRequest_InitContainersLarger(t *testing.T) {
	pod := &corev1.Pod{
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("5"),
							corev1.ResourceMemory: resource.MustParse("10Gi"),
						},
					},
				},
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("3"),
							corev1.ResourceMemory: resource.MustParse("8Gi"),
						},
					},
				},
			},
			Containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					},
				},
			},
		},
	}
	cpu, mem := calculatePodRequest(pod)
	// max(initContainers) = max(5, 3) = 5, max(10, 8) = 10
	// sum(containers) = 2, 4
	// effective = max(5, 2) = 5, max(10, 4) = 10
	if cpu.String() != "5" {
		t.Errorf("expected CPU=5 (max of initContainers), got %s", cpu.String())
	}
	if mem.String() != "10Gi" {
		t.Errorf("expected Memory=10Gi (max of initContainers), got %s", mem.String())
	}
}

func TestCalculatePodRequest_ContainersLarger(t *testing.T) {
	pod := &corev1.Pod{
		Spec: corev1.PodSpec{
			InitContainers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					},
				},
			},
			Containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("3"),
							corev1.ResourceMemory: resource.MustParse("6Gi"),
						},
					},
				},
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("3Gi"),
						},
					},
				},
			},
		},
	}
	cpu, mem := calculatePodRequest(pod)
	// max(initContainers) = 2, 4
	// sum(containers) = 3+2=5, 6+3=9
	// effective = max(2, 5) = 5, max(4, 9) = 9
	if cpu.String() != "5" {
		t.Errorf("expected CPU=5 (sum of containers), got %s", cpu.String())
	}
	if mem.String() != "9Gi" {
		t.Errorf("expected Memory=9Gi (sum of containers), got %s", mem.String())
	}
}

func TestCalculatePodRequest_WithOverhead(t *testing.T) {
	pod := &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("2"),
							corev1.ResourceMemory: resource.MustParse("4Gi"),
						},
					},
				},
			},
			Overhead: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("512Mi"),
			},
		},
	}
	cpu, mem := calculatePodRequest(pod)
	// containers = 2, 4Gi
	// overhead = 100m, 512Mi
	// effective = 2+0.1=2.1, 4Gi+512Mi=4.5Gi
	if cpu.String() != "2100m" {
		t.Errorf("expected CPU=2100m (2+100m), got %s", cpu.String())
	}
	if mem.String() != "4608Mi" {
		t.Errorf("expected Memory=4608Mi (4Gi+512Mi), got %s", mem.String())
	}
}

func TestCalculatePodRequest_NoRequests(t *testing.T) {
	pod := &corev1.Pod{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Resources: corev1.ResourceRequirements{},
				},
			},
		},
	}
	cpu, mem := calculatePodRequest(pod)
	if cpu.String() != "0" {
		t.Errorf("expected CPU=0, got %s", cpu.String())
	}
	if mem.String() != "0" {
		t.Errorf("expected Memory=0, got %s", mem.String())
	}
}

// ---------------------------------------------------------------------------
// waitForDNSPodResult
// ---------------------------------------------------------------------------

func TestWaitForDNSPodResult_PodSucceeded(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "dns-test-pod", Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodSucceeded},
	}
	client := newClient(pod)
	result := waitForDNSPodResult(context.Background(), client, "dns-test-pod")
	if !result {
		t.Error("expected true when pod is Succeeded")
	}
}

func TestWaitForDNSPodResult_PodFailed(t *testing.T) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "dns-test-pod", Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodFailed},
	}
	client := newClient(pod)
	result := waitForDNSPodResult(context.Background(), client, "dns-test-pod")
	if result {
		t.Error("expected false when pod is Failed")
	}
}

// ---------------------------------------------------------------------------
// checkDNSResolutionPerNode
// ---------------------------------------------------------------------------

func TestCheckDNSResolutionPerNode_AllFail(t *testing.T) {
	client := newClient()
	result := checkDNSResolutionPerNode(context.Background(), client, defaultDNSCheckImage, []string{"node-1", "node-2"})
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when all nodes fail DNS, got %q", result.Status)
	}
}

// ---------------------------------------------------------------------------
// CheckK8s integration
// ---------------------------------------------------------------------------

func TestCheckK8s_K01FailsNoCoreDNS(t *testing.T) {
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset())
	result := CheckK8s(client, []string{"node-1"}, "")
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when no CoreDNS, got %q", result.Status)
	}
	// Only K-01 check should be present (terminated early)
	if len(result.Checks) != 1 {
		t.Errorf("expected 1 check (K-01 only), got %d", len(result.Checks))
	}
	if result.Checks[0].ID != "K-01" {
		t.Errorf("expected first check ID K-01, got %q", result.Checks[0].ID)
	}
}

func TestCheckK8s_K02AllNodesNotReady(t *testing.T) {
	// CoreDNS pod running so K-01 proceeds to DNS check per node.
	// With fake client, DNS pods time out → K-01 fails → early termination.
	dnsPod := makePod("coredns-1", "kube-system", corev1.PodRunning,
		map[string]string{"k8s-app": "kube-dns"})
	node := makeNode("node-1", false, nil, "8", "16Gi")
	client := executor.NewK8sClientForTest(fake.NewSimpleClientset(&dnsPod, &node))

	result := CheckK8s(client, []string{"node-1"}, "")
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed result, got %q", result.Status)
	}
}

func TestCheckK8s_AllReadyRunsK03K04(t *testing.T) {
	// No CoreDNS pods → K-01 fails; we cannot easily pass K-01 with a fake
	// client because DNS pod phase never transitions. Instead verify that when
	// K-02 returns ready nodes, K-03 and K-04 are appended.
	// We test this by calling checkK02/K03/K04 directly and confirming the
	// combined result set has the right IDs.
	node := makeNode("node-1", true, nil, "8", "16Gi")
	client := newClient(&node)

	k02Results, readyNodes := checkK02(context.Background(), client, []string{"node-1"})
	k03Results := checkK03(context.Background(), client, readyNodes)
	k04Results := checkK04(context.Background(), client, readyNodes)

	allChecks := append(append(k02Results, k03Results...), k04Results...)
	ids := map[string]bool{}
	for _, c := range allChecks {
		ids[c.ID] = true
	}
	if !ids["K-02"] {
		t.Error("expected K-02 in combined checks")
	}
	if !ids["K-03"] {
		t.Error("expected K-03 in combined checks")
	}
	if !ids["K-04"] {
		t.Error("expected K-04 in combined checks")
	}
}

func TestCheckK8s_PartialReadyNodes(t *testing.T) {
	// node-1 ready, node-2 not ready → K-02 partial, K-03/K-04 run on ready nodes only.
	node1 := makeNode("node-1", true, nil, "8", "16Gi")
	node2 := makeNode("node-2", false, nil, "8", "16Gi")
	client := newClient(&node1, &node2)

	k02Results, readyNodes := checkK02(context.Background(), client, []string{"node-1", "node-2"})
	if len(readyNodes) != 1 || readyNodes[0] != "node-1" {
		t.Errorf("expected only node-1 ready, got %v", readyNodes)
	}

	k03Results := checkK03(context.Background(), client, readyNodes)
	k04Results := checkK04(context.Background(), client, readyNodes)

	if len(k02Results) != 2 {
		t.Errorf("expected 2 K-02 results, got %d", len(k02Results))
	}
	if len(k03Results) != 1 {
		t.Errorf("expected 1 K-03 result (only ready node), got %d", len(k03Results))
	}
	if len(k04Results) != 1 {
		t.Errorf("expected 1 K-04 result (only ready node), got %d", len(k04Results))
	}
}
