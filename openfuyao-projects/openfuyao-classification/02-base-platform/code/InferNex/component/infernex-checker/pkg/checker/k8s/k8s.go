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

// Package k8s implements the K8s Layer checks (K-01 through K-04), verifying
// CoreDNS health, node Ready status, node taints, and node allocatable resources
// via the Kubernetes API.
package k8s

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/log"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

const (
	defaultDNSCheckImage = "busybox:1.36" // default container image used for DNS resolution probe
)

var (
	dnsCheckInterval  = 2 * time.Second  // polling interval when waiting for DNS check pod
	dnsCheckTimeout   = 60 * time.Second // total timeout for DNS check pod to complete
	dnsCleanupTimeout = 10 * time.Second // timeout for cleaning up DNS check pods
)

// CheckK8s runs full K8s Layer checks
func CheckK8s(k8sClient *executor.K8sClient, nodeNames []string, dnsCheckImage string) *types.K8sResult {
	result := &types.K8sResult{Status: types.StatusPassed}
	ctx := context.Background()

	if dnsCheckImage == "" {
		dnsCheckImage = defaultDNSCheckImage
	}

	// K-01 CoreDNS check
	k01Result := checkK01(ctx, k8sClient, dnsCheckImage, nodeNames)
	result.Checks = append(result.Checks, k01Result)
	if k01Result.Status == types.StatusFailed {
		result.Status = types.StatusFailed
		return result
	}

	// K-02 node Ready status
	k02Results, readyNodes := checkK02(ctx, k8sClient, nodeNames)
	result.Checks = append(result.Checks, k02Results...)
	if len(readyNodes) == 0 {
		result.Status = types.StatusFailed
		return result
	}

	// K-03 node taints
	k03Results := checkK03(ctx, k8sClient, readyNodes)
	result.Checks = append(result.Checks, k03Results...)

	// K-04 node available resources
	k04Results := checkK04(ctx, k8sClient, readyNodes)
	result.Checks = append(result.Checks, k04Results...)

	return result
}

// checkK01 checks cluster CoreDNS and per-node DNS resolution
func checkK01(ctx context.Context, k8sClient *executor.K8sClient,
	dnsCheckImage string, nodeNames []string) types.CheckResult {
	pods, err := k8sClient.GetPodsInAllNamespaces(ctx, "k8s-app=kube-dns")
	if err != nil {
		log.Error(fmt.Sprintf("K-01 failed to get CoreDNS pods: %v", err))
		return types.CheckResult{
			ID:         "K-01",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("Failed to get CoreDNS pods: %v", err),
			Suggestion: "Check kubeconfig configuration and cluster connectivity",
		}
	}

	if len(pods) == 0 {
		return types.CheckResult{
			ID:         "K-01",
			Status:     types.StatusFailed,
			Message:    "No CoreDNS pods found",
			Suggestion: "Check CoreDNS pod status and logs, restart CoreDNS or investigate network policies",
		}
	}

	var notRunning []string
	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodRunning || pod.DeletionTimestamp != nil {
			notRunning = append(notRunning, fmt.Sprintf("%s(%s)", pod.Name, pod.Status.Phase))
		}
	}

	runningCount := len(pods) - len(notRunning)
	switch {
	case runningCount == 0:
		return types.CheckResult{
			ID:     "K-01",
			Status: types.StatusFailed,
			Message: fmt.Sprintf("All CoreDNS pods are not in Running state: %s",
				strings.Join(notRunning, ", ")),
			Suggestion: "Check CoreDNS pod status and logs, restart CoreDNS or investigate network policies",
		}
	case runningCount < len(pods):
		return types.CheckResult{
			ID:     "K-01",
			Status: types.StatusWarning,
			Message: fmt.Sprintf("Some CoreDNS pods are not in Running state: %s",
				strings.Join(notRunning, ", ")),
			Suggestion: "Check CoreDNS pod status and logs, restart CoreDNS or investigate network policies",
		}
	default:
		// Per-node DNS resolution check
		return checkDNSResolutionPerNode(ctx, k8sClient, dnsCheckImage, nodeNames)
	}
}

// checkDNSResolutionPerNode creates a temporary pod on each target node to verify DNS resolution.
func checkDNSResolutionPerNode(ctx context.Context, k8sClient *executor.K8sClient,
	dnsCheckImage string, nodeNames []string) types.CheckResult {
	type nodeResult struct {
		node   string
		passed bool
	}

	var results []nodeResult
	for _, nodeName := range nodeNames {
		passed := checkDNSOnNode(ctx, k8sClient, dnsCheckImage, nodeName)
		results = append(results, nodeResult{node: nodeName, passed: passed})
	}

	var failedNodes []string
	for _, r := range results {
		if !r.passed {
			failedNodes = append(failedNodes, r.node)
		}
	}

	if len(failedNodes) == 0 {
		return types.CheckResult{
			ID:      "K-01",
			Status:  types.StatusPassed,
			Message: "CoreDNS Running, all nodes DNS resolution is normal",
		}
	}

	if len(failedNodes) == len(nodeNames) {
		return types.CheckResult{
			ID:         "K-01",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("DNS resolution failed on all nodes: %s", strings.Join(failedNodes, ", ")),
			Suggestion: "Check CoreDNS pod status and logs, restart CoreDNS or investigate network policies",
		}
	}

	// Partial failure: warning, does not terminate
	return types.CheckResult{
		ID:     "K-01",
		Status: types.StatusWarning,
		Message: fmt.Sprintf("DNS resolution failed on some nodes: %s",
			strings.Join(failedNodes, ", ")),
		Suggestion: "Check DNS configuration and network policies on the failed nodes",
	}
}

// checkDNSOnNode creates a temporary pod scheduled on a specific node to verify DNS resolution
func checkDNSOnNode(ctx context.Context, k8sClient *executor.K8sClient, dnsCheckImage, nodeName string) bool {
	podName := fmt.Sprintf("infernex-dns-check-%s", uuid.New().String())
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			NodeName:      nodeName,
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:            "dns-check",
					Image:           dnsCheckImage,
					ImagePullPolicy: corev1.PullIfNotPresent,
					Command:         []string{"nslookup", "kubernetes.default.svc.cluster.local"},
				},
			},
		},
	}

	_, err := k8sClient.CreatePod(ctx, pod)
	if err != nil {
		log.Error(fmt.Sprintf("K-01 failed to create DNS check pod on node %s: %v", nodeName, err))
		return false
	}

	defer func() {
		cleanupCtx, cancel := context.WithTimeout(context.Background(), dnsCleanupTimeout)
		defer cancel()
		_ = k8sClient.DeletePod(cleanupCtx, "default", podName)
	}()

	return waitForDNSPodResult(ctx, k8sClient, podName)
}

// waitForDNSPodResult polls for the DNS check pod to complete and returns whether it succeeded
func waitForDNSPodResult(ctx context.Context, k8sClient *executor.K8sClient, podName string) bool {
	ticker := time.NewTicker(dnsCheckInterval)
	defer ticker.Stop()
	timeoutCtx, cancel := context.WithTimeout(ctx, dnsCheckTimeout)
	defer cancel()

	for {
		select {
		case <-timeoutCtx.Done():
			return false
		case <-ticker.C:
			p, err := k8sClient.GetPod(timeoutCtx, "default", podName)
			if err != nil {
				break
			}
			if p.Status.Phase == corev1.PodSucceeded {
				return true
			}
			if p.Status.Phase == corev1.PodFailed {
				return false
			}
		}
	}
}

// checkK02 checks whether nodes are Ready; returns check results and list of Ready node names
func checkK02(ctx context.Context, k8sClient *executor.K8sClient, nodeNames []string) ([]types.CheckResult, []string) {
	var results []types.CheckResult
	var readyNodes []string

	for _, name := range nodeNames {
		node, err := k8sClient.GetNode(ctx, name)
		if err != nil {
			log.Error(fmt.Sprintf("K-02 failed to get node %s: %v", name, err))
			results = append(results, types.CheckResult{
				ID:         "K-02",
				Status:     types.StatusFailed,
				Message:    fmt.Sprintf("Failed to get node %s: %v", name, err),
				Suggestion: "Please check the node name and cluster connectivity",
			})
			continue
		}

		ready := false
		for _, cond := range node.Status.Conditions {
			if cond.Type == corev1.NodeReady && cond.Status == corev1.ConditionTrue {
				ready = true
				break
			}
		}

		if ready {
			readyNodes = append(readyNodes, name)
			results = append(results, types.CheckResult{
				ID:      "K-02",
				Status:  types.StatusPassed,
				Message: fmt.Sprintf("%s Ready", name),
			})
		} else {
			results = append(results, types.CheckResult{
				ID:         "K-02",
				Status:     types.StatusFailed,
				Message:    fmt.Sprintf("%s is not in Ready state", name),
				Suggestion: "Investigate the cause of node anomaly (network, resources, component failure, etc.)",
			})
		}
	}

	return results, readyNodes
}

// checkK03 checks node taints
func checkK03(ctx context.Context, k8sClient *executor.K8sClient, nodeNames []string) []types.CheckResult {
	var results []types.CheckResult

	for _, name := range nodeNames {
		node, err := k8sClient.GetNode(ctx, name)
		if err != nil {
			log.Error(fmt.Sprintf("K-03 failed to get node %s: %v", name, err))
			results = append(results, types.CheckResult{
				ID:         "K-03",
				Status:     types.StatusFailed,
				Message:    fmt.Sprintf("Failed to get node %s: %v", name, err),
				Suggestion: "Please check the node name and cluster connectivity",
			})
			continue
		}

		if len(node.Spec.Taints) == 0 {
			results = append(results, types.CheckResult{
				ID:      "K-03",
				Status:  types.StatusPassed,
				Message: fmt.Sprintf("%s has no taints", name),
			})
		} else {
			var taintStrs []string
			for _, t := range node.Spec.Taints {
				taintStrs = append(taintStrs, fmt.Sprintf("%s=%s:%s", t.Key, t.Value, t.Effect))
			}
			results = append(results, types.CheckResult{
				ID:      "K-03",
				Status:  types.StatusInfo,
				Message: fmt.Sprintf("%s has taints: %s", name, strings.Join(taintStrs, ", ")),
				Suggestion: "For user to determine whether this affects InferNex pod scheduling; " +
					"if taints block scheduling, add corresponding tolerations for InferNex pods in values.yaml, " +
					"or remove unnecessary taints",
			})
		}
	}

	return results
}

// checkK04 checks node available resources
func checkK04(ctx context.Context, k8sClient *executor.K8sClient, nodeNames []string) []types.CheckResult {
	var results []types.CheckResult
	for _, name := range nodeNames {
		result := checkK04OneNode(ctx, k8sClient, name)
		results = append(results, result)
	}
	return results
}

// checkK04OneNode checks available resources for a single node
func checkK04OneNode(ctx context.Context, k8sClient *executor.K8sClient, nodeName string) types.CheckResult {
	node, err := k8sClient.GetNode(ctx, nodeName)
	if err != nil {
		log.Error(fmt.Sprintf("K-04 failed to get node %s: %v", nodeName, err))
		return types.CheckResult{
			ID:         "K-04",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("Failed to get node %s: %v", nodeName, err),
			Suggestion: "Please check the node name and cluster connectivity",
		}
	}

	cpuAllocatable := node.Status.Allocatable.Cpu()
	memAllocatable := node.Status.Allocatable.Memory()

	pods, err := k8sClient.GetAllPodsOnNode(ctx, nodeName)
	if err != nil {
		log.Error(fmt.Sprintf("K-04 failed to get pods on node %s: %v", nodeName, err))
		return buildK04FallbackResult(nodeName, cpuAllocatable, memAllocatable)
	}

	cpuRequested, memRequested := calculatePodRequests(pods)
	return buildK04FullResult(nodeName, cpuAllocatable, memAllocatable, cpuRequested, memRequested)
}

// calculatePodRequests sums up resource requests from all pods
func calculatePodRequests(pods []corev1.Pod) (*resource.Quantity, *resource.Quantity) {
	cpuRequested := resource.NewMilliQuantity(0, resource.DecimalSI)
	memRequested := resource.NewQuantity(0, resource.BinarySI)

	for i := range pods {
		podCPU, podMem := calculatePodRequest(&pods[i])
		cpuRequested.Add(*podCPU)
		memRequested.Add(*podMem)
	}
	return cpuRequested, memRequested
}

// calculatePodRequest calculates the effective resource request for a single Pod
// following Kubernetes scheduling rules: max(sum(containers), max(initContainers)) + overhead
func calculatePodRequest(pod *corev1.Pod) (*resource.Quantity, *resource.Quantity) {
	// 1. Calculate sum of all regular containers
	containersCPU := resource.NewMilliQuantity(0, resource.DecimalSI)
	containersMem := resource.NewQuantity(0, resource.BinarySI)
	for j := range pod.Spec.Containers {
		if req, ok := pod.Spec.Containers[j].Resources.Requests[corev1.ResourceCPU]; ok {
			containersCPU.Add(req)
		}
		if req, ok := pod.Spec.Containers[j].Resources.Requests[corev1.ResourceMemory]; ok {
			containersMem.Add(req)
		}
	}

	// 2. Calculate max of all init containers
	initCPU := resource.NewMilliQuantity(0, resource.DecimalSI)
	initMem := resource.NewQuantity(0, resource.BinarySI)
	for j := range pod.Spec.InitContainers {
		if req, ok := pod.Spec.InitContainers[j].Resources.Requests[corev1.ResourceCPU]; ok {
			if req.Cmp(*initCPU) > 0 {
				reqCopy := req.DeepCopy()
				initCPU = &reqCopy
			}
		}
		if req, ok := pod.Spec.InitContainers[j].Resources.Requests[corev1.ResourceMemory]; ok {
			if req.Cmp(*initMem) > 0 {
				reqCopy := req.DeepCopy()
				initMem = &reqCopy
			}
		}
	}

	// 3. Take max(containers, initContainers)
	effectiveCPU := containersCPU
	effectiveMem := containersMem
	if initCPU.Cmp(*containersCPU) > 0 {
		effectiveCPU = initCPU
	}
	if initMem.Cmp(*containersMem) > 0 {
		effectiveMem = initMem
	}

	// 4. Add Pod overhead if present
	if pod.Spec.Overhead != nil {
		if overhead, ok := pod.Spec.Overhead[corev1.ResourceCPU]; ok {
			effectiveCPU.Add(overhead)
		}
		if overhead, ok := pod.Spec.Overhead[corev1.ResourceMemory]; ok {
			effectiveMem.Add(overhead)
		}
	}

	return effectiveCPU, effectiveMem
}

// buildK04FallbackResult builds a K-04 result when pod information is unavailable
func buildK04FallbackResult(nodeName string, cpuAllocatable, memAllocatable *resource.Quantity) types.CheckResult {
	return types.CheckResult{
		ID:     "K-04",
		Status: types.StatusInfo,
		Message: fmt.Sprintf("%s  Allocatable: CPU %s, Memory %s (unable to calculate remaining)",
			nodeName, cpuAllocatable.String(), memAllocatable.String()),
		Detail: map[string]interface{}{
			"node":            nodeName,
			"cpu_allocatable": cpuAllocatable.String(),
			"mem_allocatable": memAllocatable.String(),
		},
		Suggestion: "Failed to retrieve pod list, unable to calculate remaining resources",
	}
}

// buildK04FullResult builds a complete K-04 result with remaining resource calculation
func buildK04FullResult(
	nodeName string,
	cpuAllocatable, memAllocatable, cpuRequested, memRequested *resource.Quantity,
) types.CheckResult {
	cpuRemaining := cpuAllocatable.DeepCopy()
	cpuRemaining.Sub(*cpuRequested)
	memRemaining := memAllocatable.DeepCopy()
	memRemaining.Sub(*memRequested)

	cpuRemainingStr := formatResourceWithOvercommit(&cpuRemaining, resource.NewMilliQuantity(0, resource.DecimalSI))
	memRemainingStr := formatResourceWithOvercommit(&memRemaining, resource.NewQuantity(0, resource.BinarySI))

	return types.CheckResult{
		ID:     "K-04",
		Status: types.StatusInfo,
		Message: fmt.Sprintf("%s  Allocatable: CPU %s, Memory %s | Remaining: CPU %s, Memory %s",
			nodeName,
			cpuAllocatable.String(), memAllocatable.String(),
			cpuRemainingStr, memRemainingStr,
		),
		Detail: map[string]interface{}{
			"node":            nodeName,
			"cpu_allocatable": cpuAllocatable.String(),
			"mem_allocatable": memAllocatable.String(),
			"cpu_requested":   cpuRequested.String(),
			"mem_requested":   memRequested.String(),
			"cpu_remaining":   cpuRemaining.String(),
			"mem_remaining":   memRemaining.String(),
		},
		Suggestion: "Please confirm this meets InferNex deployment requirements; " +
			"if insufficient, consider releasing occupied resources or adjusting workloads",
	}
}

// formatResourceWithOvercommit formats a resource quantity, adding "(overcommitted)" suffix if negative
func formatResourceWithOvercommit(remaining, zero *resource.Quantity) string {
	str := remaining.String()
	if remaining.Cmp(*zero) < 0 {
		str += " (overcommitted)"
	}
	return str
}
