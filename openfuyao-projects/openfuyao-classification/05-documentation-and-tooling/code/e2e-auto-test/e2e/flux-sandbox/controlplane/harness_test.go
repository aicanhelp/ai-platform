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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	sandboxv1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/proto/v1"
	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// Label keys.
const (
	// testLabelKey marks SandboxGroup CRs created by this test suite, so the
	// teardown can find and clean them up.
	testLabelKey = "e2e.flux-sandbox/test"
	// labelSandboxGroup is the owner-group label on SubSandboxGroups / Agent Pods.
	labelSandboxGroup = "sandbox.flux.io/sandbox-group"
	// labelSubSandboxGroup is the SSG label on Agent Pods.
	labelSubSandboxGroup = "sandbox.flux.io/sub-sandbox-group"
	// labelControllerOwned marks controller-owned objects.
	labelControllerOwned = "sandbox.flux.io/controller-owned"
	// labelScheduler marks the scheduler assigned to an SSG.
	labelScheduler = "sandbox.flux.io/scheduler"
	// appAgentPodLabel is the app label the scheduler stamps on Agent Pods.
	appAgentPodLabel = "sandbox-agent-pod"
)

// controllerHTTPURL is the base URL of the Controller HTTP server (via
// port-forward).
const controllerHTTPURL = "http://127.0.0.1:" + cpControllerHTTPPort

// ── Naming ──

// cpName derives a unique, RFC 1123-compliant name (lowercase letters/digits/-,
// <=63 chars) from the test name. It is used for SandboxGroup and Sandbox CR
// names to isolate concurrent test runs.
func cpName(t ginkgo.GinkgoTInterface) string {
	name := strings.ToLower(t.Name())
	name = strings.ReplaceAll(name, "/", "-")
	name = strings.ReplaceAll(name, "_", "-")
	if len(name) > 63 {
		name = name[len(name)-63:]
		name = strings.TrimLeft(name, "-")
	}
	return name
}

// ── SandboxGroup CR operations ──

// sgOption mutates a SandboxGroup during construction.
type sgOption func(*apiv1alpha1.SandboxGroup)

// withAgentEnv sets/overrides an agentEnv entry.
func withAgentEnv(key, val string) sgOption {
	return func(sg *apiv1alpha1.SandboxGroup) {
		if sg.Spec.AgentEnv == nil {
			sg.Spec.AgentEnv = map[string]string{}
		}
		sg.Spec.AgentEnv[key] = val
	}
}

// withoutExecd clears the ExecdImage (for the no-execd containerd case).
func withoutExecd() sgOption {
	return func(sg *apiv1alpha1.SandboxGroup) { sg.Spec.ExecdImage = "" }
}

// withExecd sets the ExecdImage.
func withExecd(img string) sgOption {
	return func(sg *apiv1alpha1.SandboxGroup) { sg.Spec.ExecdImage = img }
}

// withBuffer sets the warm-buffer min/max.
func withBuffer(min, max int) sgOption {
	return func(sg *apiv1alpha1.SandboxGroup) {
		sg.Spec.Capacity.SandboxBufferMin = min
		sg.Spec.Capacity.SandboxBufferMax = max
	}
}

// withSlotResources sets the per-slot resource requests/limits.
func withSlotResources(cpu, mem string) sgOption {
	return func(sg *apiv1alpha1.SandboxGroup) {
		sg.Spec.SandboxResources.Requests = corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse(cpu),
			corev1.ResourceMemory: resource.MustParse(mem),
		}
		sg.Spec.SandboxResources.Limits = corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse(cpu),
			corev1.ResourceMemory: resource.MustParse(mem),
		}
	}
}

// basicSandboxGroup builds a minimal SandboxGroup CR with a containerd runtime
// and execd injection enabled by default (the common precondition). The CR is
// stamped with the test label so teardown can clean it up.
func basicSandboxGroup(name string, agentPodMin, agentPodMax, maxPerPod int, opts ...sgOption) *apiv1alpha1.SandboxGroup {
	sg := &apiv1alpha1.SandboxGroup{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: cpNamespace,
			Labels: map[string]string{
				testLabelKey: "true",
			},
		},
		Spec: apiv1alpha1.SandboxGroupSpec{
			Capacity: apiv1alpha1.CapacitySpec{
				AgentPodMin:        agentPodMin,
				AgentPodMax:        agentPodMax,
				MaxSandboxesPerPod: maxPerPod,
			},
			SandboxResources: corev1.ResourceRequirements{
				// The controller validates that limits is set; requests alone
				// makes the SandboxGroup permanently un-reconcilable.
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("50m"),
					corev1.ResourceMemory: resource.MustParse("64Mi"),
				},
				Limits: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("50m"),
					corev1.ResourceMemory: resource.MustParse("64Mi"),
				},
			},
			AgentEnv: map[string]string{
				"RUNTIME_TYPE": "containerd",
			},
			ExecdImage: cpExecdImage,
		},
	}
	for _, opt := range opts {
		opt(sg)
	}
	return sg
}

// createSandboxGroup creates the SandboxGroup CR (deleting any pre-existing one
// first for idempotency) and registers a cleanup that deletes it.
func createSandboxGroup(t ginkgo.GinkgoTInterface, sg *apiv1alpha1.SandboxGroup) {
	t.Helper()
	ctx := context.Background()
	deleteSandboxGroup(t, sg.Name)
	if err := cpEnv.k8sClient.Create(ctx, sg); err != nil {
		t.Fatalf("create SandboxGroup %s: %v", sg.Name, err)
	}
	t.Cleanup(func() {
		// Use foreground deletion so the cleanup waits for cascade.
		fg := metav1.DeletePropagationForeground
		_ = cpEnv.k8sClient.Delete(ctx, &apiv1alpha1.SandboxGroup{
			ObjectMeta: metav1.ObjectMeta{Name: sg.Name, Namespace: cpNamespace},
		}, &client.DeleteOptions{PropagationPolicy: &fg})
		waitSGGone(sg.Name, 120*time.Second)
	})
}

// deleteSandboxGroup deletes the SandboxGroup CR (no-op if not found).
func deleteSandboxGroup(t ginkgo.GinkgoTInterface, name string) {
	t.Helper()
	ctx := context.Background()
	_ = cpEnv.k8sClient.Delete(ctx, &apiv1alpha1.SandboxGroup{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: cpNamespace},
	}, &client.DeleteOptions{PropagationPolicy: ptrForeground()})
	waitSGGone(name, 60*time.Second)
	// Also wait for the owned SSG CRs to disappear: the Scheduler reclaims its
	// in-memory runtime asynchronously once the SSG CR is gone. Recreating the
	// same-named SG before that would hit the stale runtime (see
	// waitSSGRuntimeCleaned).
	waitSSGRuntimeCleaned(name, 60*time.Second)
}

// ptrForeground returns a pointer to Foreground deletion propagation.
func ptrForeground() *metav1.DeletionPropagation {
	p := metav1.DeletePropagationForeground
	return &p
}

// waitSGGone blocks until the SandboxGroup CR is gone or timeout.
func waitSGGone(name string, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		err := cpEnv.k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: cpNamespace}, &apiv1alpha1.SandboxGroup{})
		if err != nil && apierrors.IsNotFound(err) {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// waitSSGRuntimeCleaned waits until no SubSandboxGroup CRs owned by the given
// SandboxGroup remain in the API server. The Scheduler reclaims its in-memory
// SSG runtime asynchronously ("SSG CR not found in API server, cleaning up
// stale runtime", ~10s poll). Recreating a same-named SG before that cleanup
// finishes lets the new create request hit the stale runtime (dead agent pod /
// released reservations), surfacing as intermittent "task already exists" or
// "no ready route" failures. Waiting for the SSG CRs to disappear closes that
// window, since the scheduler cleanup is driven by exactly that CR absence.
func waitSSGRuntimeCleaned(sgName string, timeout time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	for {
		list, err := listSSGsByGroup(sgName)
		if err == nil && len(list.Items) == 0 {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// getSandboxGroup fetches the current SandboxGroup CR.
func getSandboxGroup(t ginkgo.GinkgoTInterface, name string) *apiv1alpha1.SandboxGroup {
	t.Helper()
	ctx := context.Background()
	sg := &apiv1alpha1.SandboxGroup{}
	if err := cpEnv.k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: cpNamespace}, sg); err != nil {
		t.Fatalf("get SandboxGroup %s: %v", name, err)
	}
	return sg
}

// getSSG fetches a SubSandboxGroup CR.
func getSSG(t ginkgo.GinkgoTInterface, name string) *apiv1alpha1.SubSandboxGroup {
	t.Helper()
	ctx := context.Background()
	ssg := &apiv1alpha1.SubSandboxGroup{}
	if err := cpEnv.k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: cpNamespace}, ssg); err != nil {
		t.Fatalf("get SubSandboxGroup %s: %v", name, err)
	}
	return ssg
}

// listSSGsByGroup lists SubSandboxGroups owned by the given SandboxGroup.
func listSSGsByGroup(sgName string) (*apiv1alpha1.SubSandboxGroupList, error) {
	ctx := context.Background()
	list := &apiv1alpha1.SubSandboxGroupList{}
	if err := cpEnv.k8sClient.List(ctx, list, client.InNamespace(cpNamespace),
		client.MatchingLabels{labelSandboxGroup: sgName}); err != nil {
		return nil, err
	}
	return list, nil
}

// getSandboxCR fetches the Sandbox CR by name (= sandbox_id).
func getSandboxCR(name string) (*apiv1alpha1.Sandbox, error) {
	ctx := context.Background()
	sb := &apiv1alpha1.Sandbox{}
	if err := cpEnv.k8sClient.Get(ctx, types.NamespacedName{Name: name, Namespace: cpNamespace}, sb); err != nil {
		return nil, err
	}
	return sb, nil
}

// deleteSandboxCR deletes a Sandbox CR (no-op if not found).
func deleteSandboxCR(name string) {
	ctx := context.Background()
	_ = cpEnv.k8sClient.Delete(ctx, &apiv1alpha1.Sandbox{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: cpNamespace},
	})
}

// ── Agent Pod queries ──

// listAgentPods lists Agent Pods belonging to the given SandboxGroup (label
// app=sandbox-agent-pod + sandbox.flux.io/sandbox-group=<sgName>).
func listAgentPods(sgName string) ([]corev1.Pod, error) {
	ctx := context.Background()
	list := &corev1.PodList{}
	selector := labels.SelectorFromSet(labels.Set{
		"app":             appAgentPodLabel,
		labelSandboxGroup: sgName,
	})
	if err := cpEnv.k8sClient.List(ctx, list, client.InNamespace(cpNamespace), client.MatchingLabelsSelector{Selector: selector}); err != nil {
		return nil, err
	}
	return list.Items, nil
}

// getAgentPod returns the first ready Agent Pod for the SandboxGroup.
func getAgentPod(t ginkgo.GinkgoTInterface, sgName string) corev1.Pod {
	t.Helper()
	pods, err := listAgentPods(sgName)
	if err != nil || len(pods) == 0 {
		t.Fatalf("no agent pods for SandboxGroup %s: %v", sgName, err)
	}
	return pods[0]
}

// schedulerPods lists Scheduler Pods (label app=flux-sandbox-scheduler).
func schedulerPods() ([]corev1.Pod, error) {
	ctx := context.Background()
	list := &corev1.PodList{}
	if err := cpEnv.k8sClient.List(ctx, list, client.InNamespace(cpNamespace),
		client.MatchingLabels{"app": "flux-sandbox-scheduler"}); err != nil {
		return nil, err
	}
	return list.Items, nil
}

// ── gRPC sandbox operations ──

// preCleanSandboxIDs removes leftover Sandbox CRs (and asks the control plane
// to release any runtime) for the given IDs before a test creates them. A
// previous failed run of the same-named case can leave a stuck CR or an
// orphan node-level task that makes the create intermittently fail with
// "already exists"; cleaning first makes case reruns deterministic.
func preCleanSandboxIDs(ids ...string) {
	for _, id := range ids {
		// Ask the control plane to delete first: a leftover from a previous
		// run may exist as a live runtime (node-level task/container) with no
		// Sandbox CR, and only the gRPC delete path reaches the Agent to clean
		// it. Errors are expected and ignored (usually already deleted).
		if cpEnv != nil && cpEnv.grpcClient != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			_, _ = cpEnv.grpcClient.DeleteSandbox(ctx, &sandboxv1.DeleteSandboxRequest{
				SandboxId: id,
				Namespace: cpNamespace,
			})
			cancel()
		}
		deleteSandboxCR(id)
	}
}

// grpcCreate calls the Controller CreateSandbox RPC. sandboxID becomes the
// Sandbox CR name; sgName is the target SandboxGroup.
func grpcCreate(t ginkgo.GinkgoTInterface, sandboxID, sgName string) (*sandboxv1.CreateSandboxResponse, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	return cpEnv.grpcClient.CreateSandbox(ctx, &sandboxv1.CreateSandboxRequest{
		SandboxId:  sandboxID,
		Namespace:  cpNamespace,
		ImageSpec:  &sandboxv1.ImageSpec{Uri: cpSandboxImage},
		Entrypoint: []string{"/bin/sleep", "3600"},
		Extensions: map[string]string{"sandboxGroup": sgName},
	})
}

// grpcCreateMustCreate is grpcCreate that fatals on error.
func grpcCreateMustCreate(t ginkgo.GinkgoTInterface, sandboxID, sgName string) *sandboxv1.CreateSandboxResponse {
	t.Helper()
	resp, err := grpcCreate(t, sandboxID, sgName)
	if err != nil {
		t.Fatalf("CreateSandbox %s: %v", sandboxID, err)
	}
	return resp
}

// grpcDelete calls the Controller DeleteSandbox RPC.
func grpcDelete(t ginkgo.GinkgoTInterface, sandboxID string) (*sandboxv1.DeleteSandboxResponse, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	return cpEnv.grpcClient.DeleteSandbox(ctx, &sandboxv1.DeleteSandboxRequest{
		SandboxId: sandboxID,
		Namespace: cpNamespace,
	})
}

// grpcGet calls the Controller GetSandbox RPC.
func grpcGet(t ginkgo.GinkgoTInterface, sandboxID string) (*sandboxv1.GetSandboxResponse, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return cpEnv.grpcClient.GetSandbox(ctx, &sandboxv1.GetSandboxRequest{
		SandboxId: sandboxID,
		Namespace: cpNamespace,
	})
}

// ── HTTP helpers (Controller + Agent + execd) ──

// controllerHealthz returns (statusCode, body) from the Controller /healthz.
func controllerHealthz(t ginkgo.GinkgoTInterface) (int, string) {
	t.Helper()
	return httpGet(t, controllerHTTPURL+"/healthz")
}

// controllerReadyz returns (statusCode, body) from the Controller /readyz.
func controllerReadyz(t ginkgo.GinkgoTInterface) (int, string) {
	t.Helper()
	return httpGet(t, controllerHTTPURL+"/readyz")
}

// routerSnapshot queries the Controller /debug/router endpoint and decodes the
// RouterSnapshot JSON.
func routerSnapshot(t ginkgo.GinkgoTInterface) map[string]any {
	t.Helper()
	_, body := httpGet(t, controllerHTTPURL+"/debug/router")
	var snap map[string]any
	if err := json.Unmarshal([]byte(body), &snap); err != nil {
		t.Fatalf("unmarshal router snapshot: %v (body head: %.200s)", err, body)
	}
	return snap
}

// httpGet is a small helper for GET requests returning (code, body).
func httpGet(t ginkgo.GinkgoTInterface, url string) (int, string) {
	t.Helper()
	resp, err := http.Get(url)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	b, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read response body of %s: %v", url, readErr)
	}
	return resp.StatusCode, string(b)
}

// httpPostJSON POSTs a JSON body and returns (code, body).
func httpPostJSON(t ginkgo.GinkgoTInterface, url, body string) (int, string) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	b, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		t.Fatalf("read response body of %s: %v", url, readErr)
	}
	return resp.StatusCode, string(b)
}

// ── Port-forward (Agent Pod) ──

// portForwardAgent starts a kubectl port-forward for an Agent Pod, mapping both
// the agent HTTP port (5758) and the execd port (44772) to local ports. The
// returned strings are the local base URLs (agentHTTPBase, execdBase). The
// forward is killed when the test finishes.
//
// The fixed local ports are shared across cases that run sequentially in one
// test binary: a previous case's pf may still be tearing down when this one
// starts, so start + probe is retried (up to 3 attempts) to absorb the
// transient "port in use / connection refused" window.
func portForwardAgent(t ginkgo.GinkgoTInterface, podName string) (agentHTTPBase, execdBase string) {
	t.Helper()
	// Use fixed local ports; tests using this run sequentially within a single
	// agent pod. Use the kind-style random allocation is avoided to keep it
	// simple.
	const localAgentPort = "15758"
	const localExecdPort = "14772"

	var cmd *exec.Cmd
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		if attempt > 1 {
			time.Sleep(2 * time.Second)
		}
		cmd = exec.Command("kubectl", "port-forward", "-n", cpNamespace, podName,
			localAgentPort+":5758", localExecdPort+":44772")
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			lastErr = err
			t.Logf("start agent port-forward %s: attempt %d failed: %v", podName, attempt, err)
			continue
		}

		// Probe the agent HTTP port; on failure, kill this attempt and retry.
		probeDeadline := time.Now().Add(30 * time.Second)
		probeOK := false
		for time.Now().Before(probeDeadline) {
			resp, err := http.Get("http://127.0.0.1:" + localAgentPort + "/healthz")
			if err == nil && resp.StatusCode == 200 {
				if closeErr := resp.Body.Close(); closeErr != nil {
					t.Logf("close agent probe body: %v", closeErr)
				}
				probeOK = true
				break
			}
			if resp != nil {
				if closeErr := resp.Body.Close(); closeErr != nil {
					t.Logf("close agent probe body: %v", closeErr)
				}
			}
			time.Sleep(time.Second)
		}
		if probeOK {
			break
		}
		lastErr = fmt.Errorf("agent port-forward probe timeout on attempt %d", attempt)
		t.Logf("agent port-forward %s: attempt %d probe timeout, killing and retrying", podName, attempt)
		if cmd.Process != nil {
			if killErr := cmd.Process.Kill(); killErr != nil {
				t.Logf("kill agent port-forward: %v", killErr)
			}
			if waitErr := cmd.Wait(); waitErr != nil {
				t.Logf("reap agent port-forward: %v", waitErr)
			}
		}
	}
	if lastErr != nil {
		t.Fatalf("agent port-forward %s: %v", podName, lastErr)
	}
	t.Cleanup(func() {
		if cmd.Process != nil {
			if err := cmd.Process.Kill(); err != nil {
				t.Logf("kill agent port-forward: %v", err)
			}
			if err := cmd.Wait(); err != nil {
				t.Logf("reap agent port-forward: %v", err)
			}
		}
	})

	return "http://127.0.0.1:" + localAgentPort, "http://127.0.0.1:" + localExecdPort
}

// ── Polling helpers ──

// waitFor polls until cond returns true or timeout, failing the test on timeout.
func waitFor(t ginkgo.GinkgoTInterface, desc string, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(3 * time.Second)
	}
	t.Fatalf("waitFor timeout (%v): %s", timeout, desc)
}

// waitForNot verifies a condition stays false for the whole duration (used to
// assert something does NOT happen).
func waitForNot(t ginkgo.GinkgoTInterface, desc string, duration time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(duration)
	for {
		if !time.Now().Before(deadline) {
			// Duration elapsed with the condition never met: the assertion holds.
			break
		}
		if cond() {
			t.Fatalf("unexpected condition met: %s", desc)
		}
		time.Sleep(3 * time.Second)
	}
}

// ── kubectl helpers ──

// kubectlLogs returns the tail N lines of logs for a pod's container.
func kubectlLogs(t ginkgo.GinkgoTInterface, namespace, podName, container string, tailLines int) string {
	t.Helper()
	args := []string{"logs", "-n", namespace, podName}
	if container != "" {
		args = append(args, "-c", container)
	}
	args = append(args, fmt.Sprintf("--tail=%d", tailLines))
	out, err := exec.Command("kubectl", args...).CombinedOutput()
	if err != nil {
		t.Fatalf("kubectl logs %s/%s: %v\n%s", namespace, podName, err, out)
	}
	return string(out)
}

// kubectlExec runs a command in a pod and returns combined output.
func kubectlExec(t ginkgo.GinkgoTInterface, namespace, podName string, cmd ...string) string {
	t.Helper()
	args := append([]string{"exec", "-n", namespace, podName, "--"}, cmd...)
	out, err := exec.Command("kubectl", args...).CombinedOutput()
	if err != nil {
		t.Fatalf("kubectl exec %s/%s %v: %v\n%s", namespace, podName, cmd, err, out)
	}
	return string(out)
}

// kubectlWaitFor waits for a condition on a resource via kubectl wait.
func kubectlWaitFor(resource, name, condition string, timeout time.Duration) error {
	args := []string{"wait", resource, "-n", cpNamespace, name,
		"--for=" + condition, fmt.Sprintf("--timeout=%ds", int(timeout.Seconds()))}
	cmd := exec.Command("kubectl", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("kubectl wait %s %s: %v: %s", resource, name, err, stderr.String())
	}
	return nil
}

// requireSchedulerReplicas skips the test if the cluster does not have at least
// n scheduler replicas (per E2E_SCHEDULER_REPLICAS).
func requireSchedulerReplicas(t ginkgo.GinkgoTInterface, n int) {
	t.Helper()
	if cpSchedulerReplicas < n {
		t.Skipf("requires >=%d scheduler replicas, got E2E_SCHEDULER_REPLICAS=%d", n, cpSchedulerReplicas)
	}
}

// requireDestructive skips the test unless E2E_DESTRUCTIVE is set.
func requireDestructive(t ginkgo.GinkgoTInterface) {
	t.Helper()
	if !cpDestructive {
		t.Skip("destructive case; set E2E_DESTRUCTIVE=1 to run")
	}
}

// requireBigSharding skips the test unless E2E_BIG_SHARDING is set.
func requireBigSharding(t ginkgo.GinkgoTInterface) {
	t.Helper()
	if !cpBigSharding {
		t.Skip("large-scale sharding case; set E2E_BIG_SHARDING=1 to run")
	}
}

// conditionFor looks up a condition by type in a metav1.Condition slice.
func conditionFor(conds []metav1.Condition, condType string) (metav1.Condition, bool) {
	for _, c := range conds {
		if c.Type == condType {
			return c, true
		}
	}
	return metav1.Condition{}, false
}

// containerByName finds a container by name in a slice.
func containerByName(containers []corev1.Container, name string) (corev1.Container, bool) {
	for _, c := range containers {
		if c.Name == name {
			return c, true
		}
	}
	return corev1.Container{}, false
}

// envVar finds an env var by name in a container's Env.
func envVar(envs []corev1.EnvVar, name string) (string, bool) {
	for _, e := range envs {
		if e.Name == name {
			return e.Value, true
		}
	}
	return "", false
}

// volumeByName finds a volume by name.
func volumeByName(vols []corev1.Volume, name string) (corev1.Volume, bool) {
	for _, v := range vols {
		if v.Name == name {
			return v, true
		}
	}
	return corev1.Volume{}, false
}
