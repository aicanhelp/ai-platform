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
	"os"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ── Scenario group: Agent Pod lifecycle + execd injection ──

// waitAgentPodsReady waits until the SandboxGroup has num ready Agent Pods (2/2
// running with the sandbox-agent container ready).
func waitAgentPodsReady(t ginkgo.GinkgoTInterface, sgName string, num int) {
	t.Helper()
	waitFor(t, "agent pods ready", 180*time.Second, func() bool {
		pods, err := listAgentPods(sgName)
		if err != nil || len(pods) < num {
			return false
		}
		ready := 0
		for i := range pods {
			if isAgentPodReady(&pods[i]) {
				ready++
			}
		}
		return ready >= num
	})
}

// isAgentPodReady reports whether a Pod is 2/2 running with the sandbox-agent
// container ready.
func isAgentPodReady(pod *corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	if len(pod.Status.ContainerStatuses) != 2 {
		return false
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if !cs.Ready {
			return false
		}
	}
	return true
}

// TestAgentPod_Structure verifies FS-FUNC-006: the Agent Pod is 2/2 Running with
// sandbox-agent + resource-reserve containers and the containerd/execd volumes.
func testAgentPod_Structure(t ginkgo.GinkgoTInterface) {
	name := "sg-cp-006-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, 2, withBuffer(0, 0))
	createSandboxGroup(t, sg)
	waitAgentPodsReady(t, name, 1)

	pods, err := listAgentPods(name)
	if err != nil || len(pods) == 0 {
		t.Fatalf("no agent pods for %s: %v", name, err)
	}
	pod := &pods[0]

	if !isAgentPodReady(pod) {
		t.Fatalf("agent pod %s not 2/2 Running: %+v", pod.Name, pod.Status.ContainerStatuses)
	}

	// Containers: sandbox-agent + resource-reserve.
	names := map[string]bool{}
	for _, c := range pod.Spec.Containers {
		names[c.Name] = true
	}
	if !names["sandbox-agent"] {
		t.Errorf("expected sandbox-agent container, got %v", names)
	}
	if !names["resource-reserve"] {
		t.Errorf("expected resource-reserve container, got %v", names)
	}

	// Volumes: execd-shared (hostPath), containerd-socket (Socket), containerd-root (Directory).
	volNames := map[string]bool{}
	for _, v := range pod.Spec.Volumes {
		volNames[v.Name] = true
	}
	for _, want := range []string{"execd-shared", "containerd-socket"} {
		if !volNames[want] {
			t.Errorf("expected volume %s, got %v", want, volNames)
		}
	}
	if v, ok := volumeByName(pod.Spec.Volumes, "containerd-socket"); !ok || v.HostPath == nil {
		t.Errorf("containerd-socket volume should be hostPath")
	}
}

// envEntry returns the env entry (not just the value) for the given name.
func envEntry(envs []corev1.EnvVar, name string) (*corev1.EnvVar, bool) {
	for i := range envs {
		if envs[i].Name == name {
			return &envs[i], true
		}
	}
	return nil, false
}

// TestAgentPod_EnvInjection verifies FS-FUNC-007: the sandbox-agent container
// receives FLUXSANDBOXAGENT_CAPACITY, RUNTIME_TYPE, EXECD_BIN_PATH, and the
// Downward API env vars.
func testAgentPod_EnvInjection(t ginkgo.GinkgoTInterface) {
	name := "sg-cp-007-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, 2, withBuffer(0, 0), withAgentEnv("RUNTIME_TYPE", "containerd"), withExecd(cpExecdImage))
	createSandboxGroup(t, sg)
	waitAgentPodsReady(t, name, 1)

	pods, err := listAgentPods(name)
	if err != nil || len(pods) == 0 {
		t.Fatalf("no agent pods for %s: %v", name, err)
	}
	pod := &pods[0]

	mainContainer, ok := containerByName(pod.Spec.Containers, "sandbox-agent")
	if !ok {
		t.Fatalf("sandbox-agent container not found")
	}

	if v, ok := envVar(mainContainer.Env, "FLUXSANDBOXAGENT_CAPACITY"); !ok || v != "2" {
		t.Errorf("FLUXSANDBOXAGENT_CAPACITY: expected 2, got %q (ok=%v)", v, ok)
	}
	if v, ok := envVar(mainContainer.Env, "RUNTIME_TYPE"); !ok || v != "containerd" {
		t.Errorf("RUNTIME_TYPE: expected containerd, got %q (ok=%v)", v, ok)
	}
	if v, ok := envVar(mainContainer.Env, "EXECD_BIN_PATH"); !ok || v != "/var/lib/flux-sandbox/execd/"+name {
		t.Errorf("EXECD_BIN_PATH: expected /var/lib/flux-sandbox/execd/%s, got %q (ok=%v)", name, v, ok)
	}
	// Downward API vars are declared via fieldRef; the value is injected at
	// runtime and the Env entry's value field is empty in the pod spec. Assert
	// the fieldRef wiring instead of a non-empty value.
	for _, key := range []string{"NODE_NAME", "POD_NAME"} {
		e, ok := envEntry(mainContainer.Env, key)
		if !ok {
			t.Errorf("Downward API env %s: not found in container env", key)
			continue
		}
		if e.ValueFrom == nil || e.ValueFrom.FieldRef == nil || e.ValueFrom.FieldRef.FieldPath == "" {
			t.Errorf("Downward API env %s: expected fieldRef with non-empty fieldPath, got %+v", key, e.ValueFrom)
		}
	}
}

// TestAgentPod_AutoRebuild verifies FS-FUNC-011: deleting an Agent Pod causes the
// SSG controller to rebuild it, keeping desiredAgentPods unchanged.
func testAgentPod_AutoRebuild(t ginkgo.GinkgoTInterface) {
	name := "sg-cp-011-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, 2, withBuffer(0, 0))
	createSandboxGroup(t, sg)
	waitAgentPodsReady(t, name, 1)

	pods, err := listAgentPods(name)
	if err != nil || len(pods) == 0 {
		t.Fatalf("no agent pods for %s: %v", name, err)
	}
	oldPod := pods[0]
	oldUID := string(oldPod.UID)
	// Record desiredAgentPods before deletion.
	ssgName := getSandboxGroup(t, name).Status.SubSandboxGroups[0]
	ssgBefore := getSSG(t, ssgName)
	desiredBefore := ssgBefore.Status.DesiredAgentPods

	// Delete the Agent Pod (force).
	ctx := context.Background()
	if err := cpEnv.kubeClient.CoreV1().Pods(cpNamespace).Delete(ctx, oldPod.Name,
		metav1.DeleteOptions{}); err != nil {
		t.Fatalf("delete agent pod %s: %v", oldPod.Name, err)
	}

	// Wait for the agent pod to be rebuilt and ready. The rebuilt pod keeps the
	// SAME name (deterministic ordinal naming), so detect the rebuild by a new
	// UID rather than a different name.
	waitFor(t, "agent pod rebuilt and ready", 120*time.Second, func() bool {
		pods, err := listAgentPods(name)
		if err != nil || len(pods) == 0 {
			return false
		}
		for i := range pods {
			if string(pods[i].UID) != oldUID && isAgentPodReady(&pods[i]) {
				return true
			}
		}
		return false
	})

	// desiredAgentPods should remain unchanged (>= agentPodMin).
	ssgAfter := getSSG(t, ssgName)
	if ssgAfter.Status.DesiredAgentPods != desiredBefore {
		t.Fatalf("desiredAgentPods changed: before=%d after=%d", desiredBefore, ssgAfter.Status.DesiredAgentPods)
	}
}

// TestAgentPod_ExecdInitContainer verifies FS-FUNC-012: containerd runtime +
// execdImage generates an init-execd container that copies execd, and the main
// container mounts execd-shared -> /opt/opensandbox.
func testAgentPod_ExecdInitContainer(t ginkgo.GinkgoTInterface) {
	name := "sg-cp-012-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, 1, withBuffer(0, 0), withAgentEnv("RUNTIME_TYPE", "containerd"), withExecd(cpExecdImage))
	createSandboxGroup(t, sg)
	waitAgentPodsReady(t, name, 1)

	pods, err := listAgentPods(name)
	if err != nil || len(pods) == 0 {
		t.Fatalf("no agent pods for %s: %v", name, err)
	}
	pod := &pods[0]

	// init container init-execd, image = execdImage, command contains cp.
	var initFound bool
	for _, ic := range pod.Spec.InitContainers {
		if ic.Name == "init-execd" {
			initFound = true
			if ic.Image != cpExecdImage {
				t.Errorf("init container image: expected %s, got %s", cpExecdImage, ic.Image)
			}
			joined := strings.Join(append(ic.Command, ic.Args...), " ")
			if !strings.Contains(joined, "cp") {
				t.Errorf("init container command should contain cp: %s", joined)
			}
			break
		}
	}
	if !initFound {
		t.Fatalf("expected init-execd init container, got %+v", pod.Spec.InitContainers)
	}

	// Verify init container completed (exitCode 0).
	var initOK bool
	for _, ics := range pod.Status.InitContainerStatuses {
		if ics.Name == "init-execd" && ics.State.Terminated != nil && ics.State.Terminated.ExitCode == 0 {
			initOK = true
		}
	}
	if !initOK {
		t.Fatalf("init-execd not Completed/exitCode=0: %+v", pod.Status.InitContainerStatuses)
	}

	// Main container volumeMounts contain execd-shared -> /opt/opensandbox.
	main, ok := containerByName(pod.Spec.Containers, "sandbox-agent")
	if !ok {
		t.Fatalf("sandbox-agent container not found")
	}
	var execdMountFound bool
	for _, vm := range main.VolumeMounts {
		if vm.Name == "execd-shared" && vm.MountPath == "/opt/opensandbox" {
			execdMountFound = true
		}
	}
	if !execdMountFound {
		t.Fatalf("sandbox-agent volumeMounts missing execd-shared->/opt/opensandbox: %+v", main.VolumeMounts)
	}
}

// TestAgentPod_E2BNoExecd verifies FS-FUNC-013: E2B runtime with no execdImage
// produces no init container, no EXECD_BIN_PATH, and no execd-shared volume.
func testAgentPod_E2BNoExecd(t ginkgo.GinkgoTInterface) {
	// The agent with RUNTIME_TYPE=e2b requires the E2B CRI shim socket
	// (/run/cri-multiplex.sock) on the node; without it the agent container
	// crash-loops and the case times out. Skip on clusters without the E2B
	// runtime unless the operator explicitly opts in.
	if os.Getenv("E2E_E2B_ENABLED") == "" {
		t.Skip("cluster has no E2B CRI shim (/run/cri-multiplex.sock); set E2E_E2B_ENABLED=1 to force-run")
	}
	name := "sg-cp-013-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, 1, withBuffer(0, 0),
		withAgentEnv("RUNTIME_TYPE", "e2b"), withoutExecd())
	createSandboxGroup(t, sg)
	waitAgentPodsReady(t, name, 1)

	pods, err := listAgentPods(name)
	if err != nil || len(pods) == 0 {
		t.Fatalf("no agent pods for %s: %v", name, err)
	}
	pod := &pods[0]

	// No init-execd init container.
	for _, ic := range pod.Spec.InitContainers {
		if ic.Name == "init-execd" {
			t.Fatalf("E2B runtime should not generate init-execd, got %+v", ic)
		}
	}

	main, ok := containerByName(pod.Spec.Containers, "sandbox-agent")
	if !ok {
		t.Fatalf("sandbox-agent container not found")
	}
	// EXECD_BIN_PATH should be absent.
	if v, ok := envVar(main.Env, "EXECD_BIN_PATH"); ok && v != "" {
		t.Errorf("E2B runtime should not set EXECD_BIN_PATH, got %q", v)
	}
	// No execd-shared volume mount.
	for _, vm := range main.VolumeMounts {
		if vm.Name == "execd-shared" {
			t.Errorf("E2B runtime should not mount execd-shared, got %+v", vm)
		}
	}
}

// TestAgentPod_ContainerdNoExecdFails verifies FS-FUNC-014: containerd runtime
// with no execdImage still creates the SG + Agent Pod, but CreateSandbox fails
// because the bind-mount path /opt/opensandbox does not exist.
func testAgentPod_ContainerdNoExecdFails(t ginkgo.GinkgoTInterface) {
	name := "sg-cp-014-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, 1, withBuffer(0, 0),
		withAgentEnv("RUNTIME_TYPE", "containerd"), withoutExecd())
	createSandboxGroup(t, sg)
	waitAgentPodsReady(t, name, 1)
	// Wait until the scheduler has registered the agent (idle>0): creating
	// immediately after the pod turns Ready races the "no available agent"
	// path (allocation retries only last ~600ms).
	waitFor(t, "agent registered (idle>0)", 120*time.Second, func() bool {
		return ssgIdle(t, name+"-0") > 0
	})

	sbID := "sbx-cp-014-" + cpName(t)
	preCleanSandboxIDs(sbID)
	t.Cleanup(func() { deleteSandboxCR(sbID) })

	// CreateSandbox should return an error (bind mount path does not exist).
	_, err := grpcCreate(t, sbID, name)
	if err == nil {
		t.Fatalf("expected CreateSandbox to fail for containerd runtime without execdImage, but got nil error")
	}

	// The Agent logs should mention the missing bind-mount path.
	pod := getAgentPod(t, name)
	logs := kubectlLogs(t, cpNamespace, pod.Name, "sandbox-agent", 50)
	if !strings.Contains(logs, "/opt/opensandbox") && !strings.Contains(strings.ToLower(logs), "no such file") &&
		!strings.Contains(strings.ToLower(logs), "not exist") {
		t.Errorf("agent logs should mention the missing /opt/opensandbox path, got:\n%s", logs)
	}
}
