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
	"fmt"
	"os/exec"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// ── Scenario group: Controller ConfigMap ──

// controllerConfigMapName is the ConfigMap holding the agent defaults.
const controllerConfigMapName = "flux-sandbox-controller-config"

// TestControllerConfig_DefaultConfigMap verifies FS-FUNC-036: the
// flux-sandbox-controller-config ConfigMap contains the agent image/port keys,
// and a freshly created Agent Pod's main container image and port match them.
func testControllerConfig_DefaultConfigMap(t ginkgo.GinkgoTInterface) {
	ctx := context.Background()
	cm, err := cpEnv.kubeClient.CoreV1().ConfigMaps(cpNamespace).Get(ctx,
		controllerConfigMapName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get ConfigMap %s: %v", controllerConfigMapName, err)
	}
	for _, key := range []string{"defaultSandboxAgentImage", "defaultSandboxAgentPort"} {
		if _, ok := cm.Data[key]; !ok {
			t.Errorf("ConfigMap missing key %s, data=%+v", key, cm.Data)
		}
	}
	wantImage := cm.Data["defaultSandboxAgentImage"]
	wantPort := cm.Data["defaultSandboxAgentPort"]

	name := "sg-cp-036-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, 2, withBuffer(0, 0))
	createSandboxGroup(t, sg)
	waitAgentPodsReady(t, name, 1)

	pods, err := listAgentPods(name)
	if err != nil || len(pods) == 0 {
		t.Fatalf("no agent pods for %s: %v", name, err)
	}
	pod := &pods[0]
	main, ok := containerByName(pod.Spec.Containers, "sandbox-agent")
	if !ok {
		t.Fatalf("sandbox-agent container not found")
	}
	if main.Image != wantImage {
		t.Errorf("sandbox-agent image: expected %s (from ConfigMap), got %s", wantImage, main.Image)
	}
	if len(main.Ports) == 0 {
		t.Fatalf("sandbox-agent container has no ports")
	}
	if got := int(main.Ports[0].ContainerPort); got != atoiOr(t, wantPort, 5758) {
		t.Errorf("sandbox-agent containerPort: expected %s, got %d", wantPort, got)
	}
}

// atoiOr parses s as an int, returning fallback on error.
func atoiOr(t ginkgo.GinkgoTInterface, s string, fallback int) int {
	t.Helper()
	if s == "" {
		return fallback
	}
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return fallback
		}
		n = n*10 + int(s[i]-'0')
	}
	return n
}

// TestControllerConfig_MissingConfigMapFallback verifies FS-FUNC-037: deleting
// the ConfigMap and restarting the Controller logs a fallback warning, and the
// Controller still starts and can create Agent Pods using built-in defaults.
//
// Destructive: deletes the ConfigMap and restarts the Controller. Requires
// E2E_DESTRUCTIVE. The ConfigMap is restored in cleanup.
func testControllerConfig_MissingConfigMapFallback(t ginkgo.GinkgoTInterface) {
	requireDestructive(t)
	ctx := context.Background()

	// Snapshot the ConfigMap to restore in cleanup.
	original, err := cpEnv.kubeClient.CoreV1().ConfigMaps(cpNamespace).Get(ctx,
		controllerConfigMapName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("get ConfigMap for snapshot: %v", err)
	}
	t.Cleanup(func() {
		// Recreate the ConfigMap (CreateOrUpdate) and restart the Controller.
		_, getErr := cpEnv.kubeClient.CoreV1().ConfigMaps(cpNamespace).Get(ctx,
			controllerConfigMapName, metav1.GetOptions{})
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: controllerConfigMapName, Namespace: cpNamespace},
			Data:       original.Data,
		}
		if getErr != nil {
			_, _ = cpEnv.kubeClient.CoreV1().ConfigMaps(cpNamespace).Create(ctx, cm, metav1.CreateOptions{})
		} else {
			_, _ = cpEnv.kubeClient.CoreV1().ConfigMaps(cpNamespace).Update(ctx, cm, metav1.UpdateOptions{})
		}
		if err := restartController(); err != nil {
			t.Logf("restart controller in cleanup: %v", err)
		}
		if err := waitControllerReadyz(cpEnv.controllerPod); err != nil {
			t.Logf("wait controller readyz after cleanup restart: %v", err)
		}
	})

	// Delete the ConfigMap.
	if err := cpEnv.kubeClient.CoreV1().ConfigMaps(cpNamespace).Delete(ctx,
		controllerConfigMapName, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("delete ConfigMap: %v", err)
	}

	// Restart the Controller.
	if err := restartController(); err != nil {
		t.Fatalf("restart controller: %v", err)
	}
	if err := waitControllerReadyz(cpEnv.controllerPod); err != nil {
		t.Fatalf("controller not ready after restart: %v", err)
	}

	// Controller logs should mention the fallback warning.
	logs := kubectlLogs(t, cpNamespace, cpEnv.controllerPod, "", 50)
	if !strings.Contains(logs, "agent defaults configmap not found") {
		t.Errorf("expected controller logs to mention fallback, got:\n%s", logs)
	}

	// The Controller should still create an Agent Pod using built-in defaults.
	name := "sg-cp-037-" + cpName(t)
	sg := basicSandboxGroup(name, 1, 1, 2, withBuffer(0, 0))
	createSandboxGroup(t, sg)
	waitFor(t, "SandboxGroup Ready after ConfigMap removal", 120*time.Second, func() bool {
		return getSandboxGroup(t, name).Status.Phase == apiv1alpha1.PhaseReady
	})
	waitAgentPodsReady(t, name, 1)
}

// restartController performs a rollout restart of the Controller Deployment and
// waits for a new Running Pod.
func restartController() error {
	if _, err := execKubectl("rollout", "restart", "deployment/flux-sandbox-controller",
		"-n", cpNamespace); err != nil {
		return err
	}
	// Wait for the rollout to complete.
	if err := execKubectlRun("rollout", "status", "deployment/flux-sandbox-controller",
		"-n", cpNamespace, "--timeout=120s"); err != nil {
		return err
	}
	// Discover the new controller pod. The name-sorted pod list can return the
	// previous pod first while it is still Terminating, and probing it makes
	// the readyz wait below time out on a dead pod. Select the youngest
	// Running pod without a deletion timestamp instead, retrying briefly in
	// case the replacement pod is not observed Running yet.
	deadline := time.Now().Add(30 * time.Second)
	for {
		pods, listErr := cpEnv.kubeClient.CoreV1().Pods(cpNamespace).List(context.Background(),
			metav1.ListOptions{LabelSelector: "app=flux-sandbox-controller"})
		if listErr != nil {
			return listErr
		}
		var newest *corev1.Pod
		for i := range pods.Items {
			pod := &pods.Items[i]
			if pod.DeletionTimestamp != nil || pod.Status.Phase != corev1.PodRunning {
				continue
			}
			if newest == nil || pod.CreationTimestamp.After(newest.CreationTimestamp.Time) {
				newest = pod
			}
		}
		if newest != nil {
			cpEnv.controllerPod = newest.Name
			// The Pod was replaced: rebuild the port-forward and gRPC clients
			// so the rest of the suite in this process keeps working.
			return reconnectEnv(newest.Name)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("no running controller pod found after restart")
		}
		time.Sleep(2 * time.Second)
	}
}

// execKubectl runs kubectl with the given args and returns combined output.
func execKubectl(args ...string) (string, error) {
	out, err := exec.Command("kubectl", args...).CombinedOutput()
	return string(out), err
}

// execKubectlRun runs kubectl and discards output, returning only the error.
func execKubectlRun(args ...string) error {
	_, err := execKubectl(args...)
	return err
}
