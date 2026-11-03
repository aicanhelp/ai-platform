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
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	sandboxv1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/proto/v1"
	apiv1alpha1 "gitcode.com/openFuyao/e2e-auto-test/e2e/flux-sandbox/api/v1alpha1"
)

// setup builds the shared control-plane e2e environment. It assumes the
// Controller and Scheduler are already deployed in the cluster (see the
// common preconditions in README_cp.md). It only applies the CRDs as a
// safety net, discovers the Controller Pod, waits for /readyz, starts
// port-forwards, and builds the k8s + gRPC clients. It runs from the suite's
// BeforeSuite.
func setup() (*testEnv, error) {
	initCPEnv()

	cfg, err := clientcmd.BuildConfigFromFlags("", cpKubeconfig)
	if err != nil {
		return nil, fmt.Errorf("build kubeconfig: %w", err)
	}

	scheme := k8sruntime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		return nil, err
	}
	if err := appsv1.AddToScheme(scheme); err != nil {
		return nil, err
	}
	if err := rbacv1.AddToScheme(scheme); err != nil {
		return nil, err
	}
	if err := apiv1alpha1.AddToScheme(scheme); err != nil {
		return nil, err
	}
	k8sClient, err := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		return nil, fmt.Errorf("create k8s client: %w", err)
	}

	kubeClient, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("create kube clientset: %w", err)
	}

	// Apply CRDs as a safety net (idempotent).
	crdApplied, crdErr := applyCRDs()
	if crdErr != nil {
		// Safety net only: if the CRDs are already installed the suite proceeds;
		// later setup steps fail with a clearer error if they are missing.
		log.Printf("applyCRDs safety net failed (continuing): %v", crdErr)
	}

	// Discover / wait for the Controller Pod.
	controllerPod, err := waitForControllerPod(kubeClient)
	if err != nil {
		return nil, fmt.Errorf("wait for controller pod: %w", err)
	}

	// Wait for Controller /readyz to return {"ready":true}.
	if err := waitControllerReadyz(controllerPod); err != nil {
		return nil, fmt.Errorf("controller not ready: %w", err)
	}

	e := &testEnv{
		k8sClient:     k8sClient,
		kubeClient:    kubeClient,
		controllerPod: controllerPod,
		crdApplied:    crdApplied,
	}

	// Port-forward Controller HTTP (8080) to localhost.
	httpPF, err := startPortForward(controllerPod, "8080", cpControllerHTTPPort)
	if err != nil {
		return nil, fmt.Errorf("port-forward controller http: %w", err)
	}
	e.httpPFCmd = httpPF

	// gRPC dials the controller Pod IP directly. The kubelet SPDY port-forward
	// to the gRPC port has been observed to die on the first connection
	// ("connection reset by peer" in the pod netns) while the port itself
	// listens fine; dialing the Pod IP avoids that hop entirely.
	controllerAddr, err := controllerPodAddr(kubeClient, controllerPod)
	if err != nil {
		return nil, fmt.Errorf("resolve controller pod address: %w", err)
	}

	// Build the gRPC client.
	conn, err := grpc.NewClient(controllerAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("grpc dial: %w", err)
	}
	e.grpcConn = conn
	e.grpcClient = sandboxv1.NewSandboxProviderServiceClient(conn)

	log.Printf("control-plane e2e environment ready: controller=%s addr=%s namespace=%s schedulerReplicas=%d",
		controllerPod, controllerAddr, cpNamespace, cpSchedulerReplicas)
	return e, nil
}

// controllerPodAddr resolves the controller pod IP and returns host:port for
// the gRPC (9091) endpoint.
func controllerPodAddr(kubeClient *kubernetes.Clientset, podName string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pod, err := kubeClient.CoreV1().Pods(cpNamespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	if pod.Status.PodIP == "" {
		return "", fmt.Errorf("controller pod %s has no IP yet", podName)
	}
	return net.JoinHostPort(pod.Status.PodIP, "9091"), nil
}

func teardown() {
	if cpEnv == nil {
		return
	}
	if cpEnv.grpcConn != nil {
		cpEnv.grpcConn.Close()
	}
	stopProc(cpEnv.httpPFCmd)

	if cpCleanup {
		if err := deleteAllTestSandboxGroups(cpEnv.k8sClient); err != nil {
			log.Printf("cleanup warning: %v", err)
		}
	}
}

func stopProc(p *os.Process) {
	if p == nil {
		return
	}
	if err := p.Kill(); err != nil {
		return
	}
	if _, err := p.Wait(); err != nil {
		// Already reaped or not our child; nothing to recover.
		log.Printf("stopProc: wait after kill: %v", err)
	}
}

// reconnectEnv rebuilds the in-process controller clients after the controller
// Pod has been replaced (destructive case FS-FUNC-037): the HTTP port-forward
// targets the old pod by name and the gRPC client dials the old pod IP, so
// both must be rebuilt or every later case in the same test binary fails with
// stale "dial tcp <old-ip>" / "lost connection to pod" errors.
func reconnectEnv(newPod string) error {
	stopProc(cpEnv.httpPFCmd)

	// Re-establish the controller HTTP port-forward on the new pod. The old
	// forward may still be releasing the local port, so retry until /healthz
	// answers through the new forward.
	deadline := time.Now().Add(60 * time.Second)
	for {
		httpPF, err := startPortForward(newPod, "8080", cpControllerHTTPPort)
		if err == nil {
			cpEnv.httpPFCmd = httpPF
			if code, _ := httpGetSilent(controllerHTTPURL + "/healthz"); code == 200 {
				break
			}
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("reconnect controller http port-forward to %s: not reachable within 60s", newPod)
		}
		time.Sleep(2 * time.Second)
	}

	// Re-dial the controller gRPC on the new pod IP.
	addr, err := controllerPodAddr(cpEnv.kubeClient, newPod)
	if err != nil {
		return fmt.Errorf("re-resolve controller pod address: %w", err)
	}
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("re-dial controller grpc: %w", err)
	}
	cpEnv.grpcConn.Close()
	cpEnv.grpcConn = conn
	cpEnv.grpcClient = sandboxv1.NewSandboxProviderServiceClient(conn)
	return nil
}

// applyCRDs runs `kubectl apply -f config/crd/bases/` from the repo root. The
// path is resolved relative to this file (two parents up to the repo root).
func applyCRDs() (bool, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return false, fmt.Errorf("cannot determine source file path")
	}
	crdDir := filepath.Join(filepath.Dir(file), "..", "..", "..", "config", "crd", "bases")
	if _, err := os.Stat(crdDir); err != nil {
		return false, nil // CRD dir not found; assume already applied in-cluster.
	}
	cmd := exec.Command("kubectl", "apply", "-f", crdDir)
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("kubectl apply crd (non-fatal): %v: %s", err, out)
		return false, err
	}
	return true, nil
}

// waitForControllerPod discovers (or waits for) the Controller Pod to be
// Running and returns its name. If E2E_CONTROLLER_POD is set, that name is used
// directly (after confirming it exists).
func waitForControllerPod(kubeClient *kubernetes.Clientset) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	if cpControllerPod != "" {
		return cpControllerPod, nil
	}

	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		pods, err := kubeClient.CoreV1().Pods(cpNamespace).List(ctx, metav1.ListOptions{
			LabelSelector: "app=flux-sandbox-controller",
		})
		if err == nil && len(pods.Items) > 0 {
			pod := &pods.Items[0]
			if pod.Status.Phase == corev1.PodRunning {
				return pod.Name, nil
			}
		}
		select {
		case <-ctx.Done():
			return "", fmt.Errorf("timeout waiting for controller pod")
		case <-ticker.C:
		}
	}
}

// waitControllerReadyz polls the Controller /readyz endpoint through the API
// server pod proxy (kubectl get --raw). This needs no tooling inside the
// controller container — the minimal runtime image has no wget/curl — and
// works regardless of network reachability to the Pod IP.
func waitControllerReadyz(controllerPod string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	proxyPath := fmt.Sprintf("/api/v1/namespaces/%s/pods/%s:8080/proxy/readyz", cpNamespace, controllerPod)
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		out, err := exec.CommandContext(ctx, "kubectl", "get", "--raw", proxyPath).Output()
		if err == nil {
			var resp struct {
				Ready bool `json:"ready"`
			}
			if json.Unmarshal(out, &resp) == nil && resp.Ready {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("controller /readyz not ready after 120s")
		case <-ticker.C:
		}
	}
}

// startPortForward starts `kubectl port-forward` for the given pod, mapping the
// pod port to a local port, and returns the started process. It waits briefly
// for the port-forward to become reachable.
func startPortForward(podName, podPort, localPort string) (*os.Process, error) {
	cmd := exec.Command("kubectl", "port-forward",
		"-n", cpNamespace, podName, localPort+":"+podPort)
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start port-forward %s:%s: %w", podName, podPort, err)
	}
	// Wait for the local port to be reachable.
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get("http://127.0.0.1:" + localPort + "/healthz")
		if err == nil {
			if _, copyErr := io.Copy(io.Discard, resp.Body); copyErr != nil {
				// Drain is best-effort; the status code is what matters.
				log.Printf("drain probe body: %v", copyErr)
			}
			if closeErr := resp.Body.Close(); closeErr != nil {
				log.Printf("close probe body: %v", closeErr)
			}
			if resp.StatusCode == 200 {
				return cmd.Process, nil
			}
		}
		time.Sleep(time.Second)
	}
	// /healthz may not exist on the gRPC port (9091); accept the port-forward as
	// ready after the timeout if the process is still alive.
	if cmd.Process == nil {
		return nil, fmt.Errorf("port-forward %s:%s process died", podName, podPort)
	}
	return cmd.Process, nil
}

// deleteAllTestSandboxGroups removes all SandboxGroup CRs created by the test
// suite (identified by the controller-owned label set by the test builder). It
// relies on ownerReferences cascading the deletion to SubSandboxGroups, Agent
// Pods and Sandboxes.
func deleteAllTestSandboxGroups(k8sClient client.Client) error {
	ctx := context.Background()
	list := &apiv1alpha1.SandboxGroupList{}
	if err := k8sClient.List(ctx, list, client.InNamespace(cpNamespace),
		client.MatchingLabels{testLabelKey: "true"}); err != nil {
		return err
	}
	var errs []error
	for i := range list.Items {
		sg := &list.Items[i]
		_ = k8sClient.Delete(ctx, sg)
		// Wait briefly for cascade deletion.
		if delErr := waitDelete(k8sClient, sg.Name, 60*time.Second); delErr != nil {
			// The Get below re-verifies deletion and reports leftovers.
			log.Printf("waitDelete %s: %v", sg.Name, delErr)
		}
		if getErr := k8sClient.Get(ctx, types.NamespacedName{Name: sg.Name, Namespace: cpNamespace}, &apiv1alpha1.SandboxGroup{}); getErr == nil {
			errs = append(errs, fmt.Errorf("sandboxgroup %s still present after delete", sg.Name))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("cleanup errors: %v", errs)
	}
	return nil
}

// waitDelete blocks until the SandboxGroup CR named sgName is gone or timeout.
func waitDelete(k8sClient client.Client, sgName string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for {
		err := k8sClient.Get(ctx, types.NamespacedName{Name: sgName, Namespace: cpNamespace}, &apiv1alpha1.SandboxGroup{})
		if err != nil && apierrors.IsNotFound(err) {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for sandboxgroup %s deletion", sgName)
		case <-ticker.C:
		}
	}
}
