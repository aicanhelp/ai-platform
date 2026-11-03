//go:build e2e

/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package warmup

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	ginkgo "github.com/onsi/ginkgo/v2"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// wControllerDeployName is the Deployment name of the existing controller.
var wControllerDeployName = envOrDefault("WARM_E2E_CONTROLLER_DEPLOYMENT", "flux-sandbox-controller")

// defaultControllerNamespace returns the default controller namespace.
func defaultControllerNamespace() string {
	return "flux-system"
}

// gateEnabled reports whether a gated spec group should run. The variable
// wins when set: truthy values enable, "0"/"false"/"no"/"off" disable. When
// unset, all specs run.
func gateEnabled(key string) bool {
	switch strings.ToLower(os.Getenv(key)) {
	case "":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

// warmFlags returns the canonical --warm-* argument set pointing at the
// suite-owned mock services.
func warmFlags() []string {
	return []string{
		"--warm-enabled=true",
		fmt.Sprintf("--warm-postgres-dsn=postgres://postgres:warm@warm-postgres.%s.svc:5432/warm?sslmode=disable", wNamespace),
		"--warm-etcd-endpoints=http://warm-etcd." + wNamespace + ".svc:2379",
		"--warm-e2b-api-endpoint=http://fake-e2b." + wNamespace + ".svc:8080",
		"--warm-superpod-ids=" + strings.Join(wSuperPodIDs, ","),
		"--warm-regular-target-count=2",
		fmt.Sprintf("--warm-task-poll-interval=%s", wPollInterval),
		fmt.Sprintf("--warm-task-stale-timeout=%s", wStaleTimeout),
		fmt.Sprintf("--warm-etcd-staleness=%s", wStaleness),
	}
}

// stripWarmFlags removes every --warm-* argument from args.
func stripWarmFlags(args []string) []string {
	var kept []string
	for _, a := range args {
		if strings.HasPrefix(a, "--warm-") {
			continue
		}
		kept = append(kept, a)
	}
	return kept
}

// mergeWarmFlags strips existing --warm-* flags and appends the canonical set.
func mergeWarmFlags(args []string) []string {
	return append(stripWarmFlags(args), warmFlags()...)
}

// attachController patches the existing controller Deployment so its warm
// flags point at the suite mock services, and records the original spec for
// teardown restore. Non-warm args are preserved untouched.
func (e *warmEnv) attachController(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	dep, err := e.kube.AppsV1().Deployments(wControllerNS).Get(e.ctx, wControllerDeployName, metav1.GetOptions{})
	if err != nil {
		tb.Fatalf("find controller deployment %s/%s: %v", wControllerNS, wControllerDeployName, err)
	}
	if len(dep.Spec.Template.Spec.Containers) == 0 {
		tb.Fatalf("deployment %s/%s has no containers", wControllerNS, wControllerDeployName)
	}
	e.origController = dep.DeepCopy()
	dep.Spec.Template.Spec.Containers[0].Args = mergeWarmFlags(dep.Spec.Template.Spec.Containers[0].Args)
	if _, err := e.kube.AppsV1().Deployments(wControllerNS).Update(e.ctx, dep, metav1.UpdateOptions{}); err != nil {
		tb.Fatalf("patch controller deployment: %v", err)
	}
	e.waitForControllerAvailable(tb, 5*time.Minute)
}

// waitForControllerAvailable blocks until the controller Deployment reports
// an available updated replica set, failing the test on timeout.
func (e *warmEnv) waitForControllerAvailable(tb ginkgo.GinkgoTInterface, timeout time.Duration) {
	tb.Helper()
	waitFor(tb, fmt.Sprintf("controller %s available", wControllerDeployName), timeout, func() bool {
		d, err := e.kube.AppsV1().Deployments(wControllerNS).Get(e.ctx, wControllerDeployName, metav1.GetOptions{})
		if err != nil {
			return false
		}
		return d.Status.ObservedGeneration >= d.Generation &&
			d.Status.UpdatedReplicas >= 1 &&
			d.Status.UnavailableReplicas == 0 &&
			d.Status.AvailableReplicas >= 1
	})
}

// restoreController re-applies the controller Deployment captured at attach
// time and waits (non-fatally) for it to roll out, so teardown can continue
// cleaning up the mock namespace even if the restored controller cannot
// become ready.
func (e *warmEnv) restoreController(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	if e.origController == nil {
		return
	}
	orig := e.origController.DeepCopy()
	live, err := e.kube.AppsV1().Deployments(wControllerNS).Get(e.ctx, wControllerDeployName, metav1.GetOptions{})
	if err != nil {
		tb.Errorf("restore controller: live deployment unavailable: %v", err)
		return
	}
	orig.ResourceVersion = live.ResourceVersion
	if _, err := e.kube.AppsV1().Deployments(wControllerNS).Update(e.ctx, orig, metav1.UpdateOptions{}); err != nil {
		tb.Errorf("restore controller deployment: %v", err)
		return
	}
	deadline := time.Now().Add(5 * time.Minute)
	for {
		d, err := e.kube.AppsV1().Deployments(wControllerNS).Get(e.ctx, wControllerDeployName, metav1.GetOptions{})
		if err == nil && d.Status.ObservedGeneration >= d.Generation &&
			d.Status.UpdatedReplicas >= 1 && d.Status.UnavailableReplicas == 0 &&
			d.Status.AvailableReplicas >= 1 {
			tb.Log("controller deployment restored and available")
			return
		}
		if time.Now().After(deadline) {
			tb.Errorf("controller %s not available 5m after restore; inspect the deployment before re-running",
				wControllerDeployName)
			return
		}
		time.Sleep(2 * time.Second)
	}
}

// controllerPodSelector returns the namespace and pod label selector used to
// port-forward the controller HTTP port.
func (e *warmEnv) controllerPodSelector(tb ginkgo.GinkgoTInterface) (ns, selector string) {
	tb.Helper()
	dep, err := e.kube.AppsV1().Deployments(wControllerNS).Get(e.ctx, wControllerDeployName, metav1.GetOptions{})
	if err != nil {
		tb.Fatalf("find controller deployment %s/%s: %v", wControllerNS, wControllerDeployName, err)
	}
	var parts []string
	for k, v := range dep.Spec.Selector.MatchLabels {
		parts = append(parts, k+"="+v)
	}
	return wControllerNS, strings.Join(parts, ",")
}

// runFailFastPod starts a one-off controller pod with --warm-enabled=true
// but without --warm-postgres-dsn in the controller namespace, reusing the
// deployment's image and ServiceAccount so the crash is the same fail-fast
// the real controller would hit, and registers its cleanup. The existing
// controller Deployment is never modified.
func (e *warmEnv) runFailFastPod(tb ginkgo.GinkgoTInterface) string {
	tb.Helper()
	dep, err := e.kube.AppsV1().Deployments(wControllerNS).Get(e.ctx, wControllerDeployName, metav1.GetOptions{})
	if err != nil {
		tb.Fatalf("find controller deployment %s/%s: %v", wControllerNS, wControllerDeployName, err)
	}
	src := dep.Spec.Template.Spec.Containers[0]
	saName := dep.Spec.Template.Spec.ServiceAccountName
	name := "warm-failfast-check"
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: wControllerNS,
			Labels:    map[string]string{"app": name},
		},
		Spec: corev1.PodSpec{
			ServiceAccountName: saName,
			RestartPolicy:      corev1.RestartPolicyAlways,
			Containers: []corev1.Container{{
				Name:            "controller",
				Image:           src.Image,
				ImagePullPolicy: corev1.PullIfNotPresent,
				Args:            append(stripWarmFlags(src.Args), "--warm-enabled=true"),
			}},
		},
	}
	e.upsert(tb, pod, name)
	ginkgo.DeferCleanup(func() {
		_ = e.kube.CoreV1().Pods(wControllerNS).Delete(e.ctx, name, metav1.DeleteOptions{})
	})
	return name
}

// deleteMockNamespace deletes the mock namespace and waits (best effort) for
// it to drain, so a stuck finalizer does not hang the suite forever. When the
// static-PV fallback was used, the hostPath PV is removed afterwards so the
// cluster is left with no suite-owned storage; next runs simply recreate it.
func (e *warmEnv) deleteMockNamespace(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	err := e.kube.CoreV1().Namespaces().Delete(e.ctx, wNamespace, metav1.DeleteOptions{})
	if err != nil {
		if !apierrors.IsNotFound(err) {
			tb.Logf("delete namespace %s: %v", wNamespace, err)
		}
		return
	}
	waitCtx, cancel := context.WithTimeout(e.ctx, 3*time.Minute)
	defer cancel()
	timer := time.NewTimer(2 * time.Second)
	defer timer.Stop()
	for {
		_, err := e.kube.CoreV1().Namespaces().Get(waitCtx, wNamespace, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			tb.Log("mock namespace deleted")
			e.deleteStaticPV(tb)
			return
		}
		select {
		case <-waitCtx.Done():
			tb.Logf("namespace %s still terminating after 3m; ignoring", wNamespace)
			e.deleteStaticPV(tb)
			return
		case <-timer.C:
			timer.Reset(2 * time.Second)
		}
	}
}

// deleteStaticPV removes the suite-created hostPath PV. The PVC must be gone
// first, so this only deletes after the PVC disappeared; a PV deleted while
// still bound hangs in Terminating. Retain reclaim means no data is touched
// outside the test-owned hostPath anyway.
func (e *warmEnv) deleteStaticPV(tb ginkgo.GinkgoTInterface) {
	tb.Helper()
	if !e.staticPVCreated {
		return
	}
	// The namespace (and with it the PVC) may still be terminating: a PV
	// delete issued before the PVC is gone blocks in Terminating.
	waitCtx, cancel := context.WithTimeout(e.ctx, 2*time.Minute)
	defer cancel()
	for {
		_, err := e.kube.CoreV1().PersistentVolumeClaims(wNamespace).Get(waitCtx, wPGPVCName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			break
		}
		if err != nil {
			tb.Logf("get pvc %s during PV cleanup: %v", wPGPVCName, err)
			return
		}
		select {
		case <-waitCtx.Done():
			tb.Logf("pvc %s still present after 2m; leaving PV %s in place", wPGPVCName, wPGPVName)
			return
		case <-time.After(2 * time.Second):
		}
	}
	if err := e.kube.CoreV1().PersistentVolumes().Delete(waitCtx, wPGPVName, metav1.DeleteOptions{}); err != nil {
		if !apierrors.IsNotFound(err) {
			tb.Logf("delete static PV %s: %v", wPGPVName, err)
		}
		return
	}
	tb.Logf("static PV %s deleted", wPGPVName)
}
