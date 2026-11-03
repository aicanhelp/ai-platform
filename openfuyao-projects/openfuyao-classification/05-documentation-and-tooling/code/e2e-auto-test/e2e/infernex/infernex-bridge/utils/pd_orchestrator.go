package utils

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// WaitPDOrchestratorDeploysReady waits for all three pd-orchestrator Deployments to reach readyReplicas>=1.
func WaitPDOrchestratorDeploysReady(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	LogPhaseBegin(fitID, fmt.Sprintf("pd-orchestrator deploys ready owner=%s", owner))
	defer LogPhaseEnd(fitID, fmt.Sprintf("pd-orchestrator deploys ready owner=%s", owner))
	for i, suffix := range PDOrchestratorDeploySuffixes() {
		deploy := DeployName(owner, suffix)
		LogStep(fitID, "pd-orchestrator step %d/%d: %s", i+1, len(PDOrchestratorDeploySuffixes()), deploy)
		if SkipNPUWorkloadE2E() {
			WaitDeploymentExists(ctx, client, fitID, ns, deploy, timeout)
			continue
		}
		WaitDeploymentReadyReplicas(ctx, client, fitID, ns, deploy, 1, timeout)
	}
}

// WaitNoPDOrchestratorWorkloads waits until no pd-orchestrator Deployments or pods exist for owner.
func WaitNoPDOrchestratorWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	LogPhaseBegin(fitID, fmt.Sprintf("no pd-orchestrator workloads owner=%s", owner))
	defer LogPhaseEnd(fitID, fmt.Sprintf("no pd-orchestrator workloads owner=%s", owner))
	prefix := owner + "-pd-orchestrator"
	Eventually(func(g Gomega) {
		list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		for _, d := range list.Items {
			g.Expect(d.Name).NotTo(ContainSubstring(prefix), "unexpected deployment %s", d.Name)
		}
		pods, err := client.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("infernex.io/owner=%s", owner),
		})
		g.Expect(err).NotTo(HaveOccurred())
		for _, p := range pods.Items {
			g.Expect(p.Name).NotTo(ContainSubstring("pd-orchestrator"),
				"unexpected pd-orchestrator pod %s phase=%s", p.Name, p.Status.Phase)
		}
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
	LogStep(fitID, "no pd-orchestrator deployments or pods for owner=%s", owner)
}

// AssertNoPDOrchestratorWorkloads asserts no pd-orchestrator Deployments or Running pods for owner.
func AssertNoPDOrchestratorWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string) {
	prefix := owner + "-pd-orchestrator"
	AssertNoDeploymentSubstring(ctx, client, ns, prefix)
	list, err := client.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("infernex.io/owner=%s", owner),
	})
	Expect(err).NotTo(HaveOccurred())
	for _, p := range list.Items {
		Expect(p.Name).NotTo(ContainSubstring("pd-orchestrator"),
			"unexpected pd-orchestrator pod %s phase=%s", p.Name, p.Status.Phase)
	}
	LogStep(fitID, "no pd-orchestrator deployments or pods for owner=%s", owner)
}

func componentStatusPresent(obj *unstructured.Unstructured, component string) (bool, error) {
	_, found, err := unstructured.NestedMap(obj.Object, "status", "components", component)
	return found, err
}

// WaitPDOrchestratorComponentReady waits for status.components.pdOrchestrator.ready=true.
func WaitPDOrchestratorComponentReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	WaitComponentReady(ctx, dyn, fitID, ns, name, "pdOrchestrator", timeout)
}

// WaitPDOrchestratorComponentNotReady waits until pdOrchestrator is absent or ready=false.
func WaitPDOrchestratorComponentNotReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	phase := fmt.Sprintf("infernexservice %s component pdOrchestrator not ready", name)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		obj, err := GetInferNexService(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		present, err := componentStatusPresent(obj, "pdOrchestrator")
		g.Expect(err).NotTo(HaveOccurred())
		if !present {
			return
		}
		ok, err := ComponentReady(obj, "pdOrchestrator")
		g.Expect(err).NotTo(HaveOccurred())
		if ok {
			LogInferNexServiceStatusThrottled(ctx, dyn, fitID, ns, name)
		}
		g.Expect(ok).To(BeFalse(), "status.components.pdOrchestrator.ready should be false or absent")
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}
