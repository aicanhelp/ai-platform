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

func WaitInferNexServiceMode(ctx context.Context, dyn dynamic.Interface, fitID, ns, name, wantMode string, timeout time.Duration) {
	phase := fmt.Sprintf("infernexservice %s status.mode=%q", name, wantMode)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		obj, err := GetInferNexService(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		mode, found, err := unstructured.NestedString(obj.Object, "status", "mode")
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(found).To(BeTrue())
		LogPollThrottled(fitID, "insvc-mode", defaultPollLogInterval, "status.mode=%s (want %s)", mode, wantMode)
		if mode != wantMode {
			LogInferNexServiceStatusThrottled(ctx, dyn, fitID, ns, name)
		}
		g.Expect(mode).To(Equal(wantMode))
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

// WaitInferNexServiceSourceRefAndMode waits for spec.sourceRef.kind and status.mode (LLMISVC path).
func WaitInferNexServiceSourceRefAndMode(ctx context.Context, dyn dynamic.Interface, fitID, ns, name, wantKind, wantMode string, timeout time.Duration) {
	phase := fmt.Sprintf("infernexservice %s sourceRef.kind=%s status.mode=%s", name, wantKind, wantMode)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		obj, err := GetInferNexService(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		kind, _, _ := unstructured.NestedString(obj.Object, "spec", "sourceRef", "kind")
		mode, _, _ := unstructured.NestedString(obj.Object, "status", "mode")
		LogPollThrottled(fitID, "insvc-source-mode", defaultPollLogInterval,
			"sourceRef.kind=%q status.mode=%q (want kind=%s mode=%s)", kind, mode, wantKind, wantMode)
		if kind != wantKind || mode != wantMode {
			LogInferNexServiceStatusThrottled(ctx, dyn, fitID, ns, name)
		}
		g.Expect(kind).To(Equal(wantKind))
		g.Expect(mode).To(Equal(wantMode))
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
}

func WaitInferNexServiceReadyTrueWithOwnerFallback(ctx context.Context, dyn dynamic.Interface, client kubernetes.Interface, fitID, ns, name, owner string, timeout time.Duration) {
	phase := fmt.Sprintf("infernexservice %s status.ready=True (fallback owner=%s)", name, owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		obj, err := GetInferNexService(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		ready, found, err := unstructured.NestedBool(obj.Object, "status", "ready")
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(found).To(BeTrue())
		LogPollThrottled(fitID, "insvc-ready-fallback-"+name, defaultPollLogInterval, "infernexservice %s status.ready=%v owner=%s", name, ready, owner)
		if !ready {
			LogInferNexServiceStatusThrottled(ctx, dyn, fitID, ns, name)
			LogOwnerDeploymentsThrottled(ctx, client, fitID, ns, owner)
			RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, owner)
		}
		g.Expect(ready).To(BeTrue())
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

func WaitLLMInferenceServiceReadyTrueWithOwnerFallback(ctx context.Context, dyn dynamic.Interface, client kubernetes.Interface, fitID, ns, name, owner string, timeout time.Duration) {
	phase := fmt.Sprintf("llmisvc %s Ready=True (fallback owner=%s)", name, owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		obj, err := GetLLMInferenceService(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		conds, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(found).To(BeTrue())
		readyTrue := false
		for _, c := range conds {
			m, _ := c.(map[string]interface{})
			if m == nil {
				continue
			}
			t, _ := m["type"].(string)
			s, _ := m["status"].(string)
			if t == "Ready" && s == "True" {
				readyTrue = true
				break
			}
		}
		LogPollThrottled(fitID, "llmisvc-ready-fallback-"+name, defaultPollLogInterval, "llmisvc %s Ready condition true=%v owner=%s", name, readyTrue, owner)
		if !readyTrue {
			LogLLMISvcStatusThrottled(ctx, dyn, fitID, ns, name)
			RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, owner)
		}
		g.Expect(readyTrue).To(BeTrue())
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

func WaitDeploymentExists(ctx context.Context, client kubernetes.Interface, fitID, ns, deployName string, timeout time.Duration) {
	phase := fmt.Sprintf("deployment exists %s", deployName)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		_, err := client.AppsV1().Deployments(ns).Get(ctx, deployName, metav1.GetOptions{})
		if err != nil {
			LogPollThrottled(fitID, "deploy-exists-"+deployName, defaultPollLogInterval, "deployment %s: not found yet err=%v", deployName, err)
		}
		g.Expect(err).NotTo(HaveOccurred(), "deployment %s", deployName)
	}).WithTimeout(timeout).WithPolling(3 * time.Second).Should(Succeed())
	LogStep(fitID, "deployment exists: %s", deployName)
}

func WaitDeploymentReadyReplicas(ctx context.Context, client kubernetes.Interface, fitID, ns, deployName string, want int32, timeout time.Duration) {
	phase := fmt.Sprintf("deployment %s readyReplicas>=%d", deployName, want)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		d, err := client.AppsV1().Deployments(ns).Get(ctx, deployName, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "deployment %s", deployName)
		if d.Status.ReadyReplicas < want {
			LogDeploymentStatusThrottled(ctx, client, fitID, ns, deployName)
		} else {
			LogPollThrottled(fitID, "deploy-ready-"+deployName, defaultPollLogInterval,
				"deployment %s replicas=%d ready=%d want>=%d", deployName, d.Status.Replicas, d.Status.ReadyReplicas, want)
		}
		g.Expect(d.Status.ReadyReplicas).To(BeNumerically(">=", want))
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

func WaitDeploymentAbsent(ctx context.Context, client kubernetes.Interface, fitID, ns, deployName string, timeout time.Duration) {
	phase := fmt.Sprintf("deployment absent %s", deployName)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		_, err := client.AppsV1().Deployments(ns).Get(ctx, deployName, metav1.GetOptions{})
		g.Expect(isNotFound(err)).To(BeTrue(), "deployment %s should be gone", deployName)
	}).WithTimeout(timeout).WithPolling(3 * time.Second).Should(Succeed())
	LogStep(fitID, "deployment absent: %s", deployName)
}

func WaitDaemonSetExists(ctx context.Context, client kubernetes.Interface, fitID, ns, dsName string, timeout time.Duration) {
	phase := fmt.Sprintf("daemonset exists %s", dsName)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		_, err := client.AppsV1().DaemonSets(ns).Get(ctx, dsName, metav1.GetOptions{})
		if err != nil {
			LogPollThrottled(fitID, "ds-exists-"+dsName, defaultPollLogInterval, "daemonset %s: not found yet err=%v", dsName, err)
		}
		g.Expect(err).NotTo(HaveOccurred(), "daemonset %s", dsName)
	}).WithTimeout(timeout).WithPolling(3 * time.Second).Should(Succeed())
	LogStep(fitID, "daemonset exists: %s", dsName)
}

func WaitDaemonSetAbsent(ctx context.Context, client kubernetes.Interface, fitID, ns, dsName string, timeout time.Duration) {
	phase := fmt.Sprintf("daemonset absent %s", dsName)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		_, err := client.AppsV1().DaemonSets(ns).Get(ctx, dsName, metav1.GetOptions{})
		g.Expect(isNotFound(err)).To(BeTrue(), "daemonset %s should be gone", dsName)
	}).WithTimeout(timeout).WithPolling(3 * time.Second).Should(Succeed())
	LogStep(fitID, "daemonset absent: %s", dsName)
}

func AssertNoDeploymentSubstring(ctx context.Context, client kubernetes.Interface, ns, substr string) {
	list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	for _, d := range list.Items {
		Expect(d.Name).NotTo(ContainSubstring(substr), "unexpected deploy %s", d.Name)
	}
}

func WaitOwnerResourcesGone(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	phase := fmt.Sprintf("owner deployments gone owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("infernex.io/owner=%s", owner),
		})
		g.Expect(err).NotTo(HaveOccurred())
		if len(list.Items) > 0 {
			LogPollThrottled(fitID, "owner-gone", defaultPollLogInterval, "owner=%s still has %d deployment(s)", owner, len(list.Items))
			for _, d := range list.Items {
				LogStep(fitID, "  remaining deploy %s ready=%d", d.Name, d.Status.ReadyReplicas)
			}
		}
		g.Expect(list.Items).To(BeEmpty(), "owner deployments remain")
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
	Eventually(func(g Gomega) {
		list, err := client.AppsV1().DaemonSets(ns).List(ctx, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("infernex.io/owner=%s", owner),
		})
		g.Expect(err).NotTo(HaveOccurred())
		if len(list.Items) > 0 {
			LogPollThrottled(fitID, "owner-ds-gone", defaultPollLogInterval, "owner=%s still has %d daemonset(s)", owner, len(list.Items))
			for _, ds := range list.Items {
				LogStep(fitID, "  remaining daemonset %s ready=%d", ds.Name, ds.Status.NumberReady)
			}
		}
		g.Expect(list.Items).To(BeEmpty(), "owner daemonsets remain")
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
	LogStep(fitID, "no deployments or daemonsets with infernex.io/owner=%s", owner)
}

func ComponentReady(obj *unstructured.Unstructured, component string) (bool, error) {
	ready, found, err := unstructured.NestedBool(obj.Object, "status", "components", component, "ready")
	return found && ready, err
}

func WaitComponentReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name, component string, timeout time.Duration) {
	phase := fmt.Sprintf("infernexservice %s component %s ready", name, component)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		obj, err := GetInferNexService(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		ok, err := ComponentReady(obj, component)
		g.Expect(err).NotTo(HaveOccurred())
		if !ok {
			LogPollThrottled(fitID, "component-"+component, defaultPollLogInterval,
				"component %s not ready yet", component)
			LogInferNexServiceStatusThrottled(ctx, dyn, fitID, ns, name)
		}
		g.Expect(ok).To(BeTrue(), "status.components.%s.ready", component)
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

func DeployName(owner, suffix string) string {
	return owner + "-" + suffix
}

// BridgeWorkloadDeployName resolves the Bridge-managed Deployment name for a component suffix.
// mooncake-master/mooncake-metadata are cluster-singleton workloads shared by all
// InferNexService instances (labels infernex.io/singleton=mooncake), so their names
// carry no owner prefix.
func BridgeWorkloadDeployName(owner, suffix string) string {
	switch suffix {
	case "mooncake-master":
		return "mooncake-master-deployment"
	case "mooncake-metadata":
		return "redis-server-deployment"
	default:
		return DeployName(owner, suffix)
	}
}

func WaitInferNexServiceExists(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	phase := fmt.Sprintf("infernexservice %s exists", name)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		exists, err := InferNexServiceExists(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		if !exists {
			LogPollThrottled(fitID, "insvc-exists-"+name, defaultPollLogInterval, "infernexservice %s not found yet", name)
		}
		g.Expect(exists).To(BeTrue())
	}).WithTimeout(timeout).WithPolling(time.Second).Should(Succeed())
}

func WaitInferNexServiceAbsent(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	phase := fmt.Sprintf("infernexservice %s absent", name)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		exists, err := InferNexServiceExists(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		if exists {
			LogPollThrottled(fitID, "insvc-absent-"+name, defaultPollLogInterval, "infernexservice %s still exists", name)
		}
		g.Expect(exists).To(BeFalse())
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

// WaitInferNexCondition waits until status.conditions contains condType with status/reason.
func WaitInferNexCondition(ctx context.Context, dyn dynamic.Interface, fitID, ns, name, condType, wantStatus, wantReason string, timeout time.Duration) {
	phase := fmt.Sprintf("infernexservice %s condition %s status=%s reason=%s", name, condType, wantStatus, wantReason)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	Eventually(func(g Gomega) {
		obj, err := GetInferNexService(ctx, dyn, ns, name)
		g.Expect(err).NotTo(HaveOccurred())
		st, reason, msg, ok := ConditionStatus(obj, condType)
		g.Expect(ok).To(BeTrue(), "condition %s missing", condType)
		if st != wantStatus || reason != wantReason {
			LogPollThrottled(fitID, "cond-"+condType, defaultPollLogInterval,
				"%s status=%s reason=%s msg=%s (want status=%s reason=%s)", condType, st, reason, msg, wantStatus, wantReason)
			LogInferNexServiceStatusThrottled(ctx, dyn, fitID, ns, name)
		}
		g.Expect(st).To(Equal(wantStatus))
		if wantReason != "" {
			g.Expect(reason).To(Equal(wantReason))
		}
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}
