package utils

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// KServe LLMInferenceService workload labels (see kserve/pkg/constants and llmisvc/scheduler.go).
const (
	kserveAppPartOf                        = "llminferenceservice"
	kserveWorkloadComponentAggregate       = "llminferenceservice-workload"
	kserveWorkloadComponentPrefill         = "llminferenceservice-workload-prefill"
	kserveWorkloadComponentDecode          = "llminferenceservice-workload"
	kserveWorkloadComponentRouterScheduler = "llminferenceservice-router-scheduler"
)

// KServe deployment name suffixes (kmeta.ChildName(llmName, suffix)).
const (
	kserveSuffixAggregateWorkload = "-kserve"
	kserveSuffixPrefillWorkload   = "-kserve-prefill"
	kserveSuffixRouterScheduler   = "-kserve-router-scheduler"
)

// KServeSchedulerDeployName is the Hermes/EPP router Deployment created by KServe (not Bridge hermes-router).
func KServeSchedulerDeployName(llmName string) string {
	return llmName + kserveSuffixRouterScheduler
}

// KServeAggregateWorkloadDeployName is the aggregate inference Deployment from KServe.
func KServeAggregateWorkloadDeployName(llmName string) string {
	return llmName + kserveSuffixAggregateWorkload
}

// KServePrefillWorkloadDeployName is the PD prefill Deployment from KServe.
func KServePrefillWorkloadDeployName(llmName string) string {
	return llmName + kserveSuffixPrefillWorkload
}

// KServePDDecodeWorkloadDeployName is the PD decode Deployment (same suffix as aggregate: *-kserve).
func KServePDDecodeWorkloadDeployName(llmName string) string {
	return KServeAggregateWorkloadDeployName(llmName)
}

// KServePDInferenceEngineDeployNames returns all KServe inference engine Deployments for PD LLMISVC.
func KServePDInferenceEngineDeployNames(llmName string) []string {
	return []string{
		KServePrefillWorkloadDeployName(llmName),
		KServePDDecodeWorkloadDeployName(llmName),
	}
}

// AggregateLLMISvcBridgeDeploySuffixes are Bridge-managed Deployments when InferNexService has sourceRef.
// Router/scheduler is KServe-managed — not hermes-router.
func AggregateLLMISvcBridgeDeploySuffixes() []string {
	return appendOptionalBridgeDeploySuffixes([]string{
		"cache-indexer",
		"mooncake-master",
		"mooncake-metadata",
	})
}

// PDLLMISvcBridgeDeploySuffixes are Bridge-managed Deployments for PD LLMISVC (KServe owns prefill/decode/router).
func PDLLMISvcBridgeDeploySuffixes() []string {
	return appendOptionalBridgeDeploySuffixes([]string{
		"proxy-server",
		"cache-indexer",
		"mooncake-master",
		"mooncake-metadata",
	})
}

func kserveDeploymentSelector(llmName, component string) string {
	return fmt.Sprintf("app.kubernetes.io/name=%s,app.kubernetes.io/part-of=%s,app.kubernetes.io/component=%s",
		llmName, kserveAppPartOf, component)
}

// WaitKServeLLMInferenceWorkloads waits for KServe aggregate inference + router scheduler Deployments.
func WaitKServeLLMInferenceWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, llmName string, timeout time.Duration) {
	LogPhaseBegin(fitID, fmt.Sprintf("KServe LLMISVC aggregate workloads llm=%s", llmName))
	defer LogPhaseEnd(fitID, fmt.Sprintf("KServe LLMISVC aggregate workloads llm=%s", llmName))
	waitKServeComponent(ctx, client, fitID, ns, llmName, kserveWorkloadComponentAggregate,
		KServeAggregateWorkloadDeployName(llmName), timeout)
	waitKServeComponent(ctx, client, fitID, ns, llmName, kserveWorkloadComponentRouterScheduler,
		KServeSchedulerDeployName(llmName), timeout)
}

// WaitKServePDInferenceWorkloads waits for KServe PD prefill, decode, and router scheduler Deployments.
func WaitKServePDInferenceWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, llmName string, timeout time.Duration) {
	LogPhaseBegin(fitID, fmt.Sprintf("KServe LLMISVC PD workloads llm=%s", llmName))
	defer LogPhaseEnd(fitID, fmt.Sprintf("KServe LLMISVC PD workloads llm=%s", llmName))
	waitKServeComponent(ctx, client, fitID, ns, llmName, kserveWorkloadComponentPrefill,
		KServePrefillWorkloadDeployName(llmName), timeout)
	waitKServeComponent(ctx, client, fitID, ns, llmName, kserveWorkloadComponentDecode,
		KServeAggregateWorkloadDeployName(llmName), timeout)
	waitKServeComponent(ctx, client, fitID, ns, llmName, kserveWorkloadComponentRouterScheduler,
		KServeSchedulerDeployName(llmName), timeout)
}

func waitKServeComponent(ctx context.Context, client kubernetes.Interface, fitID, ns, llmName, component, expectedDeployName string, timeout time.Duration) {
	if component == kserveWorkloadComponentRouterScheduler {
		// router-scheduler (EPP) carries no model weights — cap its wait so a crashing
		// pod fails fast instead of burning the full engine rollout budget.
		timeout = min(lightweightBridgeDeployTimeout, timeout)
	}
	sel := kserveDeploymentSelector(llmName, component)
	phase := fmt.Sprintf("KServe deploy component=%s expectedName=%s wait=%s", component, expectedDeployName, timeout)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	waitDeploymentsReadyBySelector(ctx, client, fitID, ns, llmName, sel, expectedDeployName, timeout)
}

func waitDeploymentsReadyBySelector(ctx context.Context, client kubernetes.Interface, fitID, ns, llmName, selector, expectedDeployName string, timeout time.Duration) {
	Eventually(func(g Gomega) {
		// Always scan known inference deploys (prefill + decode) even when this selector is still empty.
		RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, llmName)

		list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{LabelSelector: selector})
		g.Expect(err).NotTo(HaveOccurred())
		if len(list.Items) == 0 {
			LogPollThrottled(fitID, "kserve-sel-"+selector, defaultPollLogInterval,
				"no deployments for selector %q in ns=%s (expected name %s, not Bridge %s-hermes-router)",
				selector, ns, expectedDeployName, llmName)
			LogOwnerDeploymentsThrottled(ctx, client, fitID, ns, llmName)
			logKServeLikeDeployments(ctx, client, fitID, ns)
			g.Expect(list.Items).NotTo(BeEmpty(), "no KServe deployments for %q", selector)
			return
		}
		for _, d := range list.Items {
			if expectedDeployName != "" && d.Name != expectedDeployName {
				LogStep(fitID, "KServe deploy %s (selector matched, expected name %s)", d.Name, expectedDeployName)
			}
			want := int32(1)
			if d.Spec.Replicas != nil && *d.Spec.Replicas > 0 {
				want = *d.Spec.Replicas
			}
			if d.Status.ReadyReplicas < want {
				LogDeploymentStatusThrottled(ctx, client, fitID, ns, d.Name)
			}
			g.Expect(d.Status.ReadyReplicas).To(BeNumerically(">=", want),
				"deployment %s ready %d want %d", d.Name, d.Status.ReadyReplicas, want)
		}
		LogPollThrottled(fitID, "kserve-ready-"+selector, defaultPollLogInterval,
			"%d deployment(s) ready for selector %q", len(list.Items), selector)
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
}

func logAllDeploymentsInNamespace(ctx context.Context, client kubernetes.Interface, fitID, ns string) {
	list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		LogStep(fitID, "list deployments in ns=%s: %v", ns, err)
		return
	}
	for _, d := range list.Items {
		LogStep(fitID, "deploy %s owner=%s component=%s ready=%d/%d",
			d.Name, d.Labels["infernex.io/owner"], d.Labels["app.kubernetes.io/component"],
			d.Status.ReadyReplicas, d.Status.Replicas)
	}
}

func logKServeLikeDeployments(ctx context.Context, client kubernetes.Interface, fitID, ns string) {
	list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/part-of=" + kserveAppPartOf,
	})
	if err != nil {
		LogStep(fitID, "list KServe-like deployments: %v", err)
		return
	}
	if len(list.Items) == 0 {
		LogStep(fitID, "no deployments with part-of=%s in ns=%s", kserveAppPartOf, ns)
		return
	}
	for _, d := range list.Items {
		LogStep(fitID, "KServe deploy %s labels name=%s component=%s ready=%d/%d",
			d.Name, d.Labels["app.kubernetes.io/name"], d.Labels["app.kubernetes.io/component"],
			d.Status.ReadyReplicas, d.Status.Replicas)
	}
}
