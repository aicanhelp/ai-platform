package infernexbridge

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
	infernexutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

const (
	fit56InsvcAggLWSFixture   = "fit56_insvc_aggregate_lws.yaml"
	fit57LLMISvcAggLWSFixture = "fit57_llmisvc_aggregate_lws.yaml"
	fit58InsvcPDLWSFixture    = "fit58_insvc_pd_lws.yaml"
	fit59LLMISvcPDLWSFixture  = "fit59_llmisvc_pd_lws.yaml"

	lwsExpectedRanks     = 2
	lwsDataParallelRanks = "0,1"
)

var lwsExpectedStartRanks = []int{0, 1}

// kserveMainLWSName returns the KServe-created LeaderWorkerSet name for an LLMInferenceService
// aggregate (decode/main) workload, matching KServe's mainLWSName
// (kmeta.ChildName(<name>, "-kserve-mn")).
func kserveMainLWSName(name string) string { return name + "-kserve-mn" }

// kservePDLWSNames returns the KServe-created LeaderWorkerSet names for an LLMInferenceService
// PD workload, matching KServe's prefillLWSName / mainLWSName
// (kmeta.ChildName(<name>, "-kserve-mn-prefill") / "-kserve-mn").
// Using the deterministic names avoids FindLeaderWorkerSetNameByRole("decode"), which can never
// match: the decode LWS is named <name>-kserve-mn and labeled app.kubernetes.io/component=
// llminferenceservice-workload (neither contains "decode").
func kservePDLWSNames(name string) []string {
	return []string{name + "-kserve-mn-prefill", name + "-kserve-mn"}
}

func verifyLWSDataParallelStartRanks(fitID, lwsName string) {
	if bridgeutils.SkipNPUWorkloadE2E() {
		return
	}
	ns := bridgeutils.E2ENS()
	Eventually(func() error {
		return verifyLWSDataParallelStartRanksOnce(
			k8sClient.RestConfig, k8sClient.Clientset, suiteCtx, ns, lwsName, lwsExpectedStartRanks)
	}, 15*time.Minute, 15*time.Second).Should(Succeed(), "%s LWS %s data-parallel-start-rank", fitID, lwsName)
	bridgeutils.LogStep(fitID, "LWS data-parallel-start-ranks=%s ok lws=%s", lwsDataParallelRanks, lwsName)
}

// verifyLWSDataParallelStartRanksOnce lists the LWS pods (sorted by name) and asserts each pod's
// running `vllm serve` command carries --data-parallel-start-rank matching expectedRanks.
//
// Bridge-local (not infernexutils.VerifyLeaderWorkerSetPodsDataParallelStartRanks): InferNex-Bridge
// MoE fixtures run vllm in the "main" container while PD pods still carry openfuyao.com/pdRole,
// so the shared util would exec into a non-existent prefill-engine/decode-engine container. Here
// the engine container is the role-derived name only when it actually exists on the pod, falling
// back to "main". The exec/ps parsing reuses the exported infernexutils.GetVLLMServeCommand.
func verifyLWSDataParallelStartRanksOnce(restConfig *rest.Config, clientset *kubernetes.Clientset, ctx context.Context, ns, lwsName string, expectedRanks []int) error {
	pods, err := listLWSPods(clientset, ctx, ns, lwsName)
	if err != nil {
		return err
	}
	if len(pods) != len(expectedRanks) {
		return fmt.Errorf("leaderworkerset %s/%s: got %d pods, want %d", ns, lwsName, len(pods), len(expectedRanks))
	}
	for i := range pods {
		pod := &pods[i]
		container := bridgeEngineContainerName(pod)
		if container == "" {
			return fmt.Errorf("pod %s: cannot determine engine container", pod.Name)
		}
		cmdline, err := infernexutils.GetVLLMServeCommand(restConfig, clientset, ctx, ns, infernexutils.PDEngineTarget{Pod: *pod, Container: container})
		if err != nil {
			return fmt.Errorf("pod %s container %s: %w", pod.Name, container, err)
		}
		rank, err := parseDataParallelStartRank(cmdline)
		if err != nil {
			return fmt.Errorf("pod %s: %w", pod.Name, err)
		}
		if rank != expectedRanks[i] {
			return fmt.Errorf("leaderworkerset %s/%s: pod %s data-parallel-start-rank=%d, want %d", ns, lwsName, pod.Name, rank, expectedRanks[i])
		}
		GinkgoWriter.Printf("pod %s container %s --data-parallel-start-rank=%d\n", pod.Name, container, rank)
	}
	return nil
}

// listLWSPods returns pods belonging to a LeaderWorkerSet, sorted by name.
func listLWSPods(clientset *kubernetes.Clientset, ctx context.Context, ns, lwsName string) ([]corev1.Pod, error) {
	for _, sel := range []string{
		"leaderworkerset.sigs.k8s.io/name=" + lwsName,
		"leaderworkerset.x-k8s.io/name=" + lwsName,
	} {
		list, err := clientset.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{LabelSelector: sel})
		if err == nil && len(list.Items) > 0 {
			pods := list.Items
			sort.Slice(pods, func(i, j int) bool { return pods[i].Name < pods[j].Name })
			return pods, nil
		}
	}
	return nil, fmt.Errorf("no pods found for leaderworkerset %s/%s", ns, lwsName)
}

// verifyLWSPodsOnDistinctNodes asserts the LWS group pods are scheduled on different nodes.
// Bridge-local: the shared infernexutils package does not provide this (kept it here so the shared
// package stays at upstream parity).
func verifyLWSPodsOnDistinctNodes(clientset *kubernetes.Clientset, ctx context.Context, ns, lwsName string) error {
	pods, err := listLWSPods(clientset, ctx, ns, lwsName)
	if err != nil {
		return err
	}
	if len(pods) < 2 {
		return nil
	}
	nodes := map[string]struct{}{}
	for _, pod := range pods {
		node := strings.TrimSpace(pod.Spec.NodeName)
		if node == "" {
			return fmt.Errorf("leaderworkerset %s/%s: pod %s has empty spec.nodeName", ns, lwsName, pod.Name)
		}
		nodes[node] = struct{}{}
	}
	if len(nodes) < 2 {
		names := make([]string, 0, len(pods))
		for _, pod := range pods {
			names = append(names, fmt.Sprintf("%s@%s", pod.Name, pod.Spec.NodeName))
		}
		return fmt.Errorf("leaderworkerset %s/%s: expected pods on distinct nodes, got %v", ns, lwsName, names)
	}
	GinkgoWriter.Printf("LeaderWorkerSet %s/%s pods spread across %d nodes OK\n", ns, lwsName, len(nodes))
	return nil
}

// bridgeEngineContainerName picks the vllm engine container for an InferNex-Bridge LWS pod.
// Bridge MoE fixtures run vllm in "main"; use prefill-engine/decode-engine only when present.
func bridgeEngineContainerName(pod *corev1.Pod) string {
	if role := pod.Labels[infernexutils.PDRoleLabelKey]; role != "" {
		var c string
		switch role {
		case infernexutils.PrefillPDRole:
			c = infernexutils.PrefillEngineContainer
		case infernexutils.DecodePDRole:
			c = infernexutils.DecodeEngineContainer
		}
		if c != "" && podHasContainer(pod, c) {
			return c
		}
	}
	for _, c := range pod.Spec.Containers {
		if c.Name == "main" || c.Name == infernexutils.PrefillEngineContainer || c.Name == infernexutils.DecodeEngineContainer {
			return c.Name
		}
	}
	if len(pod.Spec.Containers) > 0 {
		return pod.Spec.Containers[0].Name
	}
	return ""
}

func podHasContainer(pod *corev1.Pod, name string) bool {
	for _, c := range pod.Spec.Containers {
		if c.Name == name {
			return true
		}
	}
	return false
}

// parseDataParallelStartRank extracts the --data-parallel-start-rank value from a vllm serve cmdline.
func parseDataParallelStartRank(cmdline string) (int, error) {
	const flag = "--data-parallel-start-rank"
	idx := strings.Index(cmdline, flag)
	if idx == -1 {
		return 0, fmt.Errorf("container args do not contain %s", flag)
	}
	rest := strings.TrimSpace(cmdline[idx+len(flag):])
	valStr := rest
	if strings.HasPrefix(rest, "=") {
		valStr = strings.TrimSpace(rest[1:])
	} else if spaceIdx := strings.Index(rest, " "); spaceIdx != -1 {
		valStr = rest[:spaceIdx]
	}
	v, err := strconv.Atoi(valStr)
	if err != nil {
		return 0, fmt.Errorf("parse %q value %q: %w", flag, valStr, err)
	}
	return v, nil
}

func skipLWSReason() string {
	if bridgeutils.SkipLWSE2E() {
		return "SKIP_LWS_E2E=true"
	}
	if n := bridgeutils.SchedulableNodeCount("lws"); n >= 0 && n < 2 {
		return fmt.Sprintf("LWS multi-DP needs >=2 schedulable nodes (engine pod anti-affinity); cluster has %d", n)
	}
	return ""
}

func requireLWSOperator(fitID string) {
	if r := skipLWSReason(); r != "" {
		Skip(r)
	}
	Expect(bridgeutils.EnsureLWSOperator(k8sClient.Clientset)).To(Succeed(), "LWS operator required for %s", fitID)
}

func beginInsvcAggLWSTest(fitID, name, fixture string) *workloadCleanupHandle {
	engineCfg := name + "-vllm-aggregate-engine-config"
	h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, bridgeutils.AggregateInsvcBridgeOnlyDeploySuffixes())
	path := applyDeployFixtureNoDefer(fitID, fixture, name)
	h.SetManifest(path, true)
	return h
}

func waitInsvcAggLWSFullyReady(fitID, name string) {
	ns := bridgeutils.E2ENS()
	timeout := lwsWorkloadTimeout()
	bridgeutils.WaitInferNexServiceMode(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, "aggregate", lwsReadyTimeout())
	bridgeutils.WaitOwnerWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, name,
		bridgeutils.AggregateInsvcBridgeOnlyDeploySuffixes(), timeout)
	bridgeutils.WaitOwnerLeaderWorkerSetWorkloads(suiteCtx, k8sClient.DynamicClient, fitID, ns, name,
		[]string{"engine-aggregate"}, lwsExpectedRanks, timeout)
	bridgeutils.WaitOwnerDaemonSetWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, name,
		bridgeutils.EagleEyeDaemonSetSuffixes(), timeout)
	if !bridgeutils.SkipNPUWorkloadE2E() {
		bridgeutils.WaitAggregateInsvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, lwsReadyTimeout())
		bridgeutils.WaitInferNexServiceReadyTrueWithOwnerFallback(
			suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, name, lwsReadyTimeout())
	}
}

func assertInsvcAggLWSTopology(fitID, name string) {
	ns := bridgeutils.E2ENS()
	lwsName := bridgeutils.DeployName(name, "engine-aggregate")
	Expect(bridgeutils.KubectlExists(fitID, "leaderworkerset", ns, lwsName)).To(BeTrue())
	Expect(infernexutils.VerifyLeaderWorkerSetTemplateSize(
		k8sClient.DynamicClient, suiteCtx, ns, lwsName, lwsExpectedRanks)).To(Succeed())
	if !bridgeutils.SkipNPUWorkloadE2E() {
		Expect(verifyLWSPodsOnDistinctNodes(
			k8sClient.Clientset, suiteCtx, ns, lwsName)).To(Succeed())
		verifyLWSDataParallelStartRanks(fitID, lwsName)
	}
	bridgeutils.LogStep(fitID, "insvc aggregate LWS topology ok owner=%s lws=%s", name, lwsName)
}

func beginLLMISvcAggLWSTest(fitID, name, fixture string) *workloadCleanupHandle {
	h := beginLLMISvcWorkloadCleanup(fitID, name, bridgeutils.AggregateLLMISvcBridgeDeploySuffixes())
	path := applyLLMISvcFixtureNoDefer(fitID, fixture, name)
	h.SetManifest(path, true)
	return h
}

func waitLLMISvcAggLWSFullyReady(fitID, name string) {
	ns := bridgeutils.E2ENS()
	timeout := lwsWorkloadTimeout()
	bridgeutils.WaitInferNexServiceSourceRefAndMode(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, "LLMInferenceService", "aggregate", lwsReadyTimeout())
	bridgeutils.WaitOwnerWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, name, bridgeutils.AggregateLLMISvcBridgeDeploySuffixes(), timeout)
	bridgeutils.WaitOwnerDaemonSetWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, name, bridgeutils.EagleEyeDaemonSetSuffixes(), timeout)
	lwsName := kserveMainLWSName(name)
	bridgeutils.WaitLeaderWorkerSetReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, lwsName, lwsExpectedRanks, timeout)
	if !bridgeutils.SkipNPUWorkloadE2E() {
		bridgeutils.WaitAggregateLLMISvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, lwsReadyTimeout())
		bridgeutils.WaitInferNexServiceReadyTrueWithOwnerFallback(
			suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, name, lwsReadyTimeout())
		bridgeutils.WaitLLMInferenceServiceReadyTrueWithOwnerFallback(
			suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, name, lwsReadyTimeout())
	}
}

func assertLLMISvcAggLWSTopology(fitID, name string) {
	ns := bridgeutils.E2ENS()
	lwsName := kserveMainLWSName(name)
	Expect(infernexutils.VerifyLeaderWorkerSetTemplateSize(
		k8sClient.DynamicClient, suiteCtx, ns, lwsName, lwsExpectedRanks)).To(Succeed())
	if !bridgeutils.SkipNPUWorkloadE2E() {
		Expect(verifyLWSPodsOnDistinctNodes(
			k8sClient.Clientset, suiteCtx, ns, lwsName)).To(Succeed())
		verifyLWSDataParallelStartRanks(fitID, lwsName)
	}
	bridgeutils.LogStep(fitID, "llmisvc aggregate LWS topology ok owner=%s lws=%s", name, lwsName)
}

func beginInsvcPDLWSTest(fitID, name, fixture string) *workloadCleanupHandle {
	engineCfg := name + "-vllm-pd-engine-config"
	h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, bridgeutils.PDInsvcBridgeOnlyDeploySuffixes())
	path := applyDeployFixtureNoDefer(fitID, fixture, name)
	h.SetManifest(path, true)
	return h
}

func waitInsvcPDLWSFullyReady(fitID, name string) {
	ns := bridgeutils.E2ENS()
	timeout := lwsWorkloadTimeout()
	bridgeutils.WaitInferNexServiceMode(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, "pd", lwsReadyTimeout())
	bridgeutils.WaitOwnerWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, name, bridgeutils.PDInsvcBridgeOnlyDeploySuffixes(), timeout)
	bridgeutils.WaitOwnerLeaderWorkerSetWorkloads(suiteCtx, k8sClient.DynamicClient, fitID, ns, name,
		[]string{"engine-pd-prefill", "engine-pd-decode"}, lwsExpectedRanks, timeout)
	bridgeutils.WaitOwnerDaemonSetWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, name, bridgeutils.EagleEyeDaemonSetSuffixes(), timeout)
	if !bridgeutils.SkipNPUWorkloadE2E() {
		bridgeutils.WaitPDInsvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, lwsReadyTimeout())
		bridgeutils.WaitInferNexServiceReadyTrueWithOwnerFallback(
			suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, name, lwsReadyTimeout())
	}
}

func assertInsvcPDLWSTopology(fitID, name string) {
	ns := bridgeutils.E2ENS()
	for _, suffix := range []string{"engine-pd-prefill", "engine-pd-decode"} {
		lwsName := bridgeutils.DeployName(name, suffix)
		Expect(bridgeutils.KubectlExists(fitID, "leaderworkerset", ns, lwsName)).To(BeTrue())
		Expect(infernexutils.VerifyLeaderWorkerSetTemplateSize(
			k8sClient.DynamicClient, suiteCtx, ns, lwsName, lwsExpectedRanks)).To(Succeed())
		if !bridgeutils.SkipNPUWorkloadE2E() {
			Expect(verifyLWSPodsOnDistinctNodes(
				k8sClient.Clientset, suiteCtx, ns, lwsName)).To(Succeed())
			verifyLWSDataParallelStartRanks(fitID, lwsName)
		}
	}
	bridgeutils.LogStep(fitID, "insvc PD LWS topology ok owner=%s", name)
}

func beginLLMISvcPDLWSTest(fitID, name, fixture string) *workloadCleanupHandle {
	h := beginLLMISvcWorkloadCleanup(fitID, name, bridgeutils.PDLLMISvcBridgeDeploySuffixes())
	path := applyLLMISvcFixtureNoDefer(fitID, fixture, name)
	h.SetManifest(path, true)
	return h
}

func waitLLMISvcPDLWSFullyReady(fitID, name string) {
	ns := bridgeutils.E2ENS()
	timeout := lwsWorkloadTimeout()
	bridgeutils.WaitInferNexServiceSourceRefAndMode(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, "LLMInferenceService", "pd", lwsReadyTimeout())
	bridgeutils.WaitOwnerWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, name, bridgeutils.PDLLMISvcBridgeDeploySuffixes(), timeout)
	bridgeutils.WaitOwnerDaemonSetWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, name, bridgeutils.EagleEyeDaemonSetSuffixes(), timeout)
	for _, lwsName := range kservePDLWSNames(name) {
		bridgeutils.WaitLeaderWorkerSetReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, lwsName, lwsExpectedRanks, timeout)
	}
	if !bridgeutils.SkipNPUWorkloadE2E() {
		bridgeutils.WaitPDLLMISvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, lwsReadyTimeout())
		bridgeutils.WaitInferNexServiceReadyTrueWithOwnerFallback(
			suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, name, lwsReadyTimeout())
		bridgeutils.WaitLLMInferenceServiceReadyTrueWithOwnerFallback(
			suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, name, lwsReadyTimeout())
	}
}

func assertLLMISvcPDLWSTopology(fitID, name string) {
	ns := bridgeutils.E2ENS()
	for _, lwsName := range kservePDLWSNames(name) {
		Expect(infernexutils.VerifyLeaderWorkerSetTemplateSize(
			k8sClient.DynamicClient, suiteCtx, ns, lwsName, lwsExpectedRanks)).To(Succeed())
		if !bridgeutils.SkipNPUWorkloadE2E() {
			Expect(verifyLWSPodsOnDistinctNodes(
				k8sClient.Clientset, suiteCtx, ns, lwsName)).To(Succeed())
			verifyLWSDataParallelStartRanks(fitID, lwsName)
		}
	}
	bridgeutils.LogStep(fitID, "llmisvc PD LWS topology ok owner=%s", name)
}
