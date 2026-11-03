package utils

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func EnsureNamespace(ctx context.Context, client kubernetes.Interface, ns string) {
	_, err := client.CoreV1().Namespaces().Get(ctx, ns, metav1.GetOptions{})
	if err == nil {
		return
	}
	_, err = client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred(), "create namespace %s", ns)
}

func WaitDeploymentRollout(ctx context.Context, client kubernetes.Interface, fitID, ns, name string, timeout time.Duration) {
	phase := fmt.Sprintf("kubectl rollout status deploy/%s", name)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	LogDeploymentStatus(ctx, client, fitID, ns, name)
	Eventually(func(g Gomega) {
		d, err := client.AppsV1().Deployments(ns).Get(ctx, name, metav1.GetOptions{})
		g.Expect(err).NotTo(HaveOccurred(), "deployment %s", name)
		specReplicas := int32(1)
		if d.Spec.Replicas != nil && *d.Spec.Replicas > 0 {
			specReplicas = *d.Spec.Replicas
		}
		ready := d.Status.ReadyReplicas >= specReplicas &&
			d.Status.UpdatedReplicas >= specReplicas &&
			d.Status.AvailableReplicas >= specReplicas &&
			d.Status.ObservedGeneration >= d.Generation
		if !ready {
			LogDeploymentStatusThrottled(ctx, client, fitID, ns, name)
			restartFlappingInferenceEnginePods(ctx, client, fitID, ns, name)
			if owner := d.Labels["infernex.io/owner"]; owner != "" && isInferenceEngineDeployment(name, d.Labels) {
				RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, owner)
			}
		} else {
			LogPollThrottled(fitID, "rollout-ok-"+name, defaultPollLogInterval,
				"deployment %s rollout complete replicas=%d ready=%d updated=%d available=%d observed=%d generation=%d",
				name, specReplicas, d.Status.ReadyReplicas, d.Status.UpdatedReplicas, d.Status.AvailableReplicas, d.Status.ObservedGeneration, d.Generation)
		}
		g.Expect(ready).To(BeTrue(), "deployment %s rollout not complete", name)
	}).WithTimeout(timeout).WithPolling(5*time.Second).Should(Succeed(), "rollout deploy/%s", name)
}

func WaitDaemonSetRollout(ctx context.Context, client kubernetes.Interface, fitID, ns, name string, timeout time.Duration) {
	phase := fmt.Sprintf("kubectl rollout status daemonset/%s", name)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl rollout status daemonset/%s -n %s --timeout=%s",
		kc, name, ns, timeout.String())
	stdout, stderr, err := ExecLocal(cmd)
	LogStep(fitID, "rollout %s: stdout=%s stderr=%s err=%v", name, stdout, stderr, err)
	if err != nil {
		LogStep(fitID, "rollout failed for daemonset/%s — dumping status in ns=%s", name, ns)
		logDaemonSetStatus(ctx, client, fitID, ns, name)
	}
	Expect(err).NotTo(HaveOccurred(), "rollout status daemonset/%s", name)
}

func logDaemonSetStatus(ctx context.Context, client kubernetes.Interface, fitID, ns, name string) {
	ds, err := client.AppsV1().DaemonSets(ns).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		LogStep(fitID, "daemonset %s: get err=%v", name, err)
		return
	}
	LogStep(fitID, "daemonset %s desired=%d ready=%d available=%d",
		name, ds.Status.DesiredNumberScheduled, ds.Status.NumberReady, ds.Status.NumberAvailable)
}

func inferenceEngineDeployNamesForOwner(owner string) []string {
	return []string{
		DeployName(owner, "engine-aggregate"),
		DeployName(owner, "engine-pd-prefill"),
		DeployName(owner, "engine-pd-decode"),
		KServeAggregateWorkloadDeployName(owner), // aggregate LLMISVC + PD decode (*-kserve)
		KServePrefillWorkloadDeployName(owner),
	}
}

func containerStatusFlapping(name string, cs corev1.ContainerStatus) (bool, int32, string) {
	restarts := cs.RestartCount
	if w := cs.State.Waiting; w != nil {
		switch w.Reason {
		case "CrashLoopBackOff", "Error", "ImagePullBackOff", "ErrImagePull",
			"CreateContainerConfigError", "CreateContainerError", "InvalidImageName":
			return true, restarts, name + ":" + w.Reason
		}
	}
	if t := cs.State.Terminated; t != nil && t.ExitCode != 0 {
		return true, restarts, name + ":Terminated"
	}
	if restarts >= 1 {
		return true, restarts, name + ":restartCount"
	}
	return false, restarts, ""
}

func podInferenceEngineFlapping(p corev1.Pod) (flapping bool, restarts int32, reason string) {
	if p.Status.Phase == corev1.PodFailed {
		return true, 0, "PodFailed"
	}
	for _, cs := range p.Status.InitContainerStatuses {
		if ok, r, why := containerStatusFlapping("init:"+cs.Name, cs); ok {
			return true, r, why
		}
		restarts += cs.RestartCount
	}
	for _, cs := range p.Status.ContainerStatuses {
		if ok, r, why := containerStatusFlapping(cs.Name, cs); ok {
			return true, r, why
		}
		restarts += cs.RestartCount
	}
	return false, restarts, ""
}

func restartFlappingInferenceEnginePods(ctx context.Context, client kubernetes.Interface, fitID, ns, deployName string) {
	d, err := client.AppsV1().Deployments(ns).Get(ctx, deployName, metav1.GetOptions{})
	if err != nil {
		if !isNotFound(err) {
			LogPollThrottled(fitID, "fallback-engine-get-"+deployName, defaultPollLogInterval,
				"engine deploy %s get err=%v", deployName, err)
		}
		return
	}
	if d.Spec.Selector == nil {
		return
	}
	if !isInferenceEngineDeployment(deployName, d.Labels) {
		LogPollThrottled(fitID, "fallback-skip-deploy-"+deployName, defaultPollLogInterval,
			"skip engine pod delete fallback for deploy %s (not classified as inference engine)", deployName)
		return
	}
	sel, err := metav1.LabelSelectorAsSelector(d.Spec.Selector)
	if err != nil {
		return
	}
	pods, err := client.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{LabelSelector: sel.String()})
	if err != nil {
		LogPollThrottled(fitID, "fallback-engine-pods-"+deployName, defaultPollLogInterval,
			"list pods for deploy %s err=%v", deployName, err)
		return
	}
	flapped := 0
	for _, p := range pods.Items {
		if p.DeletionTimestamp != nil {
			continue
		}
		flapping, restarts, reason := podInferenceEngineFlapping(p)
		if !flapping {
			continue
		}
		flapped++
		LogStep(fitID, "inference engine pod %s %s=%d (deploy=%s) — delete pod for fast re-create fallback",
			p.Name, reason, restarts, deployName)
		delErr := client.CoreV1().Pods(ns).Delete(ctx, p.Name, metav1.DeleteOptions{})
		if delErr != nil && !isNotFound(delErr) {
			LogStep(fitID, "delete pod %s failed: %v", p.Name, delErr)
		}
	}
	if flapped == 0 && len(pods.Items) > 0 {
		maxRestarts := int32(0)
		for _, p := range pods.Items {
			_, r, _ := podInferenceEngineFlapping(p)
			if r > maxRestarts {
				maxRestarts = r
			}
		}
		LogPollThrottled(fitID, "fallback-engine-scan-"+deployName, defaultPollLogInterval,
			"engine deploy %s: %d pod(s) maxRestarts=%d (none flapping yet)", deployName, len(pods.Items), maxRestarts)
	}
}

func RestartFlappingInferenceEnginePodsByOwner(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string) {
	if client == nil || ns == "" || owner == "" {
		return
	}
	seen := map[string]struct{}{}
	scan := func(name string) {
		if name == "" {
			return
		}
		if _, ok := seen[name]; ok {
			return
		}
		seen[name] = struct{}{}
		restartFlappingInferenceEnginePods(ctx, client, fitID, ns, name)
	}
	for _, name := range inferenceEngineDeployNamesForOwner(owner) {
		scan(name)
	}
	list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("infernex.io/owner=%s", owner),
	})
	if err != nil {
		LogPollThrottled(fitID, "fallback-owner-list-"+owner, defaultPollLogInterval,
			"list owner deployments failed owner=%s err=%v", owner, err)
	} else {
		engineCandidates := 0
		for _, d := range list.Items {
			if isInferenceEngineDeployment(d.Name, d.Labels) {
				engineCandidates++
			}
			scan(d.Name)
		}
		if engineCandidates == 0 && len(seen) == len(inferenceEngineDeployNamesForOwner(owner)) {
			LogPollThrottled(fitID, "fallback-owner-no-engine-"+owner, defaultPollLogInterval,
				"owner=%s has no inference engine deployment candidates yet (known names + owner label)", owner)
		}
	}
	all, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{})
	if err == nil {
		prefix := owner + "-"
		for _, d := range all.Items {
			if !strings.HasPrefix(d.Name, prefix) {
				continue
			}
			if !isInferenceEngineDeployment(d.Name, d.Labels) {
				continue
			}
			scan(d.Name)
		}
	}
}

func isInferenceEngineDeployment(name string, labelMap map[string]string) bool {
	if labelMap != nil {
		// KServe inference engine workloads: aggregate/decode/prefill.
		// Router scheduler is intentionally excluded.
		if labelMap["app.kubernetes.io/part-of"] == kserveAppPartOf {
			component := labelMap["app.kubernetes.io/component"]
			if component == kserveWorkloadComponentAggregate || component == kserveWorkloadComponentPrefill || component == kserveWorkloadComponentDecode {
				return true
			}
		}
	}
	// Name fallback aligned with current ready paths:
	// 1) Bridge-managed engines: *-engine-*
	// 2) KServe engines: *-kserve / *-kserve-prefill (exclude router-scheduler)
	n := strings.ToLower(strings.TrimSpace(name))
	if strings.Contains(n, "-engine-") {
		return true
	}
	if strings.HasSuffix(n, "-kserve-prefill") {
		return true
	}
	// PD decode and aggregate inference share the *-kserve suffix (not router-scheduler).
	if strings.HasSuffix(n, "-kserve") {
		return !strings.Contains(n, "-kserve-router-scheduler")
	}
	return false
}
