package utils

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2/types"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

const defaultPollLogInterval = 30 * time.Second

var (
	pollLogMu   sync.Mutex
	pollLogLast = map[string]time.Time{}

	statusFingerprintMu sync.Mutex
	statusFingerprint   = map[string]string{}
)

// FitIDFromSpecReport extracts InferNexBridge_FIT_N from ginkgo labels or It title.
func FitIDFromSpecReport(report types.SpecReport) string {
	for _, l := range report.LeafNodeLabels {
		if strings.HasPrefix(l, "InferNexBridge_FIT_") {
			return l
		}
	}
	if i := strings.Index(report.LeafNodeText, " | "); i > 0 {
		return strings.TrimSpace(report.LeafNodeText[:i])
	}
	return "infernex-bridge"
}

func LogPhaseBegin(fitID, phase string) {
	LogStep(fitID, ">>> BEGIN %s", phase)
}

func LogPhaseEnd(fitID, phase string) {
	LogStep(fitID, "<<< END %s", phase)
}

// LogPollThrottled logs at most once per key per interval (reduces Eventually spam).
func LogPollThrottled(fitID, key string, interval time.Duration, format string, args ...any) {
	if interval <= 0 {
		interval = defaultPollLogInterval
	}
	now := time.Now()
	pollLogMu.Lock()
	k := fitID + "|" + key
	last := pollLogLast[k]
	if !last.IsZero() && now.Sub(last) < interval {
		pollLogMu.Unlock()
		return
	}
	pollLogLast[k] = now
	pollLogMu.Unlock()
	LogStep(fitID, format, args...)
}

func podStatusSummary(p *corev1.Pod) string {
	ready := 0
	total := len(p.Status.ContainerStatuses)
	notReady := make([]string, 0, 2)
	for _, c := range p.Status.ContainerStatuses {
		if c.Ready {
			ready++
		} else {
			w := ""
			if c.State.Waiting != nil {
				w = c.State.Waiting.Reason + ": " + c.State.Waiting.Message
			} else if c.State.Terminated != nil {
				w = c.State.Terminated.Reason
			}
			notReady = append(notReady, fmt.Sprintf("%s(%s)", c.Name, w))
		}
	}
	s := fmt.Sprintf("containers_ready=%d/%d", ready, total)
	if len(notReady) > 0 {
		s += " NOT_READY=[" + strings.Join(notReady, ", ") + "]"
	}
	return s
}

func LogDeploymentStatus(ctx context.Context, client kubernetes.Interface, fitID, ns, deployName string) {
	logDeploymentStatusInternal(ctx, client, fitID, ns, deployName, false)
}

// LogDeploymentStatusThrottled logs deploy/pod status at most once per interval while polling.
func LogDeploymentStatusThrottled(ctx context.Context, client kubernetes.Interface, fitID, ns, deployName string) {
	logDeploymentStatusInternal(ctx, client, fitID, ns, deployName, true)
}

func logDeploymentStatusInternal(ctx context.Context, client kubernetes.Interface, fitID, ns, deployName string, throttled bool) {
	if client == nil {
		return
	}
	d, err := client.AppsV1().Deployments(ns).Get(ctx, deployName, metav1.GetOptions{})
	if err != nil {
		if throttled {
			LogPollThrottled(fitID, "deploy-status-"+deployName, defaultPollLogInterval, "deploy %s/%s: get err=%v", ns, deployName, err)
		} else {
			LogStep(fitID, "deploy %s/%s: get err=%v", ns, deployName, err)
		}
		return
	}
	fp := fmt.Sprintf("r=%d/u=%d/ready=%d/avail=%d", d.Status.Replicas, d.Status.UpdatedReplicas, d.Status.ReadyReplicas, d.Status.AvailableReplicas)
	if throttled && !logStatusFingerprintChanged(fitID, "deploy-"+deployName, fp) {
		return
	}
	LogStep(fitID, "deploy %s replicas=%d updated=%d ready=%d available=%d",
		deployName, d.Status.Replicas, d.Status.UpdatedReplicas, d.Status.ReadyReplicas, d.Status.AvailableReplicas)
	logPodsForDeployment(ctx, client, fitID, ns, d)
}

func logStatusFingerprintChanged(fitID, key, fingerprint string) bool {
	statusFingerprintMu.Lock()
	defer statusFingerprintMu.Unlock()
	k := fitID + "|" + key
	if statusFingerprint[k] == fingerprint {
		return false
	}
	statusFingerprint[k] = fingerprint
	return true
}

func logPodsForDeployment(ctx context.Context, client kubernetes.Interface, fitID, ns string, d *appsv1.Deployment) {
	if d == nil || d.Spec.Selector == nil {
		return
	}
	sel, err := metav1.LabelSelectorAsSelector(d.Spec.Selector)
	if err != nil {
		return
	}
	pods, err := client.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{LabelSelector: sel.String()})
	if err != nil {
		LogStep(fitID, "pods for deploy %s: list err=%v", d.Name, err)
		return
	}
	if len(pods.Items) == 0 {
		LogStep(fitID, "pods for deploy %s: (none yet)", d.Name)
		return
	}
	for _, p := range pods.Items {
		LogStep(fitID, "  pod %s phase=%s %s", p.Name, p.Status.Phase, podStatusSummary(&p))
	}
}

// LogOwnerDeployments lists deployments labeled infernex.io/owner=owner with replica summary.
func LogOwnerDeployments(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string) {
	logOwnerDeploymentsInternal(ctx, client, fitID, ns, owner, false)
}

// LogOwnerDeploymentsThrottled logs owner deploy summary at most once per interval (for poll loops).
func LogOwnerDeploymentsThrottled(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string) {
	logOwnerDeploymentsInternal(ctx, client, fitID, ns, owner, true)
}

func logOwnerDeploymentsInternal(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, throttled bool) {
	list, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("infernex.io/owner=%s", owner),
	})
	if err != nil {
		if throttled {
			LogPollThrottled(fitID, "owner-deploys-"+owner, defaultPollLogInterval, "owner=%s deployments: list err=%v", owner, err)
		} else {
			LogStep(fitID, "owner=%s deployments: list err=%v", owner, err)
		}
		return
	}
	if len(list.Items) == 0 {
		if throttled {
			LogPollThrottled(fitID, "owner-deploys-"+owner, defaultPollLogInterval, "owner=%s: no deployments yet", owner)
		} else {
			LogStep(fitID, "owner=%s: no deployments yet", owner)
		}
		return
	}
	var b strings.Builder
	for i, d := range list.Items {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%s=%d/%d", d.Name, d.Status.ReadyReplicas, d.Status.Replicas)
	}
	summary := b.String()
	if throttled {
		if !logStatusFingerprintChanged(fitID, "owner-deploys-"+owner, summary) {
			return
		}
		LogStep(fitID, "owner=%s deployments (%d): %s", owner, len(list.Items), summary)
		return
	}
	for _, d := range list.Items {
		LogStep(fitID, "owner deploy %s ready=%d/%d", d.Name, d.Status.ReadyReplicas, d.Status.Replicas)
		logPodsForDeployment(ctx, client, fitID, ns, &d)
	}
}

func LogInferNexServiceStatusThrottled(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string) {
	logInferNexServiceStatusInternal(ctx, dyn, fitID, ns, name, true)
}

func logInferNexServiceStatusInternal(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, throttled bool) {
	obj, err := GetInferNexService(ctx, dyn, ns, name)
	if err != nil {
		if throttled {
			LogPollThrottled(fitID, "insvc-status-"+name, defaultPollLogInterval, "infernexservice %s: get err=%v", name, err)
		} else {
			LogStep(fitID, "infernexservice %s: get err=%v", name, err)
		}
		return
	}
	ready, _, _ := unstructured.NestedBool(obj.Object, "status", "ready")
	mode, _, _ := unstructured.NestedString(obj.Object, "status", "mode")
	components, _, _ := unstructured.NestedMap(obj.Object, "status", "components")
	var b strings.Builder
	fmt.Fprintf(&b, "ready=%v mode=%q", ready, mode)
	for k, v := range components {
		m, _ := v.(map[string]interface{})
		r, _ := m["ready"].(bool)
		fmt.Fprintf(&b, " %s=%v", k, r)
	}
	fp := b.String()
	if throttled {
		if !logStatusFingerprintChanged(fitID, "insvc-"+name, fp) {
			return
		}
		LogStep(fitID, "infernexservice %s %s", name, fp)
		return
	}
	LogStep(fitID, "infernexservice %s status.ready=%v status.mode=%q", name, ready, mode)
	if len(components) == 0 {
		LogStep(fitID, "  status.components: (empty)")
		return
	}
	for k, v := range components {
		m, _ := v.(map[string]interface{})
		r, _ := m["ready"].(bool)
		LogStep(fitID, "  component %s ready=%v", k, r)
	}
}

func LogLLMISvcStatusThrottled(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string) {
	logLLMISvcStatusInternal(ctx, dyn, fitID, ns, name, true)
}

func logLLMISvcStatusInternal(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, throttled bool) {
	obj, err := GetLLMInferenceService(ctx, dyn, ns, name)
	if err != nil {
		if throttled {
			LogPollThrottled(fitID, "llmisvc-status-"+name, defaultPollLogInterval, "llmisvc %s: get err=%v", name, err)
		} else {
			LogStep(fitID, "llmisvc %s: get err=%v", name, err)
		}
		return
	}
	conds, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if len(conds) == 0 {
		if throttled {
			LogPollThrottled(fitID, "llmisvc-status-"+name, defaultPollLogInterval, "llmisvc %s: no status.conditions", name)
		} else {
			LogStep(fitID, "llmisvc %s: no status.conditions", name)
		}
		return
	}
	var fp strings.Builder
	var notReady []string
	for _, c := range conds {
		m, _ := c.(map[string]interface{})
		if m == nil {
			continue
		}
		t, _ := m["type"].(string)
		s, _ := m["status"].(string)
		r, _ := m["reason"].(string)
		fmt.Fprintf(&fp, "%s=%s/%s;", t, s, r)
		if s != "True" && s != "true" {
			notReady = append(notReady, fmt.Sprintf("%s=%s reason=%s", t, s, r))
		}
	}
	if throttled {
		if !logStatusFingerprintChanged(fitID, "llmisvc-"+name, fp.String()) {
			return
		}
		if len(notReady) == 0 {
			LogStep(fitID, "llmisvc %s: all conditions True", name)
			return
		}
		LogStep(fitID, "llmisvc %s not ready: %s", name, strings.Join(notReady, "; "))
		return
	}
	for _, c := range conds {
		m, _ := c.(map[string]interface{})
		if m == nil {
			continue
		}
		LogStep(fitID, "  llmisvc condition type=%v status=%v reason=%v",
			m["type"], m["status"], m["reason"])
	}
}

// DumpE2ENamespaceSnapshot logs all pods and owner deployments in the workload namespace (on failure).
func DumpE2ENamespaceSnapshot(ctx context.Context, client kubernetes.Interface, fitID, ns string) {
	if client == nil || ns == "" {
		return
	}
	LogStep(fitID, "=== FAILURE SNAPSHOT ns=%s ===", ns)
	pods, err := client.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		LogStep(fitID, "pods list err=%v", err)
	} else {
		for _, p := range pods.Items {
			LogStep(fitID, "pod %s phase=%s %s", p.Name, p.Status.Phase, podStatusSummary(&p))
		}
	}
	deploys, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		LogStep(fitID, "deployments list err=%v", err)
	} else {
		for _, d := range deploys.Items {
			LogStep(fitID, "deploy %s ready=%d/%d", d.Name, d.Status.ReadyReplicas, d.Status.Replicas)
		}
	}
	LogStep(fitID, "=== END SNAPSHOT ===")
}
