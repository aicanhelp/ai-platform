package utils

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

const (
	aggregateTemplateName = "infernex-default-aggregate-template"
	pdTemplateName        = "infernex-default-pd-template"
)

func WaitBridgeControllerReady(ctx context.Context, client kubernetes.Interface, fitID string, timeout time.Duration) {
	ns := BridgeSystemNS()
	LogPhaseBegin(fitID, "infernex-bridge controller Ready")
	defer LogPhaseEnd(fitID, "infernex-bridge controller Ready")
	Eventually(func(g Gomega) {
		deploys, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{
			LabelSelector: "app.kubernetes.io/name=infernex-bridge",
		})
		g.Expect(err).NotTo(HaveOccurred())
		ready := false
		for _, d := range deploys.Items {
			if d.Status.ReadyReplicas >= 1 {
				ready = true
				LogStep(fitID, "controller deployment %s readyReplicas=%d", d.Name, d.Status.ReadyReplicas)
				break
			}
		}
		if !ready {
			LogPollThrottled(fitID, "bridge-controller", defaultPollLogInterval, "controller not ready in ns=%s", ns)
			for _, d := range deploys.Items {
				LogDeploymentStatusThrottled(ctx, client, fitID, ns, d.Name)
			}
		}
		g.Expect(ready).To(BeTrueBecause("timeout waiting for infernex-bridge controller Ready"))
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
}

func AssertWebhookReady(fitID string) {
	ns := BridgeSystemNS()
	kc := localKubeconfig()
	for _, kindName := range []struct{ kind, name string }{
		{"mutatingwebhookconfiguration", bridgeReleaseName + "-mutating"},
		{"validatingwebhookconfiguration", bridgeReleaseName + "-validating"},
	} {
		cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get %s %s -o name", kc, kindName.kind, kindName.name)
		out, stderr, err := ExecLocal(cmd)
		Expect(err).NotTo(HaveOccurred(), "get %s %s: %s", kindName.kind, kindName.name, stderr)
		Expect(strings.TrimSpace(out)).To(ContainSubstring(kindName.name))
		LogStep(fitID, "webhook exists: %s/%s", kindName.kind, kindName.name)
	}

	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get secret webhook-server-cert -n %s -o jsonpath='{.type}'", kc, ns)
	out, stderr, err := ExecLocal(cmd)
	Expect(err).NotTo(HaveOccurred(), "get webhook-server-cert: %s", stderr)
	Expect(strings.TrimSpace(out)).To(Equal("kubernetes.io/tls"))

	cmd = fmt.Sprintf("KUBECONFIG=%q kubectl get endpoints webhook-service -n %s -o jsonpath='{.subsets[*].addresses}'", kc, ns)
	out, stderr, err = ExecLocal(cmd)
	Expect(err).NotTo(HaveOccurred(), stderr)
	Expect(strings.TrimSpace(out)).NotTo(BeEmpty())
	LogStep(fitID, "webhook cert + service endpoints ready in ns=%s", ns)
}

func AssertDefaultTemplatesExist(ctx context.Context, dyn dynamic.Interface, fitID string) {
	ns := TemplateNS()
	for _, name := range []string{aggregateTemplateName, pdTemplateName} {
		obj, err := GetInferNexServiceConfig(ctx, dyn, ns, name)
		Expect(err).NotTo(HaveOccurred(), "get InferNexServiceConfig %s/%s", ns, name)
		Expect(obj.GetName()).To(Equal(name))
		assertTemplateOptionalComponentsEnabled(obj, fitID, name)
		LogStep(fitID, "template %s/%s exists with pdOrchestrator+eagleEye defaults enabled", ns, name)
	}
}

func assertTemplateOptionalComponentsEnabled(cfg *unstructured.Unstructured, fitID, templateName string) {
	checks := []struct {
		group string
		key   string
	}{
		{"pdOrchestrator", "elasticScaler"},
		{"pdOrchestrator", "tidal"},
		{"pdOrchestrator", "resourceScalingGroup"},
		{"eagleEye", "hardwareMonitor"},
		{"eagleEye", "hardwareDiagnosis"},
		{"eagleEye", "networkPerformanceExporter"},
	}
	for _, c := range checks {
		path := []string{"spec", "components", c.group, c.key, "enabled"}
		enabled, found, err := unstructured.NestedBool(cfg.Object, path...)
		Expect(err).NotTo(HaveOccurred(), "read %s %s", templateName, c.key)
		Expect(found).To(BeTrueBecause("template %s must define spec.components.%s.%s.enabled", templateName, c.group, c.key))
		Expect(enabled).To(BeTrueBecause("template %s must enable %s.%s by default", templateName, c.group, c.key))
	}
	LogStep(fitID, "template %s: pdOrchestrator (3) + eagleEye (3) sub-components enabled", templateName)
}

func DumpBridgeDiagnostics(fitID string, k8sClient *k8s.K8SClient, workloadNS string) {
	if k8sClient == nil {
		return
	}
	ctx := context.Background()
	ns := BridgeSystemNS()
	pods, err := k8sClient.Clientset.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/name=infernex-bridge",
	})
	if err == nil {
		for _, p := range pods.Items {
			LogFailContext(fitID, fmt.Sprintf("bridge pod %s phase=%s", p.Name, p.Status.Phase))
			tail := int64(ControllerLogTailLines())
			logs, _ := k8sClient.Clientset.CoreV1().Pods(ns).GetLogs(p.Name, &corev1.PodLogOptions{TailLines: &tail}).DoRaw(ctx)
			if len(logs) > 0 {
				GinkgoWriter.Printf("--- logs %s/%s ---\n%s\n", ns, p.Name, string(logs))
			}
		}
	}
	if workloadNS != "" {
		deploys, err := k8sClient.Clientset.AppsV1().Deployments(workloadNS).List(ctx, metav1.ListOptions{})
		if err == nil {
			for _, d := range deploys.Items {
				LogFailContext(fitID, fmt.Sprintf("deploy %s ready=%d", d.Name, d.Status.ReadyReplicas))
			}
		}
	}
}
