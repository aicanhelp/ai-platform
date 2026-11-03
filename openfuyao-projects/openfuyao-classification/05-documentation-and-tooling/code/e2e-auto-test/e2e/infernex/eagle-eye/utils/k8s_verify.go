package utils

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	podfunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
)

const (
	NSAiInference = "ai-inference"
	NSEagleEye    = "eagle-eye"
	NSNats        = "nats"
	NSMonitoring  = "monitoring"
)

// kube-prometheus-stack operator 稳定 label（不随 Helm fullname 截断变化）。
// 独立 prometheus-operator 部署常见为 app.kubernetes.io/name=prometheus-operator。
const PrometheusOperatorComponentSelector = "app.kubernetes.io/component=prometheus-operator"
const PrometheusOperatorNameSelector = "app.kubernetes.io/name=prometheus-operator"

// kube-prometheus-stack 子 chart prometheus-node-exporter 的 selector 是 chart 名；
// 部分独立/发行版部署仍用 app.kubernetes.io/name=node-exporter。
const NodeExporterChartSelector = "app.kubernetes.io/name=prometheus-node-exporter"
const NodeExporterNameSelector = "app.kubernetes.io/name=node-exporter"

const HardwareMonitorLabelSelector = "app=hardware-monitor"
const HardwareDiagnosisLabelSelector = "app=hardware-diagnosis"
const NatsLabelSelector = "app.kubernetes.io/name=nats"

var prometheusOperatorSelectors = []string{
	PrometheusOperatorComponentSelector,
	PrometheusOperatorNameSelector,
}

var nodeExporterSelectors = []string{
	NodeExporterChartSelector,
	NodeExporterNameSelector,
}

// ClusterHasPrometheusStackInMonitoring 检查 monitoring 命名空间下是否已有 Running 的 prometheus-operator 或 node-exporter Pod
// （常见于 kube-prometheus-operator 等部署在 monitoring）。若该命名空间不存在或其中无上述健康 Pod，返回 false。
func ClusterHasPrometheusStackInMonitoring(ctx context.Context, c kubernetes.Interface) (bool, error) {
	ok, err := NamespaceExists(ctx, c, NSMonitoring)
	if err != nil {
		return false, err
	}
	if !ok {
		return false, nil
	}
	ops, err := listPodsBySelectors(ctx, c, NSMonitoring, prometheusOperatorSelectors)
	if err != nil {
		return false, err
	}
	if hasHealthyRunning(ops) {
		return true, nil
	}
	exporters, err := listPodsBySelectors(ctx, c, NSMonitoring, nodeExporterSelectors)
	if err != nil {
		return false, err
	}
	return hasHealthyRunning(exporters), nil
}

// PodHasAbnormalState 检测 CrashLoopBackOff、Error 类异常（含常见 Waiting 原因）
func PodHasAbnormalState(pod corev1.Pod) bool {
	if pod.Status.Phase == corev1.PodFailed {
		return true
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if w := cs.State.Waiting; w != nil {
			switch w.Reason {
			case "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError", "InvalidImageName":
				return true
			}
		}
	}
	for _, cs := range pod.Status.InitContainerStatuses {
		if w := cs.State.Waiting; w != nil {
			switch w.Reason {
			case "CrashLoopBackOff", "ImagePullBackOff", "ErrImagePull", "CreateContainerConfigError":
				return true
			}
		}
	}
	return false
}

// NamespaceExists 命名空间是否存在
func NamespaceExists(ctx context.Context, c kubernetes.Interface, name string) (bool, error) {
	_, err := c.CoreV1().Namespaces().Get(ctx, name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// EnsureNamespace 若命名空间不存在则创建
func EnsureNamespace(ctx context.Context, c kubernetes.Interface, name string) error {
	ok, err := NamespaceExists(ctx, c, name)
	if err != nil {
		return err
	}
	if ok {
		return nil
	}
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: name},
	}
	_, err = c.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create namespace %q: %w", name, err)
	}
	return nil
}

// DeleteNamespace 删除命名空间（不存在时忽略）
func DeleteNamespace(ctx context.Context, c kubernetes.Interface, name string) error {
	err := c.CoreV1().Namespaces().Delete(ctx, name, metav1.DeleteOptions{})
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

// ListPodsSafe 命名空间不存在时返回空列表，不报错
func ListPodsSafe(ctx context.Context, c kubernetes.Interface, ns string, opts metav1.ListOptions) (*corev1.PodList, error) {
	ok, err := NamespaceExists(ctx, c, ns)
	if err != nil {
		return nil, err
	}
	if !ok {
		return &corev1.PodList{Items: nil}, nil
	}
	return c.CoreV1().Pods(ns).List(ctx, opts)
}

func listPodsBySelectors(ctx context.Context, c kubernetes.Interface, ns string, selectors []string) ([]corev1.Pod, error) {
	seen := make(map[string]struct{})
	out := make([]corev1.Pod, 0)
	for _, sel := range selectors {
		pods, err := ListPodsSafe(ctx, c, ns, metav1.ListOptions{LabelSelector: sel})
		if err != nil {
			return nil, err
		}
		if pods == nil {
			continue
		}
		for i := range pods.Items {
			p := pods.Items[i]
			key := p.Namespace + "/" + p.Name
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			out = append(out, p)
		}
	}
	return out, nil
}

func hasHealthyRunning(pods []corev1.Pod) bool {
	for i := range pods {
		p := &pods[i]
		if p.Status.Phase == corev1.PodRunning && !PodHasAbnormalState(*p) {
			return true
		}
	}
	return false
}

func assertListedPodsHealthy(kind string, pods []corev1.Pod) error {
	if len(pods) == 0 {
		return errMissing(kind)
	}
	for i := range pods {
		p := &pods[i]
		if !podfunc.VerifyPodStatus(*p) || PodHasAbnormalState(*p) {
			return errPodNotHealthy(kind, p.Name, p)
		}
	}
	return nil
}

func joinPodNames(pods []corev1.Pod) string {
	names := make([]string, 0, len(pods))
	for i := range pods {
		names = append(names, pods[i].Name)
	}
	return strings.Join(names, ",")
}

func absentResult(pods []corev1.Pod) (bool, string) {
	if len(pods) == 0 {
		return true, ""
	}
	return false, joinPodNames(pods)
}

// EagleEyeHardwarePodsAbsent eagle-eye 命名空间下不应存在 hardware-monitor / hardware-diagnosis Pod。
// 若存在，第二个返回值为全部不应存在的 Pod 名，以逗号连接。
func EagleEyeHardwarePodsAbsent(ctx context.Context, c kubernetes.Interface) (bool, string, error) {
	pods, err := listPodsBySelectors(ctx, c, NSEagleEye, []string{
		HardwareMonitorLabelSelector,
		HardwareDiagnosisLabelSelector,
	})
	if err != nil {
		return false, "", err
	}
	ok, names := absentResult(pods)
	return ok, names, nil
}

// NatsPodAbsent nats 命名空间下不应存在 nats chart Pod。
// 若存在，第二个返回值为全部不应存在的 Pod 名，以逗号连接。
func NatsPodAbsent(ctx context.Context, c kubernetes.Interface) (bool, string, error) {
	pods, err := listPodsBySelectors(ctx, c, NSNats, []string{NatsLabelSelector})
	if err != nil {
		return false, "", err
	}
	ok, names := absentResult(pods)
	return ok, names, nil
}

// ObservabilityPodsAbsentInAiInference ai-inference 中不应出现 prometheus-operator / node-exporter 相关 Pod。
// 若存在，第二个返回值为全部不应存在的 Pod 名，以逗号连接。
func ObservabilityPodsAbsentInAiInference(ctx context.Context, c kubernetes.Interface) (bool, string, error) {
	selectors := append([]string{}, prometheusOperatorSelectors...)
	selectors = append(selectors, nodeExporterSelectors...)
	pods, err := listPodsBySelectors(ctx, c, NSAiInference, selectors)
	if err != nil {
		return false, "", err
	}
	ok, names := absentResult(pods)
	return ok, names, nil
}

// EagleEyeHardwarePodsHealthy eagle-eye 下应存在 hardware-monitor、hardware-diagnosis Pod，且 Running、无异常状态
func EagleEyeHardwarePodsHealthy(ctx context.Context, c kubernetes.Interface) error {
	mon, err := listPodsBySelectors(ctx, c, NSEagleEye, []string{HardwareMonitorLabelSelector})
	if err != nil {
		return err
	}
	if err := assertListedPodsHealthy("hardware-monitor", mon); err != nil {
		return err
	}
	diag, err := listPodsBySelectors(ctx, c, NSEagleEye, []string{HardwareDiagnosisLabelSelector})
	if err != nil {
		return err
	}
	return assertListedPodsHealthy("hardware-diagnosis", diag)
}

// NatsPodHealthy nats 命名空间下应有 nats 相关 Pod 且健康
func NatsPodHealthy(ctx context.Context, c kubernetes.Interface) error {
	pods, err := listPodsBySelectors(ctx, c, NSNats, []string{NatsLabelSelector})
	if err != nil {
		return err
	}
	return assertListedPodsHealthy("nats", pods)
}

// AiInferencePrometheusStackHealthy ai-inference 中 prometheus-operator、node-exporter Pod 应存在且健康
func AiInferencePrometheusStackHealthy(ctx context.Context, c kubernetes.Interface) error {
	ops, err := listPodsBySelectors(ctx, c, NSAiInference, prometheusOperatorSelectors)
	if err != nil {
		return err
	}
	if err := assertListedPodsHealthy("prometheus-operator (in ai-inference)", ops); err != nil {
		return err
	}
	exporters, err := listPodsBySelectors(ctx, c, NSAiInference, nodeExporterSelectors)
	if err != nil {
		return err
	}
	return assertListedPodsHealthy("node-exporter (in ai-inference)", exporters)
}

// AiInferencePrometheusStackAbsent ai-inference 中不应部署 prometheus-operator / node-exporter（集群其它处已有）
func AiInferencePrometheusStackAbsent(ctx context.Context, c kubernetes.Interface) error {
	ok, name, err := ObservabilityPodsAbsentInAiInference(ctx, c)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("expected no prometheus-operator/node-exporter in %s, but found pods %q", NSAiInference, name)
	}
	return nil
}

func errMissing(kind string) error {
	return fmt.Errorf("missing expected pod kind: %s", kind)
}

func errPodNotHealthy(kind, podName string, p *corev1.Pod) error {
	return fmt.Errorf("%s pod %q not healthy: phase=%s", kind, podName, p.Status.Phase)
}
