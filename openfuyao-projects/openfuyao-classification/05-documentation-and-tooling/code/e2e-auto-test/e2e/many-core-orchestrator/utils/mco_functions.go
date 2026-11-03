package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/remotecommand"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

func helmListJSON(ctx context.Context) ([]map[string]interface{}, error) {
	cmd := exec.CommandContext(ctx, "helm", "list", "-o", "json")
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("helm list: %w", err)
	}
	out = bytes.TrimSpace(out)
	if len(out) == 0 {
		return nil, nil
	}
	var list []map[string]interface{}
	if err := json.Unmarshal(out, &list); err != nil {
		return nil, fmt.Errorf("parse helm list json: %w", err)
	}
	return list, nil
}

func HelmReleaseExists(ctx context.Context, helmConfig map[string]string) (bool, error) {
	list, err := helmListJSON(ctx)
	if err != nil {
		return false, err
	}
	for _, item := range list {
		if name, ok := item["name"].(string); ok && name == helmConfig["releaseName"] {
			return true, nil
		}
	}
	return false, nil
}

func helmReleaseStatusDeployed(ctx context.Context, releaseName string) (bool, string, error) {
	cmd := exec.CommandContext(ctx, "helm", "status", releaseName, "-o", "json")
	out, err := cmd.Output()
	if err != nil {
		return false, "", err
	}
	var parsed struct {
		Info struct {
			Status string `json:"status"`
		} `json:"info"`
	}
	if err := json.Unmarshal(bytes.TrimSpace(out), &parsed); err != nil {
		return false, "", fmt.Errorf("parse helm status json: %w", err)
	}
	st := strings.ToLower(strings.TrimSpace(parsed.Info.Status))
	return st == "deployed", st, nil
}

func WaitHelmReleaseDeployed(ctx context.Context, releaseName string, timeout, poll time.Duration) error {
	deadline := time.Now().Add(timeout)
	var last string
	for time.Now().Before(deadline) {
		ok, st, err := helmReleaseStatusDeployed(ctx, releaseName)
		if err == nil && ok {
			return nil
		}
		if err == nil {
			last = st
		}
		time.Sleep(poll)
	}
	return fmt.Errorf("helm release %q not deployed within %v; last status: %q", releaseName, timeout, last)
}

func WaitHelmReleaseDeleted(ctx context.Context, helmConfig map[string]string, timeout, poll time.Duration) error {
	releaseName := helmConfig["releaseName"]
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		exists, err := HelmReleaseExists(ctx, helmConfig)
		if err == nil && !exists {
			return nil
		}
		lastErr = err
		if sleepOrDone(ctx, poll) {
			return fmt.Errorf("helm release %q deletion wait canceled: %w", releaseName, ctx.Err())
		}
	}
	if lastErr != nil {
		return fmt.Errorf("helm release %q still exists or cannot be checked within %v: %w", releaseName, timeout, lastErr)
	}
	return fmt.Errorf("helm release %q still exists within %v", releaseName, timeout)
}

func HelmInstallLocalChart(ctx context.Context, helmConfig map[string]string, chartPath string, extraArgs ...string) error {
	releaseName := helmConfig["releaseName"]
	args := []string{"install", releaseName, chartPath}
	if IsOCIChart(chartPath) {
		args = append(args, "--version", GetChartVersion())
	}
	args = append(args, extraArgs...)
	cmd := exec.CommandContext(ctx, "helm", args...)
	if !IsOCIChart(chartPath) {
		cmd.Dir = FindRepoRoot()
	}
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm install %s from %s: %w", releaseName, chartPath, err)
	}
	return nil
}

func HelmUninstallLocal(ctx context.Context, helmConfig map[string]string) error {
	releaseName := helmConfig["releaseName"]
	cmd := exec.CommandContext(ctx, "helm", "uninstall", releaseName)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm uninstall %s: %w", releaseName, err)
	}
	return nil
}

func HelmTemplateLocal(ctx context.Context, chartPath string, extraArgs ...string) (string, error) {
	args := []string{"template", "test", chartPath}
	if IsOCIChart(chartPath) {
		args = append(args, "--version", GetChartVersion())
	}
	args = append(args, extraArgs...)
	cmd := exec.CommandContext(ctx, "helm", args...)
	if !IsOCIChart(chartPath) {
		cmd.Dir = FindRepoRoot()
	}
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("helm template: %w; stderr: %s", err, string(ee.Stderr))
		}
		return "", fmt.Errorf("helm template: %w", err)
	}
	return string(out), nil
}

func WaitAllPodsRunningOrSucceeded(c *k8s.K8SClient, ctx context.Context, namespace string, timeout, interval time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastNotReady []string
	for time.Now().Before(deadline) {
		list, err := c.Clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			if sleepOrDone(ctx, interval) {
				return fmt.Errorf("namespace %q: context canceled while listing pods within %v; not ready: %v: %w", namespace, timeout, lastNotReady, ctx.Err())
			}
			continue
		}
		if len(list.Items) == 0 {
			if sleepOrDone(ctx, interval) {
				return fmt.Errorf("namespace %q: context canceled while waiting for pods within %v; not ready: %v: %w", namespace, timeout, lastNotReady, ctx.Err())
			}
			continue
		}
		notReady := make([]string, 0)
		for _, p := range list.Items {
			ph := p.Status.Phase
			if ph != corev1.PodRunning && ph != corev1.PodSucceeded {
				notReady = append(notReady, fmt.Sprintf("%s(%s)", p.Name, ph))
			}
		}
		if len(notReady) == 0 {
			return nil
		}
		lastNotReady = notReady
		if sleepOrDone(ctx, interval) {
			return fmt.Errorf("namespace %q: context canceled while waiting for pods Running/Succeeded within %v; not ready: %v: %w", namespace, timeout, lastNotReady, ctx.Err())
		}
	}
	return fmt.Errorf("namespace %q: not all pods Running/Succeeded within %v; not ready: %v", namespace, timeout, lastNotReady)
}

func sleepOrDone(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return true
	case <-timer.C:
		return false
	}
}

func WaitDaemonSetReady(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	gvr := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "daemonsets"}
	for time.Now().Before(deadline) {
		ds, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil {
			desired, _, _ := unstructured.NestedInt64(ds.Object, "status", "desiredNumberScheduled")
			ready, _, _ := unstructured.NestedInt64(ds.Object, "status", "numberReady")
			if ready == desired && desired > 0 {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("daemonset %s/%s not ready within %v", namespace, name, timeout)
}

func WaitDeploymentReady(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		deploy, err := c.Clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil &&
			deploy.Status.ObservedGeneration >= deploy.Generation &&
			deploy.Status.Replicas > 0 &&
			deploy.Status.ReadyReplicas == deploy.Status.Replicas &&
			deploy.Status.AvailableReplicas == deploy.Status.Replicas {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("deployment %s/%s not ready within %v", namespace, name, timeout)
}

func WaitDeploymentReplicas(c *k8s.K8SClient, ctx context.Context, name, namespace string, replicas int32, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		deploy, err := c.Clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		desired := int32(0)
		if deploy.Spec.Replicas != nil {
			desired = *deploy.Spec.Replicas
		}
		if desired == replicas &&
			deploy.Status.Replicas == replicas &&
			deploy.Status.ReadyReplicas == replicas &&
			deploy.Status.AvailableReplicas == replicas {
			return nil
		}
		lastErr = fmt.Errorf("deployment replicas desired=%d status=%d ready=%d available=%d, want %d",
			desired, deploy.Status.Replicas, deploy.Status.ReadyReplicas, deploy.Status.AvailableReplicas, replicas)
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("deployment %s/%s did not reach replicas=%d within %v: %w", namespace, name, replicas, timeout, lastErr)
}

func WaitNamedResourcesDeleted(c *k8s.K8SClient, ctx context.Context, namespace string, timeout time.Duration, resources ...string) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		allDeleted := true
		for _, resource := range resources {
			deleted, err := namedResourceDeleted(c, ctx, namespace, resource)
			if err != nil {
				return err
			}
			if !deleted {
				allDeleted = false
				break
			}
		}
		if allDeleted {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("resources %v in namespace %s not deleted within %v", resources, namespace, timeout)
}

func namedResourceDeleted(c *k8s.K8SClient, ctx context.Context, namespace, resource string) (bool, error) {
	parts := strings.SplitN(resource, "/", 2)
	if len(parts) != 2 {
		return false, fmt.Errorf("invalid resource %q", resource)
	}
	kind, name := parts[0], parts[1]
	switch kind {
	case "daemonset":
		_, err := c.Clientset.AppsV1().DaemonSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	case "deployment":
		_, err := c.Clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	case "service":
		_, err := c.Clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	default:
		return false, fmt.Errorf("unsupported resource kind %q", kind)
	}
}

func WaitPodsWithLabelsDeleted(c *k8s.K8SClient, ctx context.Context, namespace string, timeout time.Duration, labelSelectors ...string) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		allDeleted := true
		for _, selector := range labelSelectors {
			pods, err := c.Clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
			if err != nil {
				return err
			}
			if len(pods.Items) > 0 {
				allDeleted = false
				break
			}
		}
		if allDeleted {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("pods with labels %v in namespace %s not deleted within %v", labelSelectors, namespace, timeout)
}

func GetDaemonSet(c *k8s.K8SClient, ctx context.Context, name, namespace string) (*appsv1.DaemonSet, error) {
	return c.Clientset.AppsV1().DaemonSets(namespace).Get(ctx, name, metav1.GetOptions{})
}

func ScaleDeployment(c *k8s.K8SClient, ctx context.Context, name, namespace string, replicas int32) error {
	deploy, err := c.Clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	deploy = deploy.DeepCopy()
	deploy.Spec.Replicas = &replicas
	_, err = c.Clientset.AppsV1().Deployments(namespace).Update(ctx, deploy, metav1.UpdateOptions{})
	return err
}

func RolloutRestartDeployment(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	deploy, err := c.Clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	deploy = deploy.DeepCopy()
	if deploy.Spec.Template.Annotations == nil {
		deploy.Spec.Template.Annotations = map[string]string{}
	}
	deploy.Spec.Template.Annotations["kubectl.kubernetes.io/restartedAt"] = time.Now().UTC().Format(time.RFC3339Nano)
	_, err = c.Clientset.AppsV1().Deployments(namespace).Update(ctx, deploy, metav1.UpdateOptions{})
	return err
}

func PatchDaemonSetContainerCapabilities(c *k8s.K8SClient, ctx context.Context, name, namespace, containerName string, add []corev1.Capability) ([]corev1.Capability, error) {
	ds, err := c.Clientset.AppsV1().DaemonSets(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	ds = ds.DeepCopy()
	for i := range ds.Spec.Template.Spec.Containers {
		if ds.Spec.Template.Spec.Containers[i].Name != containerName {
			continue
		}
		container := &ds.Spec.Template.Spec.Containers[i]
		if container.SecurityContext == nil {
			container.SecurityContext = &corev1.SecurityContext{}
		}
		if container.SecurityContext.Capabilities == nil {
			container.SecurityContext.Capabilities = &corev1.Capabilities{}
		}
		original := append([]corev1.Capability(nil), container.SecurityContext.Capabilities.Add...)
		container.SecurityContext.Capabilities.Add = append([]corev1.Capability(nil), add...)
		_, err = c.Clientset.AppsV1().DaemonSets(namespace).Update(ctx, ds, metav1.UpdateOptions{})
		return original, err
	}
	return nil, fmt.Errorf("container %s not found in daemonset %s/%s", containerName, namespace, name)
}

func AnalyzerServiceReady(c *k8s.K8SClient, ctx context.Context) error {
	svc, err := GetService(c, ctx, MCOAnalyzerServiceName, MCODeployNamespace)
	if err != nil {
		return err
	}
	for _, port := range svc.Spec.Ports {
		if port.Port == MCOAnalyzerGRPCPort {
			return nil
		}
	}
	return fmt.Errorf("service %s/%s does not expose gRPC port %d", MCODeployNamespace, MCOAnalyzerServiceName, MCOAnalyzerGRPCPort)
}

func CountSchedulableNodes(c *k8s.K8SClient, ctx context.Context) (int, error) {
	nodes, err := c.Clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return 0, err
	}
	count := 0
	for _, node := range nodes.Items {
		if !node.Spec.Unschedulable {
			count++
		}
	}
	return count, nil
}

func GetNodeLabelValue(c *k8s.K8SClient, ctx context.Context, nodeName, labelKey string) (string, bool, error) {
	node, err := c.Clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return "", false, err
	}
	val, exists := node.Labels[labelKey]
	return val, exists, nil
}

func ListNodesWithLabel(c *k8s.K8SClient, ctx context.Context, labelKey, labelValue string) ([]string, error) {
	nodes, err := c.Clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("%s=%s", labelKey, labelValue),
	})
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(nodes.Items))
	for _, n := range nodes.Items {
		names = append(names, n.Name)
	}
	return names, nil
}

func GetRuntimeClass(c *k8s.K8SClient, ctx context.Context, name string) (*unstructured.Unstructured, error) {
	gvr := schema.GroupVersionResource{
		Group:    "node.k8s.io",
		Version:  "v1",
		Resource: "runtimeclasses",
	}
	return c.DynamicClient.Resource(gvr).Get(ctx, name, metav1.GetOptions{})
}

func ListRuntimeClasses(c *k8s.K8SClient, ctx context.Context) (*unstructured.UnstructuredList, error) {
	gvr := schema.GroupVersionResource{
		Group:    "node.k8s.io",
		Version:  "v1",
		Resource: "runtimeclasses",
	}
	return c.DynamicClient.Resource(gvr).List(ctx, metav1.ListOptions{})
}

func CreateKataTestPod(c *k8s.K8SClient, ctx context.Context, podName, namespace, runtimeClassName string) (*corev1.Pod, error) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			RuntimeClassName: &runtimeClassName,
			Containers: []corev1.Container{
				{
					Name:    "busybox",
					Image:   "docker.io/library/busybox:latest",
					Command: []string{"sleep", "3600"},
				},
			},
		},
	}
	return c.Clientset.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
}

func CreateKataSchedulerTestPod(c *k8s.K8SClient, ctx context.Context, podName, namespace, runtimeClassName, schedulerName string) (*corev1.Pod, error) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			SchedulerName:    schedulerName,
			RuntimeClassName: &runtimeClassName,
			Containers: []corev1.Container{
				{
					Name:    "busybox",
					Image:   "docker.io/library/busybox:latest",
					Command: []string{"sleep", "30"},
				},
			},
			RestartPolicy: corev1.RestartPolicyNever,
		},
	}
	return c.Clientset.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
}

func CreateRuncTestPod(c *k8s.K8SClient, ctx context.Context, podName, namespace string) (*corev1.Pod, error) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:    "busybox",
					Image:   "docker.io/library/busybox:latest",
					Command: []string{"sleep", "3600"},
				},
			},
		},
	}
	return c.Clientset.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
}

func WaitPodCondition(c *k8s.K8SClient, ctx context.Context, name, namespace, condition string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pod, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil && string(pod.Status.Phase) == condition {
			return true
		}
		if sleepOrDone(ctx, 1*time.Second) {
			return false
		}
	}
	return false
}

func DeletePod(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) error {
	if name == "" {
		return nil
	}
	err := c.Clientset.CoreV1().Pods(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if timeout > 0 {
		deadline := time.Now().Add(timeout)
		for time.Now().Before(deadline) {
			if _, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{}); err != nil {
				return nil
			}
			if sleepOrDone(ctx, 1*time.Second) {
				return fmt.Errorf("context canceled waiting for pod %s/%s to be deleted: %w", namespace, name, ctx.Err())
			}
		}
		return fmt.Errorf("timeout waiting for pod %s/%s to be deleted", namespace, name)
	}
	return nil
}

func GetKataDeployPods(c *k8s.K8SClient, ctx context.Context) (*corev1.PodList, error) {
	return c.Clientset.CoreV1().Pods(MCODeployNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app=kata-deploy",
	})
}

func GetCollectorPods(c *k8s.K8SClient, ctx context.Context) (*corev1.PodList, error) {
	return c.Clientset.CoreV1().Pods(MCODeployNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: MCOCollectorAppLabel,
	})
}

func GetAnalyzerPods(c *k8s.K8SClient, ctx context.Context) (*corev1.PodList, error) {
	return c.Clientset.CoreV1().Pods(MCODeployNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: MCOAnalyzerAppLabel,
	})
}

func WaitCollectorPodRecreatedOnNode(c *k8s.K8SClient, ctx context.Context, nodeName, oldPodName string, timeout time.Duration) (*corev1.Pod, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := GetCollectorPods(c, ctx)
		if err != nil {
			return nil, err
		}
		for _, pod := range pods.Items {
			if pod.Name == oldPodName || pod.Spec.NodeName != nodeName || pod.Status.Phase != corev1.PodRunning {
				continue
			}
			ready := len(pod.Status.ContainerStatuses) > 0
			for _, cs := range pod.Status.ContainerStatuses {
				ready = ready && cs.Ready
			}
			if ready {
				return &pod, nil
			}
		}
		time.Sleep(1 * time.Second)
	}
	return nil, fmt.Errorf("collector pod on node %s was not recreated within %v", nodeName, timeout)
}

func ExecInPodOutput(c *k8s.K8SClient, ctx context.Context, namespace, podName, containerName string, command []string) (string, string, error) {
	req := c.Clientset.CoreV1().RESTClient().Post().
		Namespace(namespace).
		Resource("pods").
		Name(podName).
		SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: containerName,
			Command:   command,
			Stdin:     false,
			Stdout:    true,
			Stderr:    true,
			TTY:       false,
		}, scheme.ParameterCodec)

	executor, err := remotecommand.NewSPDYExecutor(c.RestConfig, "POST", req.URL())
	if err != nil {
		return "", "", err
	}

	var stdout, stderr bytes.Buffer
	err = executor.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdin:  nil,
		Stdout: &stdout,
		Stderr: &stderr,
		Tty:    false,
	})
	return stdout.String(), stderr.String(), err
}

func CollectorPodPressureAvailable(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod) (bool, error) {
	if len(pod.Spec.Containers) == 0 {
		return false, fmt.Errorf("collector pod %s/%s has no containers", pod.Namespace, pod.Name)
	}
	command := []string{
		"sh",
		"-c",
		"if [ -d /proc/pressure ] && [ -r /proc/pressure/io ] && [ -r /proc/pressure/memory ]; then echo available; else echo unavailable; fi",
	}
	stdout, stderr, err := ExecInPodOutput(c, ctx, pod.Namespace, pod.Name, pod.Spec.Containers[0].Name, command)
	if err != nil {
		return false, fmt.Errorf("check /proc/pressure in collector pod %s/%s: %w; stderr=%s", pod.Namespace, pod.Name, err, stderr)
	}
	switch strings.TrimSpace(stdout) {
	case "available":
		return true, nil
	case "unavailable":
		return false, nil
	default:
		return false, fmt.Errorf("unexpected /proc/pressure check output from collector pod %s/%s: stdout=%q stderr=%q", pod.Namespace, pod.Name, stdout, stderr)
	}
}

func GetService(c *k8s.K8SClient, ctx context.Context, name, namespace string) (*corev1.Service, error) {
	return c.Clientset.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
}

func GetNodeInterferenceReport(c *k8s.K8SClient, ctx context.Context, name string) (*unstructured.Unstructured, error) {
	gvr := schema.GroupVersionResource{
		Group:    "mco.many-core.io",
		Version:  "v1alpha1",
		Resource: "nodeinterferencereports",
	}
	return c.DynamicClient.Resource(gvr).Get(ctx, name, metav1.GetOptions{})
}

func WaitNodeInterferenceReportHasNodes(c *k8s.K8SClient, ctx context.Context, name string, expectedNodes []string, timeout time.Duration) (*unstructured.Unstructured, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		report, err := GetNodeInterferenceReport(c, ctx, name)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		nodes, found, err := unstructured.NestedMap(report.Object, "status", "nodes")
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		if !found || len(nodes) == 0 {
			lastErr = fmt.Errorf("nodeinterferencereport %s status.nodes is empty", name)
			time.Sleep(1 * time.Second)
			continue
		}
		missing := make([]string, 0)
		for _, node := range expectedNodes {
			if _, ok := nodes[node]; !ok {
				missing = append(missing, node)
			}
		}
		if len(missing) == 0 {
			return report, nil
		}
		lastErr = fmt.Errorf("nodeinterferencereport %s missing nodes %v", name, missing)
		time.Sleep(1 * time.Second)
	}
	return nil, fmt.Errorf("nodeinterferencereport %s did not contain expected nodes %v within %v: %w", name, expectedNodes, timeout, lastErr)
}

func NodeInterferenceReportNodes(report *unstructured.Unstructured) (map[string]interface{}, error) {
	nodes, found, err := unstructured.NestedMap(report.Object, "status", "nodes")
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("nodeinterferencereport %s has no status.nodes", report.GetName())
	}
	return nodes, nil
}

func WaitNodeInterferenceReportNode(c *k8s.K8SClient, ctx context.Context, name, node string, timeout time.Duration) (map[string]interface{}, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		report, err := GetNodeInterferenceReport(c, ctx, name)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		nodes, err := NodeInterferenceReportNodes(report)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		raw, ok := nodes[node]
		if !ok {
			lastErr = fmt.Errorf("node %s not found in report %s", node, name)
			time.Sleep(1 * time.Second)
			continue
		}
		item, ok := raw.(map[string]interface{})
		if !ok {
			lastErr = fmt.Errorf("node %s report has unexpected type %T", node, raw)
			time.Sleep(1 * time.Second)
			continue
		}
		return item, nil
	}
	return nil, fmt.Errorf("nodeinterferencereport %s node %s not ready within %v: %w", name, node, timeout, lastErr)
}

func WaitNodeInterferenceReportNodeSequence(c *k8s.K8SClient, ctx context.Context, name, node string, minSequence float64, timeout time.Duration) (map[string]interface{}, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		report, err := WaitNodeInterferenceReportNode(c, ctx, name, node, 5*time.Second)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		sequence, ok := NodeReportFloat(report, "sequence")
		if ok && sequence >= minSequence {
			return report, nil
		}
		lastErr = fmt.Errorf("node %s sequence not ready: got %v, want >= %.0f", node, sequence, minSequence)
		time.Sleep(1 * time.Second)
	}
	return nil, fmt.Errorf("nodeinterferencereport %s node %s sequence did not reach %.0f within %v: %w", name, node, minSequence, timeout, lastErr)
}

func NodeReportFloat(node map[string]interface{}, fields ...string) (float64, bool) {
	value, ok := nestedValue(node, fields...)
	if !ok {
		return 0, false
	}
	switch typed := value.(type) {
	case float64:
		return typed, true
	case int64:
		return float64(typed), true
	case int:
		return float64(typed), true
	default:
		return 0, false
	}
}

func NodeReportBool(node map[string]interface{}, fields ...string) (bool, bool) {
	value, ok := nestedValue(node, fields...)
	if !ok {
		return false, false
	}
	typed, ok := value.(bool)
	return typed, ok
}

func NodeReportDetectionsContain(node map[string]interface{}, detectionType string, messageParts ...string) bool {
	raw, ok := node["detections"].([]interface{})
	if !ok {
		return false
	}
	for _, item := range raw {
		detection, ok := item.(map[string]interface{})
		if !ok || detection["type"] != detectionType {
			continue
		}
		message, _ := detection["message"].(string)
		matched := true
		for _, part := range messageParts {
			if !strings.Contains(message, part) {
				matched = false
				break
			}
		}
		if matched {
			return true
		}
	}
	return false
}

func NodeReportAvailabilityHas(node map[string]interface{}, metric string, available bool, status string) bool {
	raw, ok := node["availability"].([]interface{})
	if !ok {
		return false
	}
	for _, item := range raw {
		availability, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if availability["metric"] == metric && availability["available"] == available && availability["status"] == status {
			return true
		}
	}
	return false
}

func nestedValue(root map[string]interface{}, fields ...string) (interface{}, bool) {
	var current interface{} = root
	for _, field := range fields {
		asMap, ok := current.(map[string]interface{})
		if !ok {
			return nil, false
		}
		current, ok = asMap[field]
		if !ok {
			return nil, false
		}
	}
	return current, true
}

func Within(value, expected, tolerance float64) bool {
	return math.Abs(value-expected) <= tolerance
}

type AnalyzerMetricSnapshot struct {
	Node              string                       `json:"node"`
	Timestamp         string                       `json:"timestamp,omitempty"`
	Status            string                       `json:"status,omitempty"`
	Sequence          uint64                       `json:"sequence,omitempty"`
	LLCMissRate       float64                      `json:"llcMissRate"`
	LLCOccupancy      float64                      `json:"llcOccupancy"`
	BioLatencyP95Ms   float64                      `json:"bioLatencyP95Ms"`
	BioLatencyP99Ms   float64                      `json:"bioLatencyP99Ms"`
	PSIAvailable      bool                         `json:"psiAvailable"`
	PSIIOAvg10        float64                      `json:"psiIoSome"`
	PSIIOFullAvg10    float64                      `json:"psiIoFull"`
	PSIMemAvg10       float64                      `json:"psiMemorySome"`
	PSIMemFullAvg10   float64                      `json:"psiMemoryFull"`
	MemAvailableRatio float64                      `json:"memAvailableRatio"`
	PageScanRate      float64                      `json:"pageScanRate"`
	Availability      []AnalyzerMetricAvailability `json:"availability,omitempty"`
}

type AnalyzerMetricAvailability struct {
	Collector string `json:"collector"`
	Metric    string `json:"metric"`
	Available bool   `json:"available"`
	Status    string `json:"status"`
	Reason    string `json:"reason,omitempty"`
}

func InjectAnalyzerMetricSnapshots(ctx context.Context, snapshots []AnalyzerMetricSnapshot) error {
	if len(snapshots) == 0 {
		return nil
	}

	dataPath := filepath.Join(os.TempDir(), fmt.Sprintf("mco-analyzer-snapshots-%d.json", time.Now().UnixNano()))
	injectorPath := filepath.Join(os.TempDir(), fmt.Sprintf("mco-analyzer-inject-%d.go", time.Now().UnixNano()))
	data, err := json.Marshal(snapshots)
	if err != nil {
		return err
	}
	if err := os.WriteFile(dataPath, data, 0600); err != nil {
		return err
	}
	if err := os.WriteFile(injectorPath, []byte(analyzerInjectorSource), 0600); err != nil {
		_ = os.Remove(dataPath)
		return err
	}
	defer func() {
		_ = os.Remove(dataPath)
		_ = os.Remove(injectorPath)
	}()

	deadline := time.Now().Add(2 * time.Minute)
	var lastErr error
	for time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("inject analyzer snapshots canceled: %w", err)
		}
		if err := injectAnalyzerMetricSnapshotsOnce(ctx, dataPath, injectorPath); err == nil {
			return nil
		} else {
			lastErr = err
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("inject analyzer snapshots not ready within 2m: %w", lastErr)
}

func injectAnalyzerMetricSnapshotsOnce(ctx context.Context, dataPath, injectorPath string) error {
	localPort, err := freeLocalPort()
	if err != nil {
		return err
	}

	pfCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	target := fmt.Sprintf("%d:%d", localPort, MCOAnalyzerGRPCPort)
	pfCmd := exec.CommandContext(pfCtx, "kubectl", "port-forward", "-n", MCODeployNamespace, "svc/"+MCOAnalyzerServiceName, target)
	var pfStderr bytes.Buffer
	pfCmd.Stderr = &pfStderr
	if err := pfCmd.Start(); err != nil {
		return fmt.Errorf("start kubectl port-forward for analyzer: %w", err)
	}
	defer func() {
		cancel()
		if pfCmd.Process != nil {
			_ = pfCmd.Process.Kill()
		}
		_ = pfCmd.Wait()
	}()

	addr := fmt.Sprintf("127.0.0.1:%d", localPort)
	if err := waitForLocalTCP(ctx, addr, 10*time.Second); err != nil {
		return fmt.Errorf("wait analyzer port-forward %s: %w; port-forward=%s", target, err, pfStderr.String())
	}

	injectCtx, injectCancel := context.WithTimeout(ctx, 45*time.Second)
	defer injectCancel()
	args := []string{"run", injectorPath, "-addr", addr, "-file", dataPath}
	cmd := exec.CommandContext(injectCtx, "go", args...)
	cmd.Dir = FindRepoRoot()
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("inject analyzer snapshots: %w; stdout=%s stderr=%s port-forward=%s", err, stdout.String(), stderr.String(), pfStderr.String())
	}
	return nil
}

func waitForLocalTCP(ctx context.Context, addr string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		if err := ctx.Err(); err != nil {
			return err
		}
		dialer := net.Dialer{Timeout: 200 * time.Millisecond}
		conn, err := dialer.DialContext(ctx, "tcp", addr)
		if err == nil {
			_ = conn.Close()
			return nil
		}
		lastErr = err
		time.Sleep(200 * time.Millisecond)
	}
	return fmt.Errorf("tcp %s not ready within %v: %w", addr, timeout, lastErr)
}

const analyzerInjectorSource = `package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/dynamicpb"
)

type snapshot struct {
	Node              string         ` + "`json:\"node\"`" + `
	Timestamp         string         ` + "`json:\"timestamp,omitempty\"`" + `
	Status            string         ` + "`json:\"status,omitempty\"`" + `
	Sequence          uint64         ` + "`json:\"sequence,omitempty\"`" + `
	LLCMissRate       float64        ` + "`json:\"llcMissRate\"`" + `
	LLCOccupancy      float64        ` + "`json:\"llcOccupancy\"`" + `
	BioLatencyP95Ms   float64        ` + "`json:\"bioLatencyP95Ms\"`" + `
	BioLatencyP99Ms   float64        ` + "`json:\"bioLatencyP99Ms\"`" + `
	PSIAvailable      bool           ` + "`json:\"psiAvailable\"`" + `
	PSIIOAvg10        float64        ` + "`json:\"psiIoSome\"`" + `
	PSIIOFullAvg10    float64        ` + "`json:\"psiIoFull\"`" + `
	PSIMemAvg10       float64        ` + "`json:\"psiMemorySome\"`" + `
	PSIMemFullAvg10   float64        ` + "`json:\"psiMemoryFull\"`" + `
	MemAvailableRatio float64        ` + "`json:\"memAvailableRatio\"`" + `
	PageScanRate      float64        ` + "`json:\"pageScanRate\"`" + `
	Availability      []availability ` + "`json:\"availability,omitempty\"`" + `
}

type availability struct {
	Collector string ` + "`json:\"collector\"`" + `
	Metric    string ` + "`json:\"metric\"`" + `
	Available bool   ` + "`json:\"available\"`" + `
	Status    string ` + "`json:\"status\"`" + `
	Reason    string ` + "`json:\"reason,omitempty\"`" + `
}

type analyzerDescriptors struct {
	snapshot     protoreflect.MessageDescriptor
	metrics      protoreflect.MessageDescriptor
	availability protoreflect.MessageDescriptor
	response     protoreflect.MessageDescriptor
}

func main() {
	addr := flag.String("addr", "127.0.0.1:50051", "analyzer address")
	file := flag.String("file", "", "snapshot json file")
	flag.Parse()
	data, err := os.ReadFile(*file)
	if err != nil {
		panic(err)
	}
	var snapshots []snapshot
	if err := json.Unmarshal(data, &snapshots); err != nil {
		panic(err)
	}
	descs, err := buildAnalyzerDescriptors()
	if err != nil {
		panic(err)
	}
	dialCtx, dialCancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer dialCancel()
	conn, err := grpc.DialContext(dialCtx, *addr, grpc.WithTransportCredentials(insecure.NewCredentials()), grpc.WithBlock())
	if err != nil {
		panic(fmt.Errorf("dial analyzer %s: %w", *addr, err))
	}
	defer conn.Close()
	sendCtx, sendCancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer sendCancel()
	if err := sendSnapshots(sendCtx, conn, descs, snapshots); err != nil {
		panic(err)
	}
}

func sendSnapshots(ctx context.Context, conn *grpc.ClientConn, descs analyzerDescriptors, snapshots []snapshot) error {
	stream, err := conn.NewStream(ctx, &grpc.StreamDesc{
		StreamName:    "CollectMetrics",
		ClientStreams: true,
	}, "/mco.analyzer.v1.Analyzer/CollectMetrics", grpc.StaticMethod())
	if err != nil {
		return fmt.Errorf("open CollectMetrics stream: %w", err)
	}
	for i, item := range snapshots {
		if err := stream.SendMsg(buildSnapshotMessage(descs, item, uint64(i+1))); err != nil {
			return fmt.Errorf("send CollectMetrics snapshot %d/%d: %w", i+1, len(snapshots), err)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err := stream.CloseSend(); err != nil {
		return fmt.Errorf("close CollectMetrics send: %w", err)
	}
	if err := stream.RecvMsg(dynamicpb.NewMessage(descs.response)); err != nil {
		return fmt.Errorf("receive CollectMetrics response: %w", err)
	}
	return nil
}

func buildSnapshotMessage(descs analyzerDescriptors, item snapshot, defaultSequence uint64) *dynamicpb.Message {
	ts := item.Timestamp
	if ts == "" {
		ts = time.Now().UTC().Format(time.RFC3339Nano)
	}
	st := item.Status
	if st == "" {
		st = "ok"
	}
	sequence := item.Sequence
	if sequence == 0 {
		sequence = defaultSequence
	}

	metrics := dynamicpb.NewMessage(descs.metrics)
	setFloat(metrics, "llc_miss_rate", item.LLCMissRate)
	setFloat(metrics, "llc_occupancy", item.LLCOccupancy)
	setFloat(metrics, "bio_latency_p95_ms", item.BioLatencyP95Ms)
	setFloat(metrics, "bio_latency_p99_ms", item.BioLatencyP99Ms)
	setBool(metrics, "psi_available", item.PSIAvailable)
	setFloat(metrics, "psi_io_some", item.PSIIOAvg10)
	setFloat(metrics, "psi_io_full", item.PSIIOFullAvg10)
	setFloat(metrics, "psi_memory_some", item.PSIMemAvg10)
	setFloat(metrics, "psi_memory_full", item.PSIMemFullAvg10)
	setFloat(metrics, "mem_available_ratio", item.MemAvailableRatio)
	setFloat(metrics, "page_scan_rate", item.PageScanRate)

	msg := dynamicpb.NewMessage(descs.snapshot)
	setString(msg, "node", item.Node)
	setString(msg, "timestamp", ts)
	setString(msg, "status", st)
	setUint64(msg, "sequence", sequence)
	setString(msg, "sample_period", "1s")
	msg.Set(msg.Descriptor().Fields().ByName("metrics"), protoreflect.ValueOfMessage(metrics))

	availabilityField := msg.Descriptor().Fields().ByName("availability")
	availabilityList := msg.Mutable(availabilityField).List()
	for _, item := range item.Availability {
		availabilityMsg := dynamicpb.NewMessage(descs.availability)
		setString(availabilityMsg, "collector", item.Collector)
		setString(availabilityMsg, "metric", item.Metric)
		setBool(availabilityMsg, "available", item.Available)
		setString(availabilityMsg, "status", item.Status)
		setString(availabilityMsg, "reason", item.Reason)
		availabilityList.Append(protoreflect.ValueOfMessage(availabilityMsg))
	}
	return msg
}

func setString(msg *dynamicpb.Message, name protoreflect.Name, value string) {
	msg.Set(msg.Descriptor().Fields().ByName(name), protoreflect.ValueOfString(value))
}

func setFloat(msg *dynamicpb.Message, name protoreflect.Name, value float64) {
	msg.Set(msg.Descriptor().Fields().ByName(name), protoreflect.ValueOfFloat64(value))
}

func setBool(msg *dynamicpb.Message, name protoreflect.Name, value bool) {
	msg.Set(msg.Descriptor().Fields().ByName(name), protoreflect.ValueOfBool(value))
}

func setUint64(msg *dynamicpb.Message, name protoreflect.Name, value uint64) {
	msg.Set(msg.Descriptor().Fields().ByName(name), protoreflect.ValueOfUint64(value))
}

func buildAnalyzerDescriptors() (analyzerDescriptors, error) {
	fd, err := protodesc.NewFile(&descriptorpb.FileDescriptorProto{
		Syntax:  proto.String("proto3"),
		Name:    proto.String("analyzer_injector.proto"),
		Package: proto.String("mco.analyzer.v1"),
		MessageType: []*descriptorpb.DescriptorProto{
			message("NodeMetrics", []*descriptorpb.FieldDescriptorProto{
				doubleField("llc_miss_rate", 1),
				doubleField("llc_occupancy", 2),
				doubleField("bio_latency_p95_ms", 3),
				doubleField("bio_latency_p99_ms", 4),
				boolField("psi_available", 5),
				doubleField("psi_io_some", 6),
				doubleField("psi_io_full", 7),
				doubleField("psi_memory_some", 8),
				doubleField("psi_memory_full", 9),
				doubleField("mem_available_ratio", 10),
				doubleField("page_scan_rate", 11),
			}),
			message("MetricAvailability", []*descriptorpb.FieldDescriptorProto{
				stringField("collector", 1),
				stringField("metric", 2),
				boolField("available", 3),
				stringField("status", 4),
				stringField("reason", 5),
			}),
			message("CollectorResult", []*descriptorpb.FieldDescriptorProto{
				stringField("name", 1),
				stringField("status", 3),
				stringField("error", 4),
			}),
			message("MetricsSnapshot", []*descriptorpb.FieldDescriptorProto{
				stringField("node", 1),
				stringField("timestamp", 2),
				messageField("metrics", 3, ".mco.analyzer.v1.NodeMetrics"),
				stringField("status", 4),
				repeatedMessageField("collectors", 5, ".mco.analyzer.v1.CollectorResult"),
				stringField("sample_period", 6),
				uint64Field("sequence", 7),
				repeatedMessageField("availability", 8, ".mco.analyzer.v1.MetricAvailability"),
			}),
			message("CollectMetricsResponse", []*descriptorpb.FieldDescriptorProto{
				uint64Field("accepted", 1),
			}),
		},
	}, protoregistry.GlobalFiles)
	if err != nil {
		return analyzerDescriptors{}, err
	}
	messages := fd.Messages()
	return analyzerDescriptors{
		metrics:      messages.ByName("NodeMetrics"),
		availability: messages.ByName("MetricAvailability"),
		snapshot:     messages.ByName("MetricsSnapshot"),
		response:     messages.ByName("CollectMetricsResponse"),
	}, nil
}

func message(name string, fields []*descriptorpb.FieldDescriptorProto) *descriptorpb.DescriptorProto {
	return &descriptorpb.DescriptorProto{Name: proto.String(name), Field: fields}
}

func stringField(name string, number int32) *descriptorpb.FieldDescriptorProto {
	return field(name, number, descriptorpb.FieldDescriptorProto_TYPE_STRING, descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL, "")
}

func doubleField(name string, number int32) *descriptorpb.FieldDescriptorProto {
	return field(name, number, descriptorpb.FieldDescriptorProto_TYPE_DOUBLE, descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL, "")
}

func boolField(name string, number int32) *descriptorpb.FieldDescriptorProto {
	return field(name, number, descriptorpb.FieldDescriptorProto_TYPE_BOOL, descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL, "")
}

func uint64Field(name string, number int32) *descriptorpb.FieldDescriptorProto {
	return field(name, number, descriptorpb.FieldDescriptorProto_TYPE_UINT64, descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL, "")
}

func messageField(name string, number int32, typeName string) *descriptorpb.FieldDescriptorProto {
	return field(name, number, descriptorpb.FieldDescriptorProto_TYPE_MESSAGE, descriptorpb.FieldDescriptorProto_LABEL_OPTIONAL, typeName)
}

func repeatedMessageField(name string, number int32, typeName string) *descriptorpb.FieldDescriptorProto {
	return field(name, number, descriptorpb.FieldDescriptorProto_TYPE_MESSAGE, descriptorpb.FieldDescriptorProto_LABEL_REPEATED, typeName)
}

func field(name string, number int32, fieldType descriptorpb.FieldDescriptorProto_Type, label descriptorpb.FieldDescriptorProto_Label, typeName string) *descriptorpb.FieldDescriptorProto {
	fd := &descriptorpb.FieldDescriptorProto{
		Name:   proto.String(name),
		Number: proto.Int32(number),
		Label:  &label,
		Type:   &fieldType,
	}
	if typeName != "" {
		fd.TypeName = proto.String(typeName)
	}
	return fd
}
`

type MetricSample struct {
	Name   string
	Labels map[string]string
	Value  float64
}

func FetchPodMetricsViaPortForward(ctx context.Context, podName, namespace string, podPort int, timeout time.Duration) (string, error) {
	localPort, err := freeLocalPort()
	if err != nil {
		return "", err
	}

	pfCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	target := fmt.Sprintf("%d:%d", localPort, podPort)
	cmd := exec.CommandContext(pfCtx, "kubectl", "port-forward", "-n", namespace, "pod/"+podName, target)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start kubectl port-forward for %s/%s: %w", namespace, podName, err)
	}
	defer func() {
		cancel()
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()

	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d/metrics", localPort)
	var lastErr error
	for time.Now().Before(deadline) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return "", err
		}
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			body, readErr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if readErr == nil && resp.StatusCode == http.StatusOK {
				return string(body), nil
			}
			if readErr != nil {
				lastErr = readErr
			} else {
				lastErr = fmt.Errorf("metrics endpoint returned HTTP %d", resp.StatusCode)
			}
		} else {
			lastErr = err
		}
		time.Sleep(200 * time.Millisecond)
	}
	return "", fmt.Errorf("fetch metrics from %s/%s timed out: %w; port-forward stderr: %s", namespace, podName, lastErr, stderr.String())
}

func FetchAnalyzerMetricSamples(ctx context.Context, timeout time.Duration) ([]MetricSample, error) {
	metricsText, err := FetchServiceMetricsViaPortForward(ctx, MCOAnalyzerServiceName, MCODeployNamespace, int(MCOAnalyzerMetricsPort), timeout)
	if err != nil {
		return nil, err
	}
	return ParsePrometheusMetrics(metricsText)
}

func FetchServiceMetricsViaPortForward(ctx context.Context, serviceName, namespace string, servicePort int, timeout time.Duration) (string, error) {
	localPort, err := freeLocalPort()
	if err != nil {
		return "", err
	}
	pfCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	target := fmt.Sprintf("%d:%d", localPort, servicePort)
	cmd := exec.CommandContext(pfCtx, "kubectl", "port-forward", "-n", namespace, "svc/"+serviceName, target)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("start kubectl port-forward for service %s/%s: %w", namespace, serviceName, err)
	}
	defer func() {
		cancel()
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()
	deadline := time.Now().Add(timeout)
	url := fmt.Sprintf("http://127.0.0.1:%d/metrics", localPort)
	var lastErr error
	for time.Now().Before(deadline) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return "", err
		}
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			body, readErr := io.ReadAll(resp.Body)
			_ = resp.Body.Close()
			if readErr == nil && resp.StatusCode == http.StatusOK {
				return string(body), nil
			}
			if readErr != nil {
				lastErr = readErr
			} else {
				lastErr = fmt.Errorf("metrics endpoint returned HTTP %d", resp.StatusCode)
			}
		} else {
			lastErr = err
		}
		time.Sleep(200 * time.Millisecond)
	}
	return "", fmt.Errorf("fetch service metrics from %s/%s timed out: %w; port-forward stderr: %s", namespace, serviceName, lastErr, stderr.String())
}

func FetchCollectorMetricSamples(ctx context.Context, pod *corev1.Pod, timeout time.Duration) ([]MetricSample, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		metricsText, err := FetchPodMetricsViaPortForward(ctx, pod.Name, pod.Namespace, MCOCollectorMetricsPort, 10*time.Second)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		samples, err := ParsePrometheusMetrics(metricsText)
		if err != nil {
			return nil, err
		}
		for _, sample := range samples {
			if sample.Labels["node"] == pod.Spec.NodeName {
				return samples, nil
			}
		}
		lastErr = fmt.Errorf("metrics for node %s not ready", pod.Spec.NodeName)
		time.Sleep(1 * time.Second)
	}
	return nil, fmt.Errorf("collector metrics for %s/%s not ready within %v: %w", pod.Namespace, pod.Name, timeout, lastErr)
}

func ParsePrometheusMetrics(metrics string) ([]MetricSample, error) {
	samples := make([]MetricSample, 0)
	for _, line := range strings.Split(metrics, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || !strings.HasPrefix(line, "mco_") {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		value, err := strconv.ParseFloat(parts[len(parts)-1], 64)
		if err != nil {
			return nil, fmt.Errorf("parse metric value from %q: %w", line, err)
		}
		name, labels := parseMetricNameAndLabels(parts[0])
		samples = append(samples, MetricSample{Name: name, Labels: labels, Value: value})
	}
	return samples, nil
}

func MetricValueForNode(samples []MetricSample, name, node string) (float64, bool) {
	for _, sample := range samples {
		if sample.Name == name && sample.Labels["node"] == node {
			return sample.Value, true
		}
	}
	return 0, false
}

func freeLocalPort() (int, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port, nil
}

func parseMetricNameAndLabels(raw string) (string, map[string]string) {
	labels := map[string]string{}
	open := strings.Index(raw, "{")
	if open < 0 {
		return raw, labels
	}
	close := strings.LastIndex(raw, "}")
	if close <= open {
		return raw[:open], labels
	}
	for _, item := range strings.Split(raw[open+1:close], ",") {
		kv := strings.SplitN(item, "=", 2)
		if len(kv) != 2 {
			continue
		}
		labels[strings.TrimSpace(kv[0])] = strings.Trim(strings.TrimSpace(kv[1]), `"`)
	}
	return raw[:open], labels
}

func GetInitContainerLogs(c *k8s.K8SClient, ctx context.Context, podName, namespace, containerName string) (string, error) {
	opts := &corev1.PodLogOptions{
		Container: containerName,
	}
	logBytes, err := c.Clientset.CoreV1().Pods(namespace).GetLogs(podName, opts).Do(ctx).Raw()
	if err != nil {
		return "", fmt.Errorf("get init container logs for %s/%s: %w", namespace, podName, err)
	}
	return string(logBytes), nil
}

func GetContainerLogs(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod) (string, error) {
	if len(pod.Spec.Containers) == 0 {
		return "", fmt.Errorf("pod %s/%s has no containers", pod.Namespace, pod.Name)
	}
	tailLines := int64(1000)
	opts := &corev1.PodLogOptions{
		Container: pod.Spec.Containers[0].Name,
		TailLines: &tailLines,
	}
	logBytes, err := c.Clientset.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, opts).Do(ctx).Raw()
	if err != nil {
		return "", fmt.Errorf("get container logs for %s/%s: %w", pod.Namespace, pod.Name, err)
	}
	return string(logBytes), nil
}

func WaitCollectorLogsHealthy(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		logs, err := GetContainerLogs(c, ctx, pod)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		if err := CollectorLogsHealthy(logs); err == nil {
			return nil
		} else {
			lastErr = err
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("collector logs for %s/%s not healthy within %v: %w", pod.Namespace, pod.Name, timeout, lastErr)
}

func WaitCollectorLogContainsAny(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod, timeout time.Duration, substrings ...string) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		logs, err := GetContainerLogs(c, ctx, pod)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		lower := strings.ToLower(logs)
		for _, substring := range substrings {
			if strings.Contains(lower, strings.ToLower(substring)) {
				return nil
			}
		}
		lastErr = fmt.Errorf("collector logs do not contain any of %v", substrings)
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("collector logs for %s/%s did not contain expected signal within %v: %w", pod.Namespace, pod.Name, timeout, lastErr)
}

func WaitCollectorSnapshotLogCount(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod, minCount int, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		logs, err := GetContainerLogs(c, ctx, pod)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		count := CollectorSnapshotLogCount(logs, pod.Spec.NodeName)
		if count >= minCount {
			return nil
		}
		lastErr = fmt.Errorf("found %d snapshot logs, need %d", count, minCount)
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("collector snapshot logs for %s/%s not ready within %v: %w", pod.Namespace, pod.Name, timeout, lastErr)
}

func CollectorSnapshotLogCount(logs, nodeName string) int {
	count := 0
	for _, line := range strings.Split(logs, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var item snapshotLog
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			continue
		}
		if collectorSnapshotLogMatches(item, nodeName) {
			count++
		}
	}
	return count
}

func WaitCollectorSnapshotLogTimes(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod, minCount int, timeout time.Duration) ([]time.Time, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		logs, err := GetContainerLogs(c, ctx, pod)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		times, err := CollectorSnapshotLogTimes(logs, pod.Spec.NodeName)
		if err != nil {
			lastErr = err
		} else if len(times) >= minCount {
			return times, nil
		} else {
			lastErr = fmt.Errorf("found %d snapshot logs, need %d", len(times), minCount)
		}
		time.Sleep(1 * time.Second)
	}
	return nil, fmt.Errorf("collector snapshot logs for %s/%s not ready within %v: %w", pod.Namespace, pod.Name, timeout, lastErr)
}

type snapshotLog struct {
	Time      string `json:"time"`
	Msg       string `json:"msg"`
	Component string `json:"component"`
	Node      string `json:"node"`
	Status    string `json:"status"`
}

func CollectorSnapshotLogTimes(logs, nodeName string) ([]time.Time, error) {
	times := make([]time.Time, 0)
	for _, line := range strings.Split(logs, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var item snapshotLog
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			continue
		}
		if !collectorSnapshotLogMatches(item, nodeName) {
			continue
		}
		parsed, err := parseCollectorLogTime(item.Time)
		if err != nil {
			return nil, fmt.Errorf("parse collector snapshot time %q: %w", item.Time, err)
		}
		times = append(times, parsed)
	}
	return times, nil
}

func collectorSnapshotLogMatches(item snapshotLog, nodeName string) bool {
	return item.Msg == "snapshot collected" &&
		item.Component == "collector" &&
		(item.Status == "ok" || item.Status == "partial") &&
		item.Node == nodeName
}

func parseCollectorLogTime(value string) (time.Time, error) {
	if parsed, err := time.Parse(time.RFC3339Nano, value); err == nil {
		return parsed, nil
	}
	return time.Parse(time.RFC3339, value)
}

func WaitCollectorPSIAvailableSnapshotLog(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		logs, err := GetContainerLogs(c, ctx, pod)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		if err := CollectorPSIAvailableSnapshotLogHealthy(logs, pod.Spec.NodeName); err == nil {
			return nil
		} else {
			lastErr = err
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("collector PSI available snapshot log for %s/%s not ready within %v: %w", pod.Namespace, pod.Name, timeout, lastErr)
}

func CollectorPSIAvailableSnapshotLogHealthy(logs, nodeName string) error {
	for _, line := range strings.Split(logs, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var item map[string]interface{}
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			continue
		}
		if item["msg"] != "snapshot collected" ||
			item["component"] != "collector" ||
			(item["status"] != "ok" && item["status"] != "partial") ||
			item["node"] != nodeName {
			continue
		}

		psiAvailable, err := snapshotBoolField(item, "psi_available")
		if err != nil || !psiAvailable {
			continue
		}
		psiIO, err := snapshotFloatField(item, "psi_io")
		if err != nil || psiIO == -1 {
			continue
		}
		psiMemory, err := snapshotFloatField(item, "psi_memory")
		if err != nil || psiMemory == -1 {
			continue
		}
		return nil
	}
	return fmt.Errorf("collector logs do not contain snapshot with node=%s psi_available=true and psi_io/psi_memory actual values", nodeName)
}

func WaitCollectorPSIUnavailableSnapshotLog(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		logs, err := GetContainerLogs(c, ctx, pod)
		if err != nil {
			lastErr = err
			time.Sleep(1 * time.Second)
			continue
		}
		if err := CollectorPSIUnavailableSnapshotLogHealthy(logs, pod.Spec.NodeName); err == nil {
			return nil
		} else {
			lastErr = err
		}
		time.Sleep(1 * time.Second)
	}
	return fmt.Errorf("collector PSI unavailable snapshot log for %s/%s not ready within %v: %w", pod.Namespace, pod.Name, timeout, lastErr)
}

func CollectorPSIUnavailableSnapshotLogHealthy(logs, nodeName string) error {
	for _, line := range strings.Split(logs, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		var item map[string]interface{}
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			continue
		}
		if item["msg"] != "snapshot collected" ||
			item["component"] != "collector" ||
			(item["status"] != "ok" && item["status"] != "partial") ||
			item["node"] != nodeName {
			continue
		}

		psiAvailable, err := snapshotBoolField(item, "psi_available")
		if err != nil || psiAvailable {
			continue
		}
		psiIO, err := snapshotFloatField(item, "psi_io")
		if err != nil || psiIO != -1 {
			continue
		}
		psiMemory, err := snapshotFloatField(item, "psi_memory")
		if err != nil || psiMemory != -1 {
			continue
		}
		return nil
	}
	return fmt.Errorf("collector logs do not contain snapshot with node=%s psi_available=false and psi_io/psi_memory fallback values", nodeName)
}

func snapshotBoolField(item map[string]interface{}, field string) (bool, error) {
	value, ok := item[field]
	if !ok {
		return false, fmt.Errorf("snapshot field %s not found", field)
	}
	switch typed := value.(type) {
	case bool:
		return typed, nil
	case string:
		parsed, err := strconv.ParseBool(typed)
		if err != nil {
			return false, fmt.Errorf("parse snapshot bool field %s=%q: %w", field, typed, err)
		}
		return parsed, nil
	default:
		return false, fmt.Errorf("snapshot field %s has unsupported type %T", field, value)
	}
}

func snapshotFloatField(item map[string]interface{}, field string) (float64, error) {
	value, ok := item[field]
	if !ok {
		return 0, fmt.Errorf("snapshot field %s not found", field)
	}
	switch typed := value.(type) {
	case float64:
		return typed, nil
	case string:
		parsed, err := strconv.ParseFloat(typed, 64)
		if err != nil {
			return 0, fmt.Errorf("parse snapshot float field %s=%q: %w", field, typed, err)
		}
		return parsed, nil
	default:
		return 0, fmt.Errorf("snapshot field %s has unsupported type %T", field, value)
	}
}

func CollectorLogsHealthy(logs string) error {
	lower := strings.ToLower(logs)
	requiredSignals := []string{
		"snapshot collected",
		"collector",
	}
	for _, signal := range requiredSignals {
		if !strings.Contains(lower, signal) {
			return fmt.Errorf("collector logs do not contain required signal %q", signal)
		}
	}
	if !strings.Contains(lower, "status\":\"ok") && !strings.Contains(lower, "status\":\"partial") {
		return fmt.Errorf("collector logs do not contain required status signal %q or %q", "status\":\"ok", "status\":\"partial")
	}

	continuousFailureSignals := []string{
		"panic",
		"fatal",
		"crash",
		"consecutive fail",
		"continuous fail",
		"failed continuously",
		"exit",
	}
	for _, signal := range continuousFailureSignals {
		if strings.Contains(lower, signal) {
			return fmt.Errorf("collector logs contain unhealthy signal %q", signal)
		}
	}
	return nil
}

func GetKataDeployResources(c *k8s.K8SClient, ctx context.Context) (map[string]bool, error) {
	found := map[string]bool{}

	dsGVR := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "daemonsets"}
	_, err := c.DynamicClient.Resource(dsGVR).Namespace(MCODeployNamespace).Get(ctx, KataDeployDaemonSetName, metav1.GetOptions{})
	if err == nil {
		found["DaemonSet/kata-deploy"] = true
	}

	cmGVR := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"}
	items, err := c.DynamicClient.Resource(cmGVR).Namespace(MCODeployNamespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, item := range items.Items {
			if strings.Contains(item.GetName(), "kata") || strings.Contains(item.GetName(), "preflight") {
				found["ConfigMap/"+item.GetName()] = true
			}
		}
	}

	saGVR := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "serviceaccounts"}
	saItems, err := c.DynamicClient.Resource(saGVR).Namespace(MCODeployNamespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, item := range saItems.Items {
			if strings.Contains(item.GetName(), "kata") {
				found["ServiceAccount/"+item.GetName()] = true
			}
		}
	}

	crGVR := schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterroles"}
	crItems, err := c.DynamicClient.Resource(crGVR).List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, item := range crItems.Items {
			if strings.Contains(item.GetName(), "kata") {
				found["ClusterRole/"+item.GetName()] = true
			}
		}
	}

	rbGVR := schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterrolebindings"}
	rbItems, err := c.DynamicClient.Resource(rbGVR).List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, item := range rbItems.Items {
			if strings.Contains(item.GetName(), "kata") {
				found["ClusterRoleBinding/"+item.GetName()] = true
			}
		}
	}

	rcGVR := schema.GroupVersionResource{Group: "node.k8s.io", Version: "v1", Resource: "runtimeclasses"}
	rcItems, err := c.DynamicClient.Resource(rcGVR).List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, item := range rcItems.Items {
			if strings.HasPrefix(item.GetName(), "kata-") {
				found["RuntimeClass/"+item.GetName()] = true
			}
		}
	}

	return found, nil
}

func GetSchedulerPods(c *k8s.K8SClient, ctx context.Context) (*corev1.PodList, error) {
	return c.Clientset.CoreV1().Pods(MCOVolcanoNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: MCOSchedulerAppLabel,
	})
}

func CreateSchedulerTestPod(c *k8s.K8SClient, ctx context.Context, podName, namespace, schedulerName string) (*corev1.Pod, error) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			SchedulerName: schedulerName,
			Containers: []corev1.Container{
				{
					Name:    "test",
					Image:   "docker.io/library/busybox:latest",
					Command: []string{"sleep", "60"},
				},
			},
			RestartPolicy: corev1.RestartPolicyNever,
		},
	}
	return c.Clientset.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
}

func CreateVolcanoJob(c *k8s.K8SClient, ctx context.Context, name, namespace string) (*unstructured.Unstructured, error) {
	job := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "batch.volcano.sh/v1alpha1",
			"kind":       "Job",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"schedulerName": "volcano",
				"minAvailable":  int64(1),
				"tasks": []interface{}{
					map[string]interface{}{
						"replicas": int64(1),
						"name":     "test",
						"template": map[string]interface{}{
							"spec": map[string]interface{}{
								"containers": []interface{}{
									map[string]interface{}{
										"name":    "test",
										"image":   "docker.io/library/busybox:latest",
										"command": []interface{}{"sleep", "30"},
									},
								},
								"restartPolicy": "Never",
							},
						},
					},
				},
			},
		},
	}
	gvr := schema.GroupVersionResource{
		Group:    "batch.volcano.sh",
		Version:  "v1alpha1",
		Resource: "jobs",
	}
	return c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, job, metav1.CreateOptions{})
}

func DeleteVolcanoJob(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	gvr := schema.GroupVersionResource{
		Group:    "batch.volcano.sh",
		Version:  "v1alpha1",
		Resource: "jobs",
	}
	err := c.DynamicClient.Resource(gvr).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !errors.IsNotFound(err) {
		return err
	}
	return nil
}

func PatchNodeInterferenceReportStatus(ctx context.Context, crName string, patchJSON string) error {
	cmd := exec.CommandContext(ctx, "kubectl", "patch",
		"nodeinterferencereports.mco.many-core.io", crName,
		"--subresource=status", "--type=merge", "-p", patchJSON)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("kubectl patch nodeinterferencereports %s status: %w", crName, err)
	}
	return nil
}

func DeleteNodeInterferenceReport(ctx context.Context, crName string) error {
	cmd := exec.CommandContext(ctx, "kubectl", "delete",
		"nodeinterferencereports.mco.many-core.io", crName,
		"--ignore-not-found")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("kubectl delete nodeinterferencereports %s: %w", crName, err)
	}
	return nil
}

func WaitSchedulerLogsContain(c *k8s.K8SClient, ctx context.Context, substring string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := GetSchedulerPods(c, ctx)
		if err != nil {
			time.Sleep(2 * time.Second)
			continue
		}
		for _, pod := range pods.Items {
			logs, err := GetContainerLogs(c, ctx, &pod)
			if err != nil {
				continue
			}
			if strings.Contains(logs, substring) {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("scheduler logs do not contain %q within %v", substring, timeout)
}

func WaitSchedulerLogsContainAny(c *k8s.K8SClient, ctx context.Context, substrings []string, timeout time.Duration) (string, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := GetSchedulerPods(c, ctx)
		if err != nil {
			time.Sleep(2 * time.Second)
			continue
		}
		for _, pod := range pods.Items {
			logs, err := GetContainerLogs(c, ctx, &pod)
			if err != nil {
				continue
			}
			for _, substring := range substrings {
				if strings.Contains(logs, substring) {
					return substring, nil
				}
			}
		}
		time.Sleep(2 * time.Second)
	}
	return "", fmt.Errorf("scheduler logs do not contain any of %v within %v", substrings, timeout)
}

func GetSchedulerResources(c *k8s.K8SClient, ctx context.Context) (map[string]bool, error) {
	found := map[string]bool{}

	deployGVR := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "deployments"}
	_, err := c.DynamicClient.Resource(deployGVR).Namespace(MCOVolcanoNamespace).Get(ctx, MCOSchedulerDeploymentName, metav1.GetOptions{})
	if err == nil {
		found["Deployment/"+MCOSchedulerDeploymentName] = true
	}

	cmGVR := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"}
	_, err = c.DynamicClient.Resource(cmGVR).Namespace(MCOVolcanoNamespace).Get(ctx, MCOSchedulerConfigMapName, metav1.GetOptions{})
	if err == nil {
		found["ConfigMap/"+MCOSchedulerConfigMapName] = true
	}

	saGVR := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "serviceaccounts"}
	_, err = c.DynamicClient.Resource(saGVR).Namespace(MCOVolcanoNamespace).Get(ctx, MCOSchedulerServiceAccount, metav1.GetOptions{})
	if err == nil {
		found["ServiceAccount/"+MCOSchedulerServiceAccount] = true
	}

	crGVR := schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterroles"}
	_, err = c.DynamicClient.Resource(crGVR).Get(ctx, MCOSchedulerServiceAccount, metav1.GetOptions{})
	if err == nil {
		found["ClusterRole/"+MCOSchedulerServiceAccount] = true
	}

	crbGVR := schema.GroupVersionResource{Group: "rbac.authorization.k8s.io", Version: "v1", Resource: "clusterrolebindings"}
	_, err = c.DynamicClient.Resource(crbGVR).Get(ctx, MCOSchedulerServiceAccount, metav1.GetOptions{})
	if err == nil {
		found["ClusterRoleBinding/"+MCOSchedulerServiceAccount] = true
	}

	return found, nil
}

func WaitPodScheduledOnNode(c *k8s.K8SClient, ctx context.Context, podName, namespace string, timeout time.Duration) (string, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pod, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err == nil && pod.Spec.NodeName != "" {
			return pod.Spec.NodeName, nil
		}
		time.Sleep(1 * time.Second)
	}
	return "", fmt.Errorf("pod %s/%s not scheduled within timeout", namespace, podName)
}

func GetAllNodeNames(c *k8s.K8SClient, ctx context.Context) ([]string, error) {
	nodes, err := c.Clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(nodes.Items))
	for _, n := range nodes.Items {
		names = append(names, n.Name)
	}
	return names, nil
}

func LabelNode(c *k8s.K8SClient, ctx context.Context, nodeName, key, value string) error {
	node, err := c.Clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	node = node.DeepCopy()
	if node.Labels == nil {
		node.Labels = map[string]string{}
	}
	node.Labels[key] = value
	_, err = c.Clientset.CoreV1().Nodes().Update(ctx, node, metav1.UpdateOptions{})
	return err
}

func UnlabelNode(c *k8s.K8SClient, ctx context.Context, nodeName, key string) error {
	node, err := c.Clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	node = node.DeepCopy()
	delete(node.Labels, key)
	_, err = c.Clientset.CoreV1().Nodes().Update(ctx, node, metav1.UpdateOptions{})
	return err
}

func WaitVolcanoAdmissionReady(c *k8s.K8SClient, ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		// 1) 检查 admission Pod 是否 Running 且 Ready
		pods, err := c.Clientset.CoreV1().Pods(MCOVolcanoNamespace).List(ctx, metav1.ListOptions{
			LabelSelector: "app=volcano-admission",
		})
		if err != nil || len(pods.Items) == 0 {
			time.Sleep(2 * time.Second)
			continue
		}
		allReady := true
		for _, pod := range pods.Items {
			if pod.Status.Phase != corev1.PodRunning {
				allReady = false
				break
			}
			for _, cond := range pod.Status.Conditions {
				if cond.Type == corev1.PodReady && cond.Status != corev1.ConditionTrue {
					allReady = false
					break
				}
			}
			if !allReady {
				break
			}
		}
		if !allReady {
			time.Sleep(2 * time.Second)
			continue
		}

		// 2) 确认 Service Endpoints 已编程（Pod Ready ≠ Endpoints 已同步）
		ep, err := c.Clientset.CoreV1().Endpoints(MCOVolcanoNamespace).Get(ctx, "volcano-admission-service", metav1.GetOptions{})
		if err != nil {
			time.Sleep(2 * time.Second)
			continue
		}
		hasReadyAddr := false
		for _, subset := range ep.Subsets {
			if len(subset.Addresses) > 0 {
				hasReadyAddr = true
				break
			}
		}
		if hasReadyAddr {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("volcano-admission not ready within %v", timeout)
}

func NewSSHExecutorForNode(c *k8s.K8SClient, ctx context.Context, nodeName string) (*executor.SSHExecutor, error) {
	node, err := c.Clientset.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("get node %s: %w", nodeName, err)
	}

	knownAddrs := make(map[string]bool)
	if hostname, ok := node.Labels["kubernetes.io/hostname"]; ok && hostname != "" {
		knownAddrs[strings.ToLower(hostname)] = true
	}
	for _, addr := range node.Status.Addresses {
		knownAddrs[strings.ToLower(addr.Address)] = true
	}

	for i := 1; i <= 10; i++ {
		ip := strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_IP", i)))
		hostname := strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_HOSTNAME", i)))
		portStr := strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_PORT", i)))
		user := strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_USER", i)))
		pass := os.Getenv(fmt.Sprintf("TEST_NODE%d_PASSWORD", i))

		if ip == "" && hostname == "" {
			continue
		}

		matched := false
		if ip != "" && knownAddrs[strings.ToLower(ip)] {
			matched = true
		}
		if hostname != "" && knownAddrs[strings.ToLower(hostname)] {
			matched = true
		}
		if !matched {
			continue
		}

		port := 22
		if portStr != "" {
			if p, err := strconv.Atoi(portStr); err == nil && p > 0 {
				port = p
			}
		}
		if user == "" {
			user = "root"
		}

		sshExec, err := executor.NewSSHExecutor(ip, port, user, pass)
		if err != nil {
			return nil, fmt.Errorf("SSH to %s (%s:%d): %w", nodeName, ip, port, err)
		}
		return sshExec, nil
	}

	return nil, fmt.Errorf("no SSH credentials found for node %s (checked TEST_NODE1-10)", nodeName)
}

func VerifyKataRuntimeOnNode(c *k8s.K8SClient, ctx context.Context, nodeName, podName, podNamespace string) error {
	sshExec, err := NewSSHExecutorForNode(c, ctx, nodeName)
	if err != nil {
		return fmt.Errorf("cannot SSH to node %s: %w", nodeName, err)
	}
	defer sshExec.Close()

	result, err := sshExec.Exec(fmt.Sprintf("ps aux | grep -E 'kata-(runtime|proxy|shim|monitor)|qemu-system' | grep -v grep | head -20"))
	if err != nil && result != nil && result.ExitCode != 0 {
		return fmt.Errorf("SSH command failed on node %s: %w; stdout=%s stderr=%s", nodeName, err, result.Stdout, result.Stderr)
	}
	if result == nil || strings.TrimSpace(result.Stdout) == "" {
		return fmt.Errorf("no kata-related processes found on node %s", nodeName)
	}
	return nil
}

func GetKataDeployPodForNode(c *k8s.K8SClient, ctx context.Context, nodeName string) (*corev1.Pod, error) {
	pods, err := GetKataDeployPods(c, ctx)
	if err != nil {
		return nil, err
	}
	for i := range pods.Items {
		if pods.Items[i].Spec.NodeName == nodeName {
			return &pods.Items[i], nil
		}
	}
	return nil, fmt.Errorf("no kata-deploy pod found on node %s", nodeName)
}

func GetPodNodeName(c *k8s.K8SClient, ctx context.Context, podName, namespace string) (string, error) {
	pod, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	return pod.Spec.NodeName, nil
}
