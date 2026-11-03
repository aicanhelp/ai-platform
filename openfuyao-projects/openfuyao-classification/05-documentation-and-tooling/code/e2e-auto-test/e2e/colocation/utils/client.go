package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/remotecommand"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

func containerStateText(state corev1.ContainerState) string {
	if state.Waiting != nil {
		return fmt.Sprintf("waiting(reason=%s message=%q)", state.Waiting.Reason, state.Waiting.Message)
	}
	if state.Running != nil {
		return fmt.Sprintf("running(startedAt=%s)", state.Running.StartedAt.Time.Format(time.RFC3339))
	}
	if state.Terminated != nil {
		return fmt.Sprintf(
			"terminated(reason=%s exitCode=%d message=%q)",
			state.Terminated.Reason,
			state.Terminated.ExitCode,
			state.Terminated.Message,
		)
	}
	return "unknown"
}

func eventTimestamp(evt corev1.Event) time.Time {
	if !evt.EventTime.Time.IsZero() {
		return evt.EventTime.Time
	}
	if !evt.LastTimestamp.Time.IsZero() {
		return evt.LastTimestamp.Time
	}
	return evt.CreationTimestamp.Time
}

func printRecentPodEvents(c *k8s.K8SClient, ctx context.Context, namespace, podName string, maxItems int) {
	events, err := c.Clientset.CoreV1().Events(namespace).List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("involvedObject.kind=Pod,involvedObject.name=%s", podName),
	})
	if err != nil {
		fmt.Printf("[WaitDebug] failed to list events for pod %s/%s: %v\n", namespace, podName, err)
		return
	}
	if len(events.Items) == 0 {
		fmt.Printf("[WaitDebug] no events found for pod %s/%s\n", namespace, podName)
		return
	}

	sort.Slice(events.Items, func(i, j int) bool {
		return eventTimestamp(events.Items[i]).Before(eventTimestamp(events.Items[j]))
	})

	start := 0
	if maxItems > 0 && len(events.Items) > maxItems {
		start = len(events.Items) - maxItems
	}
	for _, evt := range events.Items[start:] {
		ts := eventTimestamp(evt).Format(time.RFC3339)
		fmt.Printf("[WaitDebug] pod event %s/%s time=%s type=%s reason=%s message=%s\n", namespace, podName, ts, evt.Type, evt.Reason, evt.Message)
	}
}

func printPodTimeoutDiagnostics(c *k8s.K8SClient, ctx context.Context, name, namespace, expectedPhase string, timeout time.Duration) {
	fmt.Printf("[WaitPodCondition] timeout waiting for pod %s/%s to reach phase=%s within %v\n", namespace, name, expectedPhase, timeout)

	pod, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		fmt.Printf("[WaitPodCondition] failed to get pod %s/%s: %v\n", namespace, name, err)
		printRecentPodEvents(c, ctx, namespace, name, 8)
		return
	}

	fmt.Printf(
		"[WaitPodCondition] current pod phase=%s reason=%s message=%q node=%s scheduler=%s\n",
		pod.Status.Phase,
		pod.Status.Reason,
		pod.Status.Message,
		pod.Spec.NodeName,
		pod.Spec.SchedulerName,
	)

	for _, cond := range pod.Status.Conditions {
		fmt.Printf(
			"[WaitPodCondition] condition type=%s status=%s reason=%s message=%q\n",
			cond.Type,
			cond.Status,
			cond.Reason,
			cond.Message,
		)
	}

	for _, cs := range pod.Status.InitContainerStatuses {
		fmt.Printf(
			"[WaitPodCondition] init-container=%s ready=%t restarts=%d state=%s\n",
			cs.Name,
			cs.Ready,
			cs.RestartCount,
			containerStateText(cs.State),
		)
	}
	for _, cs := range pod.Status.ContainerStatuses {
		fmt.Printf(
			"[WaitPodCondition] container=%s ready=%t restarts=%d state=%s\n",
			cs.Name,
			cs.Ready,
			cs.RestartCount,
			containerStateText(cs.State),
		)
	}

	printRecentPodEvents(c, ctx, namespace, name, 8)
}

func printDaemonSetTimeoutDiagnostics(c *k8s.K8SClient, ctx context.Context, name, namespace, waitMode string, timeout time.Duration) {
	fmt.Printf("[WaitDaemonSet] timeout waiting for daemonset %s/%s mode=%s within %v\n", namespace, name, waitMode, timeout)

	gvr := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "daemonsets"}
	ds, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		fmt.Printf("[WaitDaemonSet] failed to get daemonset %s/%s: %v\n", namespace, name, err)
		return
	}

	desired, _, _ := unstructured.NestedInt64(ds.Object, "status", "desiredNumberScheduled")
	ready, _, _ := unstructured.NestedInt64(ds.Object, "status", "numberReady")
	available, _, _ := unstructured.NestedInt64(ds.Object, "status", "numberAvailable")
	updated, _, _ := unstructured.NestedInt64(ds.Object, "status", "updatedNumberScheduled")
	misscheduled, _, _ := unstructured.NestedInt64(ds.Object, "status", "numberMisscheduled")
	fmt.Printf(
		"[WaitDaemonSet] status desired=%d ready=%d available=%d updated=%d misscheduled=%d\n",
		desired,
		ready,
		available,
		updated,
		misscheduled,
	)

	pods, err := c.Clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("app=%s", name)})
	if err != nil {
		fmt.Printf("[WaitDaemonSet] failed to list daemonset pods by label app=%s: %v\n", name, err)
		return
	}
	if len(pods.Items) == 0 {
		fmt.Printf("[WaitDaemonSet] no pods found by label app=%s\n", name)
		return
	}

	for _, pod := range pods.Items {
		fmt.Printf(
			"[WaitDaemonSet] pod=%s phase=%s reason=%s message=%q node=%s\n",
			pod.Name,
			pod.Status.Phase,
			pod.Status.Reason,
			pod.Status.Message,
			pod.Spec.NodeName,
		)
		for _, cs := range pod.Status.ContainerStatuses {
			fmt.Printf(
				"[WaitDaemonSet] pod=%s container=%s ready=%t restarts=%d state=%s\n",
				pod.Name,
				cs.Name,
				cs.Ready,
				cs.RestartCount,
				containerStateText(cs.State),
			)
		}
		printRecentPodEvents(c, ctx, namespace, pod.Name, 3)
	}
}

// StressConfig 定义了 stress-ng 的配置
type StressConfig struct {
	CPUCores int
	CPULoad  int
	VM       int
	VMBytes  int
}

func DecodePodFromManifest(manifest string, name string, cpuRequest string, cpuLimit string) (*corev1.Pod, error) {
	decoder := scheme.Codecs.UniversalDeserializer()
	obj, _, err := decoder.Decode([]byte(manifest), nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decode manifest: %w", err)
	}

	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return nil, fmt.Errorf("manifest is not a Pod")
	}

	pod.Name = name
	pod.Namespace = "default"

	cpuRequest = strings.TrimSpace(cpuRequest)
	cpuLimit = strings.TrimSpace(cpuLimit)
	ctr := &pod.Spec.Containers[0]
	if cpuRequest != "" {
		if ctr.Resources.Requests == nil {
			ctr.Resources.Requests = corev1.ResourceList{}
		}
		ctr.Resources.Requests[corev1.ResourceCPU] = mustParseQuantity(cpuRequest)
	}
	if cpuLimit != "" {
		if ctr.Resources.Limits == nil {
			ctr.Resources.Limits = corev1.ResourceList{}
		}
		ctr.Resources.Limits[corev1.ResourceCPU] = mustParseQuantity(cpuLimit)
	}

	return pod, nil
}

func isTransientWebhookCreateError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	if !strings.Contains(msg, "failed calling webhook") {
		return false
	}

	transientSignals := []string{
		"connection refused",
		"no endpoints available for service",
		"i/o timeout",
		"context deadline exceeded",
		"service unavailable",
		"timeout",
		"eof",
	}

	for _, signal := range transientSignals {
		if strings.Contains(msg, signal) {
			return true
		}
	}
	return false
}

func InstallPod(c *k8s.K8SClient, ctx context.Context, pod *corev1.Pod) (*corev1.Pod, error) {
	if pod == nil {
		return nil, fmt.Errorf("pod is nil")
	}

	const maxAttempts = 4
	var lastErr error

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		createdPod, err := c.Clientset.CoreV1().Pods(pod.Namespace).Create(ctx, pod.DeepCopy(), metav1.CreateOptions{})
		if err == nil {
			return createdPod, nil
		}

		if errors.IsAlreadyExists(err) {
			existing, getErr := c.Clientset.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			if getErr == nil {
				return existing, nil
			}
			return nil, fmt.Errorf("pod %s/%s already exists but get failed: %w", pod.Namespace, pod.Name, getErr)
		}

		lastErr = err
		if !isTransientWebhookCreateError(err) || attempt == maxAttempts {
			break
		}

		time.Sleep(time.Duration(2*attempt) * time.Second)
	}

	if lastErr == nil {
		return nil, fmt.Errorf("create pod %s/%s failed with unknown error", pod.Namespace, pod.Name)
	}
	return nil, fmt.Errorf("create pod %s/%s failed: %w", pod.Namespace, pod.Name, lastErr)
}

func InstallBEPod(c *k8s.K8SClient, ctx context.Context, cpuRequest string, cpuLimit string) (*corev1.Pod, error) {
	return InstallBEPodNamed(c, ctx, "be-pod", cpuRequest, cpuLimit)
}

// InstallBEPodNamed 部署 BE Pod，可指定 metadata.name（用于同场景多 Pod）。
func InstallBEPodNamed(c *k8s.K8SClient, ctx context.Context, podName string, cpuRequest, cpuLimit string) (*corev1.Pod, error) {
	pod, err := DecodePodFromManifest(bePodManifest, podName, cpuRequest, cpuLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to decode manifest: %w", err)
	}
	return InstallPod(c, ctx, pod)
}

func InstallLSPod(c *k8s.K8SClient, ctx context.Context, cpuRequest string, cpuLimit string) (*corev1.Pod, error) {
	return InstallLSPodNamed(c, ctx, "ls-pod", cpuRequest, cpuLimit)
}

// InstallLSPodNamed 部署 LS Pod，可指定 metadata.name。
func InstallLSPodNamed(c *k8s.K8SClient, ctx context.Context, podName string, cpuRequest, cpuLimit string) (*corev1.Pod, error) {
	pod, err := DecodePodFromManifest(lsPodManifest, podName, cpuRequest, cpuLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to decode manifest: %w", err)
	}
	return InstallPod(c, ctx, pod)
}

func InstallHLSPod(c *k8s.K8SClient, ctx context.Context, cpuRequest string, cpuLimit string) (*corev1.Pod, error) {
	return InstallHLSPodNamed(c, ctx, "hls-pod", cpuRequest, cpuLimit)
}

// InstallHLSPodNamed 部署 HLS Pod，可指定 metadata.name。
func InstallHLSPodNamed(c *k8s.K8SClient, ctx context.Context, podName string, cpuRequest, cpuLimit string) (*corev1.Pod, error) {
	pod, err := DecodePodFromManifest(hlsPodManifest, podName, cpuRequest, cpuLimit)
	if err != nil {
		return nil, fmt.Errorf("failed to decode manifest: %w", err)
	}
	return InstallPod(c, ctx, pod)
}

// PodExists 查询 Pod 是否仍存在（Get 成功且无 error）。
func PodExists(c *k8s.K8SClient, ctx context.Context, name, namespace string) bool {
	_, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	return err == nil
}

// WaitPodCondition 等待 Pod 达到指定状态
func WaitPodCondition(c *k8s.K8SClient, ctx context.Context, name, namespace, condition string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pod, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil && string(pod.Status.Phase) == condition {
			return true
		}
		time.Sleep(1 * time.Second)
	}
	printPodTimeoutDiagnostics(c, ctx, name, namespace, condition, timeout)
	return false
}

// WaitPodDelete 等待 Pod 被删除
func WaitPodDelete(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{}); err != nil {
			return true
		}
		time.Sleep(1 * time.Second)
	}
	return false
}

// WaitAllPodsRunningOrSucceeded 等待命名空间内所有 Pod 均为 Running 或 Succeeded（与 e2e-frontend utils/helm.ts 中 kubectl get pods 校验一致）。
func WaitAllPodsRunningOrSucceeded(c *k8s.K8SClient, ctx context.Context, namespace string, timeout, interval time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastNotReady []string
	for time.Now().Before(deadline) {
		list, err := c.Clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			time.Sleep(interval)
			continue
		}
		if len(list.Items) == 0 {
			time.Sleep(interval)
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
		time.Sleep(interval)
	}
	return fmt.Errorf("namespace %q: not all pods Running/Succeeded within %v; not ready: %v", namespace, timeout, lastNotReady)
}

// DeletePod 删除 Pod；timeout>0 时会等待从 API 中消失。
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
	if timeout > 0 && !WaitPodDelete(c, ctx, name, namespace, timeout) {
		return fmt.Errorf("timeout waiting for pod %s/%s to be deleted", namespace, name)
	}
	return nil
}

// WaitForVolcanoRestart 等待 Volcano scheduler 重启完成
func WaitForVolcanoRestart(c *k8s.K8SClient, ctx context.Context, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := c.Clientset.CoreV1().Pods("volcano-system").List(ctx, metav1.ListOptions{
			LabelSelector: "app=volcano-scheduler",
		})
		if err != nil || len(pods.Items) == 0 {
			time.Sleep(1 * time.Second)
			continue
		}
		if pods.Items[0].Status.Phase == corev1.PodRunning {
			return true
		}
		time.Sleep(1 * time.Second)
	}
	return false
}

// ListPriorityClasses 列出所有 PriorityClass
func ListPriorityClasses(c *k8s.K8SClient, ctx context.Context) (*unstructured.UnstructuredList, error) {
	gvr := schema.GroupVersionResource{
		Group:    "scheduling.k8s.io",
		Version:  "v1",
		Resource: "priorityclasses",
	}
	return c.DynamicClient.Resource(gvr).List(ctx, metav1.ListOptions{})
}

// GetRubikLog 获取 rubik 容器的日志
func GetRubikLog(c *k8s.K8SClient, ctx context.Context) (string, error) {
	pods, err := c.Clientset.CoreV1().Pods("openfuyao-colocation").List(ctx, metav1.ListOptions{
		LabelSelector: "name=colocation-rubik",
	})
	if err != nil || len(pods.Items) == 0 {
		return "", fmt.Errorf("failed to find rubik pod: %w", err)
	}

	log, err := c.Clientset.CoreV1().Pods("openfuyao-colocation").GetLogs(
		pods.Items[0].Name,
		&corev1.PodLogOptions{Container: "colocation-rubik"},
	).Do(ctx).Raw()
	if err != nil {
		return "", fmt.Errorf("failed to get rubik log: %w", err)
	}
	return string(log), nil
}

// ColocationConfig 中混部相关 ConfigMap / data key 常量。
const (
	ColocationConfigNamespace         = "openfuyao-colocation"
	ColocationConfigName              = "colocation-config"
	ColocationVolcanoSchedulerOptsKey = "volcano-scheduler-options"
	ColocationRubikOptsKey            = "rubik-options"
)

// GetColocationConfigMap 获取混部配置 ConfigMap
func GetColocationConfigMap(c *k8s.K8SClient, ctx context.Context) (*corev1.ConfigMap, error) {
	return c.Clientset.CoreV1().ConfigMaps(ColocationConfigNamespace).Get(ctx, ColocationConfigName, metav1.GetOptions{})
}

// PatchColocationVolcanoSchedulerUsage 在现有 volcano-scheduler-options 上只改 usagePlugin（与 PatchColocationRubikEviction 同款模式）。enabled=false 时关闭 usagePlugin 与 usageThreshold；enabled=true 时开启并设置 cpu/memory 利用率阈值百分比（关闭时 cpu/mem 可传 0）。
func PatchColocationVolcanoSchedulerUsage(c *k8s.K8SClient, ctx context.Context, enabled bool, cpuPercent, memPercent int) error {
	cm, err := GetColocationConfigMap(c, ctx)
	if err != nil {
		return err
	}
	raw := ""
	if cm.Data != nil {
		raw = strings.TrimSpace(cm.Data[ColocationVolcanoSchedulerOptsKey])
	}
	var root map[string]interface{}
	if raw != "" {
		if err := json.Unmarshal([]byte(raw), &root); err != nil {
			return fmt.Errorf("parse volcano-scheduler-options: %w", err)
		}
	} else {
		root = map[string]interface{}{}
	}
	if enabled {
		root["usagePlugin"] = map[string]interface{}{
			"enable": true,
			"usageThreshold": map[string]interface{}{
				"enable": true,
				"cpu":    cpuPercent,
				"memory": memPercent,
			},
		}
	} else {
		root["usagePlugin"] = map[string]interface{}{
			"enable": false,
			"usageThreshold": map[string]interface{}{
				"enable": false,
			},
		}
	}
	out, err := json.Marshal(root)
	if err != nil {
		return fmt.Errorf("marshal volcano-scheduler-options: %w", err)
	}
	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	cm.Data[ColocationVolcanoSchedulerOptsKey] = string(out)
	_, err = c.Clientset.CoreV1().ConfigMaps(cm.Namespace).Update(ctx, cm, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("update colocation-config volcano-scheduler-options: %w", err)
	}
	return nil
}

// UpdateColocationVolcanoSchedulerOptions 写入 colocation-config.data["volcano-scheduler-options"]。
func UpdateColocationVolcanoSchedulerOptions(c *k8s.K8SClient, ctx context.Context, volcanoOptsJSON string) error {
	cm, err := GetColocationConfigMap(c, ctx)
	if err != nil {
		return fmt.Errorf("get colocation-config: %w", err)
	}
	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	cm.Data[ColocationVolcanoSchedulerOptsKey] = volcanoOptsJSON
	_, err = c.Clientset.CoreV1().ConfigMaps(cm.Namespace).Update(ctx, cm, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("update colocation-config volcano-scheduler-options: %w", err)
	}
	return nil
}

// UpdateColocationRubikOptions 覆盖写入 colocation-config.data["rubik-options"]（完整 JSON，用于恢复测试前快照）。
func UpdateColocationRubikOptions(c *k8s.K8SClient, ctx context.Context, rubikOptsJSON string) error {
	cm, err := GetColocationConfigMap(c, ctx)
	if err != nil {
		return fmt.Errorf("get colocation-config: %w", err)
	}
	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	cm.Data[ColocationRubikOptsKey] = rubikOptsJSON
	_, err = c.Clientset.CoreV1().ConfigMaps(cm.Namespace).Update(ctx, cm, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("update colocation-config rubik-options: %w", err)
	}
	return nil
}

// PatchColocationRubikEviction 在现有 rubik-options 上只改 eviction：关闭时 enable=false 且 cpuevict/memoryevict 为空对象；开启时设置阈值。
func PatchColocationRubikEviction(c *k8s.K8SClient, ctx context.Context, enabled bool, cpuEvictThreshold, memEvictThreshold int) error {
	cm, err := GetColocationConfigMap(c, ctx)
	if err != nil {
		return err
	}
	raw := ""
	if cm.Data != nil {
		raw = strings.TrimSpace(cm.Data[ColocationRubikOptsKey])
	}
	var root map[string]interface{}
	if raw != "" {
		if err := json.Unmarshal([]byte(raw), &root); err != nil {
			return fmt.Errorf("parse rubik-options: %w", err)
		}
	} else {
		root = map[string]interface{}{}
	}
	if enabled {
		root["eviction"] = map[string]interface{}{
			"enable": true,
			"cpuevict": map[string]interface{}{
				"threshold": cpuEvictThreshold,
			},
			"memoryevict": map[string]interface{}{
				"threshold": memEvictThreshold,
			},
		}
	} else {
		root["eviction"] = map[string]interface{}{
			"enable":      false,
			"cpuevict":    map[string]interface{}{},
			"memoryevict": map[string]interface{}{},
		}
	}
	out, err := json.Marshal(root)
	if err != nil {
		return fmt.Errorf("marshal rubik-options: %w", err)
	}
	cm.Data[ColocationRubikOptsKey] = string(out)
	_, err = c.Clientset.CoreV1().ConfigMaps(cm.Namespace).Update(ctx, cm, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("update colocation-config rubik-options: %w", err)
	}
	return nil
}

// UpdateVolcanoConfigMap 更新 Volcano scheduler 配置
func UpdateVolcanoConfigMap(c *k8s.K8SClient, ctx context.Context, configData string) error {
	cm, err := c.Clientset.CoreV1().ConfigMaps("volcano-system").Get(ctx, "volcano-scheduler-configmap", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get volcano configmap: %w", err)
	}
	cm.Data["volcano-scheduler.conf"] = configData
	_, err = c.Clientset.CoreV1().ConfigMaps("volcano-system").Update(ctx, cm, metav1.UpdateOptions{})
	return err
}

// GetVolcanoConfigMap 获取 Volcano scheduler 配置
func GetVolcanoConfigMap(c *k8s.K8SClient, ctx context.Context) (*corev1.ConfigMap, error) {
	return c.Clientset.CoreV1().ConfigMaps("volcano-system").Get(ctx, "volcano-scheduler-configmap", metav1.GetOptions{})
}

// ExecInPod 在 Pod 中执行命令
func ExecInPod(c *k8s.K8SClient, ctx context.Context, namespace, podName, containerName string, command []string) error {
	req := c.Clientset.CoreV1().RESTClient().Post().
		Namespace(namespace).
		Resource("pods").
		Name(podName).
		SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: containerName,
			Command:   command,
			Stdin:     false,
			Stdout:    false,
			Stderr:    false,
			TTY:       false,
		}, scheme.ParameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(c.RestConfig, "POST", req.URL())
	if err != nil {
		return err
	}

	// 只需要启动进程，无需进一步处理（不需要读写）。
	return exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdin:  nil,
		Stdout: nil,
		Stderr: nil,
		Tty:    false,
	})
}

// StopStressProcess 停止 stress-ng 进程
func StopStressProcess(c *k8s.K8SClient, ctx context.Context) error {
	pods, err := c.Clientset.CoreV1().Pods("default").List(ctx, metav1.ListOptions{
		LabelSelector: "app=stress-ng",
	})
	if err != nil {
		return err
	}

	for _, pod := range pods.Items {
		_ = ExecInPod(c, ctx, "default", pod.Name, "stress-ng", []string{"sh", "-c", "pkill -9 stress-ng || true"})
	}
	return nil
}

// InstallStressPod 部署 stress-ng Pod
func InstallStressPod(c *k8s.K8SClient, ctx context.Context, hlsResources bool) (*corev1.Pod, error) {
	decoder := scheme.Codecs.UniversalDeserializer()
	obj, _, err := decoder.Decode([]byte(stressPodManifest), nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to decode manifest: %w", err)
	}

	pod, ok := obj.(*corev1.Pod)
	if !ok {
		return nil, fmt.Errorf("manifest is not a Pod")
	}

	if hlsResources {
		// 删除可抢占标签，设置为 HLS 级别
		delete(pod.Labels, "volcano.sh/preemptable")
		delete(pod.Annotations, "volcano.sh/preemptable")
		pod.Annotations["openfuyao.com/qos-level"] = "HLS"
		pod.Spec.Containers[0].Resources = corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    mustParseQuantity("1"),
				corev1.ResourceMemory: mustParseQuantity("8Gi"),
			},
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    mustParseQuantity("1"),
				corev1.ResourceMemory: mustParseQuantity("8Gi"),
			},
		}
	}

	createdPod, err := c.Clientset.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create stress pod: %w", err)
	}

	// 等待 Pod 变为 Running
	if !WaitPodCondition(c, ctx, createdPod.Name, createdPod.Namespace, "Running", 10*time.Second) {
		_ = c.Clientset.CoreV1().Pods(createdPod.Namespace).Delete(ctx, createdPod.Name, metav1.DeleteOptions{})
		return nil, fmt.Errorf("stress pod failed to start")
	}

	return createdPod, nil
}

// StartStressProcess 启动 stress-ng 进程
func StartStressProcess(c *k8s.K8SClient, ctx context.Context, name, namespace string, config StressConfig) error {
	var cmdParts []string
	cmdParts = append(cmdParts, "stress-ng")

	if config.CPUCores > 0 && config.CPULoad > 0 {
		cmdParts = append(cmdParts, "--cpu", fmt.Sprintf("%d", config.CPUCores))
		cmdParts = append(cmdParts, "--cpu-method", "pi")
		cmdParts = append(cmdParts, "--cpu-load", fmt.Sprintf("%d", config.CPULoad))
	}

	if config.VM > 0 && config.VMBytes > 0 {
		cmdParts = append(cmdParts, "--vm", fmt.Sprintf("%d", config.VM))
		cmdParts = append(cmdParts, "--vm-bytes", fmt.Sprintf("%d%%", config.VMBytes))
	}

	command := fmt.Sprintf("%s > /dev/null 2>&1 &", strings.Join(cmdParts, " "))
	return ExecInPod(c, ctx, namespace, name, "stress-ng", []string{"sh", "-c", command})
}

// CheckNodeAllocatableCPU 检查集群中是否至少有 minMilliCPU 毫核的可分配 CPU（allocatable - 已 running pods 的 requests 总和）。
func CheckNodeAllocatableCPU(c *k8s.K8SClient, ctx context.Context, minMilliCPU int64) (bool, error) {
	nodes, err := c.Clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("list nodes: %w", err)
	}

	var totalAllocatable, totalRequested int64
	for _, node := range nodes.Items {
		totalAllocatable += node.Status.Allocatable.Cpu().MilliValue()
	}

	pods, err := c.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: "status.phase!=Succeeded,status.phase!=Failed",
	})
	if err != nil {
		return false, fmt.Errorf("list pods: %w", err)
	}
	for _, pod := range pods.Items {
		if pod.Spec.NodeName == "" {
			continue
		}
		for _, container := range pod.Spec.Containers {
			if cpuReq, ok := container.Resources.Requests[corev1.ResourceCPU]; ok {
				totalRequested += cpuReq.MilliValue()
			}
		}
	}

	available := totalAllocatable - totalRequested
	return available >= minMilliCPU, nil
}

// CheckVolcanoExists 检查 Volcano 是否已安装
func CheckVolcanoExists(c *k8s.K8SClient, ctx context.Context) bool {
	pods, err := c.Clientset.CoreV1().Pods("volcano-system").List(ctx, metav1.ListOptions{})
	if err != nil {
		return false
	}
	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "volcano") {
			return true
		}
	}
	return false
}

// CheckVolcanoNotExists 检查 Volcano 是否未安装
func CheckVolcanoNotExists(c *k8s.K8SClient, ctx context.Context) bool {
	pods, err := c.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return false
	}
	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "volcano") {
			return false
		}
	}
	return true
}

// GetRubikLogPattern 检查 rubik 日志是否匹配指定模式
func GetRubikLogPattern(c *k8s.K8SClient, ctx context.Context, pattern string) (bool, error) {
	log, err := GetRubikLog(c, ctx)
	if err != nil {
		return false, err
	}
	matched, err := regexp.MatchString(pattern, log)
	if err != nil {
		return false, err
	}
	return matched, nil
}

// WaitVolcanoWebhookReady 等待 Volcano admission webhook 服务就绪
func WaitVolcanoWebhookReady(c *k8s.K8SClient, ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		// 检查 volcano-admission 的 deployment 是否就绪。
		pods, err := c.Clientset.CoreV1().Pods("volcano-system").List(ctx, metav1.ListOptions{
			LabelSelector: "app=volcano-admission",
		})
		if err != nil {
			time.Sleep(2 * time.Second)
			continue
		}

		allReady := len(pods.Items) > 0
		for _, pod := range pods.Items {
			if pod.Status.Phase != corev1.PodRunning {
				allReady = false
				break
			}
			// 检查所有容器是否 ready
			for _, cs := range pod.Status.ContainerStatuses {
				if !cs.Ready {
					allReady = false
					break
				}
			}
		}

		if allReady {
			serviceNames := []string{"volcano-admission-service"}
			services, svcErr := c.Clientset.CoreV1().Services("volcano-system").List(ctx, metav1.ListOptions{
				LabelSelector: "app=volcano-admission",
			})
			if svcErr == nil {
				for _, svc := range services.Items {
					exists := false
					for _, name := range serviceNames {
						if name == svc.Name {
							exists = true
							break
						}
					}
					if !exists {
						serviceNames = append(serviceNames, svc.Name)
					}
				}
			}

			for _, serviceName := range serviceNames {
				endpointSlices, epErr := c.Clientset.DiscoveryV1().EndpointSlices("volcano-system").List(ctx, metav1.ListOptions{
					LabelSelector: fmt.Sprintf("kubernetes.io/service-name=%s", serviceName),
				})
				if epErr != nil {
					continue
				}

				hasAddress := false
				for _, endpointSlice := range endpointSlices.Items {
					for _, endpoint := range endpointSlice.Endpoints {
						if len(endpoint.Addresses) == 0 {
							continue
						}
						if endpoint.Conditions.Ready != nil && !*endpoint.Conditions.Ready {
							continue
						}
						hasAddress = true
						break
					}
					if hasAddress {
						break
					}
				}

				if hasAddress {
					// 额外等待几秒确保 webhook endpoint 完全可用。
					time.Sleep(5 * time.Second)
					return nil
				}
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("volcano admission webhook not ready within %v", timeout)
}

// WaitVolcanoSchedulerReady 等待 Volcano scheduler 就绪
func WaitVolcanoSchedulerReady(c *k8s.K8SClient, ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := c.Clientset.CoreV1().Pods("volcano-system").List(ctx, metav1.ListOptions{
			LabelSelector: "app=volcano-scheduler",
		})
		if err != nil {
			time.Sleep(2 * time.Second)
			continue
		}

		allReady := len(pods.Items) > 0
		for _, pod := range pods.Items {
			if pod.Status.Phase != corev1.PodRunning {
				allReady = false
				break
			}
			for _, cs := range pod.Status.ContainerStatuses {
				if !cs.Ready {
					allReady = false
					break
				}
			}
		}

		if allReady {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("volcano scheduler not ready within %v", timeout)
}

// EnsureStressPodDeleted 确保 stress pod 已被完全删除
func EnsureStressPodDeleted(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) error {
	// 先尝试删除
	err := c.Clientset.CoreV1().Pods(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !errors.IsNotFound(err) {
		return fmt.Errorf("failed to delete pod %s/%s: %w", namespace, name, err)
	}

	// 等待完全消失
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		_, err := c.Clientset.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("pod %s/%s still exists after %v", namespace, name, timeout)
}

// NUMA 亲和插件相关常量
const (
	ColocationNUMAAffinityOptsKey = "numa-affinity-options"
	// NUMA 亲和插件 Pod 的名称前缀（colocation-overquota-agent）
	NUMAPluginPodPrefix = "colocation-overquota-agent"
)

// GetNUMAPluginLogs 获取 NUMA 亲和插件的日志
func GetNUMAPluginLogs(c *k8s.K8SClient, ctx context.Context, tailLines int64) (string, error) {
	pods, err := c.Clientset.CoreV1().Pods(ColocationConfigNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to list pods in namespace %s: %w", ColocationConfigNamespace, err)
	}

	var allLogs strings.Builder
	foundPod := false
	for _, pod := range pods.Items {
		// 匹配 colocation-overquota-agent 开头的 Pod
		if !strings.HasPrefix(pod.Name, NUMAPluginPodPrefix) {
			continue
		}
		foundPod = true
		opts := &corev1.PodLogOptions{}
		if tailLines > 0 {
			opts.TailLines = &tailLines
		}
		log, err := c.Clientset.CoreV1().Pods(ColocationConfigNamespace).GetLogs(
			pod.Name, opts,
		).Do(ctx).Raw()
		if err != nil {
			continue
		}
		allLogs.WriteString(string(log))
		allLogs.WriteString("\n")
	}
	if !foundPod {
		return "", fmt.Errorf("no NUMA plugin pod found with prefix %s", NUMAPluginPodPrefix)
	}
	return allLogs.String(), nil
}

// CheckNUMALogPattern 检查 NUMA 插件日志是否包含指定模式
func CheckNUMALogPattern(c *k8s.K8SClient, ctx context.Context, pattern string, tailLines int64) (bool, error) {
	logs, err := GetNUMAPluginLogs(c, ctx, tailLines)
	if err != nil {
		return false, err
	}
	matched, err := regexp.MatchString(pattern, logs)
	if err != nil {
		return false, fmt.Errorf("invalid pattern %q: %w", pattern, err)
	}
	return matched, nil
}

// CollectNUMADiagnostics 收集 NUMA 亲和插件诊断信息，用于用例失败时留证：
// agent Pod 状态与重启次数、当前 numa-affinity-options 配置、最近日志。
func CollectNUMADiagnostics(c *k8s.K8SClient, ctx context.Context, tailLines int64) string {
	var sb strings.Builder

	pods, err := c.Clientset.CoreV1().Pods(ColocationConfigNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		fmt.Fprintf(&sb, "list pods in %s failed: %v\n", ColocationConfigNamespace, err)
		return sb.String()
	}
	for _, pod := range pods.Items {
		if !strings.HasPrefix(pod.Name, NUMAPluginPodPrefix) {
			continue
		}
		restarts := int32(0)
		for _, cs := range pod.Status.ContainerStatuses {
			restarts += cs.RestartCount
		}
		startedAt := "unknown"
		if pod.Status.StartTime != nil {
			startedAt = pod.Status.StartTime.Time.Format(time.RFC3339)
		}
		fmt.Fprintf(&sb, "agent pod %s: phase=%s restarts=%d startedAt=%s\n",
			pod.Name, pod.Status.Phase, restarts, startedAt)
	}

	opts, err := GetColocationNUMAAffinityOptions(c, ctx)
	if err != nil {
		fmt.Fprintf(&sb, "get numa-affinity-options failed: %v\n", err)
	} else {
		fmt.Fprintf(&sb, "current numa-affinity-options: %s\n", opts)
	}

	logs, err := GetNUMAPluginLogs(c, ctx, tailLines)
	if err != nil {
		fmt.Fprintf(&sb, "get agent logs failed: %v\n", err)
	} else {
		sb.WriteString("--- agent recent logs ---\n")
		sb.WriteString(logs)
	}
	return sb.String()
}

// PatchColocationNUMAAffinityEnable 启用/禁用 NUMA 亲和插件
func PatchColocationNUMAAffinityEnable(c *k8s.K8SClient, ctx context.Context, enabled bool) error {
	cm, err := GetColocationConfigMap(c, ctx)
	if err != nil {
		return err
	}
	raw := ""
	if cm.Data != nil {
		raw = strings.TrimSpace(cm.Data[ColocationNUMAAffinityOptsKey])
	}
	var root map[string]interface{}
	if raw != "" {
		if err := json.Unmarshal([]byte(raw), &root); err != nil {
			return fmt.Errorf("parse numa-affinity-options: %w", err)
		}
	} else {
		root = map[string]interface{}{}
	}
	root["enable"] = enabled

	out, err := json.Marshal(root)
	if err != nil {
		return fmt.Errorf("marshal numa-affinity-options: %w", err)
	}
	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	cm.Data[ColocationNUMAAffinityOptsKey] = string(out)
	_, err = c.Clientset.CoreV1().ConfigMaps(cm.Namespace).Update(ctx, cm, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("update colocation-config numa-affinity-options: %w", err)
	}
	return nil
}

// GetColocationNUMAAffinityOptions 获取 NUMA 亲和配置
func GetColocationNUMAAffinityOptions(c *k8s.K8SClient, ctx context.Context) (string, error) {
	cm, err := GetColocationConfigMap(c, ctx)
	if err != nil {
		return "", err
	}
	if cm.Data == nil {
		return "", nil
	}
	return cm.Data[ColocationNUMAAffinityOptsKey], nil
}

// UpdateColocationNUMAAffinityOptions 更新 NUMA 亲和配置（用于恢复）
func UpdateColocationNUMAAffinityOptions(c *k8s.K8SClient, ctx context.Context, numaOptsJSON string) error {
	cm, err := GetColocationConfigMap(c, ctx)
	if err != nil {
		return fmt.Errorf("get colocation-config: %w", err)
	}
	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	cm.Data[ColocationNUMAAffinityOptsKey] = numaOptsJSON
	_, err = c.Clientset.CoreV1().ConfigMaps(cm.Namespace).Update(ctx, cm, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("update colocation-config numa-affinity-options: %w", err)
	}
	return nil
}

// WaitNUMAPluginReady 等待 NUMA 亲和插件就绪
func WaitNUMAPluginReady(c *k8s.K8SClient, ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := c.Clientset.CoreV1().Pods(ColocationConfigNamespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			time.Sleep(2 * time.Second)
			continue
		}

		foundPod := false
		allReady := true
		for _, pod := range pods.Items {
			// 匹配 colocation-overquota-agent 开头的 Pod
			if !strings.HasPrefix(pod.Name, NUMAPluginPodPrefix) {
				continue
			}
			foundPod = true
			if pod.Status.Phase != corev1.PodRunning {
				allReady = false
				break
			}
			for _, cs := range pod.Status.ContainerStatuses {
				if !cs.Ready {
					allReady = false
					break
				}
			}
		}

		if foundPod && allReady {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("NUMA affinity plugin not ready within %v", timeout)
}

// InstallNoQoSPod 部署没有 qos-level 标签的测试 Pod
func InstallNoQoSPod(c *k8s.K8SClient, ctx context.Context, name string) (*corev1.Pod, error) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ColocationConfigNamespace,
		},
		Spec: corev1.PodSpec{
			SchedulerName: "volcano",
			Containers: []corev1.Container{
				{
					Name:    "nginx",
					Image:   "docker.io/library/nginx:latest",
					Command: []string{"sleep", "3600"},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    mustParseQuantity("100m"),
							corev1.ResourceMemory: mustParseQuantity("128Mi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    mustParseQuantity("200m"),
							corev1.ResourceMemory: mustParseQuantity("256Mi"),
						},
					},
				},
			},
		},
	}
	return c.Clientset.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
}

// InstallHLSPodInNamespace 在指定命名空间部署 HLS Pod
func InstallHLSPodInNamespace(c *k8s.K8SClient, ctx context.Context, name, namespace string) (*corev1.Pod, error) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Annotations: map[string]string{
				"openfuyao.com/qos-level": "HLS",
			},
		},
		Spec: corev1.PodSpec{
			SchedulerName: "volcano",
			Containers: []corev1.Container{
				{
					Name:            "nginx",
					Image:           "docker.io/library/nginx:latest",
					ImagePullPolicy: corev1.PullIfNotPresent,
					Command:         []string{"sleep", "3600"},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    mustParseQuantity("1"),
							corev1.ResourceMemory: mustParseQuantity("256Mi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    mustParseQuantity("1"),
							corev1.ResourceMemory: mustParseQuantity("256Mi"),
						},
					},
				},
			},
		},
	}
	return c.Clientset.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
}

// InstallLSPodInNamespace 在指定命名空间部署 LS Pod
func InstallLSPodInNamespace(c *k8s.K8SClient, ctx context.Context, name, namespace string) (*corev1.Pod, error) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Annotations: map[string]string{
				"openfuyao.com/qos-level": "LS",
			},
		},
		Spec: corev1.PodSpec{
			SchedulerName: "volcano",
			Containers: []corev1.Container{
				{
					Name:    "nginx",
					Image:   "docker.io/library/nginx:latest",
					Command: []string{"sleep", "3600"},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    mustParseQuantity("100m"),
							corev1.ResourceMemory: mustParseQuantity("128Mi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    mustParseQuantity("500m"),
							corev1.ResourceMemory: mustParseQuantity("512Mi"),
						},
					},
				},
			},
		},
	}
	return c.Clientset.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
}

// InstallBEPodInNamespace 在指定命名空间部署 BE Pod
func InstallBEPodInNamespace(c *k8s.K8SClient, ctx context.Context, name, namespace string) (*corev1.Pod, error) {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				"volcano.sh/preemptable": "true",
			},
			Annotations: map[string]string{
				"openfuyao.com/qos-level": "BE",
				"volcano.sh/preemptable":  "true",
			},
		},
		Spec: corev1.PodSpec{
			SchedulerName: "volcano",
			Containers: []corev1.Container{
				{
					Name:    "nginx",
					Image:   "docker.io/library/nginx:latest",
					Command: []string{"sleep", "3600"},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    mustParseQuantity("100m"),
							corev1.ResourceMemory: mustParseQuantity("128Mi"),
						},
						Limits: corev1.ResourceList{
							corev1.ResourceCPU:    mustParseQuantity("200m"),
							corev1.ResourceMemory: mustParseQuantity("256Mi"),
						},
					},
				},
			},
		},
	}
	return c.Clientset.CoreV1().Pods(pod.Namespace).Create(ctx, pod, metav1.CreateOptions{})
}

// InstallHLSDeployment 创建 HLS QoS 级别的 Deployment
func InstallHLSDeployment(c *k8s.K8SClient, ctx context.Context, name, namespace string, replicas int32) error {
	deploy := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"replicas": replicas,
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"app": name,
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"app": name,
						},
						"annotations": map[string]interface{}{
							"openfuyao.com/qos-level": "HLS",
						},
					},
					"spec": map[string]interface{}{
						"schedulerName": "volcano",
						"containers": []interface{}{
							map[string]interface{}{
								"name":            "nginx",
								"image":           "docker.io/library/nginx:latest",
								"imagePullPolicy": "IfNotPresent",
								"command":         []string{"sleep", "3600"},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{
										"cpu":    "1",
										"memory": "256Mi",
									},
									"limits": map[string]interface{}{
										"cpu":    "1",
										"memory": "256Mi",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, deploy, metav1.CreateOptions{})
	return err
}

// InstallLSDeployment 创建 LS QoS 级别的 Deployment
func InstallLSDeployment(c *k8s.K8SClient, ctx context.Context, name, namespace string, replicas int32) error {
	deploy := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"replicas": replicas,
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"app": name,
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"app": name,
						},
						"annotations": map[string]interface{}{
							"openfuyao.com/qos-level": "LS",
						},
					},
					"spec": map[string]interface{}{
						"schedulerName": "volcano",
						"containers": []interface{}{
							map[string]interface{}{
								"name":    "nginx",
								"image":   "docker.io/library/nginx:latest",
								"command": []string{"sleep", "3600"},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{
										"cpu":    "100m",
										"memory": "128Mi",
									},
									"limits": map[string]interface{}{
										"cpu":    "500m",
										"memory": "512Mi",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, deploy, metav1.CreateOptions{})
	return err
}

// DeleteDeployment 删除 Deployment
func DeleteDeployment(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}
	return c.DynamicClient.Resource(gvr).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

// WaitDeploymentReady 等待 Deployment 就绪
func WaitDeploymentReady(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}
	for time.Now().Before(deadline) {
		deploy, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil {
			replicas, _, _ := unstructured.NestedInt64(deploy.Object, "spec", "replicas")
			readyReplicas, _, _ := unstructured.NestedInt64(deploy.Object, "status", "readyReplicas")
			if readyReplicas == replicas && replicas > 0 {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("deployment %s not ready within %v", name, timeout)
}

// InstallLSStatefulSet 创建 LS QoS 级别的 StatefulSet
func InstallLSStatefulSet(c *k8s.K8SClient, ctx context.Context, name, namespace string, replicas int32) error {
	sts := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "StatefulSet",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"replicas":    replicas,
				"serviceName": name,
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"app": name,
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"app": name,
						},
						"annotations": map[string]interface{}{
							"openfuyao.com/qos-level": "LS",
						},
					},
					"spec": map[string]interface{}{
						"schedulerName": "volcano",
						"containers": []interface{}{
							map[string]interface{}{
								"name":    "nginx",
								"image":   "docker.io/library/nginx:latest",
								"command": []string{"sleep", "3600"},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{
										"cpu":    "100m",
										"memory": "128Mi",
									},
									"limits": map[string]interface{}{
										"cpu":    "500m",
										"memory": "512Mi",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "statefulsets",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, sts, metav1.CreateOptions{})
	return err
}

// DeleteStatefulSet 删除 StatefulSet
func DeleteStatefulSet(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "statefulsets",
	}
	return c.DynamicClient.Resource(gvr).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

// WaitStatefulSetReady 等待 StatefulSet 就绪
func WaitStatefulSetReady(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "statefulsets",
	}
	for time.Now().Before(deadline) {
		sts, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil {
			replicas, _, _ := unstructured.NestedInt64(sts.Object, "spec", "replicas")
			readyReplicas, _, _ := unstructured.NestedInt64(sts.Object, "status", "readyReplicas")
			if readyReplicas == replicas && replicas > 0 {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("statefulset %s not ready within %v", name, timeout)
}

// InstallLSJob 创建 LS QoS 级别的 Job
func InstallLSJob(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	job := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "batch/v1",
			"kind":       "Job",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"annotations": map[string]interface{}{
							"openfuyao.com/qos-level": "LS",
						},
					},
					"spec": map[string]interface{}{
						"schedulerName": "volcano",
						"restartPolicy": "Never",
						"containers": []interface{}{
							map[string]interface{}{
								"name":    "job",
								"image":   "docker.io/library/busybox:latest",
								"command": []string{"sleep", "60"},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{
										"cpu":    "100m",
										"memory": "128Mi",
									},
									"limits": map[string]interface{}{
										"cpu":    "500m",
										"memory": "512Mi",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "batch",
		Version:  "v1",
		Resource: "jobs",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, job, metav1.CreateOptions{})
	return err
}

// DeleteJob 删除 Job
func DeleteJob(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	gvr := schema.GroupVersionResource{
		Group:    "batch",
		Version:  "v1",
		Resource: "jobs",
	}
	// 使用 PropagationPolicy: Foreground 确保删除关联的 Pod
	propagationPolicy := metav1.DeletePropagationForeground
	return c.DynamicClient.Resource(gvr).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{
		PropagationPolicy: &propagationPolicy,
	})
}

// WaitJobPodRunning 等待 Job 的 Pod 开始运行
func WaitJobPodRunning(c *k8s.K8SClient, ctx context.Context, jobName, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pods, err := c.Clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("job-name=%s", jobName),
		})
		if err == nil && len(pods.Items) > 0 {
			for _, pod := range pods.Items {
				if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodSucceeded {
					return nil
				}
			}
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("job %s pod not running within %v", jobName, timeout)
}

// InstallLSDaemonSet 创建 LS QoS 级别的 DaemonSet
func InstallLSDaemonSet(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	ds := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "DaemonSet",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"app": name,
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"app": name,
						},
						"annotations": map[string]interface{}{
							"openfuyao.com/qos-level": "LS",
						},
					},
					"spec": map[string]interface{}{
						"schedulerName": "volcano",
						"tolerations": []interface{}{
							map[string]interface{}{
								"operator": "Exists",
							},
						},
						"containers": []interface{}{
							map[string]interface{}{
								"name":    "nginx",
								"image":   "docker.io/library/nginx:latest",
								"command": []string{"sleep", "3600"},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{
										"cpu":    "100m",
										"memory": "128Mi",
									},
									"limits": map[string]interface{}{
										"cpu":    "500m",
										"memory": "512Mi",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "daemonsets",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, ds, metav1.CreateOptions{})
	return err
}

// DeleteDaemonSet 删除 DaemonSet
func DeleteDaemonSet(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "daemonsets",
	}
	return c.DynamicClient.Resource(gvr).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

// WaitDaemonSetReady 等待 DaemonSet 就绪
func WaitDaemonSetReady(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "daemonsets",
	}
	for time.Now().Before(deadline) {
		ds, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil {
			desiredNumberScheduled, _, _ := unstructured.NestedInt64(ds.Object, "status", "desiredNumberScheduled")
			numberReady, _, _ := unstructured.NestedInt64(ds.Object, "status", "numberReady")
			if numberReady == desiredNumberScheduled && desiredNumberScheduled > 0 {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	printDaemonSetTimeoutDiagnostics(c, ctx, name, namespace, "all-ready", timeout)
	return fmt.Errorf("daemonset %s not ready within %v", name, timeout)
}

// WaitDaemonSetAtLeastOneReady 等待 DaemonSet 至少有一个 Ready 副本。
func WaitDaemonSetAtLeastOneReady(c *k8s.K8SClient, ctx context.Context, name, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "daemonsets",
	}
	var lastDesired, lastReady, lastAvailable int64
	for time.Now().Before(deadline) {
		ds, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil {
			desiredNumberScheduled, _, _ := unstructured.NestedInt64(ds.Object, "status", "desiredNumberScheduled")
			numberReady, _, _ := unstructured.NestedInt64(ds.Object, "status", "numberReady")
			numberAvailable, _, _ := unstructured.NestedInt64(ds.Object, "status", "numberAvailable")
			lastDesired, lastReady, lastAvailable = desiredNumberScheduled, numberReady, numberAvailable

			if desiredNumberScheduled > 0 && numberReady > 0 && numberAvailable > 0 {
				return nil
			}
		}
		time.Sleep(2 * time.Second)
	}
	printDaemonSetTimeoutDiagnostics(c, ctx, name, namespace, "at-least-one-ready", timeout)
	return fmt.Errorf(
		"daemonset %s has no ready replicas within %v (desired=%d ready=%d available=%d)",
		name,
		timeout,
		lastDesired,
		lastReady,
		lastAvailable,
	)
}

// InstallLSCronJob 创建 LS QoS 级别的 CronJob
func InstallLSCronJob(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	cronJob := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "batch/v1",
			"kind":       "CronJob",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"schedule": "* * * * *", // 每分钟执行一次
				"jobTemplate": map[string]interface{}{
					"spec": map[string]interface{}{
						"template": map[string]interface{}{
							"metadata": map[string]interface{}{
								"annotations": map[string]interface{}{
									"openfuyao.com/qos-level": "LS",
								},
							},
							"spec": map[string]interface{}{
								"schedulerName": "volcano",
								"restartPolicy": "Never",
								"containers": []interface{}{
									map[string]interface{}{
										"name":    "cronjob",
										"image":   "docker.io/library/busybox:latest",
										"command": []string{"sleep", "30"},
										"resources": map[string]interface{}{
											"requests": map[string]interface{}{
												"cpu":    "100m",
												"memory": "128Mi",
											},
											"limits": map[string]interface{}{
												"cpu":    "500m",
												"memory": "512Mi",
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "batch",
		Version:  "v1",
		Resource: "cronjobs",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, cronJob, metav1.CreateOptions{})
	return err
}

// DeleteCronJob 删除 CronJob
func DeleteCronJob(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	gvr := schema.GroupVersionResource{
		Group:    "batch",
		Version:  "v1",
		Resource: "cronjobs",
	}
	propagationPolicy := metav1.DeletePropagationForeground
	return c.DynamicClient.Resource(gvr).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{
		PropagationPolicy: &propagationPolicy,
	})
}

// WaitCronJobPodRunning 等待 CronJob 触发的 Pod 开始运行
func WaitCronJobPodRunning(c *k8s.K8SClient, ctx context.Context, cronJobName, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		// 列出所有 Job
		gvr := schema.GroupVersionResource{
			Group:    "batch",
			Version:  "v1",
			Resource: "jobs",
		}
		jobs, err := c.DynamicClient.Resource(gvr).Namespace(namespace).List(ctx, metav1.ListOptions{})
		if err == nil {
			for _, job := range jobs.Items {
				// 检查是否是该 CronJob 创建的 Job
				ownerRefs, _, _ := unstructured.NestedSlice(job.Object, "metadata", "ownerReferences")
				for _, ref := range ownerRefs {
					refMap, ok := ref.(map[string]interface{})
					if !ok {
						continue
					}
					if refMap["kind"] == "CronJob" && refMap["name"] == cronJobName {
						jobName := job.GetName()
						// 检查 Job 的 Pod 是否运行
						pods, err := c.Clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
							LabelSelector: fmt.Sprintf("job-name=%s", jobName),
						})
						if err == nil && len(pods.Items) > 0 {
							for _, pod := range pods.Items {
								if pod.Status.Phase == corev1.PodRunning || pod.Status.Phase == corev1.PodSucceeded {
									return nil
								}
							}
						}
					}
				}
			}
		}
		time.Sleep(5 * time.Second)
	}
	return fmt.Errorf("cronjob %s pod not running within %v", cronJobName, timeout)
}

// InstallNoQoSDeployment 创建没有 qos-level 标签的 Deployment
func InstallNoQoSDeployment(c *k8s.K8SClient, ctx context.Context, name, namespace string, replicas int32) error {
	deploy := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"replicas": replicas,
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"app": name,
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"app": name,
						},
					},
					"spec": map[string]interface{}{
						"schedulerName": "volcano",
						"containers": []interface{}{
							map[string]interface{}{
								"name":    "nginx",
								"image":   "docker.io/library/nginx:latest",
								"command": []string{"sleep", "3600"},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{
										"cpu":    "100m",
										"memory": "128Mi",
									},
									"limits": map[string]interface{}{
										"cpu":    "200m",
										"memory": "256Mi",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "deployments",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, deploy, metav1.CreateOptions{})
	return err
}

// InstallNoQoSStatefulSet 创建没有 qos-level 标签的 StatefulSet
func InstallNoQoSStatefulSet(c *k8s.K8SClient, ctx context.Context, name, namespace string, replicas int32) error {
	sts := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "StatefulSet",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"replicas":    replicas,
				"serviceName": name,
				"selector": map[string]interface{}{
					"matchLabels": map[string]interface{}{
						"app": name,
					},
				},
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"app": name,
						},
					},
					"spec": map[string]interface{}{
						"schedulerName": "volcano",
						"containers": []interface{}{
							map[string]interface{}{
								"name":    "nginx",
								"image":   "docker.io/library/nginx:latest",
								"command": []string{"sleep", "3600"},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{
										"cpu":    "100m",
										"memory": "128Mi",
									},
									"limits": map[string]interface{}{
										"cpu":    "200m",
										"memory": "256Mi",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "apps",
		Version:  "v1",
		Resource: "statefulsets",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, sts, metav1.CreateOptions{})
	return err
}

// InstallNoQoSJob 创建没有 qos-level 标签的 Job
func InstallNoQoSJob(c *k8s.K8SClient, ctx context.Context, name, namespace string) error {
	job := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "batch/v1",
			"kind":       "Job",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"schedulerName": "volcano",
						"restartPolicy": "Never",
						"containers": []interface{}{
							map[string]interface{}{
								"name":    "job",
								"image":   "docker.io/library/busybox:latest",
								"command": []string{"sleep", "60"},
								"resources": map[string]interface{}{
									"requests": map[string]interface{}{
										"cpu":    "100m",
										"memory": "128Mi",
									},
									"limits": map[string]interface{}{
										"cpu":    "200m",
										"memory": "256Mi",
									},
								},
							},
						},
					},
				},
			},
		},
	}

	gvr := schema.GroupVersionResource{
		Group:    "batch",
		Version:  "v1",
		Resource: "jobs",
	}
	_, err := c.DynamicClient.Resource(gvr).Namespace(namespace).Create(ctx, job, metav1.CreateOptions{})
	return err
}
