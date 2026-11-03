package utils

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	esutils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// EnsureLeaderWorkerSetAPI 探测集群已安装的 LeaderWorkerSet API 版本（需预先安装 LWS Operator）。
func EnsureLeaderWorkerSetAPI(clientset *kubernetes.Clientset) error {
	return esutils.ConfigureLeaderWorkerSetAPI(clientset)
}

// EnsureLeaderWorkerSetController 确保 LWS Operator 已安装且控制器就绪；已就绪则跳过，否则 Helm 安装。
func EnsureLeaderWorkerSetController(clientset *kubernetes.Clientset, ctx context.Context) error {
	return esutils.EnsureLeaderWorkerSetController(clientset, ctx)
}

// CleanupLeaderWorkerSetController 仅卸载本进程安装的 LWS Operator；集群里原本已有的不会动。
func CleanupLeaderWorkerSetController(clientset *kubernetes.Clientset, ctx context.Context) error {
	return esutils.CleanupLeaderWorkerSetController(clientset, ctx)
}

// FindLeaderWorkerSetNameByRole 在命名空间中查找名称包含 role（prefill/decode）的 LeaderWorkerSet。
func FindLeaderWorkerSetNameByRole(dynamicClient dynamic.Interface, ctx context.Context, namespace, role string) (string, error) {
	role = strings.ToLower(strings.TrimSpace(role))
	list, err := dynamicClient.Resource(esutils.LeaderWorkerSetGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("list leaderworkersets in %s: %w", namespace, err)
	}
	var matches []string
	for _, item := range list.Items {
		name := strings.ToLower(item.GetName())
		if strings.Contains(name, role) {
			matches = append(matches, item.GetName())
		}
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("no LeaderWorkerSet containing %q found in namespace %s", role, namespace)
	}
	sort.Strings(matches)
	if len(matches) > 1 {
		GinkgoWriter.Printf("multiple LeaderWorkerSets match role %q, using %s (candidates: %v)\n", role, matches[0], matches)
	}
	return matches[0], nil
}

func leaderWorkerSetReadyReplicas(obj *unstructured.Unstructured) (int64, error) {
	ready, found, err := unstructured.NestedInt64(obj.Object, "status", "readyReplicas")
	if err != nil {
		return -1, err
	}
	if !found {
		return -1, fmt.Errorf("status.readyReplicas not set")
	}
	return ready, nil
}

func leaderWorkerSetSpecReplicas(obj *unstructured.Unstructured) int64 {
	specRep, found, err := unstructured.NestedInt64(obj.Object, "spec", "replicas")
	if err != nil || !found || specRep <= 0 {
		return 1
	}
	return specRep
}

// WaitForLeaderWorkerSetFullyReady 等待指定 LeaderWorkerSet 的 readyReplicas 达到 spec.replicas。
func WaitForLeaderWorkerSetFullyReady(dynamicClient dynamic.Interface, ctx context.Context, namespace, name string, timeout time.Duration) {
	Eventually(func() error {
		obj, err := dynamicClient.Resource(esutils.LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		specRep := leaderWorkerSetSpecReplicas(obj)
		ready, err := leaderWorkerSetReadyReplicas(obj)
		if err != nil {
			return fmt.Errorf("LeaderWorkerSet %s/%s: %w", namespace, name, err)
		}
		if ready < specRep {
			return fmt.Errorf("LeaderWorkerSet %s/%s readyReplicas=%d spec.replicas=%d", namespace, name, ready, specRep)
		}
		return nil
	}, timeout, 10*time.Second).Should(Succeed(), "LeaderWorkerSet %s/%s 应就绪", namespace, name)
}

// CountReadyLeaderWorkerSetsByRole 统计名称包含 role 且 readyReplicas>=1 的 LeaderWorkerSet 个数。
// PD 组扩缩容会为每个 group 各建一套 prefill/decode LWS，因此应用 LWS 个数而不是 Deployment 副本。
func CountReadyLeaderWorkerSetsByRole(dynamicClient dynamic.Interface, ctx context.Context, namespace, role string) (int32, error) {
	role = strings.ToLower(strings.TrimSpace(role))
	list, err := dynamicClient.Resource(esutils.LeaderWorkerSetGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return 0, fmt.Errorf("list leaderworkersets in %s: %w", namespace, err)
	}
	var n int32
	for i := range list.Items {
		item := &list.Items[i]
		if !strings.Contains(strings.ToLower(item.GetName()), role) {
			continue
		}
		ready, err := leaderWorkerSetReadyReplicas(item)
		if err != nil || ready < 1 {
			GinkgoWriter.Printf("LWS %s/%s 尚未 Ready: %v\n", namespace, item.GetName(), err)
			continue
		}
		n++
	}
	return n, nil
}

// VerifyLeaderWorkerSetTemplateSize 校验 spec.leaderWorkerTemplate.size == expectedSize。
func VerifyLeaderWorkerSetTemplateSize(dynamicClient dynamic.Interface, ctx context.Context, namespace, lwsName string, expectedSize int64) error {
	obj, err := dynamicClient.Resource(esutils.LeaderWorkerSetGVR).Namespace(namespace).Get(ctx, lwsName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("get leaderworkerset %s/%s: %w", namespace, lwsName, err)
	}
	size, found, err := unstructured.NestedInt64(obj.Object, "spec", "leaderWorkerTemplate", "size")
	if err != nil {
		return fmt.Errorf("read leaderWorkerTemplate.size: %w", err)
	}
	if !found {
		return fmt.Errorf("leaderworkerset %s/%s missing spec.leaderWorkerTemplate.size", namespace, lwsName)
	}
	if size != expectedSize {
		return fmt.Errorf("leaderworkerset %s/%s leaderWorkerTemplate.size=%d, want %d", namespace, lwsName, size, expectedSize)
	}
	GinkgoWriter.Printf("LeaderWorkerSet %s/%s leaderWorkerTemplate.size=%d OK\n", namespace, lwsName, size)
	return nil
}

// VerifyLeaderWorkerSetPodsDataParallelStartRanks 校验 LWS 关联 Pod 的实际运行命令中 --data-parallel-start-rank 与 expectedRanks 一致（按 Pod 名称排序）。
func VerifyLeaderWorkerSetPodsDataParallelStartRanks(restConfig *rest.Config, clientset *kubernetes.Clientset, ctx context.Context, namespace, lwsName string, expectedRanks []int) error {
	pods, err := listPodsForLeaderWorkerSet(clientset, ctx, namespace, lwsName)
	if err != nil {
		return err
	}
	if len(pods) != len(expectedRanks) {
		return fmt.Errorf("leaderworkerset %s/%s: got %d pods, want %d", namespace, lwsName, len(pods), len(expectedRanks))
	}
	var gotRanks []int
	for _, pod := range pods {
		rank, err := podDataParallelStartRankFromExec(restConfig, clientset, ctx, namespace, &pod)
		if err != nil {
			return fmt.Errorf("pod %s: %w", pod.Name, err)
		}
		gotRanks = append(gotRanks, rank)
		GinkgoWriter.Printf("pod %s --data-parallel-start-rank=%d\n", pod.Name, rank)
	}
	for i := range expectedRanks {
		if gotRanks[i] != expectedRanks[i] {
			return fmt.Errorf("leaderworkerset %s/%s: data-parallel-start-ranks=%v, want %v (pods sorted by name)", namespace, lwsName, gotRanks, expectedRanks)
		}
	}
	return nil
}

func podDataParallelStartRankFromExec(restConfig *rest.Config, clientset *kubernetes.Clientset, ctx context.Context, namespace string, pod *corev1.Pod) (int, error) {
	const flag = "--data-parallel-start-rank"

	// 1. 确定 engine 容器名 (复用 FIT_38 逻辑：通过 role 或匹配容器名)
	containerName := ""
	if role := pod.Labels[PDRoleLabelKey]; role != "" {
		if c, err := engineContainerForPDRole(role); err == nil {
			containerName = c
		}
	}
	if containerName == "" {
		for _, c := range pod.Spec.Containers {
			if c.Name == PrefillEngineContainer || c.Name == DecodeEngineContainer {
				containerName = c.Name
				break
			}
		}
	}
	if containerName == "" {
		return 0, fmt.Errorf("cannot determine engine container for pod %s", pod.Name)
	}

	// 2. 复用 GetVLLMServeCommand 获取实际运行命令
	target := PDEngineTarget{Pod: *pod, Container: containerName}
	cmdline, err := GetVLLMServeCommand(restConfig, clientset, ctx, namespace, target)
	if err != nil {
		return 0, err
	}

	// 3. 解析 --data-parallel-start-rank
	idx := strings.Index(cmdline, flag)
	if idx == -1 {
		return 0, fmt.Errorf("container args do not contain %s", flag)
	}

	rest := cmdline[idx+len(flag):]
	rest = strings.TrimSpace(rest)

	valStr := rest
	if strings.HasPrefix(rest, "=") {
		valStr = strings.TrimSpace(rest[1:])
	} else {
		if spaceIdx := strings.Index(rest, " "); spaceIdx != -1 {
			valStr = rest[:spaceIdx]
		}
	}

	v, err := strconv.Atoi(valStr)
	if err != nil {
		return 0, fmt.Errorf("parse %q value %q: %w", flag, valStr, err)
	}
	return v, nil
}

func listPodsForLeaderWorkerSet(clientset *kubernetes.Clientset, ctx context.Context, namespace, lwsName string) ([]corev1.Pod, error) {
	selectors := []string{
		"leaderworkerset.sigs.k8s.io/name=" + lwsName,
		"leaderworkerset.x-k8s.io/name=" + lwsName,
	}
	var pods []corev1.Pod
	for _, sel := range selectors {
		list, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: sel})
		if err != nil {
			continue
		}
		if len(list.Items) > 0 {
			pods = list.Items
			break
		}
	}
	if len(pods) == 0 {
		all, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return nil, fmt.Errorf("list pods in %s: %w", namespace, err)
		}
		for _, p := range all.Items {
			if strings.HasPrefix(p.Name, lwsName+"-") || strings.HasPrefix(p.Name, lwsName) {
				pods = append(pods, p)
			}
		}
	}
	if len(pods) == 0 {
		return nil, fmt.Errorf("no pods found for leaderworkerset %s/%s", namespace, lwsName)
	}
	sort.Slice(pods, func(i, j int) bool { return pods[i].Name < pods[j].Name })
	return pods, nil
}

func podDataParallelStartRank(pod *corev1.Pod) (int, error) {
	const flag = "--data-parallel-start-rank"
	for _, c := range pod.Spec.Containers {
		// 拼接 Command 和 Args 以处理可能的 shell 封装情况
		fullCmd := strings.Join(append(c.Command, c.Args...), " ")

		// 查找 flag
		idx := strings.Index(fullCmd, flag)
		if idx == -1 {
			continue
		}

		// 提取 flag 之后的部分
		rest := fullCmd[idx+len(flag):]
		rest = strings.TrimSpace(rest)

		// 处理 "=" 或 " " 分隔符
		valStr := rest
		if strings.HasPrefix(rest, "=") {
			valStr = strings.TrimSpace(rest[1:])
		} else {
			// 如果是空格分隔，取第一个 token
			if spaceIdx := strings.Index(rest, " "); spaceIdx != -1 {
				valStr = rest[:spaceIdx]
			}
		}

		v, err := strconv.Atoi(valStr)
		if err != nil {
			return 0, fmt.Errorf("parse %q value %q: %w", flag, valStr, err)
		}
		return v, nil
	}
	return 0, fmt.Errorf("container args do not contain %s", flag)
}
