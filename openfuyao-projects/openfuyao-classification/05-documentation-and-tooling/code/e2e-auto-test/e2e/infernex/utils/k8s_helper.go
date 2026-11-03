package utils

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	. "github.com/onsi/ginkgo/v2"
)

// K8SClientInterface 定义 K8S 客户端接口，便于测试和扩展
type K8SClientInterface interface {
	ListPods(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.PodList, error)
	ListServices(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.ServiceList, error)
	ListEndpointSlices(ctx context.Context, namespace string, opts metav1.ListOptions) (*discoveryv1.EndpointSliceList, error)
	ListNodes(ctx context.Context, opts metav1.ListOptions) (*corev1.NodeList, error)
	ListServiceAccounts(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.ServiceAccountList, error)
	DeleteServiceAccount(ctx context.Context, namespace, name string) error
	ListNamespaces(ctx context.Context, opts metav1.ListOptions) (*corev1.NamespaceList, error)
	DeleteNamespace(ctx context.Context, name string) error
	GetNamespace(ctx context.Context, name string) (*corev1.Namespace, error)
}

// ChooseNodeIPForNodePort 选择用于 NodePort 的节点 IP
func ChooseNodeIPForNodePort(ctx context.Context, k8sClient K8SClientInterface) (string, error) {
	// 优先用显式指定的节点 IP（在很多环境里更可控）
	if ip := os.Getenv("HERMES_NODE_IP"); ip != "" {
		return ip, nil
	}

	nodes, err := k8sClient.ListNodes(ctx, metav1.ListOptions{})
	if err != nil {
		return "", err
	}
	for _, n := range nodes.Items {
		for _, addr := range n.Status.Addresses {
			if addr.Type == corev1.NodeInternalIP && addr.Address != "" {
				return addr.Address, nil
			}
		}
	}
	for _, n := range nodes.Items {
		for _, addr := range n.Status.Addresses {
			if addr.Type == corev1.NodeExternalIP && addr.Address != "" {
				return addr.Address, nil
			}
		}
	}
	return "", fmt.Errorf("no node ip found; set HERMES_NODE_IP explicitly")
}

// PickServicePort 从 Service 中选择合适的端口
func PickServicePort(svc *corev1.Service) *corev1.ServicePort {
	if svc == nil || len(svc.Spec.Ports) == 0 {
		return nil
	}
	// 优先 http/80/8080
	for i := range svc.Spec.Ports {
		p := &svc.Spec.Ports[i]
		if strings.Contains(strings.ToLower(p.Name), "http") {
			return p
		}
	}
	for i := range svc.Spec.Ports {
		p := &svc.Spec.Ports[i]
		if p.Port == 80 || p.Port == 8080 {
			return p
		}
	}
	return &svc.Spec.Ports[0]
}

// DumpPodsAndServices 打印 Pod 和 Service 状态（用于调试）
func DumpPodsAndServices(ctx context.Context, k8sClient K8SClientInterface, namespace string) {
	if k8sClient == nil {
		return
	}
	GinkgoWriter.Printf("\n===== namespace=%s pods =====\n", namespace)
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		GinkgoWriter.Printf("list pods failed: %v\n", err)
	} else if len(pods.Items) == 0 {
		GinkgoWriter.Println("No pods found.")
	} else {
		for _, p := range pods.Items {
			ready := 0
			total := len(p.Status.ContainerStatuses)
			for _, cs := range p.Status.ContainerStatuses {
				if cs.Ready {
					ready++
				}
			}
			GinkgoWriter.Printf("- pod=%s phase=%s ready=%d/%d node=%s ip=%s\n", p.Name, p.Status.Phase, ready, total, p.Spec.NodeName, p.Status.PodIP)
		}
	}

	GinkgoWriter.Printf("\n===== namespace=%s services =====\n", namespace)
	svcs, err := k8sClient.ListServices(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		GinkgoWriter.Printf("list services failed: %v\n", err)
	} else if len(svcs.Items) == 0 {
		GinkgoWriter.Println("No services found.")
	} else {
		for _, s := range svcs.Items {
			var ports []string
			for _, p := range s.Spec.Ports {
				np := ""
				if p.NodePort != 0 {
					np = fmt.Sprintf(" nodePort=%d", p.NodePort)
				}
				ports = append(ports, fmt.Sprintf("%s/%d->%s%s", p.Name, p.Port, p.TargetPort.String(), np))
			}
			GinkgoWriter.Printf("- svc=%s type=%s clusterIP=%s ports=[%s]\n", s.Name, s.Spec.Type, s.Spec.ClusterIP, strings.Join(ports, ", "))
		}
	}
	GinkgoWriter.Println("================================\n")
}

// DumpDeploymentState 打印部署状态
func DumpDeploymentState(ctx context.Context, k8sClient K8SClientInterface, infernexNamespace string) {
	// 你关心的两个命名空间：istio-system + ai-inference（Infernex/推理相关）
	DumpPodsAndServices(ctx, k8sClient, infernexNamespace)
	DumpPodsAndServices(ctx, k8sClient, "ai-inference")
}

// AllPodsRunningAndReady 检查命名空间内所有 Pod 均为 Running 且全部容器 Ready。
func AllPodsRunningAndReady(ctx context.Context, k8sClient K8SClientInterface, namespace string) error {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods failed: %w", err)
	}
	if len(pods.Items) == 0 {
		return fmt.Errorf("namespace %s 中还没有 Pod", namespace)
	}

	for _, pod := range pods.Items {
		if pod.Status.Phase != corev1.PodRunning {
			return fmt.Errorf("pod %s 状态为 %s，期望 Running", pod.Name, pod.Status.Phase)
		}
		total := len(pod.Status.ContainerStatuses)
		ready := 0
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Ready {
				ready++
			}
		}
		if ready != total {
			return fmt.Errorf("pod %s ready=%d/%d", pod.Name, ready, total)
		}
	}
	return nil
}

// NamespaceExists 检查命名空间是否存在。
func NamespaceExists(ctx context.Context, k8sClient K8SClientInterface, name string) bool {
	_, err := k8sClient.GetNamespace(ctx, name)
	return err == nil
}

// CheckInfernexPodsReady 简单检查 Infernex 相关 Pod 是否都处于 Running/Ready 状态
func CheckInfernexPodsReady(k8sClient K8SClientInterface, ctx context.Context, namespace string) (bool, error) {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("list pods failed: %v", err)
	}
	if len(pods.Items) == 0 {
		return false, nil
	}

	for _, pod := range pods.Items {
		name := pod.Name
		phase := pod.Status.Phase

		// 只关注 Infernex 相关组件（名字里包含关键字）
		if strings.Contains(name, "gateway") ||
			strings.Contains(name, "epp") ||
			strings.Contains(name, "cache") ||
			strings.Contains(name, "backend") {

			if phase != corev1.PodRunning {
				return false, nil
			}

			// 对 backend 额外要求 container ready
			if strings.Contains(name, "backend") {
				ready := false
				for _, cs := range pod.Status.ContainerStatuses {
					if cs.Ready {
						ready = true
						break
					}
				}
				if !ready {
					return false, nil
				}
			}
		}
	}
	return true, nil
}

// CleanupNamespacesByHelmRelease 删除由指定 Helm Release 创建的所有命名空间
// 通过检查命名空间的 Helm ownership 注解来识别
// 返回被删除的命名空间列表
func CleanupNamespacesByHelmRelease(ctx context.Context, k8sClient K8SClientInterface, releaseName string) ([]string, error) {
	// 列出所有命名空间
	nsList, err := k8sClient.ListNamespaces(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("列出命名空间失败: %w", err)
	}

	var deletedNamespaces []string
	// 查找并删除匹配的命名空间
	for _, ns := range nsList.Items {
		ann := ns.Annotations
		if ann == nil {
			continue
		}

		// 检查 Helm ownership 注解
		if ann["meta.helm.sh/release-name"] == releaseName {
			GinkgoWriter.Printf("发现由 release %s 创建的命名空间: %s，正在删除...\n", releaseName, ns.Name)
			err := k8sClient.DeleteNamespace(ctx, ns.Name)
			if err != nil {
				GinkgoWriter.Printf("删除命名空间 %s 失败: %v\n", ns.Name, err)
				// 继续处理其他命名空间，不因单个失败而终止
			} else {
				GinkgoWriter.Printf("命名空间 %s 已标记为删除\n", ns.Name)
				deletedNamespaces = append(deletedNamespaces, ns.Name)
			}
		}
	}

	return deletedNamespaces, nil
}

// WaitForNamespacesDeleted 等待指定的命名空间完全删除
// 使用轮询机制检查命名空间是否还存在
func WaitForNamespacesDeleted(ctx context.Context, k8sClient K8SClientInterface, namespaceNames []string, timeoutSeconds int) error {
	if len(namespaceNames) == 0 {
		return nil
	}

	// 创建超时上下文
	timeoutCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	// 轮询检查命名空间是否已删除
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-timeoutCtx.Done():
			return fmt.Errorf("等待命名空间删除超时: %v", namespaceNames)
		case <-ticker.C:
			// 检查是否所有命名空间都已删除
			remaining := []string{}

			for _, nsName := range namespaceNames {
				// 尝试获取命名空间
				ns, err := k8sClient.GetNamespace(ctx, nsName)
				if err != nil {
					// 如果获取失败（通常是 NotFound），认为已删除
					continue
				}

				// 如果命名空间存在，检查其状态
				phase := ns.Status.Phase
				if phase == corev1.NamespaceTerminating || phase == corev1.NamespaceActive {
					remaining = append(remaining, nsName)
				}
			}

			if len(remaining) == 0 {
				GinkgoWriter.Printf("所有命名空间已完全删除: %v\n", namespaceNames)
				return nil
			}

			GinkgoWriter.Printf("等待命名空间删除: %v (剩余 %d 个)\n", remaining, len(remaining))
		}
	}
}

// CleanupConflictsForInfernex 处理"资源已被其他 release 占用"的冲突场景
//
// 已知场景：
//   - ai-inference 下 SA 的 Helm ownership 属于 router/default
//   - istio-system / scaling-system 下 SA（如 istiod、infernex-elastic-scaler-controller-manager）的
//     meta.helm.sh/release-namespace 与本次 infernex 目标 ns 不一致时，Helm 无法接管
func CleanupConflictsForInfernex(ctx context.Context, k8sClient K8SClientInterface, targetRelease, targetNamespace string) error {
	// 1) 先尝试卸载可能的冲突 release（按日志给出的 router/default）
	//    卸载失败不直接终止，后面还有残留资源兜底清理。
	_ = HelmUninstallIfExists("router", "default")

	// 2) 也卸载本次要装的 release（防止上次半安装残留）
	_ = HelmUninstallIfExists(targetRelease, targetNamespace)

	// 3) 兜底：删除 ai-inference 命名空间下仍残留、且 Helm ownership 指向 router/default 的 ServiceAccount
	//    只删符合注解的 SA，避免误伤其他资源。
	saList, err := k8sClient.ListServiceAccounts(ctx, "ai-inference", metav1.ListOptions{})
	if err != nil {
		// ai-inference 不存在时，说明无该类冲突，直接返回
		return nil
	}
	for _, sa := range saList.Items {
		ann := sa.Annotations
		if ann == nil {
			continue
		}
		if ann["meta.helm.sh/release-name"] == "router" && ann["meta.helm.sh/release-namespace"] == "default" {
			_ = k8sClient.DeleteServiceAccount(ctx, "ai-inference", sa.Name)
		}
	}

	// 4) 跨 ns 子 chart 创建的 SA：若 Helm 声明的 release-namespace ≠ 本次安装 ns，先删以免 install 失败
	for _, ns := range []string{"istio-system", "scaling-system"} {
		crossList, listErr := k8sClient.ListServiceAccounts(ctx, ns, metav1.ListOptions{})
		if listErr != nil {
			continue
		}
		for _, sa := range crossList.Items {
			ann := sa.Annotations
			if ann == nil {
				continue
			}
			relNs := ann["meta.helm.sh/release-namespace"]
			if relNs != "" && relNs != targetNamespace {
				_ = k8sClient.DeleteServiceAccount(ctx, ns, sa.Name)
			}
		}
	}
	return nil
}

// ClientsetK8SAdapter 将 *kubernetes.Clientset 适配为 K8SClientInterface，供未使用 framework/k8s 的测试套件复用 utils 推理辅助函数。
type ClientsetK8SAdapter struct {
	Clientset *kubernetes.Clientset
}

func (a *ClientsetK8SAdapter) ListPods(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.PodList, error) {
	return a.Clientset.CoreV1().Pods(namespace).List(ctx, opts)
}

func (a *ClientsetK8SAdapter) ListServices(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.ServiceList, error) {
	return a.Clientset.CoreV1().Services(namespace).List(ctx, opts)
}

func (a *ClientsetK8SAdapter) ListEndpointSlices(ctx context.Context, namespace string, opts metav1.ListOptions) (*discoveryv1.EndpointSliceList, error) {
	return a.Clientset.DiscoveryV1().EndpointSlices(namespace).List(ctx, opts)
}

func (a *ClientsetK8SAdapter) ListNodes(ctx context.Context, opts metav1.ListOptions) (*corev1.NodeList, error) {
	return a.Clientset.CoreV1().Nodes().List(ctx, opts)
}

func (a *ClientsetK8SAdapter) ListServiceAccounts(ctx context.Context, namespace string, opts metav1.ListOptions) (*corev1.ServiceAccountList, error) {
	return a.Clientset.CoreV1().ServiceAccounts(namespace).List(ctx, opts)
}

func (a *ClientsetK8SAdapter) DeleteServiceAccount(ctx context.Context, namespace, name string) error {
	return a.Clientset.CoreV1().ServiceAccounts(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

func (a *ClientsetK8SAdapter) ListNamespaces(ctx context.Context, opts metav1.ListOptions) (*corev1.NamespaceList, error) {
	return a.Clientset.CoreV1().Namespaces().List(ctx, opts)
}

func (a *ClientsetK8SAdapter) DeleteNamespace(ctx context.Context, name string) error {
	return a.Clientset.CoreV1().Namespaces().Delete(ctx, name, metav1.DeleteOptions{})
}

func (a *ClientsetK8SAdapter) GetNamespace(ctx context.Context, name string) (*corev1.Namespace, error) {
	return a.Clientset.CoreV1().Namespaces().Get(ctx, name, metav1.GetOptions{})
}

func serviceNeedsReadyEndpoints(svc corev1.Service) bool {
	if svc.Spec.ClusterIP == corev1.ClusterIPNone {
		return false
	}
	if svc.Labels["app.kubernetes.io/name"] != "inference-backend" {
		return false
	}
	for _, port := range svc.Spec.Ports {
		if port.Port == 8000 {
			return true
		}
	}
	return false
}

func endpointSlicesHaveReadyAddresses(slices []discoveryv1.EndpointSlice) bool {
	for _, slice := range slices {
		for _, endpoint := range slice.Endpoints {
			if len(endpoint.Addresses) == 0 {
				continue
			}
			if endpoint.Conditions.Ready != nil && !*endpoint.Conditions.Ready {
				continue
			}
			return true
		}
	}
	return false
}

// WaitForNamespacePodsReady checks only pod-level Running/Ready state for active pods
// in the namespace. It intentionally does not validate services or endpoints.
func WaitForNamespacePodsReady(k8sClient K8SClientInterface, ctx context.Context, namespace string) error {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods failed: %v", err)
	}

	found := false
	var notReady []string
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil || pod.Status.Phase == corev1.PodSucceeded {
			continue
		}
		found = true

		ready := 0
		for _, cs := range pod.Status.ContainerStatuses {
			if cs.Ready {
				ready++
			}
		}

		if pod.Status.Phase != corev1.PodRunning {
			notReady = append(notReady, fmt.Sprintf("%s phase=%s ready=%d/%d", pod.Name, pod.Status.Phase, ready, len(pod.Spec.Containers)))
			continue
		}

		if len(pod.Status.ContainerStatuses) != len(pod.Spec.Containers) {
			notReady = append(notReady, fmt.Sprintf("%s phase=%s ready=%d/%d", pod.Name, pod.Status.Phase, ready, len(pod.Spec.Containers)))
			continue
		}

		if ready != len(pod.Spec.Containers) {
			notReady = append(notReady, fmt.Sprintf("%s phase=%s ready=%d/%d", pod.Name, pod.Status.Phase, ready, len(pod.Spec.Containers)))
		}
	}

	if !found {
		return fmt.Errorf("namespace %s has no active pods", namespace)
	}
	if len(notReady) > 0 {
		return fmt.Errorf("pods not ready yet: %s", strings.Join(notReady, "; "))
	}

	return nil
}

// CheckAllContainersReady checks that all active pods in the namespace are Running
// and that every regular container reports Ready. For inference-backend services,
// it also requires at least one ready endpoint before inference requests proceed.
func CheckAllContainersReady(k8sClient K8SClientInterface, ctx context.Context, namespace string) (bool, error) {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("list pods failed: %v", err)
	}

	found := false
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil || pod.Status.Phase == corev1.PodSucceeded {
			continue
		}
		found = true

		if pod.Status.Phase != corev1.PodRunning {
			return false, nil
		}

		if len(pod.Status.ContainerStatuses) != len(pod.Spec.Containers) {
			return false, nil
		}

		for _, cs := range pod.Status.ContainerStatuses {
			if !cs.Ready {
				return false, nil
			}
		}
	}

	svcs, err := k8sClient.ListServices(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("list services failed: %v", err)
	}
	for _, svc := range svcs.Items {
		if !serviceNeedsReadyEndpoints(svc) {
			continue
		}
		endpointSlices, err := k8sClient.ListEndpointSlices(ctx, namespace, metav1.ListOptions{
			LabelSelector: fmt.Sprintf("kubernetes.io/service-name=%s", svc.Name),
		})
		if err != nil {
			return false, fmt.Errorf("list endpointslices for service %s failed: %v", svc.Name, err)
		}
		if !endpointSlicesHaveReadyAddresses(endpointSlices.Items) {
			return false, nil
		}
	}

	return found, nil
}

// CheckSidecarsReady checks that named sidecar containers in EPP pods are Ready.
// Returns true only if all named sidecars are Ready in all EPP pods.
func CheckSidecarsReady(k8sClient K8SClientInterface, ctx context.Context, namespace string, sidecarNames []string) (bool, error) {
	pods, err := k8sClient.ListPods(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("list pods failed: %v", err)
	}

	var eppPods []corev1.Pod
	for _, pod := range pods.Items {
		if strings.Contains(pod.Name, "epp") {
			eppPods = append(eppPods, pod)
		}
	}
	if len(eppPods) == 0 {
		return false, nil
	}

	for _, pod := range eppPods {
		statusByName := make(map[string]bool)
		for _, cs := range pod.Status.ContainerStatuses {
			statusByName[cs.Name] = cs.Ready
		}
		for _, sidecar := range sidecarNames {
			ready, exists := statusByName[sidecar]
			if !exists || !ready {
				return false, nil
			}
		}
	}

	return true, nil
}
