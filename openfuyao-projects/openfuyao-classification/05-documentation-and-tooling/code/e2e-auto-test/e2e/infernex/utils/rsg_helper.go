package utils

import (
	"context"
	"fmt"
	"regexp"
	"time"

	esutils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

var (
	RSGInfernexHelmReleaseName = "infernex-rsg"
	RSGInfernexHelmNamespace   = "infernex-rsg-test"
)

const (
	RSGInfernexDefaultRSGName   = "rsg"
	RSGInfernexESCRName         = "elastic-scaler"
	RSGInfernexLoadTesterName   = "infernex-rsg-load-tester"
	RSGInfernexProbeJobName     = "infernex-rsg-probe"
	RSGInfernexPDServiceName    = "qwen3-8b-2p2d-01"
	RSGInfernexProxyName        = "qwen3-8b-2p2d-01-proxy"
	RSGInfernexPrefillName      = "qwen3-8b-2p2d-01-prefill"
	RSGInfernexDecodeName       = "qwen3-8b-2p2d-01-decode"
	RSGInfernexModelName        = "Qwen/Qwen3-8B"
	RSGInfernexHTTPServicePort  = 8000
	RSGInfernexScaleUpCPUUtil   = 10
	RSGInfernexScaleDownCPUUtil = 50
	RSGInfernexLoadWorkers      = 4
)

var infernexRSGScalingSystemDeployments = map[string]int32{
	"infernex-rsg-elastic-scaler-controller-manager": 1,
	"resourcescalinggroup-controller":                1,
	"tidal-controller-manager":                       1,
}

var infernexRSGTestDeployments = map[string]int32{
	"mooncake-master-deployment": 1,
	"redis-server-deployment":    1,
	RSGInfernexProxyName:         1,
}

var rsgInfernexLWSGroupNamePattern = regexp.MustCompile(`^qwen3-8b-2p2d-01-(prefill|decode)$|^pd-qwen3-8b-2p2d-01-(prefill|decode)-([1-9][0-9]*)$`)

// RSG 控制器为扩缩容创建工作负载时打上的标准 label key。
const (
	LabelRSGName      = "rsg.io/name"
	LabelRSGTarget    = "rsg.io/target-name"
	LabelRSGUID       = "rsg-uid"
	LabelRSGUIDAlt    = "rsg.io-uid"
	LabelRSGGroupID   = "rsg.io/group-id"
	LabelRSGGroupName = "rsg.io/group-name"
)

// rsgLabelKeys 用于判断资源是否由 RSG 创建（命中任一 key 即视为 RSG 资源）。
var rsgLabelKeys = []string{
	LabelRSGName,
	LabelRSGTarget,
	LabelRSGUID,
	LabelRSGUIDAlt,
	LabelRSGGroupID,
	LabelRSGGroupName,
}

// HasRSGLabels 判断 labels 是否包含 RSG 扩缩容标准标识。
func HasRSGLabels(labels map[string]string) bool {
	if labels == nil {
		return false
	}
	for _, key := range rsgLabelKeys {
		if _, ok := labels[key]; ok {
			return true
		}
	}
	return false
}

// DeploymentHasRSGLabels 检查 Deployment 或其 Pod 模板是否带 RSG 标签。
func DeploymentHasRSGLabels(dep *appsv1.Deployment) bool {
	if HasRSGLabels(dep.Labels) {
		return true
	}
	return HasRSGLabels(dep.Spec.Template.Labels)
}

// StatefulSetHasRSGLabels 检查 StatefulSet 或其 Pod 模板是否带 RSG 标签。
func StatefulSetHasRSGLabels(sts *appsv1.StatefulSet) bool {
	if HasRSGLabels(sts.Labels) {
		return true
	}
	return HasRSGLabels(sts.Spec.Template.Labels)
}

// DaemonSetHasRSGLabels 检查 DaemonSet 或其 Pod 模板是否带 RSG 标签。
func DaemonSetHasRSGLabels(ds *appsv1.DaemonSet) bool {
	if HasRSGLabels(ds.Labels) {
		return true
	}
	return HasRSGLabels(ds.Spec.Template.Labels)
}

// WorkloadHasRSGLabels 检查 Pod 自身或其所属 Deployment 是否带 RSG 标签。
func WorkloadHasRSGLabels(clientset kubernetes.Interface, ctx context.Context, namespace string, pod *corev1.Pod) bool {
	if HasRSGLabels(pod.Labels) {
		return true
	}
	for _, ref := range pod.OwnerReferences {
		if ref.Kind != "ReplicaSet" {
			continue
		}
		rs, err := clientset.AppsV1().ReplicaSets(namespace).Get(ctx, ref.Name, metav1.GetOptions{})
		if err != nil {
			continue
		}
		for _, rsRef := range rs.OwnerReferences {
			if rsRef.Kind != "Deployment" {
				continue
			}
			dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, rsRef.Name, metav1.GetOptions{})
			if err == nil && DeploymentHasRSGLabels(dep) {
				return true
			}
		}
	}
	return false
}

// CleanupRSGScaledResources 删除命名空间内带 RSG 标签（rsg.io/*、rsg-uid）的扩缩容工作负载。
func CleanupRSGScaledResources(
	clientset kubernetes.Interface,
	dynamicClient dynamic.Interface,
	ctx context.Context,
	namespace string,
) {
	if !namespaceExists(clientset, ctx, namespace) {
		return
	}
	propagation := metav1.DeletePropagationBackground
	delOpts := metav1.DeleteOptions{PropagationPolicy: &propagation}

	deploys, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		for i := range deploys.Items {
			dep := &deploys.Items[i]
			if !DeploymentHasRSGLabels(dep) {
				continue
			}
			GinkgoWriter.Printf("清理 RSG deployment: %s/%s\n", namespace, dep.Name)
			_ = clientset.AppsV1().Deployments(namespace).Delete(ctx, dep.Name, delOpts)
		}
	}

	stsList, err := clientset.AppsV1().StatefulSets(namespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		for i := range stsList.Items {
			sts := &stsList.Items[i]
			if !StatefulSetHasRSGLabels(sts) {
				continue
			}
			GinkgoWriter.Printf("清理 RSG statefulset: %s/%s\n", namespace, sts.Name)
			_ = clientset.AppsV1().StatefulSets(namespace).Delete(ctx, sts.Name, delOpts)
		}
	}

	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		for i := range pods.Items {
			pod := &pods.Items[i]
			if !WorkloadHasRSGLabels(clientset, ctx, namespace, pod) {
				continue
			}
			GinkgoWriter.Printf("清理 RSG pod: %s/%s\n", namespace, pod.Name)
			_ = clientset.CoreV1().Pods(namespace).Delete(ctx, pod.Name, delOpts)
		}
	}

	if dynamicClient == nil {
		return
	}
	if cs, ok := clientset.(*kubernetes.Clientset); ok {
		if err := EnsureLeaderWorkerSetAPI(cs); err != nil {
			return
		}
	}
	lwsList, err := dynamicClient.Resource(esutils.LeaderWorkerSetGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return
	}
	for _, item := range lwsList.Items {
		if !HasRSGLabels(item.GetLabels()) {
			continue
		}
		GinkgoWriter.Printf("清理 RSG LeaderWorkerSet: %s/%s\n", namespace, item.GetName())
		_ = dynamicClient.Resource(esutils.LeaderWorkerSetGVR).Namespace(namespace).Delete(ctx, item.GetName(), delOpts)
	}
}

func namespaceExists(clientset kubernetes.Interface, ctx context.Context, name string) bool {
	_, err := clientset.CoreV1().Namespaces().Get(ctx, name, metav1.GetOptions{})
	return err == nil && !apierrors.IsNotFound(err)
}

// WaitForInfernexRSGScalingSystemControllersReady 等待 RSG 场景需要的 ES/RSG 控制器 Ready。
func WaitForInfernexRSGScalingSystemControllersReady(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration) {
	ns := esutils.ElasticScalerHelmNamespace
	for name, replicas := range infernexRSGScalingSystemDeployments {
		GinkgoWriter.Printf("等待 Deployment %s/%s ReadyReplicas=%d...\n", ns, name, replicas)
		esutils.WaitForDeploymentReplicas(clientset, ctx, name, ns, replicas, timeout)
	}
}

// WaitForInfernexRSGBaseWorkloadsReady 等待当前 InferNex 默认 PD 分离配置中的基础工作负载就绪。
func WaitForInfernexRSGBaseWorkloadsReady(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, timeout time.Duration) {
	for name, replicas := range infernexRSGTestDeployments {
		GinkgoWriter.Printf("等待 Deployment %s/%s ReadyReplicas=%d...\n", RSGInfernexHelmNamespace, name, replicas)
		esutils.WaitForDeploymentReplicas(clientset, ctx, name, RSGInfernexHelmNamespace, replicas, timeout)
	}
	Eventually(func() error {
		if err := EnsureLeaderWorkerSetAPI(clientset); err != nil {
			return err
		}
		if _, err := clientset.CoreV1().Services(RSGInfernexHelmNamespace).Get(ctx, RSGInfernexPDServiceName, metav1.GetOptions{}); err != nil {
			return fmt.Errorf("get infernex proxy service %s/%s: %w", RSGInfernexHelmNamespace, RSGInfernexPDServiceName, err)
		}
		groups, err := collectRSGLeaderWorkerSetGroups(dynamicClient, ctx, RSGInfernexHelmNamespace, 1)
		if err != nil {
			return err
		}
		if !groups[0]["prefill"] || !groups[0]["decode"] {
			return fmt.Errorf("default PD LWS group is not ready: %v", groups)
		}
		return nil
	}, timeout, 10*time.Second).Should(Succeed(), "InferNex RSG 默认 PD LWS 与 proxy service 应就绪")
	GinkgoWriter.Printf("等待 LeaderWorkerSet %s/%s 就绪...\n", RSGInfernexHelmNamespace, RSGInfernexPrefillName)
	WaitForLeaderWorkerSetFullyReady(dynamicClient, ctx, RSGInfernexHelmNamespace, RSGInfernexPrefillName, timeout)
	GinkgoWriter.Printf("等待 LeaderWorkerSet %s/%s 就绪...\n", RSGInfernexHelmNamespace, RSGInfernexDecodeName)
	WaitForLeaderWorkerSetFullyReady(dynamicClient, ctx, RSGInfernexHelmNamespace, RSGInfernexDecodeName, timeout)
}

// ConfigureDefaultElasticScalerCRForRSG 修改 InferNex chart 默认 ES CR，使其满足 RSG APA 扩缩容用例。
func ConfigureDefaultElasticScalerCRForRSG(dynamicClient dynamic.Interface, ctx context.Context, minReplicas, maxReplicas, cpuUtilization int32) error {
	const maxWaitRetries = 30
	var lastErr error
	for i := 0; i < maxWaitRetries; i++ {
		if _, err := esutils.GetElasticScalerCR(dynamicClient, ctx, RSGInfernexESCRName, RSGInfernexHelmNamespace); err == nil {
			lastErr = nil
			break
		} else {
			lastErr = err
			time.Sleep(2 * time.Second)
		}
	}
	if lastErr != nil {
		return fmt.Errorf("等待默认 ElasticScaler %s/%s 创建失败: %w", RSGInfernexHelmNamespace, RSGInfernexESCRName, lastErr)
	}

	err := esutils.UpdateElasticScalerCR(dynamicClient, ctx, RSGInfernexESCRName, RSGInfernexHelmNamespace, func(obj *unstructured.Unstructured) error {
		if err := unstructured.SetNestedField(obj.Object, int64(minReplicas), "spec", "minReplicas"); err != nil {
			return err
		}
		if err := unstructured.SetNestedField(obj.Object, int64(maxReplicas), "spec", "maxReplicas"); err != nil {
			return err
		}
		if err := unstructured.SetNestedField(obj.Object, "resourcescalinggroup", "spec", "targetRef", "kind"); err != nil {
			return err
		}
		if err := unstructured.SetNestedField(obj.Object, RSGInfernexDefaultRSGName, "spec", "targetRef", "name"); err != nil {
			return err
		}
		if err := unstructured.SetNestedField(obj.Object, "autoscaling.openfuyao.com/v1alpha1", "spec", "targetRef", "apiVersion"); err != nil {
			return err
		}
		if err := unstructured.SetNestedField(obj.Object, RSGInfernexHelmNamespace, "spec", "targetRef", "namespace"); err != nil {
			return err
		}
		if err := unstructured.SetNestedField(obj.Object, "apa", "spec", "trigger", "metricsTrigger", "scalingAlgorithm"); err != nil {
			return err
		}
		metrics := []interface{}{
			map[string]interface{}{
				"type": "Resource",
				"resource": map[string]interface{}{
					"metricsName": "cpu",
					"target": map[string]interface{}{
						"type":               "Utilization",
						"averageUtilization": int64(cpuUtilization),
					},
				},
			},
		}
		if err := unstructured.SetNestedSlice(obj.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics"); err != nil {
			return err
		}
		return nil
	})
	if err == nil {
		GinkgoWriter.Printf("已更新默认 ElasticScaler %s/%s: min=%d max=%d cpuUtilization=%d\n",
			RSGInfernexHelmNamespace, RSGInfernexESCRName, minReplicas, maxReplicas, cpuUtilization)
	}
	return err
}

func DeleteElasticScalerCRForRSG(dynamicClient dynamic.Interface, ctx context.Context) error {
	return esutils.DeleteElasticScalerCR(dynamicClient, ctx, RSGInfernexESCRName, RSGInfernexHelmNamespace)
}

func WaitForRSGGroupReplicas(dynamicClient dynamic.Interface, ctx context.Context, expected int64, timeout time.Duration) {
	Eventually(func() int64 {
		obj, err := dynamicClient.Resource(esutils.ResourceScalingGroupGVR).Namespace(RSGInfernexHelmNamespace).Get(ctx, RSGInfernexDefaultRSGName, metav1.GetOptions{})
		if err != nil {
			GinkgoWriter.Printf("获取 RSG 失败: %v\n", err)
			return -1
		}
		replicas, found, err := unstructured.NestedInt64(obj.Object, "spec", "scalingStrategy", "groupConfig", "replicas")
		if err != nil || !found {
			return -1
		}
		return replicas
	}, timeout, 10*time.Second).Should(Equal(expected), "RSG groupConfig.replicas 应为 %d", expected)
}

// WaitForRSGLeaderWorkerSetGroups 等待 RSG 为每个 group 准备好 prefill/decode 两个 LWS。
func WaitForRSGLeaderWorkerSetGroups(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, namespace string, expectedGroups int64, timeout time.Duration) {
	Expect(expectedGroups).To(BeNumerically(">", 0), "expectedGroups 必须大于 0")
	Expect(EnsureLeaderWorkerSetAPI(clientset)).NotTo(HaveOccurred())
	Eventually(func() bool {
		groups, err := collectRSGLeaderWorkerSetGroups(dynamicClient, ctx, namespace, expectedGroups)
		if err != nil {
			GinkgoWriter.Printf("统计 RSG LWS group 失败: %v\n", err)
			return false
		}
		for gid := int64(0); gid < expectedGroups; gid++ {
			targets := groups[gid]
			if !targets["prefill"] || !targets["decode"] {
				GinkgoWriter.Printf("RSG group %d LWS 未就绪: targets=%v groups=%v\n", gid, targets, groups)
				return false
			}
		}
		GinkgoWriter.Printf("RSG LWS groups 已就绪: %v\n", groups)
		return true
	}, timeout, 10*time.Second).Should(BeTrue(), "RSG 应存在 %d 个包含 prefill/decode 的 LWS group", expectedGroups)
}

func collectRSGLeaderWorkerSetGroups(dynamicClient dynamic.Interface, ctx context.Context, namespace string, expectedGroups int64) (map[int64]map[string]bool, error) {
	list, err := dynamicClient.Resource(esutils.LeaderWorkerSetGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	groups := make(map[int64]map[string]bool)
	for _, item := range list.Items {
		target, gid, ok := parseRSGLeaderWorkerSetName(item.GetName())
		if !ok {
			continue
		}
		if gid < 0 || gid >= expectedGroups {
			continue
		}
		if groups[gid] == nil {
			groups[gid] = make(map[string]bool)
		}
		groups[gid][target] = true
	}
	return groups, nil
}

func parseRSGLeaderWorkerSetName(name string) (string, int64, bool) {
	matches := rsgInfernexLWSGroupNamePattern.FindStringSubmatch(name)
	if matches == nil {
		return "", 0, false
	}
	if matches[1] != "" {
		return matches[1], 0, true
	}
	groupID, err := parsePositiveInt64(matches[3])
	if err != nil {
		GinkgoWriter.Printf("忽略 group-id 非法的 LWS %s: %q\n", name, matches[3])
		return "", 0, false
	}
	return matches[2], groupID, true
}

func parsePositiveInt64(value string) (int64, error) {
	var n int64
	for _, r := range value {
		if r < '0' || r > '9' {
			return 0, fmt.Errorf("invalid digit %q", r)
		}
		n = n*10 + int64(r-'0')
	}
	return n, nil
}

func StartRSGPDLoadTest(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration) error {
	if _, err := clientset.AppsV1().Deployments(RSGInfernexHelmNamespace).Get(ctx, RSGInfernexLoadTesterName, metav1.GetOptions{}); err == nil {
		esutils.WaitForDeploymentReplicas(clientset, ctx, RSGInfernexLoadTesterName, RSGInfernexHelmNamespace, 1, timeout)
		return nil
	}

	var replicas int32 = 1
	url := fmt.Sprintf("http://%s:%d/v1/chat/completions", RSGInfernexPDServiceName, RSGInfernexHTTPServicePort)
	cmd := []string{
		"sh",
		"-c",
		fmt.Sprintf(`worker=0; while [ "$worker" -lt %d ]; do (while true; do i=0; while [ "$i" -lt 100 ]; do curl -fsS --connect-timeout 5 --max-time 180 -X POST %q -H "Content-Type: application/json" -d '{"model":%q,"messages":[{"role":"user","content":"请介绍openFuyao开源社区"}],"stream":false}' >/dev/null || true; i=$((i+1)); done; sleep 1; done) & worker=$((worker+1)); done; wait`, RSGInfernexLoadWorkers, url, RSGInfernexModelName),
	}

	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      RSGInfernexLoadTesterName,
			Namespace: RSGInfernexHelmNamespace,
			Labels:    map[string]string{"app": RSGInfernexLoadTesterName},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": RSGInfernexLoadTesterName}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": RSGInfernexLoadTesterName}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:    "load-tester",
						Image:   "curlimages/curl:8.5.0",
						Command: cmd,
					}},
				},
			},
		},
	}
	_, err := clientset.AppsV1().Deployments(RSGInfernexHelmNamespace).Create(ctx, dep, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("创建 RSG PD 压测 Deployment 失败: %w", err)
	}
	esutils.WaitForDeploymentReplicas(clientset, ctx, RSGInfernexLoadTesterName, RSGInfernexHelmNamespace, 1, timeout)
	return nil
}

func StopRSGPDLoadTest(clientset *kubernetes.Clientset, ctx context.Context) {
	err := clientset.AppsV1().Deployments(RSGInfernexHelmNamespace).Delete(ctx, RSGInfernexLoadTesterName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		GinkgoWriter.Printf("删除 RSG PD 压测 Deployment 失败: %v\n", err)
	}
}

func RunRSGPDInferenceProbe(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration) error {
	if err := deleteRSGProbeJob(clientset, ctx); err != nil {
		return err
	}

	url := fmt.Sprintf("http://%s:%d/v1/chat/completions", RSGInfernexPDServiceName, RSGInfernexHTTPServicePort)
	probeDeadlineSeconds := int(timeout.Seconds()) - 30
	if probeDeadlineSeconds < 60 {
		probeDeadlineSeconds = 60
	}
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: RSGInfernexProbeJobName, Namespace: RSGInfernexHelmNamespace},
		Spec: batchv1.JobSpec{
			BackoffLimit: func() *int32 { v := int32(0); return &v }(),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{{
						Name:  "probe",
						Image: "curlimages/curl:8.5.0",
						Command: []string{
							"sh",
							"-c",
							fmt.Sprintf(`deadline=$(( $(date +%%s) + %d )); while [ "$(date +%%s)" -lt "$deadline" ]; do if curl -fsS --connect-timeout 5 --max-time 120 -X POST %q -H "Content-Type: application/json" -d '{"model":%q,"messages":[{"role":"user","content":"请介绍openFuyao开源社区"}],"stream":false}' >/dev/null; then exit 0; fi; sleep 10; done; exit 1`, probeDeadlineSeconds, url, RSGInfernexModelName),
						},
					}},
				},
			},
		},
	}
	if _, err := clientset.BatchV1().Jobs(RSGInfernexHelmNamespace).Create(ctx, job, metav1.CreateOptions{}); err != nil {
		return err
	}

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		current, err := clientset.BatchV1().Jobs(RSGInfernexHelmNamespace).Get(ctx, RSGInfernexProbeJobName, metav1.GetOptions{})
		if err != nil {
			GinkgoWriter.Printf("获取 RSG 推理探针 Job 失败: %v\n", err)
			time.Sleep(5 * time.Second)
			continue
		}
		if current.Status.Failed > 0 {
			return fmt.Errorf("RSG PD 推理探针在 %s 内未成功返回，failed=%d", timeout, current.Status.Failed)
		}
		if current.Status.Succeeded > 0 {
			GinkgoWriter.Println("RSG PD 推理轻量探针请求成功")
			return nil
		}
		time.Sleep(5 * time.Second)
	}
	return fmt.Errorf("等待 RSG PD 推理探针成功超时: %s", timeout)
}

func deleteRSGProbeJob(clientset *kubernetes.Clientset, ctx context.Context) error {
	propagation := metav1.DeletePropagationBackground
	err := clientset.BatchV1().Jobs(RSGInfernexHelmNamespace).Delete(ctx, RSGInfernexProbeJobName, metav1.DeleteOptions{PropagationPolicy: &propagation})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("删除 RSG 推理探针 Job 失败: %w", err)
	}
	deadline := time.Now().Add(1 * time.Minute)
	for time.Now().Before(deadline) {
		_, err := clientset.BatchV1().Jobs(RSGInfernexHelmNamespace).Get(ctx, RSGInfernexProbeJobName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return err
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("等待 RSG 推理探针 Job 删除超时")
}

func CleanupRSGInfernexResources(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context) {
	StopRSGPDLoadTest(clientset, ctx)
	_ = deleteRSGProbeJob(clientset, ctx)
	_ = esutils.DeleteElasticScalerCR(dynamicClient, ctx, RSGInfernexESCRName, RSGInfernexHelmNamespace)
	if err := dynamicClient.Resource(esutils.ResourceScalingGroupGVR).Namespace(RSGInfernexHelmNamespace).Delete(ctx, RSGInfernexDefaultRSGName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		GinkgoWriter.Printf("删除默认 RSG 失败: %v\n", err)
	}
	CleanupRSGScaledResources(clientset, dynamicClient, ctx, RSGInfernexHelmNamespace)
}
