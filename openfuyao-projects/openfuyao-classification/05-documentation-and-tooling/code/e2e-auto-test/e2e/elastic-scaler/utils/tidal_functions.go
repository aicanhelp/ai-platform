package utils

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
)

// InstallTidal 执行 helm install 安装 tidal（命名空间与资源名前缀见 tidal_config.go）
func InstallTidal() error {
	cmd := exec.Command("helm", "install",
		TidalHelmReleaseName,
		TidalHelmChart,
		"--version", TidalHelmVersion,
		"-n", TidalHelmNamespace,
		"--create-namespace",
		// 与 ES 安装同样的隔离参数：命名空间 + tidal 资源名前缀，
		// 避免与同集群其它 pd-orchestrator（含 tidal）安装争抢同名的集群级对象
		"--set", "tidal.namespace.name="+TidalHelmNamespace,
		"--set", "tidal.fullnameOverride="+TidalFullnameOverride,
		"--set", "tidal.serviceAccount.name="+TidalServiceAccountName,
		"--set", "elastic-scaler.enabled=false",
		"--set", "resourcescalinggroup.enabled=false",
		"--set", "tidal.enabled=true",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("helm install 失败: %v\nOutput: %s", err, string(out))
	}
	GinkgoWriter.Printf("helm install 输出:\n%s\n", string(out))
	return nil
}

// FilterTidalPods 过滤 tidal 控制器相关的 Pod
func FilterTidalPods(clientset *kubernetes.Clientset, ctx context.Context) ([]corev1.Pod, error) {
	var filteredPods []corev1.Pod
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("列举 pods 失败: %w", err)
	}

	for _, pod := range podList.Items {
		for _, prefix := range TidalPodPrefixes {
			if strings.HasPrefix(pod.Name, prefix) {
				filteredPods = append(filteredPods, pod)
				break
			}
		}
	}

	return filteredPods, nil
}

// JudgeTidalPodsAllRunning 判断所有 tidal 相关 Pod 是否均为 Running
func JudgeTidalPodsAllRunning(clientset *kubernetes.Clientset, ctx context.Context) (bool, error) {
	pods, err := FilterTidalPods(clientset, ctx)
	if err != nil {
		return false, fmt.Errorf("获取 tidal pods 失败: %v", err)
	}

	if len(pods) == 0 {
		return false, fmt.Errorf("未找到 tidal 相关 Pod")
	}

	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodRunning {
			GinkgoWriter.Printf("Pod %s/%s 未运行 (status: %s)\n",
				pod.Namespace, pod.Name, pod.Status.Phase)
			return false, nil
		}
	}

	return true, nil
}

// VerifyTidalPodsRunning 验证 tidal 相关 Pod 均在运行
func VerifyTidalPodsRunning(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration, interval time.Duration) {
	GinkgoWriter.Println("验证 tidal Pod 运行状态...")

	Eventually(func() bool {
		allRunning, err := JudgeTidalPodsAllRunning(clientset, ctx)
		if err != nil {
			GinkgoWriter.Printf("检查失败: %v\n", err)
			return false
		}
		return allRunning
	}, timeout, interval).Should(BeTrue(),
		"tidal Pod 应在超时时间内全部变为 Running")

	pods, err := FilterTidalPods(clientset, ctx)
	Expect(err).NotTo(HaveOccurred())
	GinkgoWriter.Printf("验证成功，共 %d 个 Pod 处于 Running 状态\n", len(pods))
	for _, pod := range pods {
		GinkgoWriter.Printf("  - %s/%s\n", pod.Namespace, pod.Name)
	}
}

var tidalSchedulerGVR = schema.GroupVersionResource{
	Group:    "tidal.io",
	Version:  "v1alpha1",
	Resource: "tidals",
}

// EnsureNamespace 确保 namespace 存在且为 Active；若处于 Terminating 则等待删除完成后重建。
func EnsureNamespace(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return EnsureNamespaceReady(clientset, ctx, namespace, 3*time.Minute)
}

// EnsureNamespaceReady 确保 namespace 存在且 Phase=Active（非 Terminating）。
func EnsureNamespaceReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string, terminatingWait time.Duration) error {
	if err := WaitForNamespaceDeletionIfTerminating(clientset, ctx, namespace, terminatingWait, 5*time.Second); err != nil {
		return err
	}
	ns, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		created := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
		_, err = clientset.CoreV1().Namespaces().Create(ctx, created, metav1.CreateOptions{})
		return err
	}
	if err != nil {
		return err
	}
	if ns.Status.Phase == corev1.NamespaceTerminating {
		return fmt.Errorf("namespace %s 仍处于 Terminating，无法创建资源", namespace)
	}
	return nil
}

// VerifyTidalCRExists 验证 Tidal CR 存在
func VerifyTidalCRExists(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		_, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(),
		"Tidal %s/%s 应被成功创建", namespace, name)

	GinkgoWriter.Printf("Tidal CR %s/%s 创建成功\n", namespace, name)
}

// DeleteTidalCR 删除 Tidal CR，不存在则忽略
func DeleteTidalCR(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	if err == nil {
		GinkgoWriter.Printf("已删除 Tidal CR %s/%s\n", namespace, name)
	}
	return nil
}

// DeleteNamespaceIfExists 删除 namespace，不存在则忽略
func DeleteNamespaceIfExists(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	err := clientset.CoreV1().Namespaces().Delete(ctx, namespace, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	if err == nil {
		GinkgoWriter.Printf("已删除 namespace %s\n", namespace)
	}
	return nil
}

// TidalRule cron 规则
type TidalRule struct {
	Name        string
	Cron        string
	Replicas    int64
	Description string
}

// TidalTargetRef targetRef 配置（xlsx 用例2-8 支持 Deployment/StatefulSet/DaemonSet）
type TidalTargetRef struct {
	APIVersion string
	Kind       string
	Name       string
	Namespace  string
}

func buildTidalObject(name, namespace string, targetRef TidalTargetRef, rules []TidalRule, esName, esNamespace string) *unstructured.Unstructured {
	if targetRef.APIVersion == "" {
		targetRef = TidalTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: TidalTestDeploymentName, Namespace: namespace}
	}
	if esName == "" {
		esName = name
	}
	if esNamespace == "" {
		esNamespace = namespace
	}
	rulesSlice := make([]interface{}, 0, len(rules))
	for _, r := range rules {
		rulesSlice = append(rulesSlice, map[string]interface{}{
			"name": r.Name, "cron": r.Cron, "replicas": r.Replicas, "description": r.Description,
		})
	}
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "tidal.io/v1alpha1",
			"kind":       "Tidal",
			"metadata": map[string]interface{}{
				"name": name, "namespace": namespace,
				"labels": map[string]interface{}{
					"elasticscaler.io/name":      esName,
					"elasticscaler.io/namespace": esNamespace,
				},
			},
			"spec": map[string]interface{}{
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"triggers": map[string]interface{}{
					"times": map[string]interface{}{"rules": rulesSlice},
				},
			},
		},
	}
}

// Case2TidalRules xlsx 用例2：每天3分钟扩容到5个，每5分钟扩容到7个
var Case2TidalRules = []TidalRule{
	{Name: "scale-5", Cron: "CRON_TZ=Asia/Shanghai 0 3 0 * * *", Replicas: 5, Description: "每天0:03扩容到5"},
	{Name: "scale-7", Cron: "CRON_TZ=Asia/Shanghai 0 */5 * * * *", Replicas: 7, Description: "每5分钟扩容到7"},
}

// Case3TidalRules xlsx 用例3：每分钟触发
var Case3TidalRules = []TidalRule{
	{Name: "minute-trigger", Cron: "CRON_TZ=Asia/Shanghai 0 */1 * * * *", Replicas: 2, Description: "每分钟触发"},
}

// Case5TidalRules xlsx 用例5：缩容到零（每2分钟触发一次便于测试内验证 replicas=0）
var Case5TidalRules = []TidalRule{
	{Name: "scale-to-zero", Cron: "CRON_TZ=Asia/Shanghai 0 0 * * * *", Replicas: 0, Description: "每天0点缩容到0"},
	{Name: "scale-to-zero-2min", Cron: "CRON_TZ=Asia/Shanghai 0 */2 * * * *", Replicas: 0, Description: "每2分钟缩容到0（e2e验证用）"},
}

// Case10TidalRulesNew xlsx 用例10 修改后的新规则
var Case10TidalRulesNew = []TidalRule{
	{Name: "scale-up-new", Cron: "CRON_TZ=Asia/Shanghai 0 */2 * * * *", Replicas: 4, Description: "每2分钟扩容到4"},
}

// Case11TidalRules xlsx 用例11：上午9点、下午6点
var Case11TidalRules = []TidalRule{
	{Name: "morning-9", Cron: "CRON_TZ=Asia/Shanghai 0 0 9 * * *", Replicas: 3, Description: "上午9点"},
	{Name: "evening-6", Cron: "CRON_TZ=Asia/Shanghai 0 0 18 * * *", Replicas: 5, Description: "下午6点"},
}

// Case12TidalRules xlsx 用例12：多个同时触发，desiredReplicas 取最大值（每分钟0秒同时触发，秒分时日月周）
var Case12TidalRules = []TidalRule{
	{Name: "rule-3", Cron: "CRON_TZ=Asia/Shanghai 0 */1 * * * *", Replicas: 3, Description: "同时触发"},
	{Name: "rule-7", Cron: "CRON_TZ=Asia/Shanghai 0 */1 * * * *", Replicas: 7, Description: "同时触发取最大"},
}

// Case13TidalRules xlsx 用例13：时区敏感（Beijing）
var Case13TidalRules = []TidalRule{
	{Name: "beijing-tz", Cron: "CRON_TZ=Asia/Shanghai 0 0 10 * * *", Replicas: 2, Description: "北京时间10点"},
	// 为了在 e2e 测试时间窗口内验证 replicas 正确性，再增加一条同一时区下的高频规则
	{Name: "beijing-tz-2min", Cron: "CRON_TZ=Asia/Shanghai 0 */2 * * * *", Replicas: 2, Description: "北京时间每2分钟扩容到2（验证用）"},
}

// Case14TidalRules xlsx 用例14：相同时间规则（高频规则，每10秒，同一时间触发，便于快速验证高并发场景）
var Case14TidalRules = []TidalRule{
	{Name: "same-time", Cron: "CRON_TZ=Asia/Shanghai */10 * * * * *", Replicas: 4, Description: "每10秒"},
}

// Case15TidalRules1/2/3 xlsx 用例15：每个 CR 时间规则不同（三组不同高频规则，便于在较短时间内完成验证）
var Case15TidalRules1 = []TidalRule{
	// 每10秒一次，副本数固定为2
	{Name: "rule-a", Cron: "CRON_TZ=Asia/Shanghai */10 * * * * *", Replicas: 2, Description: "每10秒扩容到2"},
}
var Case15TidalRules2 = []TidalRule{
	// 每15秒一次，副本数固定为3
	{Name: "rule-x", Cron: "CRON_TZ=Asia/Shanghai */15 * * * * *", Replicas: 3, Description: "每15秒扩容到3"},
}
var Case15TidalRules3 = []TidalRule{
	// 每20秒一次，副本数固定为4
	{Name: "rule-p", Cron: "CRON_TZ=Asia/Shanghai */20 * * * * *", Replicas: 4, Description: "每20秒扩容到4"},
}

// CreateTidalCR 创建 Tidal CR（xlsx 用例2 基础调度规则，target 为 Deployment）
func CreateTidalCR(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	return CreateTidalCRWithTargetAndRules(dynamicClient, ctx, name, namespace, TidalTargetRef{}, Case2TidalRules)
}

// CreateTidalCRWithTargetAndRules 创建 Tidal CR，指定 targetRef 和 rules
func CreateTidalCRWithTargetAndRules(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef TidalTargetRef, rules []TidalRule) error {
	obj := buildTidalObject(name, namespace, targetRef, rules, "", "")
	_, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateTidalCRWithRules 创建 Tidal CR，支持自定义 rules（target 为 Deployment）
func CreateTidalCRWithRules(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, rules []TidalRule) error {
	obj := buildTidalObject(name, namespace, TidalTargetRef{}, rules, "", "")
	_, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateTidalCRWithTargetRulesAndESRef 创建 Tidal CR 并显式指定关联的 ES 标签（elasticscaler.io/name/namespace）
func CreateTidalCRWithTargetRulesAndESRef(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef TidalTargetRef, rules []TidalRule, esName, esNamespace string) error {
	obj := buildTidalObject(name, namespace, targetRef, rules, esName, esNamespace)
	_, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// UpdateTidalCRRules 更新 Tidal CR 的 rules（xlsx 用例10：规则动态更新），带 409 冲突重试
func UpdateTidalCRRules(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, rules []TidalRule) error {
	const maxRetries = 6
	var lastErr error

	for i := 0; i < maxRetries; i++ {
		obj, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			lastErr = err
			time.Sleep(5 * time.Second)
			continue
		}

		rulesSlice := make([]interface{}, 0, len(rules))
		for _, r := range rules {
			rulesSlice = append(rulesSlice, map[string]interface{}{
				"name": r.Name, "cron": r.Cron, "replicas": r.Replicas, "description": r.Description,
			})
		}

		if err := unstructured.SetNestedSlice(obj.Object, rulesSlice, "spec", "triggers", "times", "rules"); err != nil {
			return err
		}

		_, err = dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}

		lastErr = err
		time.Sleep(5 * time.Second)
	}

	return lastErr
}

// VerifyTidalStatusNextTriggerUpdated 验证 nextTriggerTime 已按新规则更新（xlsx 用例10）
func VerifyTidalStatusNextTriggerUpdated(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		nextTime, found, _ := unstructured.NestedString(obj.Object, "status", "nextTriggerTime")
		return found && nextTime != ""
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "nextTriggerTime 应按新规则更新")
}

// VerifyTidalStatusActiveRule 验证 status.activeRule 为期望值（xlsx 用例12）
func VerifyTidalStatusActiveRule(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, expectedRule string) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		activeRule, found, _ := unstructured.NestedString(obj.Object, "status", "activeRule")
		return found && activeRule == expectedRule
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "activeRule 应为 %s", expectedRule)
}

// CreateTestDeployment 创建测试用 Deployment（xlsx 前置条件：已有一个Deployment）
func CreateTestDeployment(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32) error {
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": name}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": name}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "nginx",
						Image: "nginx:alpine",
					}},
				},
			},
		},
	}
	_, err := clientset.AppsV1().Deployments(namespace).Create(ctx, dep, metav1.CreateOptions{})
	return err
}

// RestartTidalController 重启 tidal-controller（对应 xlsx 用例4：rollout restart tidal controller deployment）
func RestartTidalController(namespace string) error {
	for _, name := range []string{TidalFullnameOverride, "tidal-controller-manager"} {
		cmd := exec.Command("kubectl", "rollout", "restart", "deployment/"+name, "-n", namespace)
		if _, err := cmd.CombinedOutput(); err == nil {
			GinkgoWriter.Printf("已重启 deployment %s\n", name)
			return nil
		}
	}
	return fmt.Errorf("tidal controller deployment not found or restart failed")
}

// tidalConditionStatus 读取 status.conditions 中指定 type 的 status 值（未找到返回 "", false）
func tidalConditionStatus(obj *unstructured.Unstructured, conditionType string) (string, bool) {
	conditions, found, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if !found || len(conditions) == 0 {
		return "", false
	}
	for _, c := range conditions {
		cond, ok := c.(map[string]interface{})
		if !ok {
			continue
		}
		ctype, _ := cond["type"].(string)
		if ctype != conditionType {
			continue
		}
		status, _ := cond["status"].(string)
		return status, true
	}
	return "", false
}

// VerifyTidalStatusReady 验证 CR status 中 Ready=True 且 Degraded!=True（xlsx 预期结果）
func VerifyTidalStatusReady(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		readyStatus, readyFound := tidalConditionStatus(obj, "Ready")
		if !readyFound || readyStatus != "True" {
			return false
		}
		if degradedStatus, degradedFound := tidalConditionStatus(obj, "Degraded"); degradedFound && degradedStatus == "True" {
			return false
		}
		return true
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "Tidal %s Ready 应为 True 且 Degraded 不应为 True", name)
}

// VerifyTidalStatusNextTrigger 验证 nextTriggerTime/Rule 已设置（xlsx 用例2 预期）
func VerifyTidalStatusNextTrigger(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		nextTime, found, _ := unstructured.NestedString(obj.Object, "status", "nextTriggerTime")
		nextRule, found2, _ := unstructured.NestedString(obj.Object, "status", "nextTriggerRule")
		return (found && nextTime != "") || (found2 && nextRule != "")
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "nextTriggerTime/Rule 应已设置")
}

// VerifyTidalStatusNextTriggerRule 验证 status.nextTriggerRule 为期望的规则名（xlsx 用例11）
func VerifyTidalStatusNextTriggerRule(dynamicClient dynamic.Interface, ctx context.Context, name, namespace, expectedRule string) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		nextRule, found, _ := unstructured.NestedString(obj.Object, "status", "nextTriggerRule")
		return found && nextRule == expectedRule
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(),
		"nextTriggerRule 应为 %s", expectedRule)
}

// VerifyTidalLastTriggerAfter 验证 lastTriggerTime 在给定时间之后（用于重启后规则仍会触发）
func VerifyTidalLastTriggerAfter(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, after time.Time) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		tsStr, found, _ := unstructured.NestedString(obj.Object, "status", "lastTriggerTime")
		if !found || tsStr == "" {
			return false
		}
		parsed, err := time.Parse(time.RFC3339, tsStr)
		if err != nil {
			return false
		}
		return parsed.After(after)
	}, 3*time.Minute, 10*time.Second).Should(BeTrue(),
		"lastTriggerTime 应在指定时间之后（例如控制器重启之后）更新")
}

// VerifyTidalDesiredReplicas 验证 status.desiredReplicas 为期望值
func VerifyTidalDesiredReplicas(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, expected int64) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		v, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		return found && v == expected
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "desiredReplicas 应为 %d", expected)
}

// VerifyTidalDesiredReplicasInSet 验证 status.desiredReplicas 落在允许的值集合中（用于多规则不同时间的场景）
func VerifyTidalDesiredReplicasInSet(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, allowed []int64) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		v, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found {
			return false
		}
		for _, a := range allowed {
			if v == a {
				return true
			}
		}
		return false
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "desiredReplicas 应落在允许集合中")
}

// GetTargetWorkloadReplicas 根据 targetRef 获取目标工作负载的 spec.replicas（仅支持 Deployment/StatefulSet）
func GetTargetWorkloadReplicas(clientset *kubernetes.Clientset, ctx context.Context, apiVersion, kind, name, namespace string) (int32, error) {
	if apiVersion != "apps/v1" {
		return 0, fmt.Errorf("unsupported target apiVersion %s", apiVersion)
	}
	switch kind {
	case "Deployment":
		dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		if dep.Spec.Replicas == nil {
			return 0, nil
		}
		return *dep.Spec.Replicas, nil
	case "StatefulSet":
		sts, err := clientset.AppsV1().StatefulSets(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return 0, err
		}
		if sts.Spec.Replicas == nil {
			return 0, nil
		}
		return *sts.Spec.Replicas, nil
	default:
		return 0, fmt.Errorf("target kind %s replicas not verifiable (only Deployment/StatefulSet)", kind)
	}
}

// VerifyTidalTargetReplicasMatch 验证 Tidal 的 status.desiredReplicas 已正确应用到 targetRef 指向的工作负载 spec.replicas
func VerifyTidalTargetReplicasMatch(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, tidalName, namespace string) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, tidalName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found {
			return false
		}
		targetRef, found, _ := unstructured.NestedMap(obj.Object, "spec", "targetRef")
		if !found || targetRef == nil {
			return false
		}
		apiVersion, _ := targetRef["apiVersion"].(string)
		kind, _ := targetRef["kind"].(string)
		targetName, _ := targetRef["name"].(string)
		targetNs, _ := targetRef["namespace"].(string)
		if targetName == "" || targetNs == "" {
			return false
		}
		replicas, err := GetTargetWorkloadReplicas(clientset, ctx, apiVersion, kind, targetName, targetNs)
		if err != nil {
			return false
		}
		return replicas == int32(desired)
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(),
		"Tidal %s 的 target 工作负载 spec.replicas 应与 status.desiredReplicas 一致", tidalName)
}

func getTidalCR(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) (*unstructured.Unstructured, error) {
	return dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
}

// GetTidalCR 获取 Tidal CR
func GetTidalCR(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) (*unstructured.Unstructured, error) {
	return getTidalCR(dynamicClient, ctx, name, namespace)
}

// TidalCRHasStatus 检查 CR 是否有 status 字段
func TidalCRHasStatus(obj *unstructured.Unstructured) bool {
	status, found, _ := unstructured.NestedMap(obj.Object, "status")
	return found && len(status) > 0
}

// VerifyTidalCRStatusUpdated 验证 CR 的 status 已被 controller 更新
func VerifyTidalCRStatusUpdated(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		return TidalCRHasStatus(obj)
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(),
		"Tidal %s/%s 的 status 应在 1 分钟内被 controller 更新", namespace, name)
	GinkgoWriter.Printf("Tidal CR %s/%s status 已更新\n", namespace, name)
}

// UpdateTidalDesiredReplicas 直接更新 Tidal status.desiredReplicas（用于 StateTrigger e2e 场景模拟）
func UpdateTidalDesiredReplicas(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, replicas int64) error {
	const maxRetries = 6
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		obj, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			lastErr = err
			time.Sleep(5 * time.Second)
			continue
		}
		if err := unstructured.SetNestedField(obj.Object, replicas, "status", "desiredReplicas"); err != nil {
			lastErr = err
			return err
		}
		_, err = dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).UpdateStatus(ctx, obj, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		lastErr = err
		time.Sleep(5 * time.Second)
	}
	return lastErr
}

// CreateTidalCRWithInvalidCron 创建 cron 格式无效的 Tidal CR（xlsx 用例17："* * *"）
func CreateTidalCRWithInvalidCron(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	rules := []TidalRule{
		{Name: "invalid", Cron: "* * *", Replicas: 1, Description: "invalid cron format"},
	}
	return CreateTidalCRWithRules(dynamicClient, ctx, name, namespace, rules)
}

// CreateTidalCRWithoutRequiredLabels 创建缺少 required labels 的 Tidal CR（xlsx 用例16：缺 elasticscaler.io/*）
func CreateTidalCRWithoutRequiredLabels(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	obj := buildTidalObjectWithoutLabels(name, namespace, TidalTargetRef{}, Case2TidalRules)
	_, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

func buildTidalObjectWithoutLabels(name, namespace string, targetRef TidalTargetRef, rules []TidalRule) *unstructured.Unstructured {
	if targetRef.APIVersion == "" {
		targetRef = TidalTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: TidalTestDeploymentName, Namespace: namespace}
	}
	rulesSlice := make([]interface{}, 0, len(rules))
	for _, r := range rules {
		rulesSlice = append(rulesSlice, map[string]interface{}{
			"name": r.Name, "cron": r.Cron, "replicas": r.Replicas, "description": r.Description,
		})
	}
	// 故意缺失 Decision Framework 集成所需的 elasticscaler.io/* labels
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "tidal.io/v1alpha1",
			"kind":       "Tidal",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
				// 仅保留与 elasticscaler 无关的标签，触发 MissingRequiredLabels
				"labels": map[string]interface{}{
					"app.kubernetes.io/name": name,
				},
			},
			"spec": map[string]interface{}{
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"triggers": map[string]interface{}{"times": map[string]interface{}{"rules": rulesSlice}},
			},
		},
	}
}

// CreateTidalCRWithDuplicateRuleNames 创建包含重复规则名的 Tidal CR（xlsx 用例18）
func CreateTidalCRWithDuplicateRuleNames(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	rules := []TidalRule{
		{Name: "dup-rule", Cron: "CRON_TZ=Asia/Shanghai 0 */5 * * * *", Replicas: 2, Description: "rule1"},
		{Name: "dup-rule", Cron: "CRON_TZ=Asia/Shanghai 0 */10 * * * *", Replicas: 3, Description: "rule2 same name"},
	}
	return CreateTidalCRWithRules(dynamicClient, ctx, name, namespace, rules)
}

// CreateTidalCRWithNonExistentTarget 创建指向不存在目标的 Tidal CR（xlsx 用例19）
func CreateTidalCRWithNonExistentTarget(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	target := TidalTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: TidalTestNotExistDeploy, Namespace: namespace}
	return CreateTidalCRWithTargetAndRules(dynamicClient, ctx, name, namespace, target, Case2TidalRules)
}

// VerifyTidalConditionsDegradedWithReason 验证 conditions 中 Degraded=True, Ready=False 且 reason 包含指定值
func VerifyTidalConditionsDegradedWithReason(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, expectReason string) {
	Eventually(func() bool {
		obj, err := GetTidalCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		conditions, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
		for _, c := range conditions {
			cond := c.(map[string]interface{})
			ctype, _ := cond["type"].(string)
			status, _ := cond["status"].(string)
			reason, _ := cond["reason"].(string)
			if ctype == "Degraded" && status == "True" && strings.Contains(reason, expectReason) {
				return true
			}
			if ctype == "Ready" && status == "False" && strings.Contains(reason, expectReason) {
				return true
			}
		}
		return false
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "应有 condition Degraded=True 或 Ready=False 且 reason 包含 %s", expectReason)
}

// ListTidalCRs 列举 namespace 下的 Tidal CR
func ListTidalCRs(dynamicClient dynamic.Interface, ctx context.Context, namespace string) ([]unstructured.Unstructured, error) {
	list, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	return list.Items, nil
}

// VerifyTidalCRDeleted 验证 Tidal CR 已被删除
func VerifyTidalCRDeleted(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		_, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(),
		"Tidal %s/%s 应已被删除", namespace, name)
	GinkgoWriter.Printf("Tidal CR %s/%s 已删除\n", namespace, name)
}

// CreateTestStatefulSet 创建测试用 StatefulSet（xlsx 用例7），需先创建同名 headless Service
func CreateTestStatefulSet(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: corev1.ServiceSpec{
			Ports:     []corev1.ServicePort{{Port: 80}},
			Selector:  map[string]string{"app": name},
			ClusterIP: "None",
		},
	}
	if _, err := clientset.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{}); err != nil && !strings.Contains(err.Error(), "already exists") {
		return err
	}
	sts := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: appsv1.StatefulSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": name}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": name}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "nginx", Image: "nginx:alpine"}},
				},
			},
			ServiceName: name,
		},
	}
	_, err := clientset.AppsV1().StatefulSets(namespace).Create(ctx, sts, metav1.CreateOptions{})
	return err
}

// rsgGVR e2e 专用测试 CRD（用例9 自定义 CRD，仅用于验证 tidal 对 CRD targetRef 的支持）
var rsgGVR = schema.GroupVersionResource{
	Group: "e2e.tidal.io", Version: "v1alpha1", Resource: "tidaltestresources",
}

// CreateTestTidalTestResourceCRD 创建测试用 CRD 实例（用例9）
// 不依赖业务集群中的实际 CRD，而是为 e2e 单独创建一个最简 CRD，用于验证 tidal 对 CRD 目标资源的支持
func CreateTestTidalTestResourceCRD(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	// 1. 确保 e2e.tidal.io/v1alpha1, TidalTestResource 这个测试 CRD 存在
	crdGVR := schema.GroupVersionResource{
		Group: "apiextensions.k8s.io", Version: "v1", Resource: "customresourcedefinitions",
	}
	crdName := "tidaltestresources.e2e.tidal.io"

	// 尝试获取现有 CRD，若不存在则创建一个最简定义（无业务必填字段）
	_, err := dynamicClient.Resource(crdGVR).Get(ctx, crdName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		crdObj := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "apiextensions.k8s.io/v1",
				"kind":       "CustomResourceDefinition",
				"metadata": map[string]interface{}{
					"name": crdName,
				},
				"spec": map[string]interface{}{
					"group": "e2e.tidal.io",
					"names": map[string]interface{}{
						"kind":     "TidalTestResource",
						"listKind": "TidalTestResourceList",
						"plural":   "tidaltestresources",
						"singular": "tidaltestresource",
					},
					"scope": "Namespaced",
					"versions": []interface{}{
						map[string]interface{}{
							"name":    "v1alpha1",
							"served":  true,
							"storage": true,
							"schema": map[string]interface{}{
								"openAPIV3Schema": map[string]interface{}{
									// 最简结构：根对象下定义 spec 字段为任意对象，避免 unknown field "spec" 警告
									"type": "object",
									"properties": map[string]interface{}{
										"spec": map[string]interface{}{
											"type":                                 "object",
											"x-kubernetes-preserve-unknown-fields": true,
										},
									},
								},
							},
						},
					},
				},
			},
		}
		if _, err := dynamicClient.Resource(crdGVR).Create(ctx, crdObj, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
			return err
		}
		// 等待 APIServer 注册 CRD
		time.Sleep(3 * time.Second)
	} else if err != nil && !apierrors.IsNotFound(err) {
		return err
	}

	// 2. 创建一个简单的 TidalTestResource 实例，供 tidal 作为 CRD 目标引用
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "e2e.tidal.io/v1alpha1",
			"kind":       "TidalTestResource",
			"metadata":   map[string]interface{}{"name": name, "namespace": namespace},
			// 测试 CRD 不需要任何业务字段，这里不下发 spec，避免未知字段告警
		},
	}
	_, err = dynamicClient.Resource(rsgGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		return nil
	}
	return err
}

// CreateTestDaemonSet 创建测试用 DaemonSet（xlsx 用例8）
func CreateTestDaemonSet(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) error {
	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": name}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": name}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "nginx", Image: "nginx:alpine"}},
				},
			},
		},
	}
	_, err := clientset.AppsV1().DaemonSets(namespace).Create(ctx, ds, metav1.CreateOptions{})
	return err
}

// CleanupTidalTest 清理 tidal 测试资源：CR -> workload -> ns -> helm
func CleanupTidalTest(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("清理 tidal 测试资源...")
	crNames := []string{TidalTestCRName, TidalTestCRCase3Name, TidalTestCRCase5Name,
		TidalTestCRCase6Name, TidalTestCRCase7Name, TidalTestCRCase8Name,
		TidalTestCRCase9Name, TidalTestCRCase10Name, TidalTestCRCase11Name,
		TidalTestCRCase12Name, TidalTestCRCase13Name,
		TidalTestCRCase14Prefix + "-1", TidalTestCRCase14Prefix + "-2", TidalTestCRCase14Prefix + "-3",
		TidalTestCRCase15Prefix + "-1", TidalTestCRCase15Prefix + "-2", TidalTestCRCase15Prefix + "-3",
		TidalTestCRCase16Name, TidalTestCRCase17Name, TidalTestCRCase18Name, TidalTestCRCase19Name}
	for _, name := range crNames {
		_ = DeleteTidalCR(dynamicClient, ctx, name, TidalTestNamespace)
	}
	ns := TidalTestNamespace
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, TidalTestDeploymentName, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, TidalTestDeploymentC15_1, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, TidalTestDeploymentC15_2, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, TidalTestDeploymentC15_3, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, TidalTestNotExistDeploy, metav1.DeleteOptions{})
	_ = clientset.AppsV1().StatefulSets(ns).Delete(ctx, TidalTestStatefulSetName, metav1.DeleteOptions{})
	_ = clientset.AppsV1().DaemonSets(ns).Delete(ctx, TidalTestDaemonSetName, metav1.DeleteOptions{})
	_ = dynamicClient.Resource(rsgGVR).Namespace(ns).Delete(ctx, TidalTestCRDResourceName, metav1.DeleteOptions{})
	time.Sleep(2 * time.Second)
	_ = DeleteNamespaceIfExists(clientset, ctx, ns)
	_ = HelmUninstall(TidalHelmReleaseName, TidalHelmNamespace)
	GinkgoWriter.Println("清理完成")
}

func UpdateTidalLabels(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, labels map[string]string) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		obj, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		// append labels to existing labels
		existingLabels, _, _ := unstructured.NestedMap(obj.Object, "metadata", "labels")
		if existingLabels == nil {
			existingLabels = map[string]interface{}{}
		}
		for k, v := range labels {
			existingLabels[k] = v
		}
		if err := unstructured.SetNestedMap(obj.Object, existingLabels, "metadata", "labels"); err != nil {
			return err
		}
		_, err = dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		return err
	})
}
