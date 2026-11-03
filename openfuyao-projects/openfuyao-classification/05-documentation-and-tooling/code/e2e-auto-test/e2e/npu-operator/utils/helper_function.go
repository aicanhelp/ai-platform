package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/npu-operator-config"
	. "github.com/onsi/ginkgo/v2"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// FindNPUOperatorHelmRelease 验证找到 npu-operator 对应的 helm release
func FindNPUOperatorHelmRelease() (string, string, error) {
	selectedRelease, err := FindNPUOperatorHelmReleaseInfo()
	if err != nil {
		return "", "", err
	}
	return selectedRelease.Name, selectedRelease.Namespace, nil
}

func FindNPUOperatorHelmReleaseInfo() (HelmRelease, error) {
	// 获取 helm list 内容
	releases, err := GetHelmList()
	if err != nil {
		return HelmRelease{}, fmt.Errorf("get helm list error: %v", err)
	}

	// 查找 chart 名称以 "npu-operator" 开头的 release
	var foundReleases []HelmRelease
	for _, release := range releases {
		if strings.HasPrefix(release.Chart, "npu-operator") {
			foundReleases = append(foundReleases, release)
			GinkgoWriter.Printf("Found NPU Operator release: %s (chart: %s, namespace: %s, status: %s)\n",
				release.Name, release.Chart, release.Namespace, release.Status)
		}
	}

	if len(foundReleases) == 0 {
		return HelmRelease{}, fmt.Errorf("no NPU Operator release")
	}

	// 如果有多个，优先选择状态为 "deployed" 的，否则选择第一个
	var selectedRelease HelmRelease
	deployedFound := false

	for _, release := range foundReleases {
		if release.Status == "deployed" {
			selectedRelease = release
			deployedFound = true
			break
		}
	}

	if !deployedFound {
		selectedRelease = foundReleases[0]
	}

	helmReleaseName := selectedRelease.Name
	helmNamespace := selectedRelease.Namespace

	if helmReleaseName == "" {
		return HelmRelease{}, fmt.Errorf("helm release name should not be empty")
	}

	GinkgoWriter.Printf("Selected helm release: %s in namespace %s\n", helmReleaseName, helmNamespace)

	return selectedRelease, nil
}

// FilterNPUOperatorPods 过滤NPU Operator相关的Pod
func FilterNPUOperatorPods(clientset *kubernetes.Clientset, ctx context.Context) ([]corev1.Pod, error) {
	var filteredPods []corev1.Pod
	// 获取所有命名空间的Pod
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("列举pods失败: %w", err)
	}

	// 过滤 npu-operator 相关的 pods
	for _, pod := range podList.Items {
		for _, prefix := range npu_operator_config.Prefixes {
			if strings.HasPrefix(pod.Name, prefix) {
				filteredPods = append(filteredPods, pod)
				break
			}
		}
	}

	return filteredPods, nil
}

// GetSpecifiedPods 查找指定的Pod
func GetSpecifiedPods(clientset *kubernetes.Clientset, ctx context.Context, prefix string) ([]corev1.Pod, error) {
	var filteredPods []corev1.Pod
	// 获取所有命名空间的Pod
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("列举pods失败: %w", err)
	}

	// 查找指定的 Pod
	for _, pod := range podList.Items {
		if strings.HasPrefix(pod.Name, prefix) {
			filteredPods = append(filteredPods, pod)
		}
	}

	return filteredPods, nil
}

// GetNPUPolicy 获取NPUClusterPolicy
func GetNPUPolicy(dynamicClient dynamic.Interface, ctx context.Context) (unstructured.Unstructured, error, bool) {
	// 定义 NPUClusterPolicy 的 GVR (Group Version Resource)
	// 注意：需要根据实际的 CRD 定义调整这些值
	gvr := schema.GroupVersionResource{
		Group:    "npu.openfuyao.com",  // 根据实际的CRD组名调整
		Version:  "v1",                 // 根据实际的版本调整
		Resource: "npuclusterpolicies", // 复数形式
	}

	// 获取所有命名空间的 NPUClusterPolicy
	npuPolicies, err := dynamicClient.Resource(gvr).Namespace("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return unstructured.Unstructured{}, fmt.Errorf("获取NPUClusterPolicies失败: %w", err), false
	}

	GinkgoWriter.Printf("找到 %d 个 NPUClusterPolicy 资源：\n", len(npuPolicies.Items))

	// 验证只有一个 NPUClusterPolicy
	if len(npuPolicies.Items) != 1 {
		GinkgoWriter.Printf("NPUClusterPolicy数量不为1，不符合要求")
		return unstructured.Unstructured{}, nil, false
	}
	return npuPolicies.Items[0], nil, true
}

// GetNPUClusterPolicyStatus 获取NPUClusterPolicy的状态字段，具体路径根据实际的 CRD 结构调整（本用例为Status.Conditions[0].Type）
func GetNPUClusterPolicyStatus(npuPolicy unstructured.Unstructured) (string, error, bool) {
	// 获取Status字段
	status, found, err := unstructured.NestedMap(npuPolicy.Object, "status")
	if err != nil {
		return "", fmt.Errorf("获取NPUClusterPolicy Status失败: %w", err), false
	}
	if !found {
		GinkgoWriter.Printf("未找到 Status 或 Status 为空\n")
		return "", nil, false
	}

	// 最终需要识别的状态在Status.Phase中
	state, found, err := unstructured.NestedString(status, "phase")
	if err != nil {
		return "", fmt.Errorf("获取NPUClusterPolicy Phase失败: %w", err), false
	}
	if !found {
		GinkgoWriter.Printf("未找到 Phase 或 Phase 为空\n")
		return "", nil, false
	}

	return state, nil, true
}

// GetPodLogs 获取Pod日志内容
func GetPodLogs(clientset *kubernetes.Clientset, targetPod *corev1.Pod) (string, error) {
	// 获取Pod日志
	podLogOpts := corev1.PodLogOptions{}
	req := clientset.CoreV1().Pods(targetPod.Namespace).GetLogs(targetPod.Name, &podLogOpts)

	podLogs, err := req.Stream(context.TODO())
	if err != nil {
		return "", fmt.Errorf("打开log stream失败: %w", err)
	}
	defer func(podLogs io.ReadCloser) {
		err := podLogs.Close()
		if err != nil {
			GinkgoWriter.Printf("关闭pod logs失败: %v\n", err)
		}
	}(podLogs)

	// 读取日志内容
	logBuf := new(bytes.Buffer)
	_, err = io.Copy(logBuf, podLogs)
	if err != nil {
		return "", fmt.Errorf("读取log content失败: %w", err)
	}

	return logBuf.String(), nil
}

// ExtractNPUFromCapacity 提取Capacity字段中的NPU信息
func ExtractNPUFromCapacity(outputStr string) (int, error, bool) {
	// 提取Capacity部分
	lines := strings.Split(outputStr, "\n")
	inCapacity := false
	var npuCount int
	foundNPU := false

	for _, line := range lines {
		line = strings.TrimSpace(line)

		// 进入Capacity部分
		if strings.Contains(line, "Capacity:") {
			inCapacity = true
			continue
		}

		// 离开Capacity部分
		if inCapacity && (line == "Allocatable:" || line == "") {
			inCapacity = false
			break
		}

		// 在Capacity部分查找NPU
		if inCapacity && strings.Contains(line, "huawei.com/Ascend") {
			foundNPU = true
			GinkgoWriter.Printf("找到NPU字段：%s\n", line)
			parts := strings.Split(line, ":")
			if len(parts) != 2 {
				return 0, fmt.Errorf("NPU字段格式错误: %s", line), false
			}

			countStr := strings.TrimSpace(parts[1])
			count, err := strconv.Atoi(countStr)
			if err != nil {
				return 0, fmt.Errorf("NPU数量解析失败: %w", err), false
			}

			npuCount = count
			break
		}
	}

	if !foundNPU {
		GinkgoWriter.Println("未找到NPU字段")
	}
	return npuCount, nil, foundNPU
}

// GetNodeNpu 获取特定节点npu情况
func GetNodeNpu(nodeName string) (int, bool, error) {
	// 执行kubectl describe node
	cmd := exec.Command("kubectl", "describe", "node", nodeName)
	output, err := cmd.CombinedOutput()

	// 验证命令执行成功
	if err != nil {
		return 0, false, fmt.Errorf("'kubectl describe node'命令执行失败：%w", err)
	}

	outputStr := string(output)

	// 检查输出中是否包含Capacity部分
	if !strings.Contains(outputStr, "Capacity") {
		return 0, false, fmt.Errorf("输出不包含'Capacity'部分")
	}

	// 提取Capacity字段中的NPU信息
	npuCount, err, foundNPU := ExtractNPUFromCapacity(outputStr)
	if err != nil {
		return 0, false, fmt.Errorf("提取Capacity字段中的NPU信息：%w", err)
	}

	return npuCount, foundNPU, nil
}

// FindTargetPodList 获取前缀为podName的Pods
func FindTargetPodList(podName string, filteredPods []corev1.Pod) []*corev1.Pod {
	var targetPodList []*corev1.Pod
	for i, pod := range filteredPods {
		if strings.HasPrefix(pod.Name, podName) {
			targetPod := &filteredPods[i]
			GinkgoWriter.Printf("找到target pod: %s in namespace: %s\n",
				targetPod.Name, targetPod.Namespace)
			targetPodList = append(targetPodList, targetPod)
		}
	}
	return targetPodList
}

// FindNPUExporterService 查找NPU Exporter服务
func FindNPUExporterService(services *corev1.ServiceList) (string, int32, bool) {
	var found = false
	for _, service := range services.Items {
		if service.Namespace == "npu-exporter" && service.Name == "npu-exporter-service" {
			found = true
			clusterIP := service.Spec.ClusterIP
			ports := service.Spec.Ports
			if len(ports) == 0 {
				GinkgoWriter.Printf("NPU Exporter服务端口号为空，不符合规范")
				return "", 0, found
			}

			servicePort := ports[0].Port
			return clusterIP, servicePort, found
		}
	}
	return "", 0, found
}

// CreatePod 创建测试Pod对象
func CreatePod(npuType string) (*corev1.Pod, error) {
	// 转换 npuType
	convertedNpuType, err := ConvertServerType(npuType)
	if err != nil {
		return nil, fmt.Errorf("转换 npuType 失败：%w", err)
	}

	// 定义NPU资源
	npuResource := resource.MustParse("1")
	npuResourceType := corev1.ResourceName("huawei.com/" + convertedNpuType)

	// 创建Pod对象
	pod := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "npu-operator-test-pod",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:            "npu-operator-test-container",
					Image:           "docker.io/rayproject/ray",
					ImagePullPolicy: corev1.PullIfNotPresent,
					Command: []string{
						"/bin/bash",
						"-c",
						"while true;do sleep 30;done",
					},
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							npuResourceType: npuResource,
						},
						Limits: corev1.ResourceList{
							npuResourceType: npuResource,
						},
					},
				},
			},
		},
	}

	GinkgoWriter.Printf("npu-operator-test-pod Pod对象创建成功\n")
	return pod, nil
}

// CreateNPUOperatorTestPod 创建 npu-operator-test-pod
func CreateNPUOperatorTestPod(clientset *kubernetes.Clientset, ctx context.Context, npuType string) (string, string, error) {
	// 创建测试 Pod 对象
	npuOperatorTestPod, err := CreatePod(npuType)
	if err != nil {
		return "", "", fmt.Errorf("应该创建测试 Pod 对象：%v", err)
	}

	// 提取 Pod 名称
	podName := npuOperatorTestPod.ObjectMeta.Name
	GinkgoWriter.Printf("Pod名称: %s\n", podName)

	// 提取所有容器名称
	containerName := npuOperatorTestPod.Spec.Containers[0].Name
	GinkgoWriter.Printf("Container名称: %s\n", containerName)

	// 创建使用npu的pod
	GinkgoWriter.Printf("开始创建 Pod： %s ，Container: %s\n", podName, containerName)
	_, err = clientset.CoreV1().Pods("default").Create(ctx, npuOperatorTestPod, metav1.CreateOptions{})
	if err != nil {
		return "", "", fmt.Errorf("创建 Pod 失败: %w", err)
	}

	return podName, containerName, nil
}

// IdentifyMetrics 判断指标是否符合要求
func IdentifyMetrics(metricsResponse string, podName string, containerName string) bool {
	// 验证指标数据不为空
	if metricsResponse == "" {
		GinkgoWriter.Printf("指标数据为空\n")
		return false
	}

	// 验证指标数据中是否包含正确的 container_name 和 pod_name
	hasContainerName := strings.Contains(metricsResponse, "container_name=\""+containerName+"\"")
	hasPodName := strings.Contains(metricsResponse, "pod_name=\""+podName+"\"")

	if !hasContainerName || !hasPodName {
		GinkgoWriter.Printf("指标数据验证失败 - 包含 container_name: %t, 包含 pod_name: %t\n",
			hasContainerName, hasPodName)
		return false
	}

	GinkgoWriter.Printf("成功获取并验证指标数据，数据长度: %d 字节\n", len(metricsResponse))
	return true
}

func BuildNPUExporterMetricsURLForPodNode(
	clientset *kubernetes.Clientset,
	ctx context.Context,
	workloadPodName string,
) (string, error) {
	workloadPod, err := clientset.CoreV1().Pods("default").Get(ctx, workloadPodName, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("获取测试Pod失败: %w", err)
	}

	exporterPods, err := GetSpecifiedPods(clientset, ctx, "npu-exporter")
	if err != nil {
		return "", fmt.Errorf("获取npu-exporter Pod失败: %w", err)
	}

	for _, pod := range exporterPods {
		if pod.Spec.NodeName != workloadPod.Spec.NodeName {
			continue
		}
		if pod.Status.Phase != corev1.PodRunning || pod.Status.PodIP == "" {
			continue
		}

		return fmt.Sprintf("http://%s:8082/metrics", pod.Status.PodIP), nil
	}

	return "", fmt.Errorf("未找到节点 %s 上 Running 状态的 npu-exporter Pod", workloadPod.Spec.NodeName)
}

// DeleteNPUOperatorTestPod 删除 npu-operator-test-pod
func DeleteNPUOperatorTestPod(clientset *kubernetes.Clientset, ctx context.Context, podName string) error {
	// 删除 Pod
	GinkgoWriter.Printf("开始删除测试Pod，恢复原有环境：\n")
	err := clientset.CoreV1().Pods("default").Delete(ctx, podName, metav1.DeleteOptions{})
	if err != nil {
		return fmt.Errorf("删除 Pod 失败: %w", err)
	}
	return nil
}

// FilterNPUOperatorCRDs 过滤包含 "npuclusterpolicies" 且状态正常的 CRD
func FilterNPUOperatorCRDs(crds *apiextensionsv1.CustomResourceDefinitionList) ([]apiextensionsv1.CustomResourceDefinition, bool) {
	GinkgoWriter.Printf("%-50s %-10s\n", "NAME", "ESTABLISHED")
	var NPUOperatorCRDs []apiextensionsv1.CustomResourceDefinition
	found := false
	for _, crd := range crds.Items {
		// 过滤名字包含"npuclusterpolicies"
		key := "npuclusterpolicies"
		if strings.Contains(strings.ToLower(crd.Name), key) {
			established := "False"
			// 判断 CRD 是否已成功注册到 API server 并可以正常使用
			for _, condition := range crd.Status.Conditions {
				if condition.Type == apiextensionsv1.Established {
					if condition.Status == apiextensionsv1.ConditionTrue {
						established = "True"
						NPUOperatorCRDs = append(NPUOperatorCRDs, crd)
						found = true
					}
					break
				}
			}
			GinkgoWriter.Printf("%-50s %-10s\n", crd.Name, established)
		}
	}
	return NPUOperatorCRDs, found
}

// GetNPUOperatorCR 获取 NPUOperatorCR
func GetNPUOperatorCR(dynamicClient dynamic.Interface, gvr schema.GroupVersionResource) (unstructured.Unstructured, error) {
	// 找到该CRD对应的CR（取第一个，其余的无作用）
	crList, err := dynamicClient.Resource(gvr).Namespace("").List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return unstructured.Unstructured{}, fmt.Errorf("获取CR列表失败: %w", err)
	}
	if len(crList.Items) == 0 {
		return unstructured.Unstructured{}, fmt.Errorf("CR数量应为至少1个")
	}

	// 取第一个CR实例
	NPUOperatorCR := crList.Items[0]
	GinkgoWriter.Printf("找到CR: %s\n", NPUOperatorCR.GetName())

	return NPUOperatorCR, nil
}

// IsNPUOperatorCRReady 检查NPUOperatorCR状态是否为Ready
func IsNPUOperatorCRReady(NPUOperatorCR unstructured.Unstructured) (bool, error) {
	// 获取当前CR状态
	state, err, found := GetNPUClusterPolicyStatus(NPUOperatorCR)
	if err != nil {
		return false, fmt.Errorf("获取NPUClusterPolicy CR状态失败: %w", err)
	}
	if !found {
		return false, fmt.Errorf("未找到NPUClusterPolicy CR的状态字段")
	}

	// 检查CR状态是否为Ready
	GinkgoWriter.Printf("该NPUClusterPolicy的状态: %s\n", state)
	if strings.ToLower(state) == "notready" {
		return false, nil
	}
	return true, nil
}

func UpdateNPUOperatorCRWithRetry(dynamicClient dynamic.Interface, ctx context.Context,
	gvr schema.GroupVersionResource, mutator func(*unstructured.Unstructured) error) error {
	var lastErr error
	for attempt := 1; attempt <= 5; attempt++ {
		currentNPUOperatorCR, err := GetNPUOperatorCR(dynamicClient, gvr)
		if err != nil {
			return fmt.Errorf("????NPUOperatorCR??: %w", err)
		}

		if err := mutator(&currentNPUOperatorCR); err != nil {
			return err
		}

		_, err = dynamicClient.Resource(gvr).Update(ctx, &currentNPUOperatorCR, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}

		lastErr = err
		GinkgoWriter.Printf("??NPUOperatorCR??????%d???\\n", attempt)
		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("??NPUOperatorCR??????: %w", lastErr)
}

// ModifyComponentManaged ??????Managed?
func ModifyComponentManaged(componentName string, managed bool, NPUOperatorCR unstructured.Unstructured,
	dynamicClient dynamic.Interface, ctx context.Context, gvr schema.GroupVersionResource) error {
	_ = NPUOperatorCR
	err := UpdateNPUOperatorCRWithRetry(dynamicClient, ctx, gvr, func(currentNPUOperatorCR *unstructured.Unstructured) error {
		spec, found, err := unstructured.NestedMap(currentNPUOperatorCR.Object, "spec")
		if err != nil {
			return fmt.Errorf("??Spec????: %w", err)
		}
		if !found {
			return fmt.Errorf("???Spec??")
		}

		componentSpec, found := spec[componentName].(map[string]interface{})
		if !found {
			return fmt.Errorf("???%s??", componentName)
		}

		componentSpec["managed"] = managed
		spec[componentName] = componentSpec

		if err := unstructured.SetNestedMap(currentNPUOperatorCR.Object, spec, "spec"); err != nil {
			return fmt.Errorf("??????Spec??: %w", err)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("??%s???Managed??: %w", componentName, err)
	}

	GinkgoWriter.Printf("????%s???Managed???\\n", componentName)
	return nil
}

func GetComponentImageSpec(componentName string, NPUOperatorCR unstructured.Unstructured) (map[string]interface{}, error) {
	spec, found, err := unstructured.NestedMap(NPUOperatorCR.Object, "spec")
	if err != nil {
		return nil, fmt.Errorf("获取Spec字段失败: %w", err)
	}
	if !found {
		return nil, fmt.Errorf("未找到Spec字段")
	}

	componentSpec, found := spec[componentName].(map[string]interface{})
	if !found {
		return nil, fmt.Errorf("未找到%s字段", componentName)
	}

	imageSpec, found, err := unstructured.NestedMap(componentSpec, "imageSpec")
	if err != nil {
		return nil, fmt.Errorf("获取%s.imageSpec失败: %w", componentName, err)
	}
	if !found {
		return nil, fmt.Errorf("未找到%s.imageSpec字段", componentName)
	}

	return imageSpec, nil
}

func ModifyComponentManagedAndImageSpec(componentName string, managed bool, imageSpec map[string]interface{},
	NPUOperatorCR unstructured.Unstructured, dynamicClient dynamic.Interface,
	ctx context.Context, gvr schema.GroupVersionResource) error {
	_ = NPUOperatorCR
	err := UpdateNPUOperatorCRWithRetry(dynamicClient, ctx, gvr, func(currentNPUOperatorCR *unstructured.Unstructured) error {
		spec, found, err := unstructured.NestedMap(currentNPUOperatorCR.Object, "spec")
		if err != nil {
			return fmt.Errorf("??Spec????: %w", err)
		}
		if !found {
			return fmt.Errorf("???Spec??")
		}

		componentSpec, found := spec[componentName].(map[string]interface{})
		if !found {
			return fmt.Errorf("???%s??", componentName)
		}

		componentSpec["managed"] = managed
		componentSpec["imageSpec"] = imageSpec
		spec[componentName] = componentSpec

		if err := unstructured.SetNestedMap(currentNPUOperatorCR.Object, spec, "spec"); err != nil {
			return fmt.Errorf("??????Spec??: %w", err)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("??%s???managed?imageSpec??: %w", componentName, err)
	}

	GinkgoWriter.Printf("????%s???managed?imageSpec???\\n", componentName)
	return nil
}

func GetComponent(componentName string, NPUOperatorCR unstructured.Unstructured) (map[string]interface{}, error) {
	// 获取 Status 字段
	status, found, err := unstructured.NestedMap(NPUOperatorCR.Object, "status")
	if err != nil {
		return nil, fmt.Errorf("获取Status字段失败: %w", err)
	}
	if !found {
		return nil, fmt.Errorf("未找到Status字段")
	}

	// 获取 Component Statuses 数组
	componentStatuses, found, err := unstructured.NestedSlice(status, "componentStatuses")
	if err != nil {
		return nil, fmt.Errorf("获取Component Statuses失败: %w", err)
	}
	if !found {
		return nil, fmt.Errorf("未找到Component Statuses")
	}

	// 遍历 Component Statuses 查找匹配的组件
	for _, item := range componentStatuses {
		component, ok := item.(map[string]interface{})
		if !ok {
			continue // 跳过无效的条目
		}

		// 获取 Name 字段
		name, found, err := unstructured.NestedString(component, "name")
		if err != nil || !found {
			continue // 跳过没有 Name 字段的条目
		}

		// 检查 Name 是否包含 componentName
		if strings.Contains(name, componentName) {
			return component, nil
		}
	}
	return nil, fmt.Errorf("未找到包含 '%s' 的组件", componentName)
}

// GetComponentState 获取组件状态
func GetComponentState(componentName string, NPUOperatorCR unstructured.Unstructured) (string, string, error) {
	component, err := GetComponent(componentName, NPUOperatorCR)
	if err != nil {
		return "", "", fmt.Errorf("获取组件失败：%v", err)
	}

	if component == nil {
		return "", "", fmt.Errorf("未找到包含 '%s' 的组件", componentName)
	}

	// 获取 State 字段
	state, found, err := unstructured.NestedMap(component, "state")
	if err != nil || !found {
		return "", "", fmt.Errorf("找到组件但未成功获取State字段: %s", componentName)
	}

	// 获取 Reason 字段
	reason, found, err := unstructured.NestedString(state, "reason")
	if err != nil || !found {
		return "", "", fmt.Errorf("找到组件但未成功获取State.Reason字段: %s", componentName)
	}

	// 获取 Type 字段
	currentType, found, err := unstructured.NestedString(state, "type")
	if err != nil || !found {
		return "", "", fmt.Errorf("找到组件但未成功获取State.Type字段: %s", componentName)
	}

	return reason, currentType, nil
}

// GetComponentPrevState 获取组件上一个状态
func GetComponentStateDetails(componentName string, NPUOperatorCR unstructured.Unstructured) (string, string, string, error) {
	component, err := GetComponent(componentName, NPUOperatorCR)
	if err != nil {
		return "", "", "", fmt.Errorf("获取组件失败: %v", err)
	}

	if component == nil {
		return "", "", "", fmt.Errorf("未找到包含'%s' 的组件", componentName)
	}

	state, found, err := unstructured.NestedMap(component, "state")
	if err != nil || !found {
		return "", "", "", fmt.Errorf("找到组件但未成功获取State字段: %s", componentName)
	}

	reason, found, err := unstructured.NestedString(state, "reason")
	if err != nil || !found {
		return "", "", "", fmt.Errorf("找到组件但未成功获取State.Reason字段: %s", componentName)
	}

	currentType, found, err := unstructured.NestedString(state, "type")
	if err != nil || !found {
		return "", "", "", fmt.Errorf("找到组件但未成功获取State.Type字段: %s", componentName)
	}

	message, _, err := unstructured.NestedString(state, "message")
	if err != nil {
		return "", "", "", fmt.Errorf("找到组件但未成功获取State.Message字段: %s", componentName)
	}

	return reason, currentType, message, nil
}

func GetComponentPrevState(componentName string, NPUOperatorCR unstructured.Unstructured) (string, string, error) {
	component, err := GetComponent(componentName, NPUOperatorCR)
	if err != nil {
		return "", "", fmt.Errorf("获取组件失败：%v", err)
	}

	if component == nil {
		return "", "", fmt.Errorf("未找到包含 '%s' 的组件", componentName)
	}

	// 获取 prevState 字段
	prevState, found, err := unstructured.NestedMap(component, "prevState")
	if err != nil || !found {
		return "", "", fmt.Errorf("找到组件但未成功获取prevState字段: %s", componentName)
	}

	// 获取 Reason 字段
	reason, found, err := unstructured.NestedString(prevState, "reason")
	if err != nil || !found {
		return "", "", fmt.Errorf("找到组件但未成功获取prevState.Reason字段: %s", componentName)
	}

	// 获取 Type 字段
	currentType, found, err := unstructured.NestedString(prevState, "type")
	if err != nil || !found {
		return "", "", fmt.Errorf("找到组件但未成功获取prevState.Type字段: %s", componentName)
	}

	return reason, currentType, nil
}

// GetComponentReasonAndCurrentType 获取组件当前状态下的Reason和Type
func GetComponentReasonAndCurrentType(componentName string, NPUOperatorCR unstructured.Unstructured,
	verifyType string) (string, string, error) {
	var reason, currentType string
	var err error

	if verifyType == "state" {
		// 获取CR状态
		reason, currentType, err = GetComponentState(componentName, NPUOperatorCR)
		if err != nil {
			return "", "", fmt.Errorf("获取状态失败: %v\n", err)
		}
	} else if verifyType == "prevState" {
		// 获取CR上一个状态
		reason, currentType, err = GetComponentPrevState(componentName, NPUOperatorCR)
		if err != nil {
			return "", "", fmt.Errorf("获取上一个状态失败: %v\n", err)
		}
	} else {
		return "", "", fmt.Errorf("验证种类字段错误：%s\n", verifyType)
	}
	return reason, currentType, nil
}

// CompareReasonOrType 比较 reason 字段或者 type 字段
func CompareReasonOrType(reason string, expectReason string,
	currentType string, expectType string, verifyField string) bool {
	if verifyField == "reason" {
		return reason == expectReason
	} else if verifyField == "type" {
		return currentType == expectType
	} else {
		return reason == expectReason && currentType == expectType
	}
}

// HelmRelease 定义 Helm Release 结构体
type HelmRelease struct {
	Name       string `json:"name"`
	Namespace  string `json:"namespace"`
	Chart      string `json:"chart"`
	AppVersion string `json:"app_version"`
	Status     string `json:"status"`
	Updated    string `json:"updated"`
}

// 以下为安装卸载辅助函数

// GetHelmList 执行 helm list 获取信息
func GetHelmList() ([]HelmRelease, error) {
	// 执行 helm list 命令获取 JSON 输出
	cmd := exec.Command("helm", "list", "--all-namespaces", "-o", "json")
	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	err := cmd.Run()
	if err != nil {
		// 如果 helm 命令执行失败，可能是没有安装 helm 或者权限问题
		return nil, fmt.Errorf("failed to execute helm list command: %v\nStderr: %s", err, stderr.String())
	}

	// 解析 JSON 输出
	var releases []HelmRelease
	output := out.Bytes()
	if len(output) == 0 {
		return nil, fmt.Errorf("no helm releases found (empty output)")
	}

	err = json.Unmarshal(output, &releases)
	if err != nil {
		return nil, fmt.Errorf("failed to parse helm list JSON output: %v\nRaw output: %s", err, string(output))
	}

	GinkgoWriter.Printf("successfully get helm list JSON output\n")
	return releases, nil
}

// ExecuteHelmUninstall 执行 helm uninstall 命令
func ExecuteHelmUninstall(helmReleaseName string, helmNamespace string) error {
	// 构建 helm uninstall 命令
	var cmdArgs []string
	cmdArgs = append(cmdArgs, "uninstall", helmReleaseName)

	// 如果命名空间不为空，添加命名空间参数
	if helmNamespace != "" {
		cmdArgs = append(cmdArgs, "-n", helmNamespace)
	} else {
		// 如果没有指定命名空间，使用 --all-namespaces 参数
		cmdArgs = append(cmdArgs, "--all-namespaces")
	}

	GinkgoWriter.Printf("Executing: helm %s\n", strings.Join(cmdArgs, " "))
	cmd := exec.Command("helm", cmdArgs...)

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	err := cmd.Run()

	// helm uninstall 在 release 不存在时也会返回错误，需要结合后续pod卸载状况判断是否测试成功
	if err != nil {
		stderrStr := stderr.String()
		// 检查是否是 "release: not found" 错误，如果是，可能是已经被删除了
		if strings.Contains(stderrStr, "release: not found") {
			GinkgoWriter.Printf("Helm release %s not found (may have been already uninstalled)\n", helmReleaseName)
		} else {
			// 其他错误则测试失败
			return fmt.Errorf("helm uninstall failed: %s\nOutput: %s", stderrStr, out.String())
		}
	} else {
		GinkgoWriter.Printf("Successfully uninstalled helm release: %s\n", helmReleaseName)
	}

	return nil
}

// ExecuteHelmInstallFromURL 执行 helm install url 指令
func ExecuteHelmInstallFromURL() (string, string, error) {
	helmURL := strings.TrimSpace(os.Getenv("NPU_OPERATOR_HELM_URL"))
	if helmURL == "" {
		helmURL = npu_operator_config.HelmURL
	}

	helmVersion := DesiredNPUOperatorHelmVersion()

	GinkgoWriter.Println("Installing Helm chart:", helmURL)

	// 检查是否已存在相同的 release
	releases, err := GetHelmList()
	if err != nil {
		return npu_operator_config.ReleaseName, npu_operator_config.InstallationNamespace, fmt.Errorf("failed to get helm list: %v", err)
	}
	for _, release := range releases {
		if release.Name == npu_operator_config.ReleaseName {
			return "", "", fmt.Errorf("release %s already exists\n", npu_operator_config.ReleaseName)
		}
	}

	// 执行 helm install 命令，支持默认 OCI chart，也支持通过环境变量覆盖为本地待测 chart。
	args := []string{"install", npu_operator_config.ReleaseName, helmURL,
		"--create-namespace",
		"--namespace", npu_operator_config.InstallationNamespace,
		"--wait",
		"--timeout", "5m"}
	if helmVersion != "" && !isLocalHelmChart(helmURL) {
		args = append(args, "--version", helmVersion)
	}
	for _, valuesFile := range splitHelmEnvList(os.Getenv("NPU_OPERATOR_HELM_VALUES")) {
		args = append(args, "-f", valuesFile)
	}
	if setValues := strings.TrimSpace(os.Getenv("NPU_OPERATOR_HELM_SET")); setValues != "" {
		args = append(args, "--set", setValues)
	}
	if setStringValues := strings.TrimSpace(os.Getenv("NPU_OPERATOR_HELM_SET_STRING")); setStringValues != "" {
		args = append(args, "--set-string", setStringValues)
	}
	GinkgoWriter.Printf("Executing helm %s\n", strings.Join(args, " "))
	cmd := exec.Command("helm", args...)

	var out bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &stderr

	err = cmd.Run()
	if err != nil {
		return "", "", fmt.Errorf("Helm install output: %s\nHelm install error: %s\n", out.String(), stderr.String())
	}
	GinkgoWriter.Printf("Helm install successful: %s\n", out.String())

	return npu_operator_config.ReleaseName, npu_operator_config.InstallationNamespace, nil
}

func isLocalHelmChart(helmURL string) bool {
	normalized := strings.TrimSpace(helmURL)
	return normalized != "" &&
		!strings.HasPrefix(normalized, "oci://") &&
		!strings.HasPrefix(normalized, "http://") &&
		!strings.HasPrefix(normalized, "https://")
}

func DesiredNPUOperatorHelmVersion() string {
	helmVersion := npu_operator_config.HelmVersion
	if value, ok := os.LookupEnv("NPU_OPERATOR_HELM_VERSION"); ok {
		helmVersion = strings.TrimSpace(value)
	}
	if strings.EqualFold(helmVersion, "latest") {
		return ""
	}
	return helmVersion
}

func NPUOperatorHelmReleaseMatchesDesiredVersion(release HelmRelease) bool {
	desiredVersion := DesiredNPUOperatorHelmVersion()
	if desiredVersion == "" {
		return true
	}
	return release.Chart == "npu-operator-"+desiredVersion ||
		strings.HasSuffix(release.Chart, "-"+desiredVersion)
}

func splitHelmEnvList(value string) []string {
	rawItems := strings.FieldsFunc(value, func(r rune) bool {
		return r == ',' || r == ';'
	})
	items := make([]string, 0, len(rawItems))
	for _, item := range rawItems {
		item = strings.TrimSpace(item)
		if item != "" {
			items = append(items, item)
		}
	}
	return items
}

// FindAllNPUOperatorPods 查找所有NPU Operator相关的Pod（要求每种pod至少找到一个）
func FindAllNPUOperatorPods(clientset *kubernetes.Clientset, ctx context.Context) ([]corev1.Pod, error) {
	GinkgoWriter.Printf("Finding all npu-operator pods\n")
	var filteredPods []corev1.Pod
	// 获取所有命名空间的Pod
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("列举pods失败: %w", err)
	}

	// 查找所有 npu-operator 相关的 pods
	for _, prefix := range npu_operator_config.Prefixes {
		found := false
		for _, pod := range podList.Items {
			if strings.HasPrefix(pod.Name, prefix) {
				filteredPods = append(filteredPods, pod)
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("%s not found", prefix)
		}
	}

	GinkgoWriter.Printf("Found %d npu-operator pods\n", len(filteredPods))
	return filteredPods, nil
}

// BoolPtr 获得布尔类型的指针
func BoolPtr(b bool) *bool {
	return &b
}

// CheckNodeForNpu 检查单个节点上是否有指定NPU设备
func CheckNodeForNpu(clientset *kubernetes.Clientset, ctx context.Context, nodeName string, npuType string) (bool, error) {
	// 创建唯一的Pod名称
	podName := fmt.Sprintf("npu-checker-%s-%s-%d",
		strings.ToLower(nodeName),
		strings.ToLower(npuType),
		time.Now().UnixNano())

	// 创建临时Pod
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			NodeName:      nodeName,
			HostPID:       true,
			HostNetwork:   true,
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:    "checker",
					Image:   "alpine:latest",
					Command: []string{"/bin/sh"},
					Args: []string{
						"-c",
						fmt.Sprintf(`
                            echo "在节点 $HOSTNAME 上检查 %s 设备..."
                            # 安装必要的工具
                            apk add --no-cache pciutils grep > /dev/null 2>&1
                            
                            # 执行检查
                            if lspci 2>/dev/null | grep -i %s > /dev/null; then
                                echo "RESULT:FOUND"
                                exit 0
                            else
                                echo "RESULT:NOT_FOUND"
                                exit 1
                            fi
                        `, npuType, npuType),
					},
					SecurityContext: &corev1.SecurityContext{
						Privileged: BoolPtr(true),
					},
				},
			},
			Tolerations: []corev1.Toleration{
				{
					Operator: corev1.TolerationOpExists,
				},
			},
		},
	}

	// 创建Pod
	GinkgoWriter.Printf("在节点 %s 上创建检查Pod: %s\n", nodeName, podName)
	_, err := clientset.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		return false, fmt.Errorf("在节点 %s 上创建检查Pod失败: %v", nodeName, err)
	}

	// 确保Pod被清理
	defer func() {
		GinkgoWriter.Printf("清理节点 %s 上的检查Pod: %s\n", nodeName, podName)
		err := clientset.CoreV1().Pods("default").Delete(ctx, podName, metav1.DeleteOptions{})
		if err != nil {
			GinkgoWriter.Printf("清理检测pod过程出现未知异常：%v\n", err)
			return
		}
	}()

	// 等待Pod完成（最多等待30秒）
	var podPhase corev1.PodPhase
	for i := 0; i < 30; i++ {
		currentPod, err := clientset.CoreV1().Pods("default").Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			GinkgoWriter.Printf("获取Pod状态失败: %v\n", err)
			time.Sleep(1 * time.Second)
			continue
		}

		podPhase = currentPod.Status.Phase
		if podPhase == corev1.PodSucceeded || podPhase == corev1.PodFailed {
			break
		}

		time.Sleep(1 * time.Second)
	}

	// 获取Pod日志
	req := clientset.CoreV1().Pods("default").GetLogs(podName, &corev1.PodLogOptions{})
	logBytes, err := req.DoRaw(ctx)
	if err != nil {
		return false, fmt.Errorf("获取节点 %s 的日志失败: %v", nodeName, err)
	}

	logs := string(logBytes)
	GinkgoWriter.Printf("节点 %s 检查日志:\n%s\n", nodeName, logs)

	// 判断是否找到设备
	if strings.Contains(logs, "RESULT:FOUND") {
		GinkgoWriter.Printf("节点 %s: 找到 %s 设备\n", nodeName, npuType)
		return true, nil
	} else if strings.Contains(logs, "RESULT:NOT_FOUND") {
		GinkgoWriter.Printf("节点 %s: 未找到 %s 设备\n", nodeName, npuType)
		return false, nil
	}

	// 如果日志中既没有FOUND也没有NOT_FOUND，检查Pod状态
	if podPhase == corev1.PodSucceeded {
		// Pod成功执行但没有找到设备（grep返回0找到，返回1没找到）
		// 但我们的脚本设计是找到设备时返回0，没找到返回1
		// 所以Pod成功执行意味着找到了设备
		GinkgoWriter.Printf("节点 %s: Pod执行成功，找到 %s 设备\n", nodeName, npuType)
		return true, nil
	} else if podPhase == corev1.PodFailed {
		GinkgoWriter.Printf("节点 %s: Pod执行失败，未找到 %s 设备\n", nodeName, npuType)
		return false, nil
	}

	return false, fmt.Errorf("节点 %s: 检查超时或无法确定结果", nodeName)
}

func InstantiateSSH(nodeNum string) (*executor.SSHExecutor, error) {
	var nodeIP = os.Getenv("TEST_NODE" + nodeNum + "_IP")
	var nodePassword = os.Getenv("TEST_NODE" + nodeNum + "_PASSWORD")
	var nodePort = os.Getenv("TEST_NODE" + nodeNum + "_PORT")
	var nodeUser = os.Getenv("TEST_NODE" + nodeNum + "_USER")

	port, err := strconv.Atoi(nodePort)
	if err != nil {
		return nil, fmt.Errorf("failed to convert node port to int: %v", err)
	}

	sshExecutor, err := executor.NewSSHExecutor(nodeIP, port, nodeUser, nodePassword)
	if err != nil {
		return nil, fmt.Errorf("failed to init ssh executor: %v", err)
	}

	return sshExecutor, nil
}

// OSInfo 定义 OS 信息结构体
type OSInfo struct {
	Name      string
	VersionID string
}

// GetOSInfo 获取 OS 名称和版本号
func GetOSInfo(nodeNum string) (*OSInfo, error) {
	sshExecutor, err := InstantiateSSH(nodeNum)
	if err != nil {
		return nil, fmt.Errorf("failed to init ssh executor: %v", err)
	}

	// 获取os信息
	result, err := sshExecutor.Exec("cat /etc/os-release")
	if err != nil {
		return nil, fmt.Errorf("failed to execute `cat /etc/os-release` command: %v", err)
	}
	if result == nil || result.ExitCode != 0 {
		return nil, fmt.Errorf("failed to get os-release info")
	}

	// 按行分割字符串
	lines := strings.Split(result.Stdout, "\n")
	info := &OSInfo{}

	for _, line := range lines {
		// 跳过注释和空行
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}

		// 按等号分割键值
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}

		key := parts[0]
		value := strings.Trim(parts[1], `"`) // 去除引号

		switch key {
		case "NAME":
			info.Name = value
		case "VERSION_ID":
			info.VersionID = value
		}
	}

	// 验证必要字段
	if info.Name == "" {
		return nil, fmt.Errorf("无法解析OS名称")
	}
	if info.VersionID == "" {
		info.VersionID = "unknown" // 设置默认值
	}

	return info, nil
}

func GetOSInfoByNodeName(nodeName string) (*OSInfo, error) {
	nodeNum, err := FindNodeEnvNumLoose(nodeName)
	if err != nil {
		return nil, err
	}

	return GetOSInfo(nodeNum)
}

var ErrNodeSSHConfigNotFound = errors.New("node ssh config not found")

type NodePCIIdentity struct {
	Vendor          string
	Device          string
	SubsystemVendor string
	SubsystemDevice string
}

type NodePCIRecord struct {
	ClassID  string
	Identity NodePCIIdentity
}

type NodeRDMAPCIeInfo struct {
	LnkCapSpeed string
	LnkCapWidth string
	LnkStaSpeed string
	LnkStaWidth string
}

type NodeRDMAInfo struct {
	DeviceSpeeds map[string]string
	DevicePCIe   map[string]NodeRDMAPCIeInfo
}

const (
	unknownCardModel      = "UNKNOWN"
	pciAcceleratorClassID = "0x1200"
	pciBridgeClassID      = "0x0604"
	nodeArchUnknown       = "UNKNOWN"
	nodeArchAarch64       = "aarch64"
	nodeArchX8664         = "x86_64"
	maxTestNodeEnvCount   = 64
)

var nodeCardServerMap = map[NodePCIIdentity]map[string]string{
	{Vendor: "0x19e5", Device: "0xd100", SubsystemVendor: "0x0200", SubsystemDevice: "0x0100"}: {"x86_64": "A300-3010", "aarch64": "A300-3000"},
	{Vendor: "0x19e5", Device: "0xd801", SubsystemVendor: "0x0200", SubsystemDevice: "0x0100"}: {"*": "A300T-9000"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x0200", SubsystemDevice: "0x0100"}: {"*": "A900T"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3000"}: {"*": "A900T"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3001"}: {"*": "A900T"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3003"}: {"*": "A900T"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3400"}: {"*": "A900T"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3401"}: {"*": "A900T"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3402"}: {"*": "A900T"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3403"}: {"*": "A900T"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x6000"}: {"*": "A300t-a2"},
	{Vendor: "0x19e5", Device: "0xd500", SubsystemVendor: "0x0200", SubsystemDevice: "0x0100"}: {"*": "A300i-pro"},
	{Vendor: "0x19e5", Device: "0xd500", SubsystemVendor: "0x0200", SubsystemDevice: "0x0110"}: {"*": "A300i-duo"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x4000"}: {"*": "A300i-a2"},
	{Vendor: "0x19e5", Device: "0xd105", SubsystemVendor: "0x0200", SubsystemDevice: "0x0100"}: {"*": "A200i-a2"},
	{Vendor: "0x19e5", Device: "0xd107", SubsystemVendor: "0x0000", SubsystemDevice: "0x0000"}: {"*": "A200i-a2"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3002"}: {"*": "Atlas 800I A2"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3004"}: {"*": "Atlas 800I A2"},
	{Vendor: "0x19e5", Device: "0xd802", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3005"}: {"*": "Atlas 800I A2"},
	{Vendor: "0x19e5", Device: "0xd803", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3000"}: {"*": "Atlas 900 A3 Pod"},
	{Vendor: "0x19e5", Device: "0xd803", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3001"}: {"*": "Atlas 900 A3 Pod"},
	{Vendor: "0x19e5", Device: "0xd803", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3002"}: {"*": "Atlas 900 A3 Pod"},
	{Vendor: "0x19e5", Device: "0xd803", SubsystemVendor: "0x19e5", SubsystemDevice: "0x0100"}: {"*": "Atlas 800I A3"},
	{Vendor: "0x19e5", Device: "0xd803", SubsystemVendor: "0x19e5", SubsystemDevice: "0x3003"}: {"*": "Atlas 800I A3"},
}

var nodeServerBoardIDMap = map[string][]string{
	"Atlas_800":           {"0x02", "0x27", "0x21", "0x24", "0x28"},
	"Atlas_800_A2":        {"0x30", "0x31", "0x32", "0x34", "0x38"},
	"Atlas_900_A2_PoD":    {"0x30", "0x31", "0x32", "0x34"},
	"Atlas_200T_A2_Box16": {"0x50", "0x51", "0x53", "0x52"},
	"Atlas_300T":          {"0x01", "0x03", "0x06"},
	"Atlas_300T_A2":       {"0x10", "0x13", "0x12", "0x11"},
	"Atlas_800I_A3":       {"0xb3"},
}

func GetBoardIDsByServerModel(serverModel string) []string {
	boardIDs := nodeServerBoardIDMap[serverModel]
	return append([]string(nil), boardIDs...)
}

func FindNodeEnvNumLoose(nodeName string) (string, error) {
	for i := 1; i <= maxTestNodeEnvCount; i++ {
		nodeEnvName := strings.TrimSpace(os.Getenv("TEST_NODE" + strconv.Itoa(i) + "_NAME"))
		if nodeEnvName == "" {
			nodeEnvName = strings.TrimSpace(os.Getenv("TEST_NODE" + strconv.Itoa(i) + "_HOSTNAME"))
		}
		if nodeEnvName == "" {
			continue
		}
		if nodeName == nodeEnvName {
			return strconv.Itoa(i), nil
		}
	}

	return "", fmt.Errorf("%w: %s", ErrNodeSSHConfigNotFound, nodeName)
}

func InstantiateSSHByNodeName(nodeName string) (*executor.SSHExecutor, error) {
	nodeNum, err := FindNodeEnvNumLoose(nodeName)
	if err != nil {
		return nil, err
	}

	nodeIP := strings.TrimSpace(os.Getenv("TEST_NODE" + nodeNum + "_IP"))
	nodePassword := os.Getenv("TEST_NODE" + nodeNum + "_PASSWORD")
	nodePort := strings.TrimSpace(os.Getenv("TEST_NODE" + nodeNum + "_PORT"))
	nodeUser := strings.TrimSpace(os.Getenv("TEST_NODE" + nodeNum + "_USER"))
	if nodeIP == "" || nodePassword == "" || nodePort == "" || nodeUser == "" {
		return nil, fmt.Errorf("%w: incomplete SSH config for TEST_NODE%s", ErrNodeSSHConfigNotFound, nodeNum)
	}

	return InstantiateSSH(nodeNum)
}

func FindFirstConfiguredNPUNodeName(nodesNPUSituation map[string]string) (string, error) {
	var nodeNames []string
	for nodeName, npuType := range nodesNPUSituation {
		if npuType == "none" {
			continue
		}
		if _, err := FindNodeEnvNumLoose(nodeName); err != nil {
			if errors.Is(err, ErrNodeSSHConfigNotFound) {
				GinkgoWriter.Printf("节点 %s 未配置 SSH，跳过 Ascend Docker Runtime 远端验证\n", nodeName)
				continue
			}
			return "", err
		}
		nodeNames = append(nodeNames, nodeName)
	}

	sort.Strings(nodeNames)
	if len(nodeNames) == 0 {
		return "", ErrNodeSSHConfigNotFound
	}

	return nodeNames[0], nil
}

func FindRemoteAscendDockerRuntimeBinary(nodeName string) (string, error) {
	sshExecutor, err := InstantiateSSHByNodeName(nodeName)
	if err != nil {
		return "", err
	}

	command := `sh -lc 'for p in \
/var/lib/npu-container-toolkit/runtime/ascend-docker-runtime \
/var/lib/ascend-docker-runtime/ascend-docker-runtime; do \
  if [ -x "$p" ]; then echo "$p"; exit 0; fi; \
done; \
if command -v ascend-docker-runtime >/dev/null 2>&1; then command -v ascend-docker-runtime; exit 0; fi; \
exit 1'`
	result, err := sshExecutor.Exec(command)
	if err != nil || result == nil || result.ExitCode != 0 {
		return "", fmt.Errorf("failed to locate ascend-docker-runtime on node %s", nodeName)
	}

	runtimeBinary := strings.TrimSpace(result.Stdout)
	if runtimeBinary == "" {
		return "", fmt.Errorf("ascend-docker-runtime path is empty on node %s", nodeName)
	}

	return runtimeBinary, nil
}

func RemoteImageExists(nodeName string, namespace string, image string) (bool, error) {
	sshExecutor, err := InstantiateSSHByNodeName(nodeName)
	if err != nil {
		return false, err
	}

	command := fmt.Sprintf(`sh -lc 'ctr -n %s image ls -q 2>/dev/null'`, namespace)
	result, err := sshExecutor.Exec(command)
	if err != nil || result == nil {
		return false, fmt.Errorf("failed to list images on node %s", nodeName)
	}

	images := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	for _, existingImage := range images {
		existingImage = strings.TrimSpace(existingImage)
		if existingImage == "" {
			continue
		}
		if existingImage == image {
			return true, nil
		}
	}

	return false, nil
}

func PullImageOnNode(nodeName string, namespace string, image string) error {
	sshExecutor, err := InstantiateSSHByNodeName(nodeName)
	if err != nil {
		return err
	}

	command := fmt.Sprintf(`sh -lc 'ctr -n %s image pull %s'`, namespace, image)
	result, err := sshExecutor.Exec(command)
	if err != nil || result == nil || result.ExitCode != 0 {
		if result != nil {
			return fmt.Errorf("failed to pull image on node %s: %s", nodeName, result.Stderr)
		}
		return fmt.Errorf("failed to pull image on node %s", nodeName)
	}

	return nil
}

func RemoveImageOnNode(nodeName string, namespace string, image string) error {
	sshExecutor, err := InstantiateSSHByNodeName(nodeName)
	if err != nil {
		return err
	}

	command := fmt.Sprintf(`sh -lc 'ctr -n %s image rm %s >/dev/null 2>&1 || true'`, namespace, image)
	_, err = sshExecutor.Exec(command)
	return err
}

func StartContainerOnNode(nodeName string, namespace string, runtimeBinary string, image string, containerID string) error {
	sshExecutor, err := InstantiateSSHByNodeName(nodeName)
	if err != nil {
		return err
	}

	command := fmt.Sprintf(
		`sh -lc 'ctr -n %s run -d --tty=false --runtime io.containerd.runc.v2 --runc-binary %s --env ASCEND_VISIBLE_DEVICES=0 %s %s sleep 300'`,
		namespace, runtimeBinary, image, containerID,
	)
	result, err := sshExecutor.Exec(command)
	if err != nil || result == nil || result.ExitCode != 0 {
		if result != nil {
			return fmt.Errorf("failed to start container on node %s: %s", nodeName, result.Stderr)
		}
		return fmt.Errorf("failed to start container on node %s", nodeName)
	}

	time.Sleep(500 * time.Millisecond)
	return nil
}

func IsContainerRunningOnNode(nodeName string, id string, namespace string) (bool, error) {
	sshExecutor, err := InstantiateSSHByNodeName(nodeName)
	if err != nil {
		return false, err
	}

	command := fmt.Sprintf(`sh -lc 'ctr -n %s task ls'`, namespace)
	result, err := sshExecutor.Exec(command)
	if err != nil || result == nil {
		return false, fmt.Errorf("failed to list container tasks on node %s", nodeName)
	}

	lines := strings.Split(result.Stdout, "\n")
	for _, line := range lines {
		if line == "" || strings.HasPrefix(line, "TASK") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) >= 3 && fields[0] == id {
			return strings.EqualFold(fields[2], "running"), nil
		}
	}

	return false, nil
}

func ExecCommandInContainerOnNode(nodeName string, containerID string, namespace string, command ...string) (string, error) {
	sshExecutor, err := InstantiateSSHByNodeName(nodeName)
	if err != nil {
		return "", err
	}

	execID := fmt.Sprintf("exec_%d", time.Now().UnixNano())
	args := append([]string{"-n", namespace, "task", "exec", "--exec-id", execID, containerID}, command...)
	remoteCommand := fmt.Sprintf(`sh -lc 'ctr %s'`, strings.Join(args, " "))
	result, err := sshExecutor.Exec(remoteCommand)
	if err != nil || result == nil || result.ExitCode != 0 {
		if result != nil {
			return "", fmt.Errorf("failed to execute command in container on node %s: %s", nodeName, result.Stderr)
		}
		return "", fmt.Errorf("failed to execute command in container on node %s", nodeName)
	}

	return strings.TrimSpace(result.Stdout), nil
}

func CleanupContainerOnNode(nodeName string, id string, namespace string) {
	sshExecutor, err := InstantiateSSHByNodeName(nodeName)
	if err != nil {
		return
	}

	commands := []string{
		fmt.Sprintf(`sh -lc 'ctr -n %s task kill -s KILL %s >/dev/null 2>&1 || true'`, namespace, id),
		fmt.Sprintf(`sh -lc 'ctr -n %s task delete %s >/dev/null 2>&1 || true'`, namespace, id),
		fmt.Sprintf(`sh -lc 'ctr -n %s container delete %s >/dev/null 2>&1 || true'`, namespace, id),
	}

	for _, command := range commands {
		_, _ = sshExecutor.Exec(command)
	}
}

func ResolveNodeCardModels(nodes *corev1.NodeList) (map[string]string, error) {
	nodeCardModels := make(map[string]string, len(nodes.Items))
	resolvedTargetNodes := 0

	for _, node := range nodes.Items {
		if _, hasNPU := node.Labels["openfuyao.com/npu.present"]; !hasNPU {
			nodeCardModels[node.Name] = unknownCardModel
			continue
		}

		cardModel, err := GetNodeCardModel(node)
		if err != nil {
			if errors.Is(err, ErrNodeSSHConfigNotFound) {
				GinkgoWriter.Printf("skip resolving card model for %s: %v\n", node.Name, err)
				continue
			}
			return nil, err
		}
		nodeCardModels[node.Name] = cardModel
		resolvedTargetNodes++
	}

	if resolvedTargetNodes == 0 {
		return nil, fmt.Errorf("%w: no SSH-configured NPU nodes", ErrNodeSSHConfigNotFound)
	}

	return nodeCardModels, nil
}

func ResolveNodeBoardIDs(nodes *corev1.NodeList) (map[string]string, error) {
	nodeBoardIDs := make(map[string]string, len(nodes.Items))
	resolvedTargetNodes := 0

	for _, node := range nodes.Items {
		if _, hasNPU := node.Labels["openfuyao.com/npu.present"]; !hasNPU {
			nodeBoardIDs[node.Name] = ""
			continue
		}

		boardID, err := GetNodeBoardID(node)
		if err != nil {
			if errors.Is(err, ErrNodeSSHConfigNotFound) {
				GinkgoWriter.Printf("skip resolving board id for %s: %v\n", node.Name, err)
				continue
			}
			return nil, err
		}
		nodeBoardIDs[node.Name] = boardID
		resolvedTargetNodes++
	}

	if resolvedTargetNodes == 0 {
		return nil, fmt.Errorf("%w: no SSH-configured NPU nodes", ErrNodeSSHConfigNotFound)
	}

	return nodeBoardIDs, nil
}

func GetNodeBoardID(node corev1.Node) (string, error) {
	sshExecutor, err := InstantiateSSHByNodeName(node.Name)
	if err != nil {
		return "", err
	}

	return getNodeBoardID(sshExecutor)
}

func ResolveNodeRDMAInfo(nodes *corev1.NodeList) (map[string]NodeRDMAInfo, error) {
	nodeRDMAInfo := make(map[string]NodeRDMAInfo, len(nodes.Items))
	resolvedNodes := 0

	for _, node := range nodes.Items {
		info, err := GetNodeRDMAInfo(node)
		if err != nil {
			if errors.Is(err, ErrNodeSSHConfigNotFound) {
				GinkgoWriter.Printf("skip resolving RDMA info for %s: %v\n", node.Name, err)
				continue
			}
			return nil, err
		}
		nodeRDMAInfo[node.Name] = info
		resolvedNodes++
	}

	if resolvedNodes == 0 {
		return nil, fmt.Errorf("%w: no SSH-configured nodes for RDMA inspection", ErrNodeSSHConfigNotFound)
	}

	return nodeRDMAInfo, nil
}

func GetNodeRDMAInfo(node corev1.Node) (NodeRDMAInfo, error) {
	sshExecutor, err := InstantiateSSHByNodeName(node.Name)
	if err != nil {
		return NodeRDMAInfo{}, err
	}

	return getNodeRDMAInfo(sshExecutor)
}

func GetNodeCardModel(node corev1.Node) (string, error) {
	sshExecutor, err := InstantiateSSHByNodeName(node.Name)
	if err != nil {
		return "", err
	}

	pciRecords, err := getNodePCIRecords(sshExecutor)
	if err != nil {
		return "", err
	}

	return resolveNodeCardModel(pciRecords, getNodeArchKey(node)), nil
}

func getNodeBoardID(sshExecutor *executor.SSHExecutor) (string, error) {
	command := `sh -lc 'info=$(/usr/local/sbin/npu-smi info -m 2>/dev/null | awk '\''/Ascend/{print $1" "$2; exit}'\''); [ -n "$info" ] || exit 1; set -- $info; /usr/local/sbin/npu-smi info -t board -i "$1" -c "$2" 2>/dev/null | awk -F: '\''/Board/{gsub(/^[ \t]+|[ \t]+$/, "", $2); print tolower($2); exit}'\'''`
	result, err := sshExecutor.Exec(command)
	if err != nil {
		return "", fmt.Errorf("failed to inspect board id: %v", err)
	}
	if result == nil {
		return "", fmt.Errorf("failed to inspect board id: empty SSH result")
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("failed to inspect board id: %s", result.Stderr)
	}

	boardID := strings.TrimSpace(result.Stdout)
	if boardID == "" {
		return "", fmt.Errorf("failed to inspect board id: empty output")
	}

	return boardID, nil
}

func getNodeRDMAInfo(sshExecutor *executor.SSHExecutor) (NodeRDMAInfo, error) {
	deviceNames, err := getNodeRDMADeviceNames(sshExecutor)
	if err != nil {
		return NodeRDMAInfo{}, err
	}

	info := NodeRDMAInfo{
		DeviceSpeeds: make(map[string]string),
		DevicePCIe:   make(map[string]NodeRDMAPCIeInfo),
	}

	for _, deviceName := range deviceNames {
		if speed, speedErr := getNodeRDMASpeed(sshExecutor, deviceName); speedErr == nil && speed != "" {
			info.DeviceSpeeds[deviceName] = speed
		}

		if pcieInfo, pcieErr := getNodeRDMAPCIeInfo(sshExecutor, deviceName); pcieErr == nil {
			info.DevicePCIe[deviceName] = pcieInfo
		}
	}

	return info, nil
}

func getNodeRDMADeviceNames(sshExecutor *executor.SSHExecutor) ([]string, error) {
	result, err := sshExecutor.Exec(`sh -lc '/usr/sbin/rdma link 2>/dev/null || true'`)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect rdma devices: %v", err)
	}
	if result == nil {
		return nil, fmt.Errorf("failed to inspect rdma devices: empty SSH result")
	}

	var devices []string
	for _, line := range strings.Split(result.Stdout, "\n") {
		line = strings.TrimSpace(line)
		if !strings.Contains(line, "state ACTIVE") || !strings.Contains(line, "physical_state LINK_UP") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) == 0 {
			continue
		}

		deviceName := parts[len(parts)-1]
		if deviceName != "" {
			devices = append(devices, deviceName)
		}
	}

	return devices, nil
}

func getNodeRDMASpeed(sshExecutor *executor.SSHExecutor, deviceName string) (string, error) {
	result, err := sshExecutor.Exec(fmt.Sprintf(`sh -lc '/usr/sbin/ethtool %s 2>/dev/null || true'`, deviceName))
	if err != nil {
		return "", fmt.Errorf("failed to inspect rdma speed for %s: %v", deviceName, err)
	}
	if result == nil {
		return "", fmt.Errorf("failed to inspect rdma speed for %s: empty SSH result", deviceName)
	}

	re := regexp.MustCompile(`Speed:\s*(\d+\S+)`)
	match := re.FindStringSubmatch(result.Stdout)
	if len(match) < 2 {
		return "", fmt.Errorf("failed to inspect rdma speed for %s: speed information not found", deviceName)
	}

	return strings.ReplaceAll(match[1], "/", "p"), nil
}

func getNodeRDMAPCIeInfo(sshExecutor *executor.SSHExecutor, deviceName string) (NodeRDMAPCIeInfo, error) {
	pciAddress, err := getNodeRDMAPCIAddress(sshExecutor, deviceName)
	if err != nil {
		return NodeRDMAPCIeInfo{}, err
	}

	result, err := sshExecutor.Exec(fmt.Sprintf(`sh -lc '/usr/bin/lspci -vv -s %s 2>/dev/null || true'`, pciAddress))
	if err != nil {
		return NodeRDMAPCIeInfo{}, fmt.Errorf("failed to inspect pcie info for %s: %v", deviceName, err)
	}
	if result == nil {
		return NodeRDMAPCIeInfo{}, fmt.Errorf("failed to inspect pcie info for %s: empty SSH result", deviceName)
	}

	re := regexp.MustCompile(`(?i)(LnkCap|LnkSta):.*?Speed\s+([0-9.]+[A-Za-z/]+).*?Width\s+(x\d+)`)
	matches := re.FindAllStringSubmatch(result.Stdout, -1)
	info := NodeRDMAPCIeInfo{}

	for _, match := range matches {
		if len(match) < 4 {
			continue
		}

		speed := strings.ReplaceAll(match[2], "/", "p")
		width := match[3]
		if strings.EqualFold(match[1], "LnkCap") {
			info.LnkCapSpeed = speed
			info.LnkCapWidth = width
			continue
		}

		info.LnkStaSpeed = speed
		info.LnkStaWidth = width
	}

	if info.LnkCapSpeed == "" || info.LnkCapWidth == "" || info.LnkStaSpeed == "" || info.LnkStaWidth == "" {
		return NodeRDMAPCIeInfo{}, fmt.Errorf("failed to inspect pcie info for %s: incomplete lspci output", deviceName)
	}

	return info, nil
}

func getNodeRDMAPCIAddress(sshExecutor *executor.SSHExecutor, deviceName string) (string, error) {
	result, err := sshExecutor.Exec(fmt.Sprintf(`sh -lc '/usr/sbin/udevadm info /sys/class/net/%s 2>/dev/null || true'`, deviceName))
	if err != nil {
		return "", fmt.Errorf("failed to inspect pci address for %s: %v", deviceName, err)
	}
	if result == nil {
		return "", fmt.Errorf("failed to inspect pci address for %s: empty SSH result", deviceName)
	}

	for _, line := range strings.Split(result.Stdout, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "P:") {
			continue
		}

		parts := strings.Split(line, "/")
		if len(parts) < 3 {
			break
		}

		pciAddress := parts[len(parts)-3]
		if pciAddress != "" {
			return pciAddress, nil
		}
	}

	return "", fmt.Errorf("failed to inspect pci address for %s: address not found", deviceName)
}

func getNodePCIRecords(sshExecutor *executor.SSHExecutor) ([]NodePCIRecord, error) {
	command := `sh -lc 'for d in /sys/bus/pci/devices/*; do
  [ -f "$d/class" ] || continue
  class=$(cat "$d/class" 2>/dev/null)
  case "$class" in
    0x1200*|0x0604*)
      vendor=$(cat "$d/vendor" 2>/dev/null)
      device=$(cat "$d/device" 2>/dev/null)
      subsystem_vendor=$(cat "$d/subsystem_vendor" 2>/dev/null)
      subsystem_device=$(cat "$d/subsystem_device" 2>/dev/null)
      printf "%s,%s,%s,%s,%s\n" "$class" "$vendor" "$device" "$subsystem_vendor" "$subsystem_device"
      ;;
  esac
done'`
	result, err := sshExecutor.Exec(command)
	if err != nil {
		return nil, fmt.Errorf("failed to inspect PCI devices: %v", err)
	}
	if result == nil {
		return nil, fmt.Errorf("failed to inspect PCI devices: empty SSH result")
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("failed to inspect PCI devices: %s", result.Stderr)
	}

	var pciRecords []NodePCIRecord
	for _, line := range strings.Split(result.Stdout, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		parts := strings.Split(line, ",")
		if len(parts) != 5 {
			continue
		}

		pciRecords = append(pciRecords, NodePCIRecord{
			ClassID: parts[0],
			Identity: NodePCIIdentity{
				Vendor:          parts[1],
				Device:          parts[2],
				SubsystemVendor: parts[3],
				SubsystemDevice: parts[4],
			},
		})
	}

	return pciRecords, nil
}

func resolveNodeCardModel(pciRecords []NodePCIRecord, archKey string) string {
	tmpValue := unknownCardModel
	for _, pciRecord := range pciRecords {
		isAccelerator := strings.HasPrefix(pciRecord.ClassID, pciAcceleratorClassID)
		isBridge := strings.HasPrefix(pciRecord.ClassID, pciBridgeClassID)
		if !isAccelerator && !isBridge {
			continue
		}

		cardNames, exists := nodeCardServerMap[pciRecord.Identity]
		if !exists {
			continue
		}

		cardName := resolveNodeCardName(cardNames, archKey)
		if cardName == "" {
			continue
		}

		if isBridge {
			tmpValue = cardName
			continue
		}

		return cardName
	}

	return tmpValue
}

func resolveNodeCardName(cardNames map[string]string, archKey string) string {
	if archKey == nodeArchUnknown {
		return ""
	}

	if value := cardNames[archKey]; value != "" {
		return value
	}

	return cardNames["*"]
}

func getNodeArchKey(node corev1.Node) string {
	arch, exists := node.Labels["kubernetes.io/arch"]
	if !exists {
		return nodeArchUnknown
	}

	if arch == "arm64" {
		return nodeArchAarch64
	}

	return nodeArchX8664
}

// JudgePodsAllRunning 判断是否所有 pod 状态都为 running
func JudgePodsAllRunning(clientset *kubernetes.Clientset, ctx context.Context) (bool, error) {
	pods, err := FindAllNPUOperatorPods(clientset, ctx)
	if err != nil {
		return false, fmt.Errorf("error getting all npu-operator related pods: %v", err)
	}

	if len(pods) == 0 {
		return false, fmt.Errorf("no NPU Operator pods found")
	}

	allRunning := true
	for _, pod := range pods {
		if pod.Status.Phase != corev1.PodRunning {
			GinkgoWriter.Printf("Pod %s is not running (status: %s)\n",
				pod.Name, pod.Status.Phase)
			allRunning = false
		}
	}

	return allRunning, nil
}

// FindNodesNPUSituation 确定节点NPU设备情况
func FindNodesNPUSituation(clientset *kubernetes.Clientset, ctx context.Context) (map[string]string, error) {
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("'kubectl get nodes' 执行失败：%v", err)
	}
	if len(nodes.Items) == 0 {
		return nil, fmt.Errorf("在集群中没有找到任何节点")
	}

	nodeNPUSituation := make(map[string]string)

	for _, node := range nodes.Items {
		nodeName := node.Name
		GinkgoWriter.Printf("\n=== 处理节点: %s ===\n", nodeName)

		// 检查该节点是否有NPU设备
		success := false
		foundNPUType := "none"
		for _, npuType := range npu_operator_config.NpuTypes {
			success, err = CheckNodeForNpu(clientset, ctx, nodeName, npuType)
			if err != nil {
				GinkgoWriter.Printf("节点 %s: %s检查失败: %v\n", nodeName, npuType, err)
				continue
			}
			if !success {
				GinkgoWriter.Printf("节点 %s: 未找到 %s NPU设备\n", nodeName, npuType)
				continue
			}

			// 找到设备，验证节点标签
			success = true
			foundNPUType = npuType
			GinkgoWriter.Printf("节点 %s: 找到 %s NPU设备\n", nodeName, npuType)
			break
		}

		nodeNPUSituation[nodeName] = foundNPUType
		GinkgoWriter.Printf("节点：%s   NPU：%s\n", nodeName, nodeNPUSituation[nodeName])
	}

	return nodeNPUSituation, nil
}

// StartContainer 创建并启动容器
func StartContainer() error {
	cmd := exec.Command("ctr", "-n", npu_operator_config.K8SNamespace, "run",
		"-d", "--tty=false",
		"--runtime", "io.containerd.runc.v2",
		"--runc-binary", "/var/lib/npu-container-toolkit/runtime/ascend-docker-runtime",
		"--env", "ASCEND_VISIBLE_DEVICES=0",
		npu_operator_config.ContainerImage,
		npu_operator_config.ContainerID,
		"sleep", "300",
	)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	// 创建容器
	err := cmd.Run()
	if err != nil {
		return fmt.Errorf("failed to start container: %v\nstderr: %s", err, stderr.String())
	}

	// 等待一小段时间确保进程启动
	time.Sleep(500 * time.Millisecond)
	GinkgoWriter.Println("容器创建并启动启动命令执行成功")

	return nil
}

// IsContainerRunning 检查容器是否运行
func IsContainerRunning(id string, namespace string) (bool, error) {
	// 执行 ctr task ls 命令
	cmd := exec.Command("ctr", "-n", namespace, "task", "ls")

	output, err := cmd.Output()
	if err != nil {
		return false, fmt.Errorf("查看容器运行状态失败: %v", err)
	}

	lines := strings.Split(string(output), "\n")

	for _, line := range lines {
		// 跳过空行和标题行
		if line == "" || strings.HasPrefix(line, "TASK") {
			continue
		}

		// 按空格分割行
		fields := strings.Fields(line)
		if len(fields) >= 1 {
			// 检查容器ID是否匹配
			if fields[0] == id {
				// 获取状态（通常是第三列）
				var status string
				if len(fields) >= 3 {
					status = fields[2]
				}
				if strings.ToLower(status) == "running" {
					GinkgoWriter.Printf("容器已开始运行，状态：%s", status)
					return true, nil
				} else {
					GinkgoWriter.Printf("容器未开始运行，状态：%s", status)
					return false, nil
				}
			}
		}
	}

	GinkgoWriter.Println("未找到指定容器")
	return false, nil
}

// ExecCommandInContainer 在容器内执行命令
func ExecCommandInContainer(containerID string, namespace string, command ...string) (string, error) {
	execID := fmt.Sprintf("exec_%d", time.Now().UnixNano())
	args := []string{"-n", namespace, "task", "exec", "--exec-id", execID, containerID}
	args = append(args, command...)

	cmd := exec.Command("ctr", args...)
	output, err := cmd.CombinedOutput()
	outputStr := strings.TrimSpace(string(output))

	// 错误处理：暴露具体的退出码和输出
	if err != nil {
		// 解析 exit status（比如 exit status 1）
		exitErr, ok := err.(*exec.ExitError)
		if ok {
			return "", fmt.Errorf("command error: exit status %d, output: %s", exitErr.ExitCode(), outputStr)
		}
		// 非退出码错误（如命令找不到、执行超时等）
		return "", fmt.Errorf("command execute failed: %v, output: %s", err, outputStr)
	}

	return outputStr, nil
}

// CleanupContainer 清理容器
func CleanupContainer(id string, namespace string) {
	commands := []*exec.Cmd{
		exec.Command("ctr", "-n", namespace, "task", "kill", "-s", "KILL", id),
		exec.Command("ctr", "-n", namespace, "task", "delete", id),
		exec.Command("ctr", "-n", namespace, "container", "delete", id),
	}

	for _, cmd := range commands {
		cmd.Run() // 忽略错误，因为容器可能不存在
	}
}

// ImageManager 镜像管理器
type ImageManager struct {
	Namespace string
	Timeout   time.Duration
}

// NewImageManager 创建镜像管理器
func NewImageManager(namespace string) *ImageManager {
	return &ImageManager{
		Namespace: namespace,
		Timeout:   10 * time.Minute,
	}
}

// ImageExists 检查镜像是否存在于指定命名空间
func (im *ImageManager) ImageExists(image string) bool {
	cmd := exec.Command("ctr", "-n", im.Namespace, "image", "ls", "-q")
	output, err := cmd.Output()
	if err != nil {
		return false
	}

	images := strings.Split(strings.TrimSpace(string(output)), "\n")

	// 检查镜像是否在列表中
	for _, img := range images {
		if im.imageMatch(img, image) {
			return true
		}
	}

	return false
}

// imageMatch 检查两个镜像名称是否匹配
func (im *ImageManager) imageMatch(existingImage, targetImage string) bool {
	// 如果完全相等，直接返回true
	if existingImage == targetImage {
		return true
	}

	// 处理简写形式
	// 例如：targetImage = "ubuntu:22.04"，existingImage = "npu_operator_config.ContainerImagePrefix + ubuntu:22.04"
	if !strings.Contains(targetImage, "/") {
		// 尝试加上 docker.io/library/ 前缀
		fullImage := npu_operator_config.ContainerImagePrefix + targetImage
		if existingImage == fullImage {
			return true
		}
	}

	return false
}

// PullIfNotExists 如果镜像不存在则拉取
func (im *ImageManager) PullIfNotExists(image string) (bool, error) {
	// 检查镜像是否已存在
	if im.ImageExists(image) {
		fmt.Printf("Image %s already exists in namespace %s, skipping pull\n", image, im.Namespace)
		return false, nil
	}

	fmt.Printf("Image %s not found in namespace %s, pulling...\n", image, im.Namespace)

	// 拉取镜像
	err := im.PullImage(image)
	if err != nil {
		return false, fmt.Errorf("failed to pull image %s: %w", image, err)
	}

	fmt.Printf("Successfully pulled image %s to namespace %s\n", image, im.Namespace)
	return true, nil
}

// PullImage 拉取镜像
func (im *ImageManager) PullImage(image string) error {
	ctx, cancel := context.WithTimeout(context.Background(), im.Timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "ctr", "-n", im.Namespace, "image", "pull", image)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to pull image: %v\nstdout: %s\nstderr: %s",
			err, stdout.String(), stderr.String())
	}

	return nil
}

// RemoveImage 删除镜像
func (im *ImageManager) RemoveImage(image string) error {
	cmd := exec.Command("ctr", "-n", im.Namespace, "image", "rm", image)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to remove image %s: %v\nstderr: %s",
			image, err, stderr.String())
	}

	fmt.Printf("Removed image %s from namespace %s\n", image, im.Namespace)
	return nil
}

// ListImages 列出命名空间中的所有镜像
func (im *ImageManager) ListImages() ([]string, error) {
	cmd := exec.Command("ctr", "-n", im.Namespace, "image", "ls", "-q")
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("failed to list images: %v", err)
	}

	images := strings.Split(strings.TrimSpace(string(output)), "\n")
	return images, nil
}

// FindNodeEnvNum 找到node在环境变量中对应的序号
func FindNodeEnvNum(nodeName string, nodeSum int) (string, error) {
	return FindNodeEnvNumLoose(nodeName)
}

// ConvertServerType 转化npu类型
func ConvertServerType(servertype string) (string, error) {
	if servertype == "d100" {
		return "Ascend310", nil
	}
	if servertype == "d500" {
		return "Ascend310P", nil
	}
	if servertype == "d801" {
		return "Ascend910", nil
	}
	if servertype == "d802" {
		return "Ascend910", nil
	}
	if servertype == "d803" {
		return "Ascend910", nil
	}
	return "", fmt.Errorf("servertype %s not recognized", servertype)
}

// JudgeInstallationEnvironment 判断安装环境是否符合要求
func JudgeInstallationEnvironment(nodes *corev1.NodeList, verifyEnvironments []*VerifyEnvironment, NodesNPUSituation map[string]string) (bool, error) {
	var verifyEnvironmentMap = make(map[VerifyEnvironment]bool)
	for _, verifyEnvironment := range verifyEnvironments {
		verifyEnvironmentMap[*verifyEnvironment] = false
	}
	for _, node := range nodes.Items {
		if NodesNPUSituation[node.Name] == "none" {
			GinkgoWriter.Printf("当前节点 %s 没有npu设备，跳过验证\n", node.Name)
		} else {
			servertype := NodesNPUSituation[node.Name]
			servertype, err := ConvertServerType(servertype)
			if err != nil {
				return false, fmt.Errorf("node %s failed to get server type", node.Name)
			}

			systemtype, exists := node.Labels["kubernetes.io/arch"]
			if !exists {
				return false, fmt.Errorf("node %s failed to find system type", node.Name)
			}

			osInfo, err := GetOSInfoByNodeName(node.Name)
			if err != nil {
				if errors.Is(err, ErrNodeSSHConfigNotFound) {
					GinkgoWriter.Printf("%s 节点未配置SSH环境，跳过多节点环境匹配\n", node.Name)
					continue
				}
				return false, fmt.Errorf("failed to get %s OS info: %v", node.Name, err)
			}

			testEnvironment := VerifyEnvironment{
				NpuType:    servertype,
				EnvName:    strings.ToLower(osInfo.Name),
				VersionID:  osInfo.VersionID,
				SystemType: systemtype,
			}

			_, included := verifyEnvironmentMap[testEnvironment]
			if included {
				verifyEnvironmentMap[testEnvironment] = true
				GinkgoWriter.Printf("%s 节点: %s %s %s %s在验证环境范围内\n", node.Name, testEnvironment.NpuType, testEnvironment.EnvName, testEnvironment.VersionID, testEnvironment.SystemType)
			} else {
				GinkgoWriter.Printf("%s 节点: %s %s %s %s不在验证环境范围内\n", node.Name, testEnvironment.NpuType, testEnvironment.EnvName, testEnvironment.VersionID, testEnvironment.SystemType)
			}
		}
	}

	for _, included := range verifyEnvironmentMap {
		if !included {
			GinkgoWriter.Println("验证环境未完全满足")
			return false, nil
		}
	}
	GinkgoWriter.Println("验证环境完全满足")
	return true, nil
}
