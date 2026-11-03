package utils

import (
	"context"
	stderrs "errors"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/npu-operator-config"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// VerifyHelmUninstall 验证 helm uninstall 成功执行
func VerifyHelmUninstall(helmReleaseName string, helmNamespace string) {
	Expect(helmReleaseName != "", "helm release name should not be empty")

	// 执行 helm uninstall 命令
	err := ExecuteHelmUninstall(helmReleaseName, helmNamespace)
	Expect(err).NotTo(HaveOccurred(), "should successfully execute helm uninstall")
}

// VerifyAllNPUOperatorRelatedPodsRemoved 验证所有 npu-operator 相关 Pod 都被卸载
func VerifyAllNPUOperatorRelatedPodsRemoved(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("Checking for NPU-related pods in all namespaces...")

	// 使用 client-go 检查所有命名空间的 pod
	// 设置超时时间为 2 分钟
	Eventually(func() bool {
		npuPods, err := FilterNPUOperatorPods(clientset, ctx)
		if err != nil {
			GinkgoWriter.Printf("Error finding npu-operator related pods: %v\n", err)
			return false
		}

		if len(npuPods) > 0 {
			GinkgoWriter.Printf("Found %d NPU-related pods still running:\n", len(npuPods))
			for _, pod := range npuPods {
				GinkgoWriter.Printf("  - %s\n", pod.Name)
			}
			return false
		}

		return true
	}, 5*time.Minute, 5*time.Second).Should(BeTrue(),
		"NPU-related pods should be removed within 5 minutes")

	GinkgoWriter.Println("All NPU-related pods have been removed")
}

// VerifyNPUSmiInfo 执行 npu-smi info 命令查看驱动安装情况
func VerifyNPUSmiInfo(npuDriverExists bool) {
	// 检查 npu-smi 命令是否存在
	cmd := exec.Command("which", "npu-smi")
	err := cmd.Run()
	Expect(err).NotTo(HaveOccurred(), "should find npu-smi command")
	GinkgoWriter.Printf("npu-smi info 命令存在，开始执行···")

	// 执行 npu-smi info 命令
	cmd = exec.Command("npu-smi", "info")
	output, err := cmd.CombinedOutput()

	// 验证命令执行无错误且输出不为空
	Expect(err).NotTo(HaveOccurred(), "'npu-smi info'命令应执行成功")
	if npuDriverExists {
		Expect(string(output)).NotTo(BeEmpty(), "npu驱动仍然存在，'npu-smi info' 命令的输出结果不应为空")
	} else {
		Expect(string(output)).To(BeEmpty(), "npu驱动已被卸载，'npu-smi info' 命令的输出结果应为空")
	}
	GinkgoWriter.Printf("当前驱动存在情况：%t，npu-smi info 命令得到字符串长度：%d，验证成功\n", npuDriverExists, len(output))
}

// VerifyNPUOperatorUninstallation 验证 npu-operator 卸载流程
func VerifyNPUOperatorUninstallation(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("Verifying NPU Operator uninstallation ···")

	// 找到 npu-operator 对应的 helm release
	helmReleaseName, helmNamespace, err := FindNPUOperatorHelmRelease()
	if err != nil {
		if strings.Contains(err.Error(), "no NPU Operator release") {
			GinkgoWriter.Println("No npu-operator helm release found, verify only residual pod cleanup")
			VerifyAllNPUOperatorRelatedPodsRemoved(clientset, ctx)
			GinkgoWriter.Println("npu-operator 插件卸载验证成功")
			return
		}
		Expect(err).NotTo(HaveOccurred(), "should successfully find helm release information")
	}

	// 验证 helm uninstall 成功执行
	VerifyHelmUninstall(helmReleaseName, helmNamespace)

	// 等待一段时间让卸载操作完成
	GinkgoWriter.Println("wait for the operation for ten seconds ···")
	time.Sleep(10 * time.Second)
	GinkgoWriter.Println("end of waiting")

	// 验证所有 npu-operator 相关 Pod 都被卸载
	VerifyAllNPUOperatorRelatedPodsRemoved(clientset, ctx)

	// 额外等待一段时间确保所有资源被清理
	GinkgoWriter.Println("Waiting additional time for resource cleanup ···")
	time.Sleep(15 * time.Second)

	GinkgoWriter.Println("npu-operator 插件卸载验证成功")
}

// VerifyHelmReleaseStatus 验证 Helm Release 状态
func VerifyHelmReleaseStatus(helmReleaseName string, helmNamespace string) {
	GinkgoWriter.Println("Verifying Helm release status ···")

	Eventually(func() string {
		releases, err := GetHelmList()
		if err != nil {
			return ""
		}
		for _, release := range releases {
			if release.Name == helmReleaseName && release.Namespace == helmNamespace {
				GinkgoWriter.Printf("Helm release status: %s\n", release.Status)
				return release.Status
			}
		}
		GinkgoWriter.Printf("Helm release status: \"\"\n")
		return ""
	}, 2*time.Minute, 5*time.Second).Should(Equal("deployed"),
		"Helm release should be in deployed status in 2 minutes")

	GinkgoWriter.Println("Helm release status verified: deployed")
}

// VerifyNPUOperatorPodsRunning 验证 npu-operator 相关 pods 状态
func VerifyNPUOperatorPodsRunning(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("Verifying NPU Operator pods are running...")

	Eventually(func() bool {
		allRunning, err := JudgePodsAllRunning(clientset, ctx)
		if err != nil {
			GinkgoWriter.Printf("Error verifying pods: %v\n", err)
			return false
		}

		if allRunning {
			GinkgoWriter.Printf("All NPU Operator pods are running\n")
		}

		return allRunning
	}, 10*time.Minute, 10*time.Second).Should(BeTrue(),
		"All NPU Operator pods should be running in 10 minutes")
}

// VerifyPodsExpected 验证 Pod 集合符合预期
func VerifyPodsExpected(clientset *kubernetes.Clientset, ctx context.Context, prefix string, expectedLength int) {
	Eventually(func(g Gomega) {
		pods, err := GetSpecifiedPods(clientset, ctx, prefix)
		g.Expect(err).NotTo(HaveOccurred(), "should successfully get specified pods")
		readyCount := 0
		notReadyPods := make([]string, 0)
		for _, pod := range pods {
			if pod.DeletionTimestamp != nil {
				continue
			}
			if isPodReady(pod) {
				readyCount++
				continue
			}
			notReadyPods = append(notReadyPods, fmt.Sprintf("%s/%s:%s", pod.Namespace, pod.Name, pod.Status.Phase))
		}
		g.Expect(readyCount).To(BeNumerically(">=", expectedLength),
			"should find at least %d ready pods, current pods: %v", expectedLength, podStateSummary(pods))
		g.Expect(notReadyPods).To(BeEmpty(), "all active filtered pods should be ready")
	}, 180*time.Second, 2*time.Second).Should(Succeed(), fmt.Sprintf("%s pods should reach expected state", prefix))

	GinkgoWriter.Printf("%s 组件 pods 验证成功\n", prefix)
}

func isPodReady(pod corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func podStateSummary(pods []corev1.Pod) []string {
	summary := make([]string, 0, len(pods))
	for _, pod := range pods {
		ready := "not-ready"
		if isPodReady(pod) {
			ready = "ready"
		}
		summary = append(summary, fmt.Sprintf("%s/%s:%s/%s", pod.Namespace, pod.Name, pod.Status.Phase, ready))
	}
	return summary
}

// VerifyComponentsInstallation 验证各组件安装情况
func VerifyComponentsInstallation(clientset *kubernetes.Clientset, ctx context.Context, nodesNPUSituation map[string]string) {
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})

	Expect(err).NotTo(HaveOccurred(), "'kubectl get nodes' 执行失败")
	Expect(nodes.Items).NotTo(BeEmpty(), "在集群中没有找到任何节点")

	GinkgoWriter.Printf("检测到nodes数量:\n%d\n", len(nodes.Items))

	// 验证 Deployments 各组件运行状态
	for _, prefix := range npu_operator_config.Deployments {
		var expectedLength = 1
		VerifyPodsExpected(clientset, ctx, prefix, expectedLength)
	}

	// 验证 Daemonsets 各组件运行状态
	for _, prefix := range npu_operator_config.Daemonsets {
		expectedLength := GetExpectedDaemonsetPodCount(clientset, ctx, prefix, nodes, nodesNPUSituation)
		VerifyPodsExpected(clientset, ctx, prefix, expectedLength)
	}

	GinkgoWriter.Println("所有组件的 pods 均验证成功")
}

func GetExpectedDaemonsetPodCount(
	clientset *kubernetes.Clientset,
	ctx context.Context,
	prefix string,
	nodes *corev1.NodeList,
	nodesNPUSituation map[string]string,
) int {
	daemonsets, err := clientset.AppsV1().DaemonSets("").List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, daemonset := range daemonsets.Items {
			if strings.HasPrefix(daemonset.Name, prefix) {
				return int(daemonset.Status.DesiredNumberScheduled)
			}
		}
	}

	expectedLength := 0
	for _, withNPU := range nodesNPUSituation {
		if withNPU != "none" {
			expectedLength++
		}
	}

	if prefix == "npu-feature-discovery" || prefix == "mindio-acp" || prefix == "mindio-tft" {
		return len(nodes.Items)
	}

	return expectedLength
}

// VerifyNPUOperatorInstallation 验证 npu-operator 安装流程
func VerifyNPUOperatorInstallation(clientset *kubernetes.Clientset, ctx context.Context, nodesNPUSituation map[string]string) {
	var err error
	GinkgoWriter.Println("Verifying NPU Operator installation ···")

	// 执行 helm install url 指令
	helmReleaseName, helmNamespace, err := ExecuteHelmInstallFromURL()
	Expect(err).NotTo(HaveOccurred(), "should successfully execute helm install")

	// 验证 Helm Release 状态
	VerifyHelmReleaseStatus(helmReleaseName, helmNamespace)

	// 验证 npu-operator 相关 pods 状态
	VerifyNPUOperatorPodsRunning(clientset, ctx)

	// 验证各组件安装情况
	VerifyComponentsInstallation(clientset, ctx, nodesNPUSituation)

	GinkgoWriter.Println("npu-operator 插件安装验证成功\n")
}

func EnsureNPUOperatorInstallation(
	clientset *kubernetes.Clientset,
	ctx context.Context,
	nodesNPUSituation map[string]string,
) bool {
	GinkgoWriter.Println("Verifying NPU Operator installation...")

	helmRelease, err := FindNPUOperatorHelmReleaseInfo()
	if err == nil {
		if !NPUOperatorHelmReleaseMatchesDesiredVersion(helmRelease) {
			GinkgoWriter.Printf(
				"Found npu-operator release %s with chart %s, reinstall expected chart version %s\n",
				helmRelease.Name,
				helmRelease.Chart,
				DesiredNPUOperatorHelmVersion(),
			)
			Expect(ExecuteHelmUninstall(helmRelease.Name, helmRelease.Namespace)).To(Succeed())
			VerifyNPUOperatorInstallation(clientset, ctx, nodesNPUSituation)
			return true
		}
		GinkgoWriter.Printf(
			"Found existing npu-operator release %s in namespace %s, reuse current environment\n",
			helmRelease.Name,
			helmRelease.Namespace,
		)
		VerifyHelmReleaseStatus(helmRelease.Name, helmRelease.Namespace)
		VerifyNPUOperatorPodsRunning(clientset, ctx)
		VerifyComponentsInstallation(clientset, ctx, nodesNPUSituation)
		GinkgoWriter.Println("npu-operator installation verification succeeded")
		return false
	}

	VerifyNPUOperatorInstallation(clientset, ctx, nodesNPUSituation)
	return true
}

func VerifyPullContainerImage(imageManager *ImageManager) {
	Eventually(func() bool {
		// 确保测试镜像存在
		pulled, err := imageManager.PullIfNotExists(npu_operator_config.ContainerImage)
		if err != nil {
			GinkgoWriter.Printf("should successfully pull container image: %v", err)
			return false
		}

		if pulled {
			GinkgoWriter.Printf("Pulled image %s\n", npu_operator_config.ContainerImage)
		} else {
			GinkgoWriter.Printf("Image %s already exists\n", npu_operator_config.ContainerImage)
		}

		images, err := imageManager.ListImages()
		if err != nil {
			GinkgoWriter.Printf("Error listing images: %v\n", err)
			return false
		}

		for _, img := range images {
			if strings.Contains(img, npu_operator_config.ContainerImage) {
				GinkgoWriter.Printf("Images in %s namespace:\n", npu_operator_config.K8SNamespace)
				GinkgoWriter.Printf("  - %s\n", img)
				break
			}
		}

		// 确保我们的测试镜像在列表中
		if slices.Contains(images, npu_operator_config.ContainerImage) {
			GinkgoWriter.Printf("%s has been already pulled in %s\n", npu_operator_config.ContainerImage, npu_operator_config.K8SNamespace)
			return true
		} else {
			GinkgoWriter.Printf("%s not found in %s\n", npu_operator_config.ContainerImage, npu_operator_config.K8SNamespace)
			return false
		}
	}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(BeTrue(),
		"Container image should be pulled in 60 seconds")

	GinkgoWriter.Printf("Successfully pulled %s in %s\n", npu_operator_config.ContainerImage, npu_operator_config.K8SNamespace)
}

// VerifyStartContainer 验证启动容器
func VerifyStartContainer() {
	// 启动容器
	GinkgoWriter.Println("Starting container")
	err := StartContainer()
	Expect(err).NotTo(HaveOccurred(), "should successfully start container")

	// 等待容器启动并运行
	GinkgoWriter.Println("Waiting for container to be ready")
	Eventually(func() bool {
		running, err := IsContainerRunning(npu_operator_config.ContainerID, npu_operator_config.K8SNamespace)
		if err != nil {
			GinkgoWriter.Printf("Error checking if container is running: %v\n", err)
			return false
		}
		return running
	}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(BeTrue(),
		"Container should be running in 60 seconds")

	GinkgoWriter.Printf("Container %s is ready\n", npu_operator_config.ContainerID)
}

// VerifyLSDEVOutput 执行 ls /dev 并检查输出
func VerifyLSDEVOutput() {
	GinkgoWriter.Println("Checking devices in container")
	var output string
	var err error

	// 检查输出是否符合预期
	Eventually(func() string {
		output, err = ExecCommandInContainer(npu_operator_config.ContainerID, npu_operator_config.K8SNamespace, "ls", "/dev")
		if err != nil {
			GinkgoWriter.Printf("Failed to execute command in container: %v\n", err)
			return ""
		}
		return output
	}).WithTimeout(30*time.Second).WithPolling(1*time.Second).ShouldNot(BeEmpty(),
		"Should get output from container in 30 seconds")

	GinkgoWriter.Println("Verifying davinci0 device is present")

	// 验证 davinci0 设备存在
	Expect(output).To(ContainSubstring("davinci0"),
		fmt.Sprintf("Expected 'davinci0' in device list, got:%s\n", output))

	GinkgoWriter.Println("davinci0 device is present")
}

// VerifyAscendDriverAndFirmware 验证Ascend Driver & Firmware组件
func VerifyAscendDriverAndFirmware(nodes *corev1.NodeList, nodesNPUSituation map[string]string) {
	for _, node := range nodes.Items {
		nodeName := node.Name
		withNPU := nodesNPUSituation[nodeName]

		if withNPU == "none" {
			GinkgoWriter.Printf("%s 节点未检测到npu设备，跳过验证\n", nodeName)
			continue
		}

		sshExecutor, err := InstantiateSSHByNodeName(nodeName)
		if err != nil {
			if stderrs.Is(err, ErrNodeSSHConfigNotFound) {
				GinkgoWriter.Printf("%s 节点未配置SSH环境，跳过 Ascend Driver & Firmware 验证\n", nodeName)
				continue
			}
			Expect(err).NotTo(HaveOccurred(), "should successfully instantiate SSH executor")
		}

		GinkgoWriter.Printf("%s 节点检测到npu设备，开始验证\n", nodeName)
		result, err := sshExecutor.Exec("npu-smi info")
		Expect(err).NotTo(HaveOccurred(), "should successfully execute `executor.Exec` command")
		Expect(result).NotTo(BeNil(), "should get values from `executor.Exec` command")
		Expect(result.ExitCode).To(Equal(0), "should successfully execute `npu-smi info` command")
		Expect(result.Stdout).NotTo(BeNil(), "should get output from `npu-smi info` command")
		GinkgoWriter.Printf("%s 节点npu-smi info 的执行结果不为空\n", nodeName)
	}

	GinkgoWriter.Println("Ascend Driver & Firmware组件验证成功")
}

// VerifyAscendDockerRuntime 验证Ascend Docker Runtime组件
func VerifyAscendDockerRuntime() {
	// 创建镜像管理器
	imageManager := NewImageManager(npu_operator_config.K8SNamespace)
	// 验证拉取镜像成功
	VerifyPullContainerImage(imageManager)

	var err error

	// 清理可能存在的同名容器
	CleanupContainer(npu_operator_config.ContainerID, npu_operator_config.K8SNamespace)

	// 验证启动容器
	VerifyStartContainer()

	// 给容器一些额外时间完全启动
	GinkgoWriter.Println("Wait for extra 3 seconds ···")
	time.Sleep(3 * time.Second)

	// 执行 ls /dev 并检查输出
	VerifyLSDEVOutput()

	// 测试完成后清理容器和镜像
	CleanupContainer(npu_operator_config.ContainerID, npu_operator_config.K8SNamespace)
	err = imageManager.RemoveImage(npu_operator_config.ContainerImage)
	Expect(err).NotTo(HaveOccurred(), "should successfully remove image")

	GinkgoWriter.Printf("ls /dev 的执行结果中包含davinci0，Ascend Docker Runtime组件验证成功\n")
}

// VerifyAscendDevicePlugin 验证Ascend Device Plugin组件
func VerifyPullContainerImageOnNode(nodeName string) {
	Eventually(func() bool {
		exists, err := RemoteImageExists(nodeName, npu_operator_config.K8SNamespace, npu_operator_config.ContainerImage)
		if err != nil {
			GinkgoWriter.Printf("failed to list images on node %s: %v\n", nodeName, err)
			return false
		}
		if exists {
			return true
		}
		if err := PullImageOnNode(nodeName, npu_operator_config.K8SNamespace, npu_operator_config.ContainerImage); err != nil {
			GinkgoWriter.Printf("failed to pull image on node %s: %v\n", nodeName, err)
			return false
		}
		exists, err = RemoteImageExists(nodeName, npu_operator_config.K8SNamespace, npu_operator_config.ContainerImage)
		return err == nil && exists
	}).WithTimeout(3*time.Minute).WithPolling(5*time.Second).Should(BeTrue(),
		"Container image should be pulled on target node")
}

func VerifyStartContainerOnNode(nodeName string, runtimeBinary string) {
	GinkgoWriter.Printf("Starting container on node %s\n", nodeName)
	err := StartContainerOnNode(
		nodeName,
		npu_operator_config.K8SNamespace,
		runtimeBinary,
		npu_operator_config.ContainerImage,
		npu_operator_config.ContainerID,
	)
	Expect(err).NotTo(HaveOccurred(), "should successfully start container")

	Eventually(func() bool {
		running, err := IsContainerRunningOnNode(
			nodeName,
			npu_operator_config.ContainerID,
			npu_operator_config.K8SNamespace,
		)
		if err != nil {
			GinkgoWriter.Printf("Error checking if container is running: %v\n", err)
			return false
		}
		return running
	}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(BeTrue(),
		"Container should be running in 60 seconds")
}

func VerifyLSDEVOutputOnNode(nodeName string) {
	var output string
	var err error

	Eventually(func() string {
		output, err = ExecCommandInContainerOnNode(
			nodeName,
			npu_operator_config.ContainerID,
			npu_operator_config.K8SNamespace,
			"ls", "/dev",
		)
		if err != nil {
			GinkgoWriter.Printf("Failed to execute command in container on node %s: %v\n", nodeName, err)
			return ""
		}
		return output
	}).WithTimeout(30*time.Second).WithPolling(1*time.Second).ShouldNot(BeEmpty(),
		"Should get output from container in 30 seconds")

	Expect(output).To(ContainSubstring("davinci0"),
		fmt.Sprintf("Expected 'davinci0' in device list, got:%s\n", output))
}

func VerifyAscendDockerRuntimeOnNode(nodesNPUSituation map[string]string) {
	nodeName, err := FindFirstConfiguredNPUNodeName(nodesNPUSituation)
	if err != nil {
		if stderrs.Is(err, ErrNodeSSHConfigNotFound) {
			Skip("当前没有已配置 SSH 的含 NPU 节点，跳过 Ascend Docker Runtime 远端验证")
		}
		Expect(err).NotTo(HaveOccurred(), "should find a target node for Ascend Docker Runtime verification")
	}

	runtimeBinary, err := FindRemoteAscendDockerRuntimeBinary(nodeName)
	Expect(err).NotTo(HaveOccurred(), "should locate ascend-docker-runtime on target node")

	VerifyPullContainerImageOnNode(nodeName)
	CleanupContainerOnNode(nodeName, npu_operator_config.ContainerID, npu_operator_config.K8SNamespace)
	VerifyStartContainerOnNode(nodeName, runtimeBinary)
	GinkgoWriter.Println("Wait for extra 3 seconds ...")
	time.Sleep(3 * time.Second)
	VerifyLSDEVOutputOnNode(nodeName)
	CleanupContainerOnNode(nodeName, npu_operator_config.ContainerID, npu_operator_config.K8SNamespace)
	err = RemoveImageOnNode(nodeName, npu_operator_config.K8SNamespace, npu_operator_config.ContainerImage)
	Expect(err).NotTo(HaveOccurred(), "should successfully remove image")
}

func VerifyAscendDevicePlugin(clientset *kubernetes.Clientset, ctx context.Context, nodesNPUSituation map[string]string) {
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})

	Expect(err).NotTo(HaveOccurred(), "'kubectl get nodes' 执行失败")
	Expect(nodes.Items).NotTo(BeEmpty(), "在集群中没有找到任何节点")

	for _, node := range nodes.Items {
		nodeName := node.Name
		withNPU := nodesNPUSituation[nodeName]

		// 只检测有NPU的节点
		if withNPU != "none" {
			var npuCount int
			var foundNPU bool
			npuCount, foundNPU, err = GetNodeNpu(nodeName)

			// 验证是否找到NPU
			Expect(err).NotTo(HaveOccurred(), "应该成功提取Capacity字段中的NPU信息")
			Expect(foundNPU).To(BeTrue(), "应该成功找到 'NPU' 部分字段")

			GinkgoWriter.Printf("%s 节点找到NPU，个数为：%d\n", nodeName, npuCount)
		} else {
			GinkgoWriter.Printf("%s 节点未找到NPU设备，跳过验证", nodeName)
		}
	}

	GinkgoWriter.Printf("Ascend Device Plugin组件验证成功\n")
}

// VerifyPodLogs 通过 kubectl logs pod 验证组件
func VerifyPodLogs(clientset *kubernetes.Clientset, ctx context.Context, podName string, keyWords []string) {
	// 过滤 npu-operator 相关的 Pod
	filteredPods, err := FilterNPUOperatorPods(clientset, ctx)
	Expect(err).NotTo(HaveOccurred(), "应该成功过滤出NPU Operator相关的Pod")
	// 至少找到一个npu-operator Pod
	Expect(len(filteredPods)).Should(BeNumerically(">", 0))
	GinkgoWriter.Printf("找到 %d 个包含'npu-operator'的Pod\n", len(filteredPods))

	// 获取对应的Pod
	targetPodList := FindTargetPodList(podName, filteredPods)
	Expect(len(targetPodList)).NotTo(Equal(0), "应该在filtered pods集合中找到"+podName+" pods")

	for _, targetPod := range targetPodList {
		Eventually(func() bool {
			// 获取日志
			logContent, err := GetPodLogs(clientset, targetPod)
			if err != nil {
				GinkgoWriter.Printf("获取 %s pod的日志失败：%v\n", targetPod.Name, err)
				return false
			}
			if logContent == "" {
				GinkgoWriter.Printf("%s pod的日志内容不应为空\n", targetPod.Name)
				return false
			}

			// 检查日志中是否包含关键字
			for _, keyWord := range keyWords {
				if !strings.Contains(logContent, keyWord) {
					GinkgoWriter.Printf("%s pod的日志不包含关键字：%s\n", podName, keyWord)
					return false
				}
				GinkgoWriter.Printf("%s pod的日志包含关键字：%s\n", podName, keyWord)
			}
			return true
		}).WithTimeout(10*time.Second).WithPolling(1*time.Second).Should(BeTrue(),
			fmt.Sprintf("在重试10次后仍未能成功比对 %s 日志内容", targetPod.Name))
	}

	GinkgoWriter.Printf("%s Pod logs 验证结束\n", podName)
}

// VerifyAscendOperator 验证Ascend Operator组件
func VerifyAscendOperator(clientset *kubernetes.Clientset, ctx context.Context) {
	podName := "ascend-operator-manager"
	keyWords := []string{"init success", "starting manager"}
	VerifyPodLogs(clientset, ctx, podName, keyWords)

	// 打印确认信息
	GinkgoWriter.Printf(podName + "日志包含所有需要的关键字，Ascend Operator组件验证成功\n")
}

// VerifyNodeD 验证NodeD组件
func VerifyNodeD(clientset *kubernetes.Clientset, ctx context.Context) {
	podName := "noded"
	keyWords := []string{"init success", "init config from local json file success"}
	VerifyPodLogs(clientset, ctx, podName, keyWords)

	// 打印确认信息
	GinkgoWriter.Printf(podName + "日志包含所有需要的关键字，NodeD组件验证成功\n")
}

// VerifyClusterD 验证ClusterD组件
func VerifyClusterD(clientset *kubernetes.Clientset, ctx context.Context) {
	podName := "clusterd"
	keyWords := []string{"init success", "start listen"}
	VerifyPodLogs(clientset, ctx, podName, keyWords)

	// 打印确认信息
	GinkgoWriter.Printf(podName + "日志包含所有需要的关键字，ClusterD组件验证成功\n")
}

// VerifyMindIO 验证MindIO组件
func VerifyMindIO(clientset *kubernetes.Clientset, ctx context.Context) {
	podName := "mindio"
	keyWords := []string{"Successfully installed"}
	VerifyPodLogs(clientset, ctx, podName, keyWords)

	// 打印确认信息
	GinkgoWriter.Printf(podName + "日志包含所有需要的关键字，MindIO组件验证成功\n")
}

// VerifyNPUExporter 验证NPU Exporter组件
func VerifyNPUExporter(clientset *kubernetes.Clientset, ctx context.Context) {
	podName := "npu-exporter"
	keyWords := []string{"update Cache,key is npu-exporter"}
	VerifyPodLogs(clientset, ctx, podName, keyWords)

	// 打印确认信息
	GinkgoWriter.Printf(podName + "日志包含所有需要的关键字，NPU Exporter组件验证成功\n")
}

// VerifyVolcanoControllers 验证VolcanoControllers
func VerifyVolcanoControllers(clientset *kubernetes.Clientset, ctx context.Context, nodes *corev1.NodeList) {
	// 验证有volcano-controllers
	podName := "volcano-controllers"
	volcanoControllersPods, err := GetSpecifiedPods(clientset, ctx, podName)
	Expect(err).NotTo(HaveOccurred(), "应该成功找到 volcano-controllers 相关的Pod")
	Expect(len(volcanoControllersPods)).To(Equal(1), "volcano-controllers 相关的Pod应该只有1个")
	volcanoControllersPod := volcanoControllersPods[0]

	sshExecutor, err := InstantiateSSHByNodeName(volcanoControllersPod.Spec.NodeName)
	if err != nil {
		if stderrs.Is(err, ErrNodeSSHConfigNotFound) {
			GinkgoWriter.Printf("%s 节点未配置SSH环境，跳过 volcano-controllers 日志验证\n", volcanoControllersPod.Spec.NodeName)
			return
		}
		Expect(err).NotTo(HaveOccurred(), "should successfully instantiate SSH executor")
	}

	// 查看volcano-controllers日志
	catLogFilePath := "cat /var/log/mindx-dl/volcano-controller/volcano-controller.log"
	result, err := sshExecutor.Exec(catLogFilePath)
	Expect(err).NotTo(HaveOccurred(), "`executor.Exec` 命令执行失败")
	Expect(result).NotTo(BeNil(), "`executor.Exec` 命令应获得非空结果")
	Expect(result.ExitCode).To(Equal(0), "应成功读取 volcano-controllers 日志文件")

	GinkgoWriter.Printf("开始验证volcano-controllers日志中包含'running'和'start'字段\n")

	// 检查特定字段
	Expect(result.Stdout).To(ContainSubstring("running"), "volcano-controllers日志中应包含'running'")
	Expect(result.Stdout).To(ContainSubstring("start"), "volcano-controllers日志中应包含'start'")

	GinkgoWriter.Printf("volcano-controllers日志中包含'running'和'start'字段验证成功\n")
}

// VerifyVolcanoScheduler 验证VolcanoScheduler
func VerifyVolcanoScheduler(clientset *kubernetes.Clientset, ctx context.Context, nodes *corev1.NodeList) {
	// 验证有volcano-scheduler
	podName := "volcano-scheduler"
	volcanoSchedulerPods, err := GetSpecifiedPods(clientset, ctx, podName)
	Expect(err).NotTo(HaveOccurred(), "应该成功找到 volcano-scheduler 相关的Pod")
	Expect(len(volcanoSchedulerPods)).To(Equal(1), "volcano-scheduler 相关的Pod应该只有1个")
	volcanoSchedulerPod := volcanoSchedulerPods[0]

	sshExecutor, err := InstantiateSSHByNodeName(volcanoSchedulerPod.Spec.NodeName)
	if err != nil {
		if stderrs.Is(err, ErrNodeSSHConfigNotFound) {
			GinkgoWriter.Printf("%s 节点未配置SSH环境，跳过 volcano-scheduler 日志验证\n", volcanoSchedulerPod.Spec.NodeName)
			return
		}
		Expect(err).NotTo(HaveOccurred(), "should successfully instantiate SSH executor")
	}

	// 查看volcano-scheduler日志
	catLogFilePath := "cat /var/log/mindx-dl/volcano-scheduler/volcano-scheduler.log"
	result, err := sshExecutor.Exec(catLogFilePath)
	Expect(err).NotTo(HaveOccurred(), "`executor.Exec` 命令执行失败")
	Expect(result).NotTo(BeNil(), "`executor.Exec` 命令应获得非空结果")
	Expect(result.ExitCode).To(Equal(0), "应成功读取 volcano-scheduler 日志文件")

	GinkgoWriter.Printf("开始验证volcano-scheduler日志中包含'Running on machine'和'Binary: Built with gc'字段\n")

	// 检查特定字段
	Expect(result.Stdout).To(ContainSubstring("Running on machine"), "volcano-scheduler 日志中应包含'Running on machine'")
	Expect(result.Stdout).To(ContainSubstring("Binary: Built with gc"), "volcano-scheduler 日志中应包含'Binary: Built with gc'")

	GinkgoWriter.Printf("volcano-scheduler日志中包含'Running on machine'和'Binary: Built with gc'字段验证成功\n")
}

// VerifyVolcano 验证Volcano组件
func VerifyVolcano(clientset *kubernetes.Clientset, ctx context.Context, nodes *corev1.NodeList) {
	// 验证VolcanoControllers
	VerifyVolcanoControllers(clientset, ctx, nodes)

	// 验证VolcanoScheduler
	VerifyVolcanoScheduler(clientset, ctx, nodes)

	GinkgoWriter.Printf("Volcano组件验证成功\n")
}

// VerifyNPUOperatorComponents 验证 npu-operator 各组件安装后状态
func VerifyNPUOperatorComponents(clientset *kubernetes.Clientset, ctx context.Context, nodesNPUSituation map[string]string, nodes *corev1.NodeList) {
	VerifyAscendDriverAndFirmware(nodes, nodesNPUSituation)
	VerifyAscendDockerRuntimeOnNode(nodesNPUSituation)
	VerifyAscendDevicePlugin(clientset, ctx, nodesNPUSituation)
	VerifyAscendOperator(clientset, ctx)
	VerifyNodeD(clientset, ctx)
	VerifyClusterD(clientset, ctx)
	VerifyVolcano(clientset, ctx, nodes)
	VerifyMindIO(clientset, ctx)
	VerifyNPUExporter(clientset, ctx)
}

// VerifyEnvironment 验证有/无驱动部署环境的必要信息
type VerifyEnvironment struct {
	NpuType    string
	EnvName    string
	VersionID  string
	SystemType string
}

// VerifyInstallationInfo 验证有/无驱动部署的必要信息
type VerifyInstallationInfo struct {
	Clientset          *kubernetes.Clientset
	Ctx                context.Context
	NodesNPUSituation  map[string]string
	VerifyEnvironments []*VerifyEnvironment
}

// VerifyComponentsAfterInstallationSingleNode 验证单节点有/无驱动部署
func VerifyComponentsAfterInstallationSingleNode(info *VerifyInstallationInfo) bool {
	var nodes *corev1.NodeList
	var err error
	nodes, err = info.Clientset.CoreV1().Nodes().List(info.Ctx, metav1.ListOptions{})

	Expect(err).NotTo(HaveOccurred(), "'kubectl get nodes' 执行失败")
	Expect(nodes.Items).NotTo(BeEmpty(), "在集群中没有找到任何节点")

	var needUninstallation bool
	if len(nodes.Items) != 1 {
		GinkgoWriter.Printf("检测环境应为单节点，实际节点数量为：%d，跳过验证", len(nodes.Items))
	} else {
		Expect(len(info.VerifyEnvironments)).To(Equal(1), "should have only one verify environment")
		GinkgoWriter.Printf("检测到nodes数量:\n%d\n", len(nodes.Items))
		node := nodes.Items[0]
		verifyEnvironment := info.VerifyEnvironments[0]

		if info.NodesNPUSituation[node.Name] == "none" {
			GinkgoWriter.Printf("当前节点 %s 没有npu设备，跳过验证\n", node.Name)
		} else {
			servertype := info.NodesNPUSituation[node.Name]
			servertype, err = ConvertServerType(servertype)
			Expect(err).NotTo(HaveOccurred(), "should successfully get server type")

			systemtype, exists := node.Labels["kubernetes.io/arch"]
			Expect(exists).To(BeTrue(), "should find system type")

			var osInfo *OSInfo
			osInfo, err = GetOSInfoByNodeName(node.Name)
			if err != nil {
				if stderrs.Is(err, ErrNodeSSHConfigNotFound) {
					needUninstallation = false
					GinkgoWriter.Printf("%s 节点未配置SSH环境，跳过当前单节点安装验证\n", node.Name)
					return needUninstallation
				}
				Expect(err).NotTo(HaveOccurred(), "should successfully get OS info")
			}

			if strings.Contains(servertype, verifyEnvironment.NpuType) &&
				strings.ToLower(osInfo.Name) == verifyEnvironment.EnvName &&
				osInfo.VersionID == verifyEnvironment.VersionID &&
				strings.Contains(systemtype, verifyEnvironment.SystemType) {
				GinkgoWriter.Printf("%s节点存在servertype，值为%s，系统环境为%s %s %s，符合环境要求，继续验证\n", node.Name, servertype, osInfo.Name, osInfo.VersionID, systemtype)

				// 验证 npu-operator 安装流程
				needUninstallation = EnsureNPUOperatorInstallation(info.Clientset, info.Ctx, info.NodesNPUSituation)

				// 验证安装后各组件状态
				VerifyNPUOperatorComponents(info.Clientset, info.Ctx, info.NodesNPUSituation, nodes)
			} else {
				needUninstallation = false
				GinkgoWriter.Printf("%s节点存在servertype，值为%s，系统环境为%s %s %s，不符合环境要求，跳过验证\n", node.Name, servertype, osInfo.Name, osInfo.VersionID, systemtype)
			}
			GinkgoWriter.Printf("NPU类型：%s，系统环境为%s %s %s，安装部署验证成功\n",
				verifyEnvironment.NpuType, verifyEnvironment.EnvName, verifyEnvironment.VersionID, verifyEnvironment.SystemType)
		}
	}
	return needUninstallation
}

// VerifyComponentsAfterInstallationMultipleNodes 验证多节点有/无驱动部署
func VerifyComponentsAfterInstallationMultipleNodes(info *VerifyInstallationInfo) bool {
	var nodes *corev1.NodeList
	var err error
	nodes, err = info.Clientset.CoreV1().Nodes().List(info.Ctx, metav1.ListOptions{})

	Expect(err).NotTo(HaveOccurred(), "'kubectl get nodes' 执行失败")
	Expect(nodes.Items).NotTo(BeEmpty(), "在集群中没有找到任何节点")

	var needUninstallation bool
	if len(nodes.Items) == 1 {
		GinkgoWriter.Printf("检测环境应为多节点，实际节点数量为：%d，跳过验证", len(nodes.Items))
	} else {
		Expect(len(info.VerifyEnvironments)).To(BeNumerically(">", 0), "should have at least one verify environments")
		GinkgoWriter.Printf("检测到nodes数量:\n%d\n", len(nodes.Items))

		// 判断是否符合环境要求
		var included bool
		included, err = JudgeInstallationEnvironment(nodes, info.VerifyEnvironments, info.NodesNPUSituation)
		Expect(err).NotTo(HaveOccurred(), "should successfully judge installation environment")

		if included {
			GinkgoWriter.Printf("当前集群环境符合环境要求，继续验证\n")

			// 验证 npu-operator 安装流程
			needUninstallation = EnsureNPUOperatorInstallation(info.Clientset, info.Ctx, info.NodesNPUSituation)

			// 验证安装后各组件状态
			VerifyNPUOperatorComponents(info.Clientset, info.Ctx, info.NodesNPUSituation, nodes)
		} else {
			needUninstallation = false
			GinkgoWriter.Printf("当前集群环境不符合环境要求，跳过验证\n")
		}
		GinkgoWriter.Printf("多节点安装部署验证成功\n")
	}
	return needUninstallation
}

// VerifyNPUClusterPolicyStatus 验证 NPUClusterPolicy 状态
func VerifyNPUClusterPolicyStatus(dynamicClient dynamic.Interface, ctx context.Context, allRunning bool) {
	// 获取NPUClusterPolicy
	npuPolicy, err, onlyOneNPUPolicy := GetNPUPolicy(dynamicClient, ctx)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取NPUClusterPolicy")
	Expect(onlyOneNPUPolicy).To(BeTrue(), "应该只有一个NPUClusterPolicy")

	// 输出NPUClusterPolicy对应信息
	policyName := npuPolicy.GetName()
	policyNamespace := npuPolicy.GetNamespace()
	GinkgoWriter.Printf("NPUClusterPolicy %s/%s\n", policyNamespace, policyName)

	// 查询NPUClusterPolicy对应状态
	state, err, found := GetNPUClusterPolicyStatus(npuPolicy)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取NPUClusterPolicy对应状态")
	Expect(found).To(BeTrue(), "应该能够找到NPUClusterPolicy的状态字段")

	GinkgoWriter.Printf("该NPUClusterPolicy的状态: %s\n", state)

	if allRunning {
		GinkgoWriter.Printf("All NPU Operator pods are running\n")

		// 检查状态是否为 ready
		GinkgoWriter.Printf("  - 命名空间: %s, NPUClusterPolicy: %s, 状态: %s\n",
			policyNamespace, policyName, state)
		Expect(strings.ToLower(state)).To(Equal("ready"), "NPUClusterPolicy的状态应该为Ready")
		GinkgoWriter.Println("当前NPUClusterPolicy状态验证成功")
	} else {
		GinkgoWriter.Printf("Some NPU Operator pods are not running\n")

		// 检查状态是否为 notReady
		GinkgoWriter.Printf("  - 命名空间: %s, NPUClusterPolicy: %s, 状态: %s\n",
			policyNamespace, policyName, state)
		Expect(strings.ToLower(state)).To(Equal("notready"), "NPUClusterPolicy的状态应该为notReady")
		GinkgoWriter.Println("当前NPUClusterPolicy状态验证成功")
	}
}

// VerifyNodeLabels 验证节点标签是否符合预期
func VerifyNodeLabels(node corev1.Node, label string, expectValue string) {
	value, exists := node.Labels[label]

	Expect(exists).To(BeTrue(), label+"标签应当存在")
	Expect(value).To(Equal(expectValue), label+"标签的值应为"+expectValue)
	GinkgoWriter.Printf("%s节点存在%s标签，值为%s\n", node.Name, label, value)
}

var npuFeatureDiscoveryLabels = []string{
	"feature.node.kubernetes.io/pci-1200_19e5_d100.present",
	"feature.node.kubernetes.io/pci-1200_19e5_d500.present",
	"feature.node.kubernetes.io/pci-1200_19e5_d801.present",
	"feature.node.kubernetes.io/pci-1200_19e5_d802.present",
	"feature.node.kubernetes.io/pci-1200_19e5_d803.present",
}

var npuFeatureDiscoveryLabelByType = map[string]string{
	"d100": "feature.node.kubernetes.io/pci-1200_19e5_d100.present",
	"d500": "feature.node.kubernetes.io/pci-1200_19e5_d500.present",
	"d801": "feature.node.kubernetes.io/pci-1200_19e5_d801.present",
	"d802": "feature.node.kubernetes.io/pci-1200_19e5_d802.present",
	"d803": "feature.node.kubernetes.io/pci-1200_19e5_d803.present",
}

// VerifyNodeLabelAbsent 验证节点标签不存在
func VerifyNodeLabelAbsent(node corev1.Node, label string) {
	_, exists := node.Labels[label]

	Expect(exists).To(BeFalse(), label+"标签不应当存在")
	GinkgoWriter.Printf("%s节点不存在%s标签\n", node.Name, label)
}

// IsManagerNode 判断节点是否为管理节点
func IsManagerNode(node corev1.Node) bool {
	_, controlPlaneExists := node.Labels["node-role.kubernetes.io/control-plane"]
	_, masterExists := node.Labels["node-role.kubernetes.io/master"]

	return controlPlaneExists || masterExists
}

// FindNodesByRoleAndNPU 按节点角色和是否带 NPU 过滤节点
func FindNodesByRoleAndNPU(nodes *corev1.NodeList, nodesNPUSituation map[string]string, manager bool, hasNPU bool) []corev1.Node {
	var matchedNodes []corev1.Node

	for _, node := range nodes.Items {
		nodeNPUType, exists := nodesNPUSituation[node.Name]
		nodeHasNPU := exists && nodeNPUType != "none"

		if IsManagerNode(node) != manager || nodeHasNPU != hasNPU {
			continue
		}

		matchedNodes = append(matchedNodes, node)
	}

	return matchedNodes
}

// FindNodesByNPUType 按卡型过滤节点
func FindNodesByNPUType(nodes *corev1.NodeList, nodesNPUSituation map[string]string, npuType string) []corev1.Node {
	var matchedNodes []corev1.Node

	for _, node := range nodes.Items {
		if nodesNPUSituation[node.Name] == npuType {
			matchedNodes = append(matchedNodes, node)
		}
	}

	return matchedNodes
}

// VerifyNodeExpectedNPUFeatureLabels 按检测到的卡型验证节点的 NPU Feature Discovery 标签
func VerifyNodeExpectedNPUFeatureLabels(node corev1.Node, npuType string) {
	for _, label := range npuFeatureDiscoveryLabels {
		expectLabel, knownType := npuFeatureDiscoveryLabelByType[npuType]
		if npuType != "none" && knownType && label == expectLabel {
			VerifyNodeLabels(node, label, "true")
			continue
		}
		VerifyNodeLabelAbsent(node, label)
	}
}

// VerifyNodeHasNoNPUFeatureLabels 验证节点不存在任何 NPU Feature Discovery 标签
func VerifyNodeHasNoNPUFeatureLabels(node corev1.Node) {
	VerifyNodeExpectedNPUFeatureLabels(node, "none")
}

// VerifyNodeRoleLabels 验证节点角色相关标签
func VerifyNodeRoleLabels(node corev1.Node, expectMasterSelector bool, expectWorkerSelector bool, expectWorkerRole bool) {
	if expectMasterSelector {
		VerifyNodeLabels(node, "masterselector", "dls-master-node")
	} else {
		VerifyNodeLabelAbsent(node, "masterselector")
	}

	if expectWorkerSelector {
		VerifyNodeLabels(node, "workerselector", "dls-worker-node")
	} else {
		VerifyNodeLabelAbsent(node, "workerselector")
	}

	if expectWorkerRole {
		VerifyNodeLabels(node, "node-role.kubernetes.io/worker", "worker")
	} else {
		VerifyNodeLabelAbsent(node, "node-role.kubernetes.io/worker")
	}
}

// VerifyHostArchLabel 验证 host-arch 标签与 kubernetes.io/arch 的映射关系
func VerifyHostArchLabel(node corev1.Node) {
	arch, exists := node.Labels["kubernetes.io/arch"]
	expectValue := "unknown"

	if exists {
		switch arch {
		case "amd64":
			expectValue = "huawei-x86"
		case "arm64":
			expectValue = "huawei-arm"
		default:
			expectValue = arch
		}
	}

	VerifyNodeLabels(node, "host-arch", expectValue)
}

var npuFeatureDiscoveryManagedLabelKeys = []string{
	"openfuyao.com/container.runtime",
	"masterselector",
	"workerselector",
	"node-role.kubernetes.io/worker",
	"openfuyao.com/npu.present",
	"host-arch",
	"accelerator",
	"accelerator-type",
	"server-usage",
	"card-side-roce-bandwidth",
}

var npuFeatureDiscoveryRecoverableLabelKeys = []string{
	"openfuyao.com/container.runtime",
	"openfuyao.com/npu.present",
	"host-arch",
	"accelerator",
	"accelerator-type",
	"server-usage",
	"card-side-roce-bandwidth",
}

var npuFeatureDiscoveryDynamicLabelPrefixes = []string{
	"node-side-rdma-speed-",
	"node-side-rdma-pcie-",
}

var nodeSideRDMAPCIeWidthPattern = regexp.MustCompile(`^x\d+$`)

func LoadNodeCardModelsOrSkip(nodes *corev1.NodeList) map[string]string {
	nodeCardModels, err := ResolveNodeCardModels(nodes)
	if err != nil {
		if stderrs.Is(err, ErrNodeSSHConfigNotFound) {
			Skip(err.Error())
		}
		Expect(err).NotTo(HaveOccurred(), "should successfully resolve node card models")
	}

	return nodeCardModels
}

func LoadNodeBoardIDsOrSkip(nodes *corev1.NodeList) map[string]string {
	nodeBoardIDs, err := ResolveNodeBoardIDs(nodes)
	if err != nil {
		if stderrs.Is(err, ErrNodeSSHConfigNotFound) {
			Skip(err.Error())
		}
		Expect(err).NotTo(HaveOccurred(), "should successfully resolve node board ids")
	}

	return nodeBoardIDs
}

func LoadNodeRDMAInfoOrSkip(nodes *corev1.NodeList) map[string]NodeRDMAInfo {
	nodeRDMAInfo, err := ResolveNodeRDMAInfo(nodes)
	if err != nil {
		if stderrs.Is(err, ErrNodeSSHConfigNotFound) {
			Skip(err.Error())
		}
		Expect(err).NotTo(HaveOccurred(), "should successfully resolve node rdma info")
	}

	return nodeRDMAInfo
}

func FindNodesByCardModels(nodes *corev1.NodeList, nodeCardModels map[string]string, cardModels ...string) []corev1.Node {
	var matchedNodes []corev1.Node
	cardModelSet := make(map[string]struct{}, len(cardModels))
	for _, cardModel := range cardModels {
		cardModelSet[cardModel] = struct{}{}
	}

	for _, node := range nodes.Items {
		if _, exists := cardModelSet[nodeCardModels[node.Name]]; exists {
			matchedNodes = append(matchedNodes, node)
		}
	}

	return matchedNodes
}

func FindNodesByBoardIDs(nodes *corev1.NodeList, nodeBoardIDs map[string]string, boardIDs ...string) []corev1.Node {
	var matchedNodes []corev1.Node
	boardIDSet := make(map[string]struct{}, len(boardIDs))
	for _, boardID := range boardIDs {
		boardIDSet[strings.ToLower(boardID)] = struct{}{}
	}

	for _, node := range nodes.Items {
		if _, exists := boardIDSet[strings.ToLower(nodeBoardIDs[node.Name])]; exists {
			matchedNodes = append(matchedNodes, node)
		}
	}

	return matchedNodes
}

func FindNodesByRDMAInfo(nodes *corev1.NodeList, nodeRDMAInfo map[string]NodeRDMAInfo) []corev1.Node {
	var matchedNodes []corev1.Node

	for _, node := range nodes.Items {
		info := nodeRDMAInfo[node.Name]
		if len(info.DeviceSpeeds) == 0 && len(info.DevicePCIe) == 0 {
			continue
		}
		matchedNodes = append(matchedNodes, node)
	}

	return matchedNodes
}

func FindNodesByRDMASpeedInfo(nodes *corev1.NodeList, nodeRDMAInfo map[string]NodeRDMAInfo) []corev1.Node {
	var matchedNodes []corev1.Node

	for _, node := range nodes.Items {
		if len(nodeRDMAInfo[node.Name].DeviceSpeeds) == 0 {
			continue
		}
		matchedNodes = append(matchedNodes, node)
	}

	return matchedNodes
}

func FindNodesByRDMAPCIeInfo(nodes *corev1.NodeList, nodeRDMAInfo map[string]NodeRDMAInfo) []corev1.Node {
	var matchedNodes []corev1.Node

	for _, node := range nodes.Items {
		if len(nodeRDMAInfo[node.Name].DevicePCIe) == 0 {
			continue
		}
		matchedNodes = append(matchedNodes, node)
	}

	return matchedNodes
}

func FindNodesExcludingCardModels(nodes *corev1.NodeList, nodeCardModels map[string]string, cardModels ...string) []corev1.Node {
	var matchedNodes []corev1.Node
	cardModelSet := make(map[string]struct{}, len(cardModels))
	for _, cardModel := range cardModels {
		cardModelSet[cardModel] = struct{}{}
	}

	for _, node := range nodes.Items {
		cardModel, exists := nodeCardModels[node.Name]
		if !exists {
			continue
		}
		if _, exists := cardModelSet[cardModel]; exists {
			continue
		}
		matchedNodes = append(matchedNodes, node)
	}

	return matchedNodes
}

func FindNodesWithLabelPrefix(nodes *corev1.NodeList, prefix string) []corev1.Node {
	return FindNodesWithAnyLabelPrefix(nodes, prefix)
}

func FindNodesWithAnyLabelPrefix(nodes *corev1.NodeList, prefixes ...string) []corev1.Node {
	var matchedNodes []corev1.Node

	for _, node := range nodes.Items {
		for key := range node.Labels {
			for _, prefix := range prefixes {
				if strings.HasPrefix(key, prefix) {
					matchedNodes = append(matchedNodes, node)
					goto nextNode
				}
			}
		}
	nextNode:
	}

	return matchedNodes
}

func SnapshotManagedNPUFeatureDiscoveryLabels(node corev1.Node) map[string]string {
	return snapshotNodeLabels(node, npuFeatureDiscoveryManagedLabelKeys, npuFeatureDiscoveryDynamicLabelPrefixes)
}

func SnapshotRecoverableNPUFeatureDiscoveryLabels(node corev1.Node) map[string]string {
	return snapshotNodeLabels(node, npuFeatureDiscoveryRecoverableLabelKeys, npuFeatureDiscoveryDynamicLabelPrefixes)
}

func snapshotNodeLabels(node corev1.Node, staticKeys []string, dynamicPrefixes []string) map[string]string {
	snapshot := make(map[string]string)

	for _, key := range staticKeys {
		if value, exists := node.Labels[key]; exists {
			snapshot[key] = value
		}
	}

	for key, value := range node.Labels {
		for _, prefix := range dynamicPrefixes {
			if strings.HasPrefix(key, prefix) {
				snapshot[key] = value
				break
			}
		}
	}

	return snapshot
}

func VerifyManagedNPUFeatureDiscoveryLabels(node corev1.Node, expectedLabels map[string]string) {
	currentLabels := SnapshotManagedNPUFeatureDiscoveryLabels(node)
	Expect(currentLabels).To(Equal(expectedLabels), node.Name+" managed npu-feature-discovery labels should remain unchanged")
}

func VerifyRecoverableNPUFeatureDiscoveryLabels(node corev1.Node, expectedLabels map[string]string) {
	currentLabels := SnapshotRecoverableNPUFeatureDiscoveryLabels(node)
	Expect(currentLabels).To(Equal(expectedLabels), node.Name+" recoverable npu-feature-discovery labels should be restored")
}

func VerifyNodeRDMASpeedLabels(node corev1.Node) {
	rdmaSpeedLabels := make(map[string]string)
	for key, value := range node.Labels {
		if strings.HasPrefix(key, "node-side-rdma-speed-") {
			rdmaSpeedLabels[key] = value
		}
	}

	Expect(rdmaSpeedLabels).NotTo(BeEmpty(), node.Name+" should have node-side-rdma-speed-* labels")
	for key, value := range rdmaSpeedLabels {
		Expect(strings.TrimPrefix(key, "node-side-rdma-speed-")).NotTo(BeEmpty(), key+" should include device name suffix")
		Expect(value).NotTo(BeEmpty(), key+" should not be empty")
		Expect(value).NotTo(ContainSubstring("/"), key+" should use p instead of /")
	}
}

func VerifyNodeExpectedRDMASpeedLabels(node corev1.Node, info NodeRDMAInfo) {
	Expect(info.DeviceSpeeds).NotTo(BeEmpty(), node.Name+" should have rdma devices discovered independently")

	for deviceName, speed := range info.DeviceSpeeds {
		VerifyNodeLabels(node, "node-side-rdma-speed-"+deviceName, speed)
	}
}

func VerifyNodeRDMAPCIeLabels(node corev1.Node) {
	deviceNames := make(map[string]struct{})
	prefixes := []string{
		"node-side-rdma-pcie-lnk-cap-speed-",
		"node-side-rdma-pcie-lnk-cap-width-",
		"node-side-rdma-pcie-lnk-sta-speed-",
		"node-side-rdma-pcie-lnk-sta-width-",
	}

	for key := range node.Labels {
		for _, prefix := range prefixes {
			if strings.HasPrefix(key, prefix) {
				deviceNames[strings.TrimPrefix(key, prefix)] = struct{}{}
			}
		}
	}

	Expect(deviceNames).NotTo(BeEmpty(), node.Name+" should have node-side-rdma-pcie-* labels")
	for deviceName := range deviceNames {
		Expect(deviceName).NotTo(BeEmpty(), "rdma pcie label suffix should include device name")

		capSpeedKey := "node-side-rdma-pcie-lnk-cap-speed-" + deviceName
		capWidthKey := "node-side-rdma-pcie-lnk-cap-width-" + deviceName
		staSpeedKey := "node-side-rdma-pcie-lnk-sta-speed-" + deviceName
		staWidthKey := "node-side-rdma-pcie-lnk-sta-width-" + deviceName

		VerifyNodeLabelsNotEmpty(node, capSpeedKey)
		VerifyNodeLabelsNotEmpty(node, capWidthKey)
		VerifyNodeLabelsNotEmpty(node, staSpeedKey)
		VerifyNodeLabelsNotEmpty(node, staWidthKey)

		Expect(node.Labels[capSpeedKey]).NotTo(ContainSubstring("/"), capSpeedKey+" should use p instead of /")
		Expect(node.Labels[staSpeedKey]).NotTo(ContainSubstring("/"), staSpeedKey+" should use p instead of /")
		Expect(node.Labels[capWidthKey]).To(MatchRegexp(nodeSideRDMAPCIeWidthPattern.String()), capWidthKey+" should be in xN format")
		Expect(node.Labels[staWidthKey]).To(MatchRegexp(nodeSideRDMAPCIeWidthPattern.String()), staWidthKey+" should be in xN format")
	}
}

func VerifyNodeExpectedRDMAPCIeLabels(node corev1.Node, info NodeRDMAInfo) {
	Expect(info.DevicePCIe).NotTo(BeEmpty(), node.Name+" should have rdma pcie info discovered independently")

	for deviceName, pcieInfo := range info.DevicePCIe {
		VerifyNodeLabels(node, "node-side-rdma-pcie-lnk-cap-speed-"+deviceName, pcieInfo.LnkCapSpeed)
		VerifyNodeLabels(node, "node-side-rdma-pcie-lnk-cap-width-"+deviceName, pcieInfo.LnkCapWidth)
		VerifyNodeLabels(node, "node-side-rdma-pcie-lnk-sta-speed-"+deviceName, pcieInfo.LnkStaSpeed)
		VerifyNodeLabels(node, "node-side-rdma-pcie-lnk-sta-width-"+deviceName, pcieInfo.LnkStaWidth)
	}
}

func VerifyNodeLabelsNotEmpty(node corev1.Node, label string) {
	value, exists := node.Labels[label]

	Expect(exists).To(BeTrue(), label+" label should exist")
	Expect(value).NotTo(BeEmpty(), label+" label should not be empty")
	GinkgoWriter.Printf("%s node has %s label with value %s\n", node.Name, label, value)
}

// VerifyAllNodesLabels 验证每个节点标签是否符合预期
func VerifyAllNodesLabels(nodes *corev1.NodeList, nodesNPUSituation map[string]string, label string, expectValue string, npuType string) {
	targetNodes := FindNodesByNPUType(nodes, nodesNPUSituation, npuType)
	if len(targetNodes) == 0 {
		Skip("当前环境不存在 " + npuType + " 卡型节点")
	}

	GinkgoWriter.Printf("开始检查 %d 个节点上的 %s 设备标签 %s...\n", len(targetNodes), npuType, label)

	// 检查每个节点的标签
	for _, node := range nodes.Items {
		nodeName := node.Name
		foundNPU := nodesNPUSituation[nodeName]

		if foundNPU == npuType {
			// 找到设备，验证节点标签
			GinkgoWriter.Printf("节点 %s: 找到 %s 设备，开始验证标签\n", nodeName, npuType)
			VerifyNodeLabels(node, label, expectValue)
		} else {
			GinkgoWriter.Printf("%s 节点不是 %s 设备，验证 %s 标签不存在\n", nodeName, npuType, label)
			VerifyNodeLabelAbsent(node, label)
		}
	}

	GinkgoWriter.Println("所有节点检查完成")
}

// VerifyTestLabel 验证每个节点的标签是否符合预期
func VerifyTestLabel(nodes *corev1.NodeList, nodesNPUSituation map[string]string, testNpuType string, testSystemType string, label string, expectValue string) {
	for _, node := range nodes.Items {
		foundNPU := nodesNPUSituation[node.Name]

		// 只检测有NPU的节点
		if foundNPU == "none" {
			GinkgoWriter.Printf("%s节点不存在NPU\n", node.Name)
			continue
		} else {
			GinkgoWriter.Printf("%s 节点未找到NPU设备，跳过验证", node.Name)
		}

		servertype, exists := node.Labels["servertype"]
		Expect(exists).To(BeTrue(), "should find server type")

		systemtype, exists := node.Labels["kubernetes.io/arch"]
		Expect(exists).To(BeTrue(), "should find system type")

		if strings.Contains(servertype, testNpuType) && strings.Contains(systemtype, testSystemType) {
			GinkgoWriter.Printf("%s节点存在servertype，值为%s，系统环境为%s，符合环境要求，继续验证\n", node.Name, servertype, systemtype)

			// 检查节点标签
			VerifyNodeLabels(node, label, expectValue)
		} else {
			GinkgoWriter.Printf("%s节点存在servertype，值为%s，系统环境为%s，不符合环境要求，跳过验证\n", node.Name, servertype, systemtype)
			continue
		}
	}
}

// VerifyWorkerLabel 验证 worker 标签
func VerifyWorkerLabel(nodes *corev1.NodeList, nodesNPUSituation map[string]string, testNpuType string, testSystemType string) {
	// 需要验证的标签及期望值
	var label = "node-role.kubernetes.io/worker"
	var expectValue = "worker"

	VerifyTestLabel(nodes, nodesNPUSituation, testNpuType, testSystemType, label, expectValue)
}

// VerifyPodRunning 验证Pod运行状态
func VerifyPodRunning(clientset *kubernetes.Clientset, ctx context.Context, podName string) {
	GinkgoWriter.Printf("等待 Pod：%s 运行...\n", podName)

	Eventually(func() (corev1.PodPhase, error) {
		currentPod, err := clientset.CoreV1().Pods("default").Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			GinkgoWriter.Printf("获取 Pod：%s 失败: %v\n", podName, err)
			return "", err
		}

		GinkgoWriter.Printf("当前 Pod：%s 状态: %s\n", currentPod.Name, currentPod.Status.Phase)
		return currentPod.Status.Phase, nil
	}).WithTimeout(60*time.Second).WithPolling(2*time.Second).Should(Equal(corev1.PodRunning),
		"Pod：%s 未在指定时间内运行", podName)
}

// VerifyMetricsThroughURL 通过url查看指标
func VerifyMetricsThroughURL(metricsUrl string, podName string, containerName string) {
	var metricsResponse string
	GinkgoWriter.Printf("开始获取指标数据，URL: %s\n", metricsUrl)

	Eventually(func() bool {
		// 创建带超时的 HTTP 客户端
		client := &http.Client{
			Timeout: 30 * time.Second,
		}

		// 发送请求
		resp, err := client.Get(metricsUrl)
		if err != nil {
			GinkgoWriter.Printf("访问指标端点失败: %v\n", err)
			return false
		}
		defer func() {
			if err := resp.Body.Close(); err != nil {
				GinkgoWriter.Printf("关闭响应体失败: %v\n", err)
			}
		}()

		// 检查状态码
		if resp.StatusCode != http.StatusOK {
			GinkgoWriter.Printf("指标端点返回非200状态码: %d\n", resp.StatusCode)
			return false
		}

		// 读取响应
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			GinkgoWriter.Printf("读取响应体失败: %v\n", err)
			return false
		}

		// 判断指标是否符合要求
		metricsResponse = string(body)
		identification := IdentifyMetrics(metricsResponse, podName, containerName)
		return identification
	}).WithTimeout(60*time.Second).WithPolling(3*time.Second).Should(BeTrue(),
		"在重试10次后仍未能成功获取有效的指标数据")
}

// VerifyNPUOperatorTestPodDeleted 验证 Pod 已被删除
func VerifyNPUOperatorTestPodDeleted(clientset *kubernetes.Clientset, ctx context.Context, podName string) {
	// 验证 Pod 已被删除
	Eventually(func() (bool, error) {
		currentPod, err := clientset.CoreV1().Pods("default").Get(ctx, podName, metav1.GetOptions{})
		if currentPod != nil && err == nil {
			GinkgoWriter.Printf("当前 Pod：%s 状态: %s\n", currentPod.Name, currentPod.Status.Phase)
		}
		if errors.IsNotFound(err) {
			return errors.IsNotFound(err), nil
		} else if err != nil {
			return false, fmt.Errorf("获取 Pod：%s 失败： %v\n", podName, err)
		}
		return false, nil
	}, 60*time.Second, 2*time.Second).Should(BeTrue(), "Pod应该被删除")
}

// VerifyModification 验证修改操作
func VerifyModification(modificationName string, managed bool, NPUOperatorCR unstructured.Unstructured,
	dynamicClient dynamic.Interface, ctx context.Context, gvr schema.GroupVersionResource) {
	// 修改 Spec.componentName.Managed 为 managed
	err := ModifyComponentManaged(modificationName, managed, NPUOperatorCR, dynamicClient, ctx, gvr)
	Expect(err).NotTo(HaveOccurred(), "应成功将"+modificationName+"的Managed值修改为"+strconv.FormatBool(managed))
	GinkgoWriter.Printf("成功将%s的Managed值修改为%t\n", modificationName, managed)
}

// VerifyModificationEffectivenessInfo 验证修改有效性的必要信息
type VerifyModificationEffectivenessInfo struct {
	DynamicClient dynamic.Interface
	Gvr           schema.GroupVersionResource
	ComponentName string
	ExpectReason  string
	ExpectType    string
	VerifyType    string
	VerifyField   string
}

// VerifyModificationEffectiveness 执行修改操作并验证修改有效性
func VerifyModificationEffectiveness(info *VerifyModificationEffectivenessInfo) {
	var finalReason, finalCurrentType string

	// 验证修改后结果是否符合预期
	Eventually(func() bool {
		// 获取最新的CR
		currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
		if err != nil {
			GinkgoWriter.Printf("获取最新CR失败：%v\n", err)
			return false
		}
		isReady, err := IsNPUOperatorCRReady(currentNPUOperatorCR)
		if err != nil {
			GinkgoWriter.Printf("检验当前CR状态失败：%v\n", err)
			return false
		}
		if !isReady {
			GinkgoWriter.Printf("当前CR状态不为Ready\n")
			return false
		}

		// 获取组件当前状态下的Reason和Type
		reason, currentType, err := GetComponentReasonAndCurrentType(info.ComponentName, currentNPUOperatorCR, info.VerifyType)
		if err != nil {
			GinkgoWriter.Printf("获取组件当前状态下的Reason和Type出现错误：%v\n", err)
			return false
		}

		// 验证CR状态是否符合预期
		finalReason = reason
		finalCurrentType = currentType
		success := CompareReasonOrType(reason, info.ExpectReason, currentType, info.ExpectType, info.VerifyField)
		if !success {
			GinkgoWriter.Printf("状态不匹配: currentReason=%s, currentType=%s\n", reason, currentType)
		}

		return success
	}, 120*time.Second, 2*time.Second).Should(BeTrue(), "组件状态应该在120秒内达到预期")
	GinkgoWriter.Printf("状态匹配: reason=%s, currentType=%s\n", finalReason, finalCurrentType)
}

// VerifyModificationANDEffectivenessInfo 验证修改操作及有效性的必要信息
type VerifyModificationANDEffectivenessInfo struct {
	DynamicClient    dynamic.Interface
	Ctx              context.Context
	NPUOperatorCR    unstructured.Unstructured
	Gvr              schema.GroupVersionResource
	ModificationName string
	ComponentName    string
}

// VerifyRestoreEnvironment 恢复原来的环境
func VerifyRestoreEnvironment(info *VerifyModificationANDEffectivenessInfo) {
	// managed 设置回true，恢复原来的环境（state恢复正常）
	verifyModificationEffectivenessInfo := VerifyModificationEffectivenessInfo{
		DynamicClient: info.DynamicClient,
		Gvr:           info.Gvr,
		ComponentName: info.ComponentName,
		ExpectReason:  "Reconciled",
		ExpectType:    "running",
		VerifyType:    "state",
		VerifyField:   "all",
	}
	var managed = true

	VerifyModification(info.ModificationName, managed, info.NPUOperatorCR, info.DynamicClient, info.Ctx, info.Gvr)
	VerifyModificationEffectiveness(&verifyModificationEffectivenessInfo)
}

// VerifyComponentManagedFalseState 通过修改Managed值验证组件State字段
func VerifyComponentManagedFalseState(info *VerifyModificationANDEffectivenessInfo, verifyField string) {
	verifyModificationEffectivenessInfo := VerifyModificationEffectivenessInfo{
		DynamicClient: info.DynamicClient,
		Gvr:           info.Gvr,
		ComponentName: info.ComponentName,
		ExpectReason:  "ComponentUnmanaged",
		ExpectType:    "unmanaged",
		VerifyType:    "state",
		VerifyField:   verifyField,
	}

	// 验证修改 Spec.componentName.Managed 为 false
	var managed = false

	VerifyModification(info.ModificationName, managed, info.NPUOperatorCR, info.DynamicClient, info.Ctx, info.Gvr)
	VerifyModificationEffectiveness(&verifyModificationEffectivenessInfo)

	GinkgoWriter.Printf("%s组件Managed-false-State：%s验证成功，正在恢复环境\n", info.ComponentName, verifyField)

	// 获取最新的CR
	currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
	info.NPUOperatorCR = currentNPUOperatorCR

	// 设置回true，恢复原来的环境
	VerifyRestoreEnvironment(info)

	GinkgoWriter.Printf("%s组件Managed-false-State：%s验证成功\n", info.ComponentName, verifyField)
}

// VerifyComponentManagedFalsePrevState 通过修改Managed值验证组件PrevState字段
func VerifyComponentManagedFalsePrevState(info *VerifyModificationANDEffectivenessInfo, verifyField string) {
	verifyModificationEffectivenessInfo := VerifyModificationEffectivenessInfo{
		DynamicClient: info.DynamicClient,
		Gvr:           info.Gvr,
		ComponentName: info.ComponentName,
		ExpectReason:  "Reconciled",
		ExpectType:    "running",
		VerifyType:    "prevState",
		VerifyField:   verifyField,
	}

	// 验证修改 Spec.componentName.Managed 为 false
	var managed = false

	VerifyModification(info.ModificationName, managed, info.NPUOperatorCR, info.DynamicClient, info.Ctx, info.Gvr)
	VerifyModificationEffectiveness(&verifyModificationEffectivenessInfo)

	GinkgoWriter.Printf("%s组件Managed-false-prevState：%s验证成功，正在恢复环境\n", info.ComponentName, verifyField)

	// 获取最新的CR
	currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
	info.NPUOperatorCR = currentNPUOperatorCR

	// 设置回true，恢复原来的环境
	VerifyRestoreEnvironment(info)

	GinkgoWriter.Printf("%s组件Managed-false-prevState：%s验证成功\n", info.ComponentName, verifyField)
}

// VerifyPodStateInfo 验证 Pod 状态的必要信息
type VerifyPodStateInfo struct {
	Clientset         *kubernetes.Clientset
	DynamicClient     dynamic.Interface
	Ctx               context.Context
	NPUOperatorCR     unstructured.Unstructured
	Gvr               schema.GroupVersionResource
	ModificationNames []string
	PodNames          []string
}

type VerifyManagedAndInvalidImageInfo struct {
	Clientset               *kubernetes.Clientset
	DynamicClient           dynamic.Interface
	Ctx                     context.Context
	Gvr                     schema.GroupVersionResource
	ModificationName        string
	ComponentName           string
	PodPrefix               string
	InvalidRepositorySuffix string
}

func VerifyComponentManagedAndInvalidImageState(info *VerifyManagedAndInvalidImageInfo) {
	VerifyComponentRunningAndReconciled(info)
	expectedPodCount := GetExpectedComponentPodCount(info)
	VerifyPodsExpected(info.Clientset, info.Ctx, info.PodPrefix, expectedPodCount)

	originalImageSpec := GetOriginalComponentImageSpec(info)
	invalidImageSpec := BuildInvalidComponentImageSpec(originalImageSpec, info)

	DeferCleanup(func() {
		RestoreComponentManagedAndImageSpecByCount(info, originalImageSpec, expectedPodCount)
	})

	VerifyComponentUnmanagedAfterManagedFalse(info)
	ApplyInvalidImageSpecAndVerifyFailure(info, invalidImageSpec)
}

func GetExpectedComponentPodCount(info *VerifyManagedAndInvalidImageInfo) int {
	deployments, err := info.Clientset.AppsV1().Deployments("").List(info.Ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "should successfully list deployments")
	for _, deployment := range deployments.Items {
		if strings.HasPrefix(deployment.Name, info.PodPrefix) {
			if deployment.Spec.Replicas != nil {
				return int(*deployment.Spec.Replicas)
			}
			return 1
		}
	}

	daemonsets, err := info.Clientset.AppsV1().DaemonSets("").List(info.Ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "should successfully list daemonsets")
	for _, daemonset := range daemonsets.Items {
		if strings.HasPrefix(daemonset.Name, info.PodPrefix) {
			return int(daemonset.Status.DesiredNumberScheduled)
		}
	}

	pods, err := GetSpecifiedPods(info.Clientset, info.Ctx, info.PodPrefix)
	Expect(err).NotTo(HaveOccurred(), "should successfully get specified pods")
	Expect(pods).NotTo(BeEmpty(), "component should have existing pods before modification")
	return len(pods)
}

func VerifyComponentRunningAndReconciled(info *VerifyManagedAndInvalidImageInfo) {
	Eventually(func(g Gomega) {
		currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")

		reason, currentType, err := GetComponentState(info.ComponentName, currentNPUOperatorCR)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取组件当前状态")
		g.Expect(currentType).To(Equal("running"))
		g.Expect(reason).To(Equal("Reconciled"))
	}, 120*time.Second, 2*time.Second).Should(Succeed(), "组件初始状态应为 running/Reconciled")
}

func GetOriginalComponentImageSpec(info *VerifyManagedAndInvalidImageInfo) map[string]interface{} {
	currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")

	imageSpec, err := GetComponentImageSpec(info.ModificationName, currentNPUOperatorCR)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取组件imageSpec")
	return imageSpec
}

func BuildInvalidComponentImageSpec(
	originalImageSpec map[string]interface{}, info *VerifyManagedAndInvalidImageInfo,
) map[string]interface{} {
	invalidImageSpec := make(map[string]interface{}, len(originalImageSpec))
	for key, value := range originalImageSpec {
		invalidImageSpec[key] = value
	}

	if repository, ok := invalidImageSpec["repository"].(string); ok && repository != "" {
		invalidImageSpec["repository"] = repository + info.InvalidRepositorySuffix
		return invalidImageSpec
	}

	invalidImageSpec["repository"] = "openfuyao/" + info.PodPrefix + info.InvalidRepositorySuffix
	return invalidImageSpec
}

func RestoreComponentManagedAndImageSpecByCount(
	info *VerifyManagedAndInvalidImageInfo,
	originalImageSpec map[string]interface{},
	expectedPodCount int,
) {
	currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
	if err != nil && strings.Contains(err.Error(), "CR数量应为至少1个") {
		GinkgoWriter.Printf(
			"skip restoring %s because NPUOperatorCR has already been removed\n",
			info.ComponentName,
		)
		return
	}
	Expect(err).NotTo(HaveOccurred(), "should successfully get latest NPUOperatorCR")

	err = ModifyComponentManagedAndImageSpec(
		info.ModificationName,
		true,
		originalImageSpec,
		currentNPUOperatorCR,
		info.DynamicClient,
		info.Ctx,
		info.Gvr,
	)
	Expect(err).NotTo(HaveOccurred(), "should successfully restore component managed and imageSpec")

	Eventually(func(g Gomega) {
		currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
		g.Expect(err).NotTo(HaveOccurred(), "should successfully get restored NPUOperatorCR")

		pods, err := GetSpecifiedPods(info.Clientset, info.Ctx, info.PodPrefix)
		g.Expect(err).NotTo(HaveOccurred(), "should successfully get component pods")
		g.Expect(len(pods)).To(Equal(expectedPodCount), "restored component pod count should match original")
		for _, pod := range pods {
			g.Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "restored component pods should be Running")
		}

		reason, currentType, err := GetComponentState(info.ComponentName, currentNPUOperatorCR)
		g.Expect(err).NotTo(HaveOccurred(), "should successfully get restored component state")
		g.Expect(currentType).To(Equal("running"))
		g.Expect(reason).To(Equal("Reconciled"))
	}, 180*time.Second, 2*time.Second).Should(Succeed(), "restored component should be running again")
}

func VerifyComponentUnmanagedAfterManagedFalse(info *VerifyManagedAndInvalidImageInfo) {
	currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")

	err = ModifyComponentManaged(
		info.ModificationName, false, currentNPUOperatorCR, info.DynamicClient, info.Ctx, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "应该成功将组件managed修改为false")

	Eventually(func(g Gomega) {
		currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")

		pods, err := GetSpecifiedPods(info.Clientset, info.Ctx, info.PodPrefix)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取组件相关Pod")
		g.Expect(len(pods)).To(Equal(0), "managed=false后组件Pod应被卸载")

		reason, currentType, err := GetComponentState(info.ComponentName, currentNPUOperatorCR)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取组件当前状态")
		g.Expect(currentType).To(Equal("unmanaged"))
		g.Expect(reason).To(Equal("ComponentUnmanaged"))
	}, 120*time.Second, 2*time.Second).Should(Succeed(), "managed=false后组件应为unmanaged")
}

func ApplyInvalidImageSpecAndVerifyFailure(
	info *VerifyManagedAndInvalidImageInfo, invalidImageSpec map[string]interface{},
) {
	currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")

	err = ModifyComponentManagedAndImageSpec(
		info.ModificationName, true, invalidImageSpec, currentNPUOperatorCR, info.DynamicClient, info.Ctx, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "应该成功写入无效镜像并设置managed=true")

	seenIntermediateState := WaitForComponentTerminated(info)
	VerifyComponentTerminatedFailure(info)
	VerifyComponentIntermediateStateLenient(info, seenIntermediateState)
}

func WaitForComponentTerminated(info *VerifyManagedAndInvalidImageInfo) bool {
	seenIntermediateState := false

	Eventually(func(g Gomega) bool {
		currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")

		_, currentType, _, err := GetComponentStateDetails(info.ComponentName, currentNPUOperatorCR)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取组件当前状态详情")
		if currentType == "deploying" || currentType == "pending" {
			seenIntermediateState = true
		}
		return currentType == "terminated"
	}, 120*time.Second, 500*time.Millisecond).Should(BeTrue(), "组件应最终进入terminated")

	return seenIntermediateState
}

func VerifyComponentTerminatedFailure(info *VerifyManagedAndInvalidImageInfo) {
	Eventually(func(g Gomega) {
		currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")

		reason, currentType, message, err := GetComponentStateDetails(info.ComponentName, currentNPUOperatorCR)
		g.Expect(err).NotTo(HaveOccurred(), "应该成功获取组件失败状态详情")
		g.Expect(currentType).To(Equal("terminated"))
		g.Expect(reason).To(Equal("ReconcileFailed"))
		g.Expect(message).To(ContainSubstring("not ready"))
	}, 120*time.Second, 2*time.Second).Should(Succeed(), "无效镜像下组件应进入ReconcileFailed/terminated状态")
}

func VerifyComponentIntermediateState(info *VerifyManagedAndInvalidImageInfo, seenIntermediateState bool) {
	currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")

	_, prevType, err := GetComponentPrevState(info.ComponentName, currentNPUOperatorCR)
	Expect(err).NotTo(HaveOccurred(), "应该成功获取组件上一状态")
	Expect(
		seenIntermediateState || prevType == "deploying" || prevType == "pending",
	).To(BeTrue(), "在进入terminated之前，state.type或prevState.type应至少出现过deploying或pending")
}

// VerifyComponentManagedFalsePodState 验证 managed=false Pod 卸载
func VerifyComponentIntermediateStateLenient(info *VerifyManagedAndInvalidImageInfo, seenIntermediateState bool) {
	currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
	Expect(err).NotTo(HaveOccurred(), "should successfully get latest NPUOperatorCR")

	_, prevType, err := GetComponentPrevState(info.ComponentName, currentNPUOperatorCR)
	Expect(err).NotTo(HaveOccurred(), "should successfully get component previous state")
	if seenIntermediateState || prevType == "deploying" || prevType == "pending" {
		GinkgoWriter.Printf("%s component observed intermediate state before terminated, prevState.type=%s\n", info.ComponentName, prevType)
		return
	}

	GinkgoWriter.Printf(
		"%s component did not expose a stable deploying/pending intermediate state in CR, prevState.type=%s; relying on terminated/ReconcileFailed instead\n",
		info.ComponentName,
		prevType,
	)
}

func VerifyComponentManagedFalsePodState(info *VerifyPodStateInfo) {
	// 验证修改 Spec.componentName.Managed 为 false
	var managed = false
	for _, modificationName := range info.ModificationNames {
		VerifyModification(modificationName, managed, info.NPUOperatorCR, info.DynamicClient, info.Ctx, info.Gvr)

		// 获取最新的CR
		currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
		Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
		info.NPUOperatorCR = currentNPUOperatorCR
	}

	// 验证修改后结果是否符合预期
	Eventually(func() bool {
		success := true

		for _, podName := range info.PodNames {
			pods, err := GetSpecifiedPods(info.Clientset, info.Ctx, podName)
			if err != nil {
				GinkgoWriter.Printf("获取 %s 相关 Pod 失败：%v\n", podName, err)
				success = false
				continue
			}
			if len(pods) != 0 {
				GinkgoWriter.Printf("%s 相关 Pod 还未完成卸载\n", podName)
				success = false
			} else {
				GinkgoWriter.Printf("%s 相关 Pod 已完成卸载\n", podName)
			}
		}

		return success
	}, 60*time.Second, 1*time.Second).Should(BeTrue(), "组件相关 Pod 应该在60秒内完成卸载")
	GinkgoWriter.Printf("%v 已成功卸载\n", info.PodNames)
}

// VerifyComponentManagedTruePodState 验证 managed=true Pod 卸载
func VerifyComponentManagedTruePodState(info *VerifyPodStateInfo) {
	// 验证修改 Spec.componentName.Managed 为 true
	var managed = true
	for _, modificationName := range info.ModificationNames {
		VerifyModification(modificationName, managed, info.NPUOperatorCR, info.DynamicClient, info.Ctx, info.Gvr)

		// 获取最新的CR
		currentNPUOperatorCR, err := GetNPUOperatorCR(info.DynamicClient, info.Gvr)
		Expect(err).NotTo(HaveOccurred(), "应该成功获取最新NPUOperatorCR")
		info.NPUOperatorCR = currentNPUOperatorCR
	}

	// 验证修改后结果是否符合预期
	Eventually(func() bool {
		success := true

		for _, podName := range info.PodNames {
			pods, err := GetSpecifiedPods(info.Clientset, info.Ctx, podName)
			if err != nil {
				GinkgoWriter.Printf("获取 %s 相关 Pod 失败：%v\n", podName, err)
				success = false
				continue
			}
			if len(pods) == 0 {
				GinkgoWriter.Printf("%s 相关 Pod 还未开始安装\n", podName)
				success = false
			} else {
				for _, pod := range pods {
					if pod.Status.Phase != corev1.PodRunning {
						GinkgoWriter.Printf("Pod %s 还未完成安装(状态: %s)\n",
							pod.Name, pod.Status.Phase)
						success = false
					}
				}
			}

			if success {
				GinkgoWriter.Printf("%s 相关 Pod 已完成安装\n", podName)
			}
		}

		return success
	}, 60*time.Second, 1*time.Second).Should(BeTrue(), "组件相关 Pod 应该在60秒内完成安装")

	GinkgoWriter.Printf("%v 已成功安装\n", info.PodNames)
}
