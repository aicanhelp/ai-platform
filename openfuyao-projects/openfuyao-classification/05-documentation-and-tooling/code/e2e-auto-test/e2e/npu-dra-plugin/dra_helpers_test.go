package npu_dra_plugin

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/remotecommand"
)

type npuSmiDevice struct {
	PhysicalID    int
	ChipName      string
	MemCapacityMB int64
}

var usagePairPattern = regexp.MustCompile(`(\d+)\s*/\s*(\d+)`)
var memoryQuotaRe = regexp.MustCompile(`Memory Limit quota\(MB\)\D*(\d+)`)

func ensureNamespace() {
	_, err := k8sClient.GetNamespace(ctx, testNamespace)
	if err != nil {
		err = k8sClient.CreateNamespaceIfNotExists(ctx, testNamespace)
		Expect(err).NotTo(HaveOccurred(), "创建 namespace 应当成功")
		GinkgoWriter.Printf("已创建 namespace %s\n", testNamespace)
	}
}

func setNodeVnpuMode(nodeName string, configs string) {
	cmd := exec.Command("kubectl", "create", "configmap", "dra-profile-config",
		fmt.Sprintf("--from-file=dra.config=/dev/stdin"),
		"-n", pluginNamespace,
		"-o", "yaml", "--dry-run=client")
	cmd.Stdin = strings.NewReader(configs)
	var cmOut bytes.Buffer
	cmd.Stdout = &cmOut
	cmd.Stderr = &cmBuffer
	Expect(cmd.Run()).NotTo(HaveOccurred())

	replaceCmd := exec.Command("kubectl", "replace", "-f", "-")
	replaceCmd.Stdin = &cmOut
	replaceCmd.Stderr = &cmBuffer
	Expect(replaceCmd.Run()).NotTo(HaveOccurred())

	exec.Command("rm", "-f", "/var/run/cdi/k8s.npu.huawei.com-*.yaml").Run()
	exec.Command("kubectl", "rollout", "restart", "ds", "npu-dra-plugin", "-n", pluginNamespace).Run()
	time.Sleep(15 * time.Second)

	Eventually(func() bool {
		slices, err := k8sClient.Clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
		if err != nil {
			return false
		}
		for _, rs := range slices.Items {
			if rs.Spec.Driver == driverName && rs.Spec.NodeName != nil && *rs.Spec.NodeName == nodeName {
				return len(rs.Spec.Devices) > 0
			}
		}
		return false
	}, 60*time.Second, pollInterval).Should(BeTrue(), "插件重启后 ResourceSlice 应当重新上报")
	GinkgoWriter.Printf("节点 %s ConfigMap 已更新，插件已重启\n", nodeName)
}

func setShareCount(shareCount int) {
	By(fmt.Sprintf("设置 --share-count=%d 并重启 DRA 插件", shareCount))
	ds, err := k8sClient.Clientset.AppsV1().DaemonSets(pluginNamespace).Get(ctx, "npu-dra-plugin", metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取 DRA 插件 DaemonSet 应当成功")

	args := ds.Spec.Template.Spec.Containers[0].Args
	found := false
	for i, arg := range args {
		if strings.HasPrefix(arg, "--share-count=") {
			args[i] = fmt.Sprintf("--share-count=%d", shareCount)
			found = true
			break
		}
	}
	if !found {
		args = append(args, fmt.Sprintf("--share-count=%d", shareCount))
	}
	ds.Spec.Template.Spec.Containers[0].Args = args

	_, err = k8sClient.Clientset.AppsV1().DaemonSets(pluginNamespace).Update(ctx, ds, metav1.UpdateOptions{})
	Expect(err).NotTo(HaveOccurred(), "更新 DaemonSet --share-count 应当成功")

	// 等待 Pod 重启
	time.Sleep(20 * time.Second)
	Eventually(func() bool {
		pods, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).List(ctx, metav1.ListOptions{LabelSelector: "app=npu-dra-plugin"})
		if err != nil || len(pods.Items) == 0 {
			return false
		}
		return pods.Items[0].Status.Phase == corev1.PodRunning && pods.Items[0].Status.ContainerStatuses[0].Ready
	}, 120*time.Second, pollInterval).Should(BeTrue(), "DRA 插件应当重启完成")
	GinkgoWriter.Printf("DRA 插件已重启，--share-count=%d\n", shareCount)
}

func restoreShareCount() {
	By("恢复 --share-count=16 并重启 DRA 插件")
	setShareCount(16)
}

// 注意: K8s 1.34 PR #133706 bug 导致调度器总是检查 slice.Spec.Devices[0] (npu-0) 的 capacity。
// 因此所有配置函数必须确保 npu-0 (physicalId=0) 被设为目标模式来规避 bug。
// 其他卡可以配为不同模式，但 npu-0 决定了调度器 capacity 检查的类型。

func softConfig(nodeName string, physicalIDs ...int) string {
	// 确保 npu-0 在列表中
	ids := ensurePhysicalID0(physicalIDs)
	var lines []string
	lines = append(lines, "softShareMounts:")
	lines = append(lines, "- hostPath: /opt/xpu/lib/libboundscheck.so")
	lines = append(lines, "  containerPath: /lib/libboundscheck.so")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, "- hostPath: /opt/xpu/bin/npu-monitor")
	lines = append(lines, "  containerPath: /opt/enpu/vcann-rt/tools/enpu-monitor")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, "- hostPath: /opt/xpu/bin/systemd-detect-virt")
	lines = append(lines, "  containerPath: /usr/bin/systemd-detect-virt")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, fmt.Sprintf("nodes:\n  %s:", nodeName))
	for _, id := range ids {
		lines = append(lines, fmt.Sprintf("    - physicalId: %d", id))
		lines = append(lines, "      vnpuMode: soft")
		lines = append(lines, "      schedulingPolicy: elastic")
	}
	return strings.Join(lines, "\n")
}

func hardConfig(nodeName string, physicalIDs ...int) string {
	// 确保 npu-0 在列表中
	ids := ensurePhysicalID0(physicalIDs)
	var lines []string
	lines = append(lines, "softShareMounts:")
	lines = append(lines, "- hostPath: /opt/xpu/lib/libboundscheck.so")
	lines = append(lines, "  containerPath: /lib/libboundscheck.so")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, "- hostPath: /opt/xpu/bin/npu-monitor")
	lines = append(lines, "  containerPath: /opt/enpu/vcann-rt/tools/enpu-monitor")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, "- hostPath: /opt/xpu/bin/systemd-detect-virt")
	lines = append(lines, "  containerPath: /usr/bin/systemd-detect-virt")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, fmt.Sprintf("nodes:\n  %s:", nodeName))
	for _, id := range ids {
		lines = append(lines, fmt.Sprintf("    - physicalId: %d", id))
		lines = append(lines, "      vnpuMode: hard")
	}
	return strings.Join(lines, "\n")
}

func fullConfig(nodeName string, physicalIDs ...int) string {
	// 确保 npu-0 在列表中
	ids := ensurePhysicalID0(physicalIDs)
	var lines []string
	lines = append(lines, fmt.Sprintf("nodes:\n  %s:", nodeName))
	for _, id := range ids {
		lines = append(lines, fmt.Sprintf("    - physicalId: %d", id))
		lines = append(lines, "      vnpuMode: full")
	}
	return strings.Join(lines, "\n")
}

// ensurePhysicalID0 ensures physicalId=0 is always in the list
// (workaround for K8s 1.34 PR #133706 bug)
func ensurePhysicalID0(ids []int) []int {
	for _, id := range ids {
		if id == 0 {
			return ids
		}
	}
	// npu-0 not in list, prepend it
	result := []int{0}
	result = append(result, ids...)
	return result
}

func mixedConfig(nodeName string, configs ...string) string {
	var lines []string
	lines = append(lines, "softShareMounts:")
	lines = append(lines, "- hostPath: /opt/xpu/lib/libboundscheck.so")
	lines = append(lines, "  containerPath: /lib/libboundscheck.so")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, "- hostPath: /opt/xpu/bin/npu-monitor")
	lines = append(lines, "  containerPath: /opt/enpu/vcann-rt/tools/enpu-monitor")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, "- hostPath: /opt/xpu/bin/systemd-detect-virt")
	lines = append(lines, "  containerPath: /usr/bin/systemd-detect-virt")
	lines = append(lines, "  options: [rw, rbind]")
	lines = append(lines, fmt.Sprintf("nodes:\n  %s:", nodeName))
	for _, c := range configs {
		lines = append(lines, c)
	}
	return strings.Join(lines, "\n")
}

func cleanupAllTestPods() {
	pods, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return
	}
	for _, pod := range pods.Items {
		_ = k8sClient.Clientset.CoreV1().Pods(testNamespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
	}
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(ctx context.Context) (bool, error) {
		pods, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, nil
		}
		return len(pods.Items) == 0, nil
	})
	GinkgoWriter.Printf("已清理 namespace %s 中的所有 Pod\n", testNamespace)
}

func cleanupAllResourceClaimTemplates() {
	templates, err := k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return
	}
	for _, tpl := range templates.Items {
		_ = k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Delete(ctx, tpl.Name, metav1.DeleteOptions{})
	}
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(ctx context.Context) (bool, error) {
		templates, err := k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, nil
		}
		return len(templates.Items) == 0, nil
	})
	GinkgoWriter.Printf("已清理所有 ResourceClaimTemplate\n")
}

func ptrStr(v string) *string { return &v }
func ptrHostPathType(t corev1.HostPathType) *corev1.HostPathType { return &t }

func hostPathVolume(name, path string, hostPathType corev1.HostPathType) corev1.Volume {
	return corev1.Volume{
		Name: name,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: path,
				Type: ptrHostPathType(hostPathType),
			},
		},
	}
}

func buildHardVNpuPod(podName, claimTemplateName string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: testNamespace},
		Spec: corev1.PodSpec{
			HostNetwork:       true,
			DNSPolicy:          corev1.DNSClusterFirstWithHostNet,
			NodeSelector:      map[string]string{"kubernetes.io/hostname": localNodeName},
			RuntimeClassName: ptrStr("ascend"),
			ResourceClaims: []corev1.PodResourceClaim{{
				Name: "npu", ResourceClaimTemplateName: ptrStr(claimTemplateName),
			}},
			Containers: []corev1.Container{{
				Name: "app", Image: testPodImage, ImagePullPolicy: corev1.PullIfNotPresent,
				Command: []string{"sleep", "3600"},
				Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "npu"}}},
				Env: []corev1.EnvVar{{Name: "LD_LIBRARY_PATH", Value: "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64"}},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "ascend-driver", MountPath: "/usr/local/Ascend", ReadOnly: true},
					{Name: "npu-smi-command", MountPath: "/usr/local/bin/npu-smi", ReadOnly: true},
				},
				SecurityContext: &corev1.SecurityContext{Capabilities: &corev1.Capabilities{Add: []corev1.Capability{"SYS_ADMIN"}}},
			}},
			Volumes: []corev1.Volume{
				hostPathVolume("ascend-driver", "/usr/local/Ascend", corev1.HostPathDirectory),
				hostPathVolume("npu-smi-command", "/usr/local/bin/npu-smi", corev1.HostPathFile),
			},
		},
	}
}

func buildSoftVNpuPod(podName, claimTemplateName string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: testNamespace},
		Spec: corev1.PodSpec{
			HostNetwork:  true,
			DNSPolicy:    corev1.DNSClusterFirstWithHostNet,
			NodeSelector: map[string]string{"kubernetes.io/hostname": localNodeName},
			ResourceClaims: []corev1.PodResourceClaim{{
				Name: "npu", ResourceClaimTemplateName: ptrStr(claimTemplateName),
			}},
			Containers: []corev1.Container{{
				Name: "app", Image: testPodImage, ImagePullPolicy: corev1.PullIfNotPresent,
				Command: []string{"sleep", "3600"},
				Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "npu"}}},
				Env: []corev1.EnvVar{{Name: "LD_LIBRARY_PATH", Value: "/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64"}},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "npu-libs-driver", MountPath: "/usr/local/Ascend/driver/lib64/driver"},
					{Name: "npu-libs-common", MountPath: "/usr/local/Ascend/driver/lib64/common"},
					{Name: "ascend-toolkit", MountPath: "/usr/local/Ascend/ascend-toolkit", ReadOnly: true},
					{Name: "npu-smi", MountPath: "/usr/local/sbin/npu-smi"},
				},
			}},
			Volumes: []corev1.Volume{
				hostPathVolume("npu-libs-driver", "/usr/local/Ascend/driver/lib64/driver", corev1.HostPathDirectoryOrCreate),
				hostPathVolume("npu-libs-common", "/usr/local/Ascend/driver/lib64/common", corev1.HostPathDirectoryOrCreate),
				hostPathVolume("ascend-toolkit", "/usr/local/Ascend/ascend-toolkit", corev1.HostPathDirectory),
				hostPathVolume("npu-smi", "/usr/local/sbin/npu-smi", corev1.HostPathFile),
			},
		},
	}
}

func buildFullNPUPod(podName, claimTemplateName string) *corev1.Pod {
	pod := buildSoftVNpuPod(podName, claimTemplateName)
	pod.Spec.Containers[0].Env[0].Value = "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64"
	pod.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{
		{Name: "ascend-driver", MountPath: "/usr/local/Ascend", ReadOnly: true},
		{Name: "npu-smi-command", MountPath: "/usr/local/bin/npu-smi", ReadOnly: true},
	}
	pod.Spec.Volumes = []corev1.Volume{
		hostPathVolume("ascend-driver", "/usr/local/Ascend", corev1.HostPathDirectory),
		hostPathVolume("npu-smi-command", "/usr/local/bin/npu-smi", corev1.HostPathFile),
	}
	return pod
}

func createFullClaimTemplate(name string) {
	createClaimTemplate(name, deviceClassFull, nil, "")
}

func createHardClaimTemplate(name string, aiCore, aiCPU int, celExpr string) {
	caps := map[resourcev1.QualifiedName]apiresource.Quantity{
		"aiCore": apiresource.MustParse(fmt.Sprintf("%d", aiCore)),
		"aiCPU":  apiresource.MustParse(fmt.Sprintf("%d", aiCPU)),
	}
	createClaimTemplate(name, deviceClassHard, caps, celExpr)
}

func createSoftClaimTemplate(name string, memCapacityBytes int64, coreCapacity int, celExpr string) {
	caps := map[resourcev1.QualifiedName]apiresource.Quantity{
		"memCapacity":   apiresource.MustParse(fmt.Sprintf("%d", memCapacityBytes)),
		"coreCapacity":  apiresource.MustParse(fmt.Sprintf("%d", coreCapacity)),
	}
	createClaimTemplate(name, deviceClassElastic, caps, celExpr)
}

func createSoftClaimTemplateWithClass(name, deviceClass string, memCapacityBytes int64, coreCapacity int, celExpr string) {
	caps := map[resourcev1.QualifiedName]apiresource.Quantity{
		"memCapacity":   apiresource.MustParse(fmt.Sprintf("%d", memCapacityBytes)),
		"coreCapacity":  apiresource.MustParse(fmt.Sprintf("%d", coreCapacity)),
	}
	createClaimTemplate(name, deviceClass, caps, celExpr)
}

func createClaimTemplate(name, deviceClass string, caps map[resourcev1.QualifiedName]apiresource.Quantity, celExpr string) {
	cleanupBeforeCreate(fmt.Sprintf("ResourceClaimTemplate %s", name),
		func() error { return k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Delete(ctx, name, metav1.DeleteOptions{}) },
		func() error { _, err := k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Get(ctx, name, metav1.GetOptions{}); return err })

	var selectors []resourcev1.DeviceSelector
	if celExpr != "" {
		selectors = append(selectors, resourcev1.DeviceSelector{CEL: &resourcev1.CELDeviceSelector{Expression: celExpr}})
	}

	tpl := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: testNamespace},
		Spec: resourcev1.ResourceClaimTemplateSpec{
			Spec: resourcev1.ResourceClaimSpec{
				Devices: resourcev1.DeviceClaim{
					Requests: []resourcev1.DeviceRequest{{
						Name: "npu",
						Exactly: &resourcev1.ExactDeviceRequest{
							DeviceClassName: deviceClass,
							Count:           1,
							Capacity: &resourcev1.CapacityRequirements{Requests: caps},
							Selectors:       selectors,
						},
					}},
				},
			},
		},
	}
	_, err := k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Create(ctx, tpl, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred(), "创建 ResourceClaimTemplate %s 应当成功", name)
	GinkgoWriter.Printf("ResourceClaimTemplate %s 已创建 (deviceClass=%s)\n", name, deviceClass)
}

func createPodAndWaitRunning(podName, claimTemplate string, podBuilder func(string, string) *corev1.Pod) {
	if podBuilder == nil {
		podBuilder = buildFullNPUPod
	}
	pod := podBuilder(podName, claimTemplate)
	_, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Create(ctx, pod, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred(), "创建 Pod %s 应当成功", podName)
	waitForPodRunning(testNamespace, podName)
}

func waitForPodRunning(namespace, podName string) {
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		pod, err := k8sClient.Clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil { return false, nil }
		if pod.Status.Phase != corev1.PodRunning { return false, nil }
		for _, c := range pod.Status.ContainerStatuses { if !c.Ready { return false, nil } }
		return true, nil
	})
	Expect(err).NotTo(HaveOccurred(), "Pod %s/%s 应当达到 Running 状态", namespace, podName)
	GinkgoWriter.Printf("Pod %s/%s 已 Running\n", namespace, podName)
}

func waitForPodPending(namespace, podName string) {
	err := wait.PollUntilContextTimeout(ctx, pollInterval, 120*time.Second, true, func(ctx context.Context) (bool, error) {
		pod, err := k8sClient.Clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil { return false, nil }
		return pod.Status.Phase == corev1.PodPending, nil
	})
	Expect(err).NotTo(HaveOccurred(), "Pod %s/%s 应当保持 Pending", namespace, podName)
}

func waitForResourceSlice(nodeName string) *resourcev1.ResourceSlice {
	var found *resourcev1.ResourceSlice
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(ctx context.Context) (bool, error) {
		slices, err := k8sClient.Clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
		if err != nil { return false, nil }
		for i := range slices.Items {
			rs := &slices.Items[i]
			if rs.Spec.Driver == driverName && rs.Spec.NodeName != nil && *rs.Spec.NodeName == nodeName {
				found = rs
				return true, nil
			}
		}
		return false, nil
	})
	Expect(err).NotTo(HaveOccurred(), "节点 %s 的 ResourceSlice 应当存在", nodeName)
	return found
}

func cleanupBeforeCreate(resourceName string, deleteFunc, getFunc func() error) {
	_ = deleteFunc()
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(ctx context.Context) (bool, error) {
		err := getFunc()
		return apierrors.IsNotFound(err), nil
	})
}

func deletePodAndWait(namespace, podName string) {
	_ = k8sClient.Clientset.CoreV1().Pods(namespace).Delete(ctx, podName, metav1.DeleteOptions{})
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(ctx context.Context) (bool, error) {
		_, err := k8sClient.Clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	})
	GinkgoWriter.Printf("Pod %s/%s 已删除\n", namespace, podName)
}

func deleteResourceClaimTemplateAndWait(name string) {
	_ = k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Delete(ctx, name, metav1.DeleteOptions{})
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(ctx context.Context) (bool, error) {
		_, err := k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	})
	GinkgoWriter.Printf("ResourceClaimTemplate %s 已删除\n", name)
}

func cleanupPodAndClaimTemplate(podName, claimTemplate string) {
	deletePodAndWait(testNamespace, podName)
	deleteResourceClaimTemplateAndWait(claimTemplate)
}

func execCommandInPod(namespace, podName, containerName, command string) (string, string, error) {
	req := k8sClient.Clientset.CoreV1().RESTClient().Post().
		Resource("pods").Namespace(namespace).Name(podName).SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: containerName, Command: []string{"sh", "-c", command},
			Stdin: false, Stdout: true, Stderr: true,
		}, scheme.ParameterCodec)
	executor, err := remotecommand.NewSPDYExecutor(k8sClient.RestConfig, "POST", req.URL())
	if err != nil { return "", "", fmt.Errorf("create SPDY executor: %w", err) }
	var stdout, stderr bytes.Buffer
	err = executor.StreamWithContext(ctx, remotecommand.StreamOptions{Stdout: &stdout, Stderr: &stderr})
	return stdout.String(), stderr.String(), err
}

func verifyNPUSMIInPod(podName string) string {
	var stdout string
	_ = wait.PollUntilContextTimeout(ctx, 5*time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		var err error
		stdout, _, err = execCommandInPod(testNamespace, podName, "app", "npu-smi info")
		return err == nil && stdout != "", nil
	})
	return stdout
}

func verifyVNPUInfoInPod(podName string) string {
	var result string
	_ = wait.PollUntilContextTimeout(ctx, 5*time.Second, 120*time.Second, true, func(ctx context.Context) (bool, error) {
		npuSmiOutput := verifyNPUSMIInPod(podName)
		if npuSmiOutput == "" {
			return false, nil
		}
		devices := parseNpuSmiInfo(npuSmiOutput)
		for _, dev := range devices {
			if strings.Contains(dev.ChipName, "vir") {
				result = fmt.Sprintf("%s (ID=%d, Mem=%dMB)", dev.ChipName, dev.PhysicalID, dev.MemCapacityMB)
				return true, nil
			}
		}
		return false, nil
	})
	return result
}

func requireVNPUInPod(podName string) string {
	result := verifyVNPUInfoInPod(podName)
	Expect(result).NotTo(BeEmpty(), "Pod %s 应当包含 VNPU 设备", podName)
	return result
}

func requireWholeCardInPod(podName string) string {
	output := verifyNPUSMIInPod(podName)
	Expect(output).NotTo(BeEmpty(), "Pod %s 应当可见 NPU", podName)
	Expect(verifyVNPUInfoInPod(podName)).To(BeEmpty(), "Pod %s 应当使用整卡而非 VNPU", podName)
	return output
}

func verifyEnpuMonitorInPod(podName string) string {
	var stdout string
	_ = wait.PollUntilContextTimeout(ctx, 5*time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		var err error
		stdout, _, err = execCommandInPod(testNamespace, podName, "app", "/opt/enpu/vcann-rt/tools/enpu-monitor 2>&1 || true")
		if err != nil {
			return false, nil
		}
		return strings.Contains(stdout, "Aicore Limit Quota"), nil
	})
	return stdout
}

// verifyEnpuMonitorMemoryQuota 从 enpu-monitor 输出中提取 Memory Limit quota(MB) 数值，
// 并校验其与申请 memCapacity 的换算值（expectedMB）一致。
func verifyEnpuMonitorMemoryQuota(output string, expectedMB int64) {
	matches := memoryQuotaRe.FindStringSubmatch(output)
	Expect(matches).NotTo(BeNil(), "enpu-monitor 输出应包含 Memory Limit quota(MB) 字段")
	actualMB, err := strconv.ParseInt(matches[1], 10, 64)
	Expect(err).NotTo(HaveOccurred(), "解析 memory-quota 数值应当成功")
	Expect(actualMB).To(Equal(expectedMB),
		"memory-quota 应为 %d MB，实际 %d MB", expectedMB, actualMB)
	GinkgoWriter.Printf("memory-quota 校验通过: %d MB\n", actualMB)
}

func findNPUNode(chipNamePrefix string) string {
	slices, err := k8sClient.Clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	for _, rs := range slices.Items {
		if rs.Spec.Driver != driverName { continue }
		for _, device := range rs.Spec.Devices {
			if attr, ok := device.Attributes["chipName"]; ok && attr.StringValue != nil {
				if strings.HasPrefix(*attr.StringValue, chipNamePrefix) {
					if rs.Spec.NodeName != nil { return *rs.Spec.NodeName }
				}
			}
		}
	}
	return ""
}

func parseNpuSmiInfo(output string) map[int]npuSmiDevice {
	result := make(map[int]npuSmiDevice)
	var pendingHead *npuSmiDeviceHead
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.Contains(line, "Process id") { break }
		columns := npuSmiParseTableLine(line)
		if len(columns) < 3 { continue }
		if head, ok := npuSmiParseDeviceHead(columns[0]); ok {
			pendingHead = &head
			continue
		}
		if pendingHead == nil { continue }
		if memCapacityMB, ok := npuSmiParseDeviceMemCapacity(columns[len(columns)-1]); ok {
			result[pendingHead.index] = npuSmiDevice{PhysicalID: pendingHead.index, ChipName: pendingHead.chipName, MemCapacityMB: memCapacityMB}
			pendingHead = nil
		}
	}
	return result
}

type npuSmiDeviceHead struct{ index int; chipName string }

func npuSmiParseDeviceHead(left string) (npuSmiDeviceHead, bool) {
	fields := strings.Fields(left)
	if len(fields) < 2 { return npuSmiDeviceHead{}, false }
	idx, err := strconv.Atoi(fields[0])
	if err != nil { return npuSmiDeviceHead{}, false }
	chipName := fields[1]
	if chipName == "Name" || chipName == "Chip" { return npuSmiDeviceHead{}, false }
	// Exclude pure numeric chip names (e.g. "0" from 310P tail line "0    0")
	if _, parseErr := strconv.Atoi(chipName); parseErr == nil {
		return npuSmiDeviceHead{}, false
	}
	return npuSmiDeviceHead{index: idx, chipName: chipName}, true
}

func npuSmiParseDeviceMemCapacity(right string) (int64, bool) {
	pairs := usagePairPattern.FindAllStringSubmatch(right, -1)
	if len(pairs) == 0 { return 0, false }
	last := pairs[len(pairs)-1]
	totalMB, err := strconv.ParseInt(last[2], 10, 64)
	if err != nil || totalMB <= 0 { return 0, false }
	return totalMB, true
}

func npuSmiParseTableLine(line string) []string {
	if !strings.HasPrefix(line, "|") { return nil }
	parts := strings.Split(line, "|")
	if len(parts) < 3 { return nil }
	columns := make([]string, 0, len(parts)-2)
	for i := 1; i < len(parts)-1; i++ { columns = append(columns, strings.TrimSpace(parts[i])) }
	return columns
}

func destroyVNPU(ref string) {
	parts := strings.Split(ref, ":")
	if len(parts) != 3 { return }
	exec.Command("npu-smi", "destroy-vnpu", "-i", parts[0], "-c", parts[1], "-v", parts[2]).Run()
}

func verifyVNPUCleanedUp(npuID, chipID string) {
	_ = wait.PollUntilContextTimeout(ctx, 5*time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		output, err := exec.Command("npu-smi", "info", "-t", "info-vnpu", "-i", npuID, "-c", chipID).CombinedOutput()
		if err != nil { return false, nil }
		return !strings.Contains(string(output), "vir"), nil
	})
}

func deleteAllResourceSlices() {
	slices, _ := k8sClient.Clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
	for _, rs := range slices.Items {
		if rs.Spec.Driver == driverName {
			_ = k8sClient.Clientset.ResourceV1().ResourceSlices().Delete(ctx, rs.Name, metav1.DeleteOptions{})
		}
	}
}

func cleanupTestEnvironment() {
	cleanupAllTestPods()
	cleanupAllResourceClaimTemplates()
}

var cmBuffer bytes.Buffer

func getPluginPod() *corev1.Pod {
	pods, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).List(
		ctx, metav1.ListOptions{LabelSelector: "app=npu-dra-plugin"})
	Expect(err).NotTo(HaveOccurred(), "获取插件 Pod 列表应当成功")
	Expect(pods.Items).NotTo(BeEmpty(), "应当存在至少一个 npu-dra-plugin Pod")
	return &pods.Items[0]
}

func getPluginLogs(podName, containerName string) string {
	logs, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).GetLogs(
		podName, &corev1.PodLogOptions{Container: containerName}).Do(ctx).Raw()
	Expect(err).NotTo(HaveOccurred(), "获取插件 %s 日志应当成功", podName)
	return string(logs)
}
