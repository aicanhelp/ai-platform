package npu_dra_dfx

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
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

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

// 共享常量与全局变量（声明在此非 _test 文件中，使 go build 可编译；
// 变量赋值在 dfx_suite_test.go 的 BeforeSuite 中完成）。

const (
	testNamespace  = "ascend-dra"
	driverName     = "npu.huawei.com"
	testPodImage   = "docker.io/library/ubuntu:22.04"
	kubeConfigPath = "/root/.kube/config"

	pluginNamespace = "npu-dra-driver"

	deviceClassFull       = "full.npu.huawei.com"
	deviceClassHard       = "hard.npu.huawei.com"
	deviceClassElastic    = "elastic.npu.huawei.com"
	deviceClassFixed      = "fixed.npu.huawei.com"
	deviceClassBestEffort = "best-effort.npu.huawei.com"

	pollInterval  = 5 * time.Second
	pollTimeout   = 300 * time.Second
	deleteTimeout = 120 * time.Second

	memCapacity4GiB  = 4 * 1024 * 1024 * 1024
	memCapacity16GiB = 16 * 1024 * 1024 * 1024

	celPhysicalID0 = `device.attributes["npu.huawei.com"].physicalID == 0`

	// checkpoint 文件在 dra-plugin 容器内的路径
	checkpointPath = "/var/lib/kubelet/plugins/npu.huawei.com/checkpoint.json"
	// chip-capabilities ConfigMap 名称（与 helm chart values.yaml 一致）
	chipCapsConfigMap = "vnpu-template-config"
	// dra profile ConfigMap 名称
	draProfileConfigMap = "dra-profile-config"
)

var (
	k8sClient     *k8s.K8SClient
	ctx           context.Context
	cancel        context.CancelFunc
	localNodeName string
)

// klog v2 默认日志行前缀，如 "I0817 11:24:00.123456"
var klogPrefixPattern = regexp.MustCompile(`^[IWEF]\d{4}\s\d{2}:\d{2}:\d{2}\.\d+`)

// ---------------- namespace ----------------

func ensureNamespace() {
	_, err := k8sClient.GetNamespace(ctx, testNamespace)
	if err != nil {
		err = k8sClient.CreateNamespaceIfNotExists(ctx, testNamespace)
		Expect(err).NotTo(HaveOccurred(), "创建 namespace 应当成功")
		GinkgoWriter.Printf("已创建 namespace %s\n", testNamespace)
	}
}

// ---------------- node mode config ----------------

// setNodeVnpuMode 更新 dra-profile-config ConfigMap 并重启插件使节点模式生效。
// 注意: K8s 1.34 PR #133706 bug 导致调度器总检查 npu-0，需确保 npu-0 配目标模式。
func setNodeVnpuMode(nodeName string, configs string) {
	By(fmt.Sprintf("更新 dra-profile-config ConfigMap 并重启 DRA 插件"))
	cm, err := k8sClient.Clientset.CoreV1().ConfigMaps(pluginNamespace).Get(ctx, draProfileConfigMap, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		cm = &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: draProfileConfigMap, Namespace: pluginNamespace},
			Data:       map[string]string{"dra.config": configs},
		}
		_, err = k8sClient.Clientset.CoreV1().ConfigMaps(pluginNamespace).Create(ctx, cm, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred(), "创建 dra-profile-config ConfigMap 应当成功")
	} else {
		Expect(err).NotTo(HaveOccurred(), "获取 dra-profile-config ConfigMap 应当成功")
		cm.Data["dra.config"] = configs
		_, err = k8sClient.Clientset.CoreV1().ConfigMaps(pluginNamespace).Update(ctx, cm, metav1.UpdateOptions{})
		Expect(err).NotTo(HaveOccurred(), "更新 dra-profile-config ConfigMap 应当成功")
	}

	// 清理旧 CDI spec 并重启插件
	_ = exec.Command("rm", "-f", "/var/run/cdi/k8s.npu.huawei.com-*.yaml").Run()
	_ = exec.Command("kubectl", "rollout", "restart", "ds", "npu-dra-plugin", "-n", pluginNamespace).Run()
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

// setShareCount 更新 DaemonSet 的 --share-count 参数并等待 Pod 重启完成。
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

	time.Sleep(20 * time.Second)
	waitForPluginPodReady()
	GinkgoWriter.Printf("DRA 插件已重启，--share-count=%d\n", shareCount)
}

func restoreShareCount() {
	By("恢复 --share-count=16 并重启 DRA 插件")
	setShareCount(16)
}

// ---------------- config builders ----------------

func softConfig(nodeName string, physicalIDs ...int) string {
	ids := ensurePhysicalID0(physicalIDs)
	var lines []string
	lines = append(lines, "softShareMounts:")
	lines = append(lines, softShareMountsLines()...)
	lines = append(lines, fmt.Sprintf("nodes:\n  %s:", nodeName))
	for _, id := range ids {
		lines = append(lines, fmt.Sprintf("    - physicalId: %d", id))
		lines = append(lines, "      vnpuMode: soft")
		lines = append(lines, "      schedulingPolicy: elastic")
	}
	return strings.Join(lines, "\n")
}

func fullConfig(nodeName string, physicalIDs ...int) string {
	ids := ensurePhysicalID0(physicalIDs)
	var lines []string
	lines = append(lines, fmt.Sprintf("nodes:\n  %s:", nodeName))
	for _, id := range ids {
		lines = append(lines, fmt.Sprintf("    - physicalId: %d", id))
		lines = append(lines, "      vnpuMode: full")
	}
	return strings.Join(lines, "\n")
}

func softShareMountsLines() []string {
	return []string{
		"- hostPath: /opt/xpu/lib/libboundscheck.so",
		"  containerPath: /lib/libboundscheck.so",
		"  options: [rw, rbind]",
		"- hostPath: /opt/xpu/bin/npu-monitor",
		"  containerPath: /opt/enpu/vcann-rt/tools/enpu-monitor",
		"  options: [rw, rbind]",
		"- hostPath: /opt/xpu/bin/systemd-detect-virt",
		"  containerPath: /usr/bin/systemd-detect-virt",
		"  options: [rw, rbind]",
	}
}

// ensurePhysicalID0 确保 physicalId=0 在列表中（K8s 1.34 PR #133706 bug workaround）
func ensurePhysicalID0(ids []int) []int {
	for _, id := range ids {
		if id == 0 {
			return ids
		}
	}
	result := []int{0}
	return append(result, ids...)
}

// ---------------- pod builders ----------------

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

func buildFullNPUPod(podName, claimTemplateName string) *corev1.Pod {
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

// ---------------- claim templates ----------------

func createFullClaimTemplate(name string) {
	createClaimTemplate(name, deviceClassFull, nil, "", nil)
}

func createSoftClaimTemplate(name string, memCapacityBytes int64, coreCapacity int, celExpr string) {
	caps := map[resourcev1.QualifiedName]apiresource.Quantity{
		"memCapacity":  apiresource.MustParse(fmt.Sprintf("%d", memCapacityBytes)),
		"coreCapacity": apiresource.MustParse(fmt.Sprintf("%d", coreCapacity)),
	}
	createClaimTemplate(name, deviceClassElastic, caps, celExpr, nil)
}

func createSoftClaimTemplateWithClass(name, deviceClass string, memCapacityBytes int64, coreCapacity int, celExpr string) {
	caps := map[resourcev1.QualifiedName]apiresource.Quantity{
		"memCapacity":  apiresource.MustParse(fmt.Sprintf("%d", memCapacityBytes)),
		"coreCapacity": apiresource.MustParse(fmt.Sprintf("%d", coreCapacity)),
	}
	createClaimTemplate(name, deviceClass, caps, celExpr, nil)
}

func createClaimTemplate(name, deviceClass string, caps map[resourcev1.QualifiedName]apiresource.Quantity, celExpr string, annotations map[string]string) {
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
			ObjectMeta: metav1.ObjectMeta{Annotations: annotations},
			Spec: resourcev1.ResourceClaimSpec{
				Devices: resourcev1.DeviceClaim{
					Requests: []resourcev1.DeviceRequest{{
						Name: "npu",
						Exactly: &resourcev1.ExactDeviceRequest{
							DeviceClassName: deviceClass,
							Count:            1,
							Capacity:         &resourcev1.CapacityRequirements{Requests: caps},
							Selectors:        selectors,
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

// ---------------- pod lifecycle ----------------

func createPodAndWaitRunning(podName, claimTemplate string, podBuilder func(string, string) *corev1.Pod) {
	if podBuilder == nil {
		podBuilder = buildFullNPUPod
	}
	pod := podBuilder(podName, claimTemplate)
	_, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Create(ctx, pod, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred(), "创建 Pod %s 应当成功", podName)
	waitForPodRunning(podName)
}

func waitForPodRunning(podName string) {
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(context.Context) (bool, error) {
		pod, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		if pod.Status.Phase != corev1.PodRunning {
			return false, nil
		}
		for _, c := range pod.Status.ContainerStatuses {
			if !c.Ready {
				return false, nil
			}
		}
		return true, nil
	})
	Expect(err).NotTo(HaveOccurred(), "Pod %s/%s 应当达到 Running 状态", testNamespace, podName)
	GinkgoWriter.Printf("Pod %s/%s 已 Running\n", testNamespace, podName)
}

func deletePodAndWait(podName string) {
	_ = k8sClient.Clientset.CoreV1().Pods(testNamespace).Delete(ctx, podName, metav1.DeleteOptions{})
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(context.Context) (bool, error) {
		_, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Get(ctx, podName, metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	})
	GinkgoWriter.Printf("Pod %s/%s 已删除\n", testNamespace, podName)
}

func deleteResourceClaimTemplateAndWait(name string) {
	_ = k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Delete(ctx, name, metav1.DeleteOptions{})
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(context.Context) (bool, error) {
		_, err := k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err), nil
	})
	GinkgoWriter.Printf("ResourceClaimTemplate %s 已删除\n", name)
}

func cleanupBeforeCreate(resourceName string, deleteFunc, getFunc func() error) {
	_ = deleteFunc()
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(context.Context) (bool, error) {
		err := getFunc()
		return apierrors.IsNotFound(err), nil
	})
}

func cleanupTestEnvironment() {
	pods, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, pod := range pods.Items {
			_ = k8sClient.Clientset.CoreV1().Pods(testNamespace).Delete(ctx, pod.Name, metav1.DeleteOptions{})
		}
	}
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(context.Context) (bool, error) {
		pods, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, nil
		}
		return len(pods.Items) == 0, nil
	})

	templates, err := k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, tpl := range templates.Items {
			_ = k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).Delete(ctx, tpl.Name, metav1.DeleteOptions{})
		}
	}
	_ = wait.PollUntilContextTimeout(ctx, pollInterval, deleteTimeout, true, func(context.Context) (bool, error) {
		templates, err := k8sClient.Clientset.ResourceV1().ResourceClaimTemplates(testNamespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, nil
		}
		return len(templates.Items) == 0, nil
	})
	GinkgoWriter.Printf("已清理 namespace %s 中的 Pod 和 ResourceClaimTemplate\n", testNamespace)
}

// ---------------- exec in pod ----------------

func execCommandInPod(podName, containerName, command string) (string, string, error) {
	req := k8sClient.Clientset.CoreV1().RESTClient().Post().
		Resource("pods").Namespace(testNamespace).Name(podName).SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: containerName, Command: []string{"sh", "-c", command},
			Stdin: false, Stdout: true, Stderr: true,
		}, scheme.ParameterCodec)
	executor, err := remotecommand.NewSPDYExecutor(k8sClient.RestConfig, "POST", req.URL())
	if err != nil {
		return "", "", fmt.Errorf("create SPDY executor: %w", err)
	}
	var stdout, stderr bytes.Buffer
	err = executor.StreamWithContext(ctx, remotecommand.StreamOptions{Stdout: &stdout, Stderr: &stderr})
	return stdout.String(), stderr.String(), err
}

func verifyNPUSMIInPod(podName string) string {
	var stdout string
	_ = wait.PollUntilContextTimeout(ctx, 5*time.Second, 60*time.Second, true, func(context.Context) (bool, error) {
		var err error
		stdout, _, err = execCommandInPod(podName, "app", "npu-smi info")
		return err == nil && stdout != "", nil
	})
	return stdout
}

// ---------------- plugin pod helpers ----------------

func getPluginPodName() string {
	pods, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).List(ctx, metav1.ListOptions{LabelSelector: "name=npu-dra-plugin"})
	Expect(err).NotTo(HaveOccurred(), "列出插件 Pod 应当成功")
	Expect(pods.Items).NotTo(BeEmpty(), "应当存在 npu-dra-plugin Pod")
	return pods.Items[0].Name
}

func waitForPluginPodReady() {
	Eventually(func() bool {
		pods, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).List(ctx, metav1.ListOptions{LabelSelector: "name=npu-dra-plugin"})
		if err != nil || len(pods.Items) == 0 {
			return false
		}
		pod := pods.Items[0]
		if pod.Status.Phase != corev1.PodRunning {
			return false
		}
		for _, c := range pod.Status.ContainerStatuses {
			if !c.Ready {
				return false
			}
		}
		return true
	}, 120*time.Second, pollInterval).Should(BeTrue(), "DRA 插件应当重启完成并 Ready")
}

func restartPluginDaemonSet() {
	By("滚动重启 DRA 插件 DaemonSet")
	_ = exec.Command("kubectl", "rollout", "restart", "ds", "npu-dra-plugin", "-n", pluginNamespace).Run()
	time.Sleep(15 * time.Second)
	waitForPluginPodReady()
}

func getPluginLogs() string {
	podName := getPluginPodName()
	// 不限制 --since，因为插件可能已长时间运行无新日志
	cmd := exec.Command("kubectl", "logs", podName, "-c", "dra-plugin", "-n", pluginNamespace, "--tail=200")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	_ = cmd.Run()
	return stdout.String()
}

func getPluginLogsSinceRestart() string {
	podName := getPluginPodName()
	// 等待新 Pod 的启动日志产生
	time.Sleep(5 * time.Second)
	cmd := exec.Command("kubectl", "logs", podName, "-c", "dra-plugin", "-n", pluginNamespace, "--tail=200")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	_ = cmd.Run()
	return stdout.String()
}

// ---------------- ResourceSlice helpers ----------------

func waitForResourceSlice(nodeName string) *resourcev1.ResourceSlice {
	var found *resourcev1.ResourceSlice
	err := wait.PollUntilContextTimeout(ctx, pollInterval, pollTimeout, true, func(context.Context) (bool, error) {
		slices, err := k8sClient.Clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, nil
		}
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

// getSchedulingPolicy 读取 ResourceSlice 中指定设备的 schedulingPolicy 属性。
func getSchedulingPolicy(nodeName, deviceName string) string {
	rs := waitForResourceSlice(nodeName)
	for _, device := range rs.Spec.Devices {
		if device.Name == deviceName {
			if attr, ok := device.Attributes["schedulingPolicy"]; ok && attr.StringValue != nil {
				return *attr.StringValue
			}
		}
	}
	return ""
}

// ---------------- ConfigMap helpers ----------------

// backupConfigMap 备份指定 ConfigMap 的 yaml 到临时文件，返回文件路径。
func backupConfigMap(name, namespace string) string {
	cmd := exec.Command("kubectl", "get", "configmap", name, "-n", namespace, "-o", "yaml")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	Expect(cmd.Run()).To(Succeed(), "备份 ConfigMap %s/%s 应当成功: %s", namespace, name, stderr.String())
	tmpFile := "/tmp/" + name + "-backup.yaml"
	Expect(os.WriteFile(tmpFile, stdout.Bytes(), 0644)).To(Succeed())
	GinkgoWriter.Printf("已备份 ConfigMap %s/%s 到 %s\n", namespace, name, tmpFile)
	return tmpFile
}

// restoreConfigMapFromBackup 从备份 yaml 恢复 ConfigMap（幂等：先 delete 再 apply，避免 Conflict）。
func restoreConfigMapFromBackup(backupPath string) {
	// 先删除可能残留的 ConfigMap（如果已存在且被修改过）
	delCmd := exec.Command("kubectl", "delete", "configmap", chipCapsConfigMap, "-n", pluginNamespace, "--ignore-not-found=true")
	_ = delCmd.Run()
	// 再从备份创建
	cmd := exec.Command("kubectl", "apply", "-f", backupPath)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	Expect(cmd.Run()).To(Succeed(), "从 %s 恢复 ConfigMap 应当成功: %s", backupPath, stderr.String())
	GinkgoWriter.Printf("已从 %s 恢复 ConfigMap\n", backupPath)
}

// ---------------- checkpoint helpers ----------------

// countPreparedClaims 解析 checkpoint JSON 并返回 preparedClaims map 中的 claim 数量。
func countPreparedClaims(checkpointJSON string) int {
	var cp struct {
		V1 struct {
			PreparedClaims map[string]json.RawMessage `json:"preparedClaims"`
		} `json:"v1"`
	}
	if err := json.Unmarshal([]byte(checkpointJSON), &cp); err != nil {
		GinkgoWriter.Printf("解析 checkpoint JSON 失败: %v\n", err)
		return -1
	}
	return len(cp.V1.PreparedClaims)
}

// readCheckpointInPlugin 在 dra-plugin 容器内读取 checkpoint.json 内容。
func readCheckpointInPlugin() string {
	podName := getPluginPodName()
	stdout, stderr, err := execCommandInPlugin(podName, "cat "+checkpointPath)
	if err != nil {
		GinkgoWriter.Printf("读取 checkpoint 失败: %v, stderr: %s\n", err, stderr)
		return ""
	}
	return stdout
}

// execCommandInPlugin 在 pluginNamespace 的插件 Pod 的 dra-plugin 容器内执行命令。
func execCommandInPlugin(podName, command string) (string, string, error) {
	req := k8sClient.Clientset.CoreV1().RESTClient().Post().
		Resource("pods").Namespace(pluginNamespace).Name(podName).SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: "dra-plugin", Command: []string{"sh", "-c", command},
			Stdin: false, Stdout: true, Stderr: true,
		}, scheme.ParameterCodec)
	executor, err := remotecommand.NewSPDYExecutor(k8sClient.RestConfig, "POST", req.URL())
	if err != nil {
		return "", "", fmt.Errorf("create SPDY executor: %w", err)
	}
	var stdout, stderr bytes.Buffer
	err = executor.StreamWithContext(ctx, remotecommand.StreamOptions{Stdout: &stdout, Stderr: &stderr})
	return stdout.String(), stderr.String(), err
}

// listCDISpecFiles 列出插件 Pod 中 /var/run/cdi/ 目录下所有 CDI spec 文件。
func listCDISpecFiles() []string {
	podName := getPluginPodName()
	stdout, _, _ := execCommandInPlugin(podName, "ls /var/run/cdi/*.yaml 2>/dev/null")
	var files []string
	for _, f := range strings.Split(strings.TrimSpace(stdout), "\n") {
		f = strings.TrimSpace(f)
		if f != "" {
			files = append(files, f)
		}
	}
	return files
}

// diffFileLists 返回 after 中存在但 before 中不存在的文件（新增的文件）。
func diffFileLists(before, after []string) []string {
	beforeSet := make(map[string]bool)
	for _, f := range before {
		beforeSet[f] = true
	}
	var diff []string
	for _, f := range after {
		if !beforeSet[f] {
			diff = append(diff, f)
		}
	}
	return diff
}

// parseNpuName 从 npu-smi info 输出中提取 NPU 型号名（如 "310P3"、"910B4"）。
// 用于比对重启前后 NPU 设备是否一致，忽略 Memory-Usage 等动态值。
func parseNpuName(output string) string {
	for _, line := range strings.Split(output, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "|") {
			continue
		}
		cols := npuSmiParseTableLine(line)
		if len(cols) < 2 {
			continue
		}
		for _, col := range cols[1:] {
			col = strings.TrimSpace(col)
			if strings.Contains(col, "310P") || strings.Contains(col, "910") || strings.Contains(col, "Ascend") {
				return col
			}
		}
	}
	return ""
}

// npuSmiParseTableLine 解析 npu-smi 表格行，返回列切片。
func npuSmiParseTableLine(line string) []string {
	if !strings.HasPrefix(line, "|") {
		return nil
	}
	parts := strings.Split(line, "|")
	if len(parts) < 3 {
		return nil
	}
	cols := make([]string, 0, len(parts)-2)
	for i := 1; i < len(parts)-1; i++ {
		cols = append(cols, strings.TrimSpace(parts[i]))
	}
	return cols
}

// ---------------- assertions ----------------

// expectKlogPrefix 验证日志行符合 klog v2 单字母前缀格式。
func expectKlogPrefix(logs string) {
	lines := strings.Split(strings.TrimSpace(logs), "\n")
	Expect(lines).NotTo(BeEmpty(), "日志不应当为空")
	matchedCount := 0
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if klogPrefixPattern.MatchString(line) {
			matchedCount++
		}
	}
	Expect(matchedCount).To(BeNumerically(">", 0), "应当至少有一行日志符合 klog v2 前缀格式 (如 I0817 11:24:00.123)")
	GinkgoWriter.Printf("klog 前缀格式校验通过: %d/%d 行匹配\n", matchedCount, len(lines))
}

// expectVnpuID 验证 Pod 内 ASCEND_VNPU_ID 环境变量等于期望值。
func expectVnpuID(podName, expected string) {
	stdout, _, _ := execCommandInPod(podName, "app", "echo $ASCEND_VNPU_ID")
	actual := strings.TrimSpace(stdout)
	Expect(actual).To(Equal(expected), "Pod %s 的 ASCEND_VNPU_ID 应当为 %s", podName, expected)
	GinkgoWriter.Printf("Pod %s ASCEND_VNPU_ID=%s\n", podName, actual)
}

// strconv import guard (used by parseNpuSmiInfo in parent but kept for future use)
var _ = strconv.Itoa
