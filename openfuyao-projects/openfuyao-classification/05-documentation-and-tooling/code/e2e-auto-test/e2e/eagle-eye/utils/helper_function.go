package utils

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

var DefaultDCMIPath = "/usr/local/Ascend/driver/lib64/driver/libdcmi.so"

func FindNodesNPUSituationEagleEye(clientset *kubernetes.Clientset, ctx context.Context) (map[string]string, error) {
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("list nodes failed: %w", err)
	}
	if len(nodes.Items) == 0 {
		return nil, fmt.Errorf("no nodes in cluster")
	}

	nodeNPUSituation := make(map[string]string)
	for _, node := range nodes.Items {
		nodeName := node.Name
		hasPath, err := nodeHasDCMIPath(clientset, ctx, nodeName, DefaultDCMIPath)
		if err != nil {
			GinkgoWriter.Printf("节点 %s: 检查 DCMI 路径失败: %v\n", nodeName, err)
			nodeNPUSituation[nodeName] = "none"
			continue
		}
		if hasPath {
			GinkgoWriter.Printf("节点 %s: 检测到 DCMI 路径，可采集 HBM/DDR/NPU 指标\n", nodeName)
			nodeNPUSituation[nodeName] = "dcmi"
		} else {
			GinkgoWriter.Printf("节点 %s: 未检测到 DCMI 路径，跳过指标采集\n", nodeName)
			nodeNPUSituation[nodeName] = "none"
		}
	}
	return nodeNPUSituation, nil
}

// nodeHasDCMIPath 在指定节点上通过临时 Pod 检查给定文件路径是否存在。
func nodeHasDCMIPath(clientset *kubernetes.Clientset, ctx context.Context, nodeName string, path string) (bool, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return false, nil
	}
	script := fmt.Sprintf("if test -f %q; then echo RESULT:FOUND; exit 0; else echo RESULT:NOT_FOUND; exit 1; fi", path)

	// Mount host's Ascend path so the container can see DCMI at the same path.
	hostAscendDir := "/usr/local/Ascend"
	podName := fmt.Sprintf("eagle-eye-dcmi-check-%s-%d",
		strings.ToLower(strings.ReplaceAll(nodeName, ".", "-")), time.Now().UnixNano())
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: "default"},
		Spec: corev1.PodSpec{
			NodeName:      nodeName,
			HostPID:       true,
			HostNetwork:   true,
			RestartPolicy: corev1.RestartPolicyNever,
			Volumes: []corev1.Volume{
				{
					Name: "host-ascend",
					VolumeSource: corev1.VolumeSource{
						HostPath: &corev1.HostPathVolumeSource{
							Path: hostAscendDir,
							Type: ptrHostPathType(corev1.HostPathDirectoryOrCreate),
						},
					},
				},
			},
			Containers: []corev1.Container{
				{
					Name:    "checker",
					Image:   "alpine:latest",
					Command: []string{"/bin/sh", "-c", script},
					VolumeMounts: []corev1.VolumeMount{
						{Name: "host-ascend", MountPath: hostAscendDir, ReadOnly: true},
					},
					SecurityContext: &corev1.SecurityContext{
						Privileged: ptr(true),
					},
				},
			},
			Tolerations: []corev1.Toleration{{Operator: corev1.TolerationOpExists}},
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

	req := clientset.CoreV1().Pods("default").GetLogs(podName, &corev1.PodLogOptions{})
	logBytes, err := req.DoRaw(ctx)
	if err != nil {
		return false, fmt.Errorf("get pod logs: %w", err)
	}

	logs := string(logBytes)
	GinkgoWriter.Printf("节点 %s 检查日志:\n%s\n", nodeName, logs)

	// 判断是否找到设备
	if strings.Contains(logs, "RESULT:FOUND") {
		GinkgoWriter.Printf("节点 %s: 找到 %s 设备\n", nodeName)
		return true, nil
	} else if strings.Contains(logs, "RESULT:NOT_FOUND") {
		GinkgoWriter.Printf("节点 %s: 未找到 %s 设备\n", nodeName)
		return false, nil
	}

	return false, fmt.Errorf("节点 %s: 检查超时或无法确定结果", nodeName)
}

func ptr(b bool) *bool { return &b }

func ptrHostPathType(t corev1.HostPathType) *corev1.HostPathType { return &t }

// HelmInstall runs "helm install <release> <chart> -n <ns> --create-namespace --wait --timeout 5m".
// Pass additional flags (e.g. "--version", "1.0.0", "--set", "enabled=true") via extraArgs.
func HelmInstall(releaseName, chartURL, namespace string, extraArgs ...string) {
	args := []string{"install", releaseName, chartURL, "-n", namespace, "--create-namespace", "--wait", "--timeout", "30m","--set","tracing.enabled=false"}
	args = append(args, extraArgs...)
	cmd := exec.Command("helm", args...)
	out, err := cmd.CombinedOutput()
	GinkgoWriter.Printf("helm install %s output:\n%s\n", releaseName, string(out))
	Expect(err).NotTo(HaveOccurred(), "helm install %s should succeed\noutput:\n%s", releaseName, string(out))
}

// HelmUninstall runs "helm uninstall <release> -n <ns> --wait", silently ignoring "release: not found".
func HelmUninstall(releaseName, namespace string) {
	GinkgoWriter.Printf("helm uninstall %s -n %s\n", releaseName, namespace)
	cmd := exec.Command("helm", "uninstall", releaseName, "-n", namespace, "--wait")
	out, err := cmd.CombinedOutput()
	if err != nil && strings.Contains(string(out), "release: not found") {
		GinkgoWriter.Printf("release %s not found (already uninstalled)\n", releaseName)
		return
	}
	if err != nil {
		GinkgoWriter.Printf("helm uninstall %s failed (continuing): %v. Output: %s\n", releaseName, err, string(out))
	}
}

// UninstallEagleEye uninstalls the Eagle-Eye helm release and waits for the given
// namespaces to be fully deleted before returning.
func UninstallEagleEye(clientset *kubernetes.Clientset, ctx context.Context, releaseName, namespace string, waitNamespaces ...string) {
	HelmUninstall(releaseName, namespace)
	for _, ns := range waitNamespaces {
		WaitForNamespaceGone(clientset, ctx, ns)
	}
}

// WaitForNamespaceGone blocks until the namespace itself is fully deleted.
func WaitForNamespaceGone(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	GinkgoWriter.Printf("Waiting for namespace %s to be deleted...\n", namespace)
	Eventually(func() bool {
		_, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
		return err != nil // namespace gone when Get returns NotFound
	}, 3*time.Minute, 5*time.Second).Should(BeTrue(),
		"namespace %s should be fully deleted", namespace)
}

// InstallEagleEyeIfNeeded installs Eagle-Eye via helm when hardware-monitor pod is not already running.
// Returns true if the install was performed (so the caller can track whether to uninstall later).
func InstallEagleEyeIfNeeded(clientset *kubernetes.Clientset, ctx context.Context,
	releaseName, chartURL, namespace, version string) bool {
	if GetHardwarePod(clientset, ctx, "monitor") != nil {
		return false
	}

	GinkgoWriter.Printf("Eagle-Eye not installed; installing via helm: %s %s -n %s\n",
		releaseName, chartURL, namespace)
	extraArgs := []string{}
	if version != "" {
		extraArgs = append(extraArgs, "--version", version)
	}
	HelmInstall(releaseName, chartURL, namespace, extraArgs...)

	GinkgoWriter.Printf("Waiting for hardware-monitor and hardware-diagnosis pod to be Running...\n")
	Eventually(func() *corev1.Pod {
		return GetHardwarePod(clientset, ctx, "monitor")
	}, 3*time.Minute, 10*time.Second).ShouldNot(BeNil(), "hardware-monitor pod should appear after helm install")

	monitorPod := GetHardwarePod(clientset, ctx, "monitor")
	Expect(monitorPod.Status.Phase).To(Equal(corev1.PodRunning),
		"hardware-monitor pod should be Running; current phase: %s", monitorPod.Status.Phase)

	diagPod := GetHardwarePod(clientset, ctx, "diagnosis")
	Expect(diagPod.Status.Phase).To(Equal(corev1.PodRunning),
		"hardware-diagnosis pod should be Running; current phase: %s", diagPod.Status.Phase)

	GinkgoWriter.Printf("Waiting 60s for hardware-monitor to collect initial metrics...\n")
	time.Sleep(60 * time.Second)
	GinkgoWriter.Printf("Eagle-Eye is installed and Running\n")
	return true
}

// GetHardwarePod returns the first Running pod whose name starts with "hardware-<component>-", or nil.
func GetHardwarePod(clientset *kubernetes.Clientset, ctx context.Context, component string) *corev1.Pod {
	prefix := "hardware-" + component + "-"
	podList, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "list pods should succeed")
	for i := range podList.Items {
		p := &podList.Items[i]
		if strings.HasPrefix(p.Name, prefix) && p.Status.Phase == corev1.PodRunning {
			return p
		}
	}
	return nil
}
