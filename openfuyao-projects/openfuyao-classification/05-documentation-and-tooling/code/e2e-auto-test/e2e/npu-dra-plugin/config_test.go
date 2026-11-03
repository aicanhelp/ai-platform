package npu_dra_plugin

import (
	"os/exec"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("NPU DRA Plugin 配置边界测试",
	Ordered, Label("npu-dra-plugin", "26.08"), Label("P1"), Label("npu"), func() {

		BeforeEach(func() {
			ensureNamespace()
		})

		AfterEach(func() {
			cleanupTestEnvironment()
			time.Sleep(10 * time.Second)
		})

		It("昇腾-npu-dra-IT-023：share-count设置超过上限", func() {
			if findNPUNode("910B") == "" {
				Skip("未找到 910B 节点，本用例需 910B 芯片（share-count 校验需最新镜像）")
			}
			By("设置 --share-count=200 (超过上限 63)")
			ds, err := k8sClient.Clientset.AppsV1().DaemonSets(pluginNamespace).Get(ctx, "npu-dra-plugin", metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred(), "获取 DRA 插件 DaemonSet 应当成功")

			args := ds.Spec.Template.Spec.Containers[0].Args
			found := false
			for i, arg := range args {
				if strings.HasPrefix(arg, "--share-count=") {
					args[i] = "--share-count=200"
					found = true
					break
				}
			}
			if !found {
				args = append(args, "--share-count=200")
			}
			ds.Spec.Template.Spec.Containers[0].Args = args
			_, err = k8sClient.Clientset.AppsV1().DaemonSets(pluginNamespace).Update(ctx, ds, metav1.UpdateOptions{})
			Expect(err).NotTo(HaveOccurred(), "更新 DaemonSet --share-count=200 应当成功")

			DeferCleanup(func() {
				By("恢复 --share-count=16")
				restoreShareCount()
			})

			By("等待插件 Pod 进入异常状态")
			time.Sleep(30 * time.Second)
			Eventually(func() bool {
				pods, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).List(ctx, metav1.ListOptions{LabelSelector: "app=npu-dra-plugin"})
				if err != nil || len(pods.Items) == 0 {
					return true
				}
				pod := pods.Items[0]
				if pod.Status.Phase != "Running" {
					return true
				}
				for _, cs := range pod.Status.ContainerStatuses {
					if !cs.Ready {
						return true
					}
				}
				return false
			}, 120*time.Second, pollInterval).Should(BeTrue(), "插件应当因 share-count 超限而异常")

			By("验证插件日志包含报错信息")
			pods, _ := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).List(ctx, metav1.ListOptions{LabelSelector: "app=npu-dra-plugin"})
			if len(pods.Items) > 0 {
				logs, err := k8sClient.Clientset.CoreV1().Pods(pluginNamespace).GetLogs(pods.Items[0].Name, &corev1.PodLogOptions{Container: "dra-plugin"}).Do(ctx).Raw()
				if err == nil {
					logStr := string(logs)
					Expect(logStr).To(ContainSubstring("invalid share-count"), "插件日志应包含 share-count 报错")
					Expect(logStr).To(ContainSubstring("must be between 1 and 63"), "插件日志应包含上限范围信息")
				}
			}
			GinkgoWriter.Printf("IT-023 验证通过: share-count=200 时插件异常\n")
		})

		It("昇腾-npu-dra-IT-024：切分模式优先级验证 (ConfigMap > 节点标签 > 默认值)", func() {
			// 清理可能存在的节点标签
			exec.Command("kubectl", "label", "node", localNodeName, "vnpu-mode=full", "--overwrite").Run()
			exec.Command("kubectl", "label", "node", localNodeName, "schedulingPolicy-").Run()

			By("步骤1: 设置节点标签 vnpu-mode=soft, schedulingPolicy=fixed, 作为默认模式")
			exec.Command("kubectl", "label", "node", localNodeName, "vnpu-mode=soft", "schedulingPolicy=fixed", "--overwrite").Run()

			By("步骤2: 重启DRA插件Pod使节点标签生效")
			emptyConfig := "nodes:\n  " + localNodeName + ": []"
			setNodeVnpuMode(localNodeName, emptyConfig)

			By("步骤3: ConfigMap中只设置 physicalId=0 为 soft(elastic), physicalId=1 为 hard, 其余卡不设置")
			mixedConfig := softShareMountsConfig(localNodeName,
				"    - physicalId: 0\n      vnpuMode: soft\n      schedulingPolicy: elastic",
				"    - physicalId: 1\n      vnpuMode: hard",
			)
			setNodeVnpuMode(localNodeName, mixedConfig)

			By("步骤4: 验证 npu-0 为 soft,elastic (ConfigMap 优先于节点标签)")
			Eventually(func() string {
				return getDeviceVnpuMode(0)
			}, 60*time.Second, pollInterval).Should(Equal("soft"), "npu-0 应为 soft (ConfigMap 设置)")
			Eventually(func() string {
				return getDeviceSchedulingPolicy(0)
			}, 60*time.Second, pollInterval).Should(Equal("elastic"), "npu-0 schedulingPolicy 应为 elastic (ConfigMap 设置)")

			By("步骤5: 验证 npu-1 为 hard (ConfigMap 优先于节点标签)")
			Eventually(func() string {
				return getDeviceVnpuMode(1)
			}, 60*time.Second, pollInterval).Should(Equal("hard"), "npu-1 应为 hard (ConfigMap 设置)")

			By("步骤6: 验证 npu-2 为 soft,fixed (未在ConfigMap设置, 使用节点标签)")
			Eventually(func() string {
				return getDeviceVnpuMode(2)
			}, 60*time.Second, pollInterval).Should(Equal("soft"), "npu-2 应为 soft (节点标签 vnpu-mode=soft)")
			Eventually(func() string {
				return getDeviceSchedulingPolicy(2)
			}, 60*time.Second, pollInterval).Should(Equal("fixed"), "npu-2 schedulingPolicy 应为 fixed (节点标签)")

			By("步骤7: 验证 npu-3 为 soft,fixed (同上)")
			Eventually(func() string {
				return getDeviceVnpuMode(3)
			}, 60*time.Second, pollInterval).Should(Equal("soft"), "npu-3 应为 soft (节点标签 vnpu-mode=soft)")
			Eventually(func() string {
				return getDeviceSchedulingPolicy(3)
			}, 60*time.Second, pollInterval).Should(Equal("fixed"), "npu-3 schedulingPolicy 应为 fixed (节点标签)")

			By("步骤8: 清理节点标签, 验证未设置的卡回退到默认 full")
			exec.Command("kubectl", "label", "node", localNodeName, "vnpu-mode-").Run()
			exec.Command("kubectl", "label", "node", localNodeName, "schedulingPolicy-").Run()
			setNodeVnpuMode(localNodeName, emptyConfig)
		Eventually(func() string {
			return getDeviceVnpuMode(2)
		}, 60*time.Second, pollInterval).Should(Equal("full"), "清理节点标签后 npu-2 应回退到默认 full")
		Eventually(func() string {
			return getDeviceVnpuMode(3)
		}, 60*time.Second, pollInterval).Should(Equal("full"), "清理节点标签后 npu-3 应回退到默认 full")

			By("步骤9: 恢复 ConfigMap 为默认配置")
			setNodeVnpuMode(localNodeName, fullConfig(localNodeName, 0))

			GinkgoWriter.Printf("IT-024 验证通过: 优先级 ConfigMap > 节点标签 > 默认值\n")
		})
	})

func getDeviceVnpuMode(physicalID int) string {
	return getDeviceAttr(physicalID, "vnpuMode")
}

func getDeviceSchedulingPolicy(physicalID int) string {
	return getDeviceAttr(physicalID, "schedulingPolicy")
}

func getDeviceAttr(physicalID int, attrName string) string {
	slices, err := k8sClient.Clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
	if err != nil {
		return ""
	}
	for _, rs := range slices.Items {
		if rs.Spec.Driver != driverName {
			continue
		}
		for _, device := range rs.Spec.Devices {
			if attr, ok := device.Attributes["physicalID"]; ok && attr.IntValue != nil {
				if int(*attr.IntValue) == physicalID {
					if val, ok := device.Attributes[resourcev1.QualifiedName(attrName)]; ok && val.StringValue != nil {
						return *val.StringValue
					}
					return ""
				}
			}
		}
	}
	return ""
}

func softShareMountsConfig(nodeName string, cardConfigs ...string) string {
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
	lines = append(lines, "nodes:")
	lines = append(lines, "  "+nodeName+":")
	for _, c := range cardConfigs {
		lines = append(lines, c)
	}
	return strings.Join(lines, "\n")
}
