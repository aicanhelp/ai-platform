package npu_dra_dfx

import (
	"os"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("NPU DRA Plugin 可诊断性测试 (DFD)",
	Ordered, Label("npu-dra-plugin", "26.06"), Label("npu"), Label("dfd"), func() {

		BeforeEach(func() {
			ensureNamespace()
		})

		AfterEach(func() {
			cleanupTestEnvironment()
			time.Sleep(10 * time.Second)
		})

		It("昇腾-npu-dra-DFX-003：插件启动日志包含设备发现信息且使用 klog 级别前缀", Label("P1"), func() {
			By("获取插件 Pod 名称: kubectl get pods -n npu-dra-driver -l name=npu-dra-plugin")
			podName := getPluginPodName()
			GinkgoWriter.Printf("插件 Pod: %s\n", podName)

			By("获取插件容器日志: kubectl logs <pod-name> -c dra-plugin -n npu-dra-driver")
			logs := getPluginLogs()

			By("验证日志非空")
			Expect(logs).NotTo(BeEmpty(), "插件日志不应当为空")

			By("验证日志包含设备发现关键字")
			Expect(strings.Contains(logs, "Device discovery finished") ||
				strings.Contains(logs, "Enumerated device") ||
				strings.Contains(logs, "Device enumeration complete")).To(BeTrue(),
				"日志应当包含设备发现关键字 (Device discovery finished / Enumerated device / Device enumeration complete)")

			By("验证日志行使用 klog 标准级别前缀 (单字母 I/W/E/F 开头，后跟月日时间戳)")
			expectKlogPrefix(logs)

			By("验证日志包含节点名或设备数量信息")
			Expect(strings.Contains(logs, "node=") || strings.Contains(logs, "devices=") ||
				strings.Contains(logs, localNodeName)).To(BeTrue(),
				"日志应当包含 node 名称或 devices 数量信息")

			GinkgoWriter.Printf("DFX-003 验证通过: 插件启动日志包含设备发现信息且符合 klog v2 格式\n")
		})

		It("昇腾-npu-dra-DFX-005：ConfigMap 加载失败时插件降级使用默认模板并记录告警日志", Label("P2"), func() {
			By("前置：确认 chip-capabilities ConfigMap (vnpu-template-config) 存在")
			_, err := k8sClient.Clientset.CoreV1().ConfigMaps(pluginNamespace).Get(ctx, chipCapsConfigMap, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred(), "chip-capabilities ConfigMap %s 应当存在", chipCapsConfigMap)

			By("备份 ConfigMap 以便恢复")
			backupPath := backupConfigMap(chipCapsConfigMap, pluginNamespace)
			DeferCleanup(func() {
				By("恢复 ConfigMap 并重启插件")
				restoreConfigMapFromBackup(backupPath)
				restartPluginDaemonSet()
				_ = os.Remove(backupPath)
			})

			By("删除 chip-capabilities ConfigMap: kubectl delete configmap vnpu-template-config -n npu-dra-driver")
			err = k8sClient.Clientset.CoreV1().ConfigMaps(pluginNamespace).Delete(ctx, chipCapsConfigMap, metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred(), "删除 ConfigMap 应当成功")
			GinkgoWriter.Printf("ConfigMap %s/%s 已删除\n", pluginNamespace, chipCapsConfigMap)

			By("重启插件 DaemonSet 使 ConfigMap 加载失败")
			restartPluginDaemonSet()

			By("获取插件日志（重启后部分）: kubectl logs <pod> -c dra-plugin -n npu-dra-driver --tail=200")
			logs := getPluginLogsSinceRestart()
			Expect(logs).NotTo(BeEmpty(), "重启后插件日志不应当为空")

			By("验证日志包含降级告警关键字")
			Expect(strings.Contains(logs, "failed to get ConfigMap")).To(BeTrue(),
				"日志应当包含 klog.Warningf 输出的 'failed to get ConfigMap' 关键字")
			Expect(strings.Contains(logs, "using default vnpu templates")).To(BeTrue(),
				"日志应当包含 'using default vnpu templates' 关键字")

			By("验证插件仍正常运行")
			waitForPluginPodReady()

			By("验证 ResourceSlice 已重新上报 (降级后使用内置默认模板，不影响设备发现)")
			rs := waitForResourceSlice(localNodeName)
			Expect(len(rs.Spec.Devices)).To(BeNumerically(">", 0),
				"降级后 ResourceSlice 中设备数量应当大于 0")

			By("恢复 ConfigMap 并重启插件")
			restoreConfigMapFromBackup(backupPath)
			restartPluginDaemonSet()

			By("验证恢复后日志包含 'loaded vnpu templates from ConfigMap' (Infof 级别)，无告警")
			recoveredLogs := getPluginLogsSinceRestart()
			Expect(strings.Contains(recoveredLogs, "loaded vnpu templates from ConfigMap")).To(BeTrue(),
				"ConfigMap 恢复后重启插件，日志应当包含 'loaded vnpu templates from ConfigMap' (Infof 级别)")

			GinkgoWriter.Printf("DFX-005 验证通过: ConfigMap 加载失败时降级告警正常，恢复后正常加载\n")

			// 防止 lint 未使用引用告警
			_ = func() (*corev1.ConfigMap, error) {
				return nil, apierrors.NewNotFound(corev1.Resource("configmaps"), "unused")
			}
		})
	})
