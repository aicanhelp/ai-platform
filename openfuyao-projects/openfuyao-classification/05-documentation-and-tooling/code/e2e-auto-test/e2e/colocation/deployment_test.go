package colocation

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/colocation/utils"
)

var _ = SIGDescribe("混部部署测试", Label("colocation-deployment"), func() {

	It("【在离线混部-005/015/018/021/030】验证HLS业务部署", func() {
		// 前置环境检查
		Expect(CheckVolcanoExists(k8sClient, ctx)).To(BeTrue(), "Volcano scheduler not found in cluster")
		Expect(WaitVolcanoWebhookReady(k8sClient, ctx, 120*time.Second)).NotTo(HaveOccurred(), "Volcano admission webhook not ready")

		// HLS 需要 1 整核 CPU（Guaranteed QoS 绑核）
		enoughCPU, err := CheckNodeAllocatableCPU(k8sClient, ctx, 1100)
		if err != nil || !enoughCPU {
			Skip("节点可分配 CPU 不足 1 核，跳过 HLS 用例")
		}

		// 005-验证HLS（高要求在线业务）为在线业务
		// 015-验证HLS业务正确使用默认priority-hls
		// 018-验证HLS业务默认使用vlocano调度
		// 021-验证HLS业务的资源为requests.limits.cpu和requests.request.cpu相等且为整数（节点有无分配任何资源的CPU核）
		// 030-验证HLS pod通过准入器会增加node.openfuyao.com/colocation节点标签亲和性
		hlsPod, err := InstallHLSPodNamed(k8sClient, ctx, fmt.Sprintf("hls-pod-%d", time.Now().UnixNano()), "", "")
		Expect(err).NotTo(HaveOccurred())
		defer k8sClient.Clientset.CoreV1().Pods(hlsPod.Namespace).Delete(ctx, hlsPod.Name, metav1.DeleteOptions{})

		// 等待Pod达到Running状态，WaitPodCondition内部会自动输出诊断信息
		Expect(WaitPodCondition(k8sClient, ctx, hlsPod.Name, hlsPod.Namespace, "Running", 360*time.Second)).To(BeTrue(), "HLS Pod failed to reach Running state")

		// 获取最新的 Pod 状态
		freshPod, err := k8sClient.Clientset.CoreV1().Pods(hlsPod.Namespace).Get(ctx, hlsPod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		// 验证 QoS 级别
		Expect(freshPod.Status.QOSClass).To(Equal(corev1.PodQOSGuaranteed))

		// 验证注解
		Expect(freshPod.Annotations["openfuyao.com/qos-level"]).To(Equal("HLS"))

		// 验证调度器名称
		Expect(freshPod.Spec.SchedulerName).To(Equal("volcano"))

		// 验证优先级
		Expect(*freshPod.Spec.Priority).To(Equal(int32(10000)))
		Expect(freshPod.Spec.PriorityClassName).To(Equal("priority-hls"))

		// 验证节点亲和性配置
		Expect(freshPod.Spec.Affinity).NotTo(BeNil())
		Expect(freshPod.Spec.Affinity.NodeAffinity).NotTo(BeNil())

		// 验证容器资源限制
		container := freshPod.Spec.Containers[0]
		Expect(container.Resources.Limits.Cpu()).To(Equal(container.Resources.Requests.Cpu()))
	})

	It("【在离线混部-009/016/019/031】验证LS业务部署", func() {
		// 前置环境检查
		Expect(CheckVolcanoExists(k8sClient, ctx)).To(BeTrue(), "Volcano scheduler not found in cluster")
		Expect(WaitVolcanoWebhookReady(k8sClient, ctx, 120*time.Second)).NotTo(HaveOccurred(), "Volcano admission webhook not ready")

		// Test 9, 16, 19, 31
		// 009-验证LS（低要求在线业务）为在线业务
		// 016-验证LS业务正确使用默认priority-ls
		// 019-验证LS业务默认使用vlocano调度
		// 031-验证LS pod通过准入器会增加node.openfuyao.com/colocation节点标签亲和性
		lsPod, err := InstallLSPodNamed(k8sClient, ctx, fmt.Sprintf("ls-pod-%d", time.Now().UnixNano()), "", "")
		Expect(err).NotTo(HaveOccurred())
		defer k8sClient.Clientset.CoreV1().Pods(lsPod.Namespace).Delete(ctx, lsPod.Name, metav1.DeleteOptions{})

		// 等待Pod达到Running状态，WaitPodCondition内部会自动输出诊断信息
		Expect(WaitPodCondition(k8sClient, ctx, lsPod.Name, lsPod.Namespace, "Running", 240*time.Second)).To(BeTrue(), "LS Pod failed to reach Running state")

		freshPod, err := k8sClient.Clientset.CoreV1().Pods(lsPod.Namespace).Get(ctx, lsPod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		Expect(freshPod.Status.QOSClass).To(Equal(corev1.PodQOSBurstable))
		Expect(freshPod.Annotations["openfuyao.com/qos-level"]).To(Equal("LS"))
		Expect(freshPod.Spec.SchedulerName).To(Equal("volcano"))
		Expect(*freshPod.Spec.Priority).To(Equal(int32(1000)))
		Expect(freshPod.Spec.PriorityClassName).To(Equal("priority-ls"))
		Expect(freshPod.Spec.Affinity).NotTo(BeNil())
		Expect(freshPod.Spec.Affinity.NodeAffinity).NotTo(BeNil())
	})

	It("【在离线混部-010/017/020/032/033】验证BE业务部署", func() {
		// 前置环境检查
		Expect(CheckVolcanoExists(k8sClient, ctx)).To(BeTrue(), "Volcano scheduler not found in cluster")
		Expect(WaitVolcanoWebhookReady(k8sClient, ctx, 120*time.Second)).NotTo(HaveOccurred(), "Volcano admission webhook not ready")

		// 010-验证BE（可抢占离线业务）为离线业务
		// 017-验证BE业务正确使用默认priority-be
		// 020-验证BE业务默认使用vlocano调度
		// 032-验证BE pod通过准入器会增加node.openfuyao.com/colocation节点标签亲和性
		// 033-验证BE pod通过准入器会增加volcano.sh/preemptable注解
		bePod, err := InstallBEPodNamed(k8sClient, ctx, fmt.Sprintf("be-pod-%d", time.Now().UnixNano()), "", "")
		Expect(err).NotTo(HaveOccurred())
		defer k8sClient.Clientset.CoreV1().Pods(bePod.Namespace).Delete(ctx, bePod.Name, metav1.DeleteOptions{})

		Expect(WaitPodCondition(k8sClient, ctx, bePod.Name, bePod.Namespace, "Running", 240*time.Second)).To(BeTrue(), "BE Pod failed to reach Running state")

		freshPod, err := k8sClient.Clientset.CoreV1().Pods(bePod.Namespace).Get(ctx, bePod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		Expect(freshPod.Status.QOSClass).To(Equal(corev1.PodQOSBestEffort))
		Expect(freshPod.Annotations["openfuyao.com/qos-level"]).To(Equal("BE"))
		Expect(freshPod.Spec.SchedulerName).To(Equal("volcano"))
		Expect(*freshPod.Spec.Priority).To(Equal(int32(-1000)))
		Expect(freshPod.Spec.PriorityClassName).To(Equal("priority-be"))
		Expect(freshPod.Spec.Affinity).NotTo(BeNil())
		Expect(freshPod.Spec.Affinity.NodeAffinity).NotTo(BeNil())
		Expect(freshPod.Annotations["volcano.sh/preemptable"]).To(Equal("true"))
		Expect(freshPod.Annotations).To(HaveKey("openfuyao.com/extender-resource-cfg"))
	})

	It("【在离线混部-011/012/013/014】验证默认 PriorityClass 资源存在性", func() {
		// 011-验证默认 PriorityClass 资源存在性
		// 012-验证默认 PriorityClass 资源存在性
		// 013-验证默认 PriorityClass 资源存在性
		// 014-验证默认 PriorityClass 资源存在性
		pcList, err := ListPriorityClasses(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		expectedPriorities := map[string]int32{
			"priority-hls": 10000,
			"priority-ls":  1000,
			"priority-be":  -1000,
		}

		foundPriorities := make(map[string]int32)
		for _, item := range pcList.Items {
			name := item.GetName()
			value, _, _ := unstructured.NestedInt64(item.Object, "value")
			foundPriorities[name] = int32(value)
		}

		for name, value := range expectedPriorities {
			Expect(foundPriorities).To(HaveKeyWithValue(name, value))
		}
	})

	It("【在离线混部-029】colocation-configmap优化", func() {
		cm, err := GetColocationConfigMap(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		Expect(cm.Data).To(HaveKey("colocation-options"))
		Expect(cm.Data).To(HaveKey("rubik-options"))
		Expect(cm.Data).To(HaveKey("volcano-scheduler-options"))
		Expect(cm.Data).NotTo(HaveKey("overread_namespaced_config_mapsubscription-options"))
	})
})
