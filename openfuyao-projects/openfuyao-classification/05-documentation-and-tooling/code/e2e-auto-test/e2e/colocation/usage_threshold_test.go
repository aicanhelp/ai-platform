package colocation

import (
	"context"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/colocation/utils"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

// 真实负载 usageThreshold 开启时的 CPU/内存利用率阈值（百分比）。按需直接改此处。
var (
	usageThresholdCPUPercent    = 60
	usageThresholdMemPercent    = 60
	usageThresholdStressCPULoad = 75
	usageThresholdStressMemLoad = 75
)

var _ = SIGDescribe("混部真实负载调度测试", Label("colocation-usage-threshold", "skip-temporarily"), func() {
	var client *k8s.K8SClient
	var ctx context.Context
	var stressPod *corev1.Pod

	BeforeEach(func() {
		// 注意：真实负载调度测试需要通过 exec 进入 Pod 执行 stress 命令来模拟高负载，
		// 这依赖于特定的集群配置和 stress-ng Pod 的正确运行，在自动化环境中不稳定
		// 如需启用，请移除下面的 Skip 并确保 stress-ng Pod 可以正常接收 exec 命令
		Skip("真实负载调度测试依赖 Pod exec 和 stress 负载模拟，在自动化环境中不稳定，已跳过")

		var err error
		client, err = k8s.NewK8SClientFromLocalKubeconfig()
		Expect(err).NotTo(HaveOccurred())
		ctx = context.Background()

		if !CheckVolcanoExists(client, ctx) {
			Skip("Volcano 未安装，跳过此测试")
		}

		// 等待 Volcano webhook 就绪
		Expect(WaitVolcanoWebhookReady(client, ctx, 30*time.Second)).To(Succeed())
	})

	AfterEach(func() {
		if client != nil {
			_ = StopStressProcess(client, ctx)
		}
		if stressPod != nil && client != nil {
			_ = DeletePod(client, ctx, stressPod.Name, stressPod.Namespace, 30*time.Second)
		}
		stressPod = nil
	})

	Context("真实负载调度关闭时的测试", Ordered, func() {
		var savedVolcanoSchedulerOptions string
		var orderedClient *k8s.K8SClient
		var orderedCtx context.Context

		BeforeAll(func() {
			var err error
			orderedClient, err = k8s.NewK8SClientFromLocalKubeconfig()
			Expect(err).NotTo(HaveOccurred())
			orderedCtx = context.Background()

			cm, err := GetColocationConfigMap(orderedClient, orderedCtx)
			Expect(err).NotTo(HaveOccurred())
			if cm.Data != nil {
				savedVolcanoSchedulerOptions = cm.Data[ColocationVolcanoSchedulerOptsKey]
			}
			Expect(PatchColocationVolcanoSchedulerUsage(orderedClient, orderedCtx, false, 0, 0)).To(Succeed())
			time.Sleep(3 * time.Second)
		})

		BeforeEach(func() {
			// 确保旧的 stress pod 已删除
			_ = EnsureStressPodDeleted(client, ctx, "stress-ng-1", "default", 30*time.Second)

			var instErr error
			stressPod, instErr = InstallStressPod(client, ctx, false)
			Expect(instErr).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			if orderedClient == nil {
				return
			}
			Expect(UpdateColocationVolcanoSchedulerOptions(orderedClient, orderedCtx, savedVolcanoSchedulerOptions)).To(Succeed())
		})

		It("CPU 达到阈值时不应该发生驱逐", func() {
			var podNames []struct {
				name, ns string
			}

			defer func() {
				if client == nil {
					return
				}
				for _, p := range podNames {
					_ = DeletePod(client, ctx, p.name, p.ns, 30*time.Second)
				}
			}()

			err := StartStressProcess(client, ctx, stressPod.Name, stressPod.Namespace, StressConfig{
				CPUCores: 4,
				CPULoad:  usageThresholdStressCPULoad,
			})
			Expect(err).NotTo(HaveOccurred())
			time.Sleep(10 * time.Second)

			bePod, err := InstallBEPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{bePod.Name, bePod.Namespace})
			Expect(WaitPodCondition(client, ctx, bePod.Name, bePod.Namespace, "Running", 60*time.Second)).To(BeTrue())

			lsPod, err := InstallLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{lsPod.Name, lsPod.Namespace})
			Expect(WaitPodCondition(client, ctx, lsPod.Name, lsPod.Namespace, "Running", 60*time.Second)).To(BeTrue())

			hlsPod, err := InstallHLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{hlsPod.Name, hlsPod.Namespace})
			Expect(WaitPodCondition(client, ctx, hlsPod.Name, hlsPod.Namespace, "Running", 60*time.Second)).To(BeTrue())
		})

		It("内存达到阈值时不应该发生驱逐", func() {
			var podNames []struct {
				name, ns string
			}

			defer func() {
				if client == nil {
					return
				}
				for _, p := range podNames {
					_ = DeletePod(client, ctx, p.name, p.ns, 30*time.Second)
				}
			}()

			err := StartStressProcess(client, ctx, stressPod.Name, stressPod.Namespace, StressConfig{
				VM:      1,
				VMBytes: usageThresholdStressMemLoad,
			})
			Expect(err).NotTo(HaveOccurred())
			time.Sleep(10 * time.Second)

			bePod, err := InstallBEPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{bePod.Name, bePod.Namespace})
			Expect(WaitPodCondition(client, ctx, bePod.Name, bePod.Namespace, "Running", 60*time.Second)).To(BeTrue())

			lsPod, err := InstallLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{lsPod.Name, lsPod.Namespace})
			Expect(WaitPodCondition(client, ctx, lsPod.Name, lsPod.Namespace, "Running", 60*time.Second)).To(BeTrue())

			hlsPod, err := InstallHLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{hlsPod.Name, hlsPod.Namespace})
			Expect(WaitPodCondition(client, ctx, hlsPod.Name, hlsPod.Namespace, "Running", 60*time.Second)).To(BeTrue())
		})
	})

	Context("真实负载调度开启时的测试", Ordered, func() {
		var savedVolcanoSchedulerOptions string
		var orderedClient2 *k8s.K8SClient
		var orderedCtx2 context.Context

		BeforeAll(func() {
			var err error
			orderedClient2, err = k8s.NewK8SClientFromLocalKubeconfig()
			Expect(err).NotTo(HaveOccurred())
			orderedCtx2 = context.Background()

			cm, err := GetColocationConfigMap(orderedClient2, orderedCtx2)
			Expect(err).NotTo(HaveOccurred())
			if cm.Data != nil {
				savedVolcanoSchedulerOptions = cm.Data[ColocationVolcanoSchedulerOptsKey]
			}
			Expect(PatchColocationVolcanoSchedulerUsage(orderedClient2, orderedCtx2, true,
				usageThresholdCPUPercent,
				usageThresholdMemPercent,
			)).To(Succeed())
			time.Sleep(3 * time.Second)
		})

		BeforeEach(func() {
			// 确保旧的 stress pod 已删除
			_ = EnsureStressPodDeleted(client, ctx, "stress-ng-1", "default", 30*time.Second)

			var instErr error
			stressPod, instErr = InstallStressPod(client, ctx, false)
			Expect(instErr).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			if orderedClient2 == nil {
				return
			}
			Expect(UpdateColocationVolcanoSchedulerOptions(orderedClient2, orderedCtx2, savedVolcanoSchedulerOptions)).To(Succeed())
		})

		It("CPU 达到阈值时应该阻止调度", func() {
			var podNames []struct {
				name, ns string
			}

			defer func() {
				if client == nil {
					return
				}
				for _, p := range podNames {
					_ = DeletePod(client, ctx, p.name, p.ns, 30*time.Second)
				}
			}()

			err := StartStressProcess(client, ctx, stressPod.Name, stressPod.Namespace, StressConfig{
				CPUCores: 4,
				CPULoad:  usageThresholdStressCPULoad,
			})
			Expect(err).NotTo(HaveOccurred())
			time.Sleep(10 * time.Second)

			bePod, err := InstallBEPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{bePod.Name, bePod.Namespace})
			Expect(WaitPodCondition(client, ctx, bePod.Name, bePod.Namespace, "Pending", 60*time.Second)).To(BeTrue())

			lsPod, err := InstallLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{lsPod.Name, lsPod.Namespace})
			Expect(WaitPodCondition(client, ctx, lsPod.Name, lsPod.Namespace, "Pending", 60*time.Second)).To(BeTrue())

			hlsPod, err := InstallHLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{hlsPod.Name, hlsPod.Namespace})
			Expect(WaitPodCondition(client, ctx, hlsPod.Name, hlsPod.Namespace, "Pending", 60*time.Second)).To(BeTrue())
		})

		It("内存达到阈值时应该阻止调度", func() {
			var podNames []struct {
				name, ns string
			}

			defer func() {
				if client == nil {
					return
				}
				for _, p := range podNames {
					_ = DeletePod(client, ctx, p.name, p.ns, 30*time.Second)
				}
			}()

			err := StartStressProcess(client, ctx, stressPod.Name, stressPod.Namespace, StressConfig{
				VM:      1,
				VMBytes: usageThresholdStressMemLoad,
			})
			Expect(err).NotTo(HaveOccurred())
			time.Sleep(10 * time.Second)

			bePod, err := InstallBEPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{bePod.Name, bePod.Namespace})
			Expect(WaitPodCondition(client, ctx, bePod.Name, bePod.Namespace, "Pending", 60*time.Second)).To(BeTrue())

			lsPod, err := InstallLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{lsPod.Name, lsPod.Namespace})
			Expect(WaitPodCondition(client, ctx, lsPod.Name, lsPod.Namespace, "Pending", 60*time.Second)).To(BeTrue())

			hlsPod, err := InstallHLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			podNames = append(podNames, struct{ name, ns string }{hlsPod.Name, hlsPod.Namespace})
			Expect(WaitPodCondition(client, ctx, hlsPod.Name, hlsPod.Namespace, "Pending", 60*time.Second)).To(BeTrue())
		})
	})
})
