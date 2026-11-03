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

// Rubik eviction 水位（与 colocation-config rubik-options 中 cpuevict/memoryevict.threshold 一致）。按需直接改此处。
var (
	evictionCPUThreshold  = 60
	evictionMemThreshold  = 60
	evictionStressCPULoad = 75
	evictionStressMemLoad = 75
)

var _ = SIGDescribe("混部离线负载水位线驱逐测试", Label("colocation-eviction", "skip-temporarily"), func() {
	var client *k8s.K8SClient
	var ctx context.Context
	var stressPod *corev1.Pod

	BeforeEach(func() {
		// 注意：驱逐测试需要通过 exec 进入 Pod 执行 stress 命令来模拟高负载，
		// 这依赖于特定的集群配置和 stress-ng Pod 的正确运行，在自动化环境中不稳定
		// 如需启用，请移除下面的 Skip 并确保 stress-ng Pod 可以正常接收 exec 命令
		Skip("驱逐测试依赖 Pod exec 和 stress 负载模拟，在自动化环境中不稳定，已跳过")

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

	Context("离线负载水位线驱逐开启时的调度", Ordered, func() {
		var savedRubikOptions string
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
				savedRubikOptions = cm.Data[ColocationRubikOptsKey]
			}
			Expect(PatchColocationRubikEviction(orderedClient, orderedCtx, true, evictionCPUThreshold, evictionMemThreshold)).To(Succeed())
			time.Sleep(3 * time.Second)
		})

		BeforeEach(func() {
			// 确保旧的 stress pod 已删除
			_ = EnsureStressPodDeleted(client, ctx, "stress-ng-1", "default", 30*time.Second)

			var instErr error
			stressPod, instErr = InstallStressPod(client, ctx, true)
			Expect(instErr).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			if orderedClient == nil {
				return
			}
			Expect(UpdateColocationRubikOptions(orderedClient, orderedCtx, savedRubikOptions)).To(Succeed())
		})

		It("测试 CPU 达到阈值时的驱逐行为", func() {
			var bePodName, bePodNs string
			var lsPodName, lsPodNs string
			var hlsPodName, hlsPodNs string

			defer func() {
				if client == nil {
					return
				}
				if bePodName != "" {
					_ = DeletePod(client, ctx, bePodName, bePodNs, 30*time.Second)
				}
				if lsPodName != "" {
					_ = DeletePod(client, ctx, lsPodName, lsPodNs, 30*time.Second)
				}
				if hlsPodName != "" {
					_ = DeletePod(client, ctx, hlsPodName, hlsPodNs, 30*time.Second)
				}
			}()

			err := StartStressProcess(client, ctx, stressPod.Name, stressPod.Namespace, StressConfig{
				CPUCores: 4,
				CPULoad:  evictionStressCPULoad,
			})
			Expect(err).NotTo(HaveOccurred())
			time.Sleep(10 * time.Second)

			bePod, err := InstallBEPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			bePodName, bePodNs = bePod.Name, bePod.Namespace
			Expect(WaitPodDelete(client, ctx, bePodName, bePodNs, 10*time.Second)).To(BeTrue())

			lsPod, err := InstallLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			lsPodName, lsPodNs = lsPod.Name, lsPod.Namespace
			Expect(WaitPodCondition(client, ctx, lsPodName, lsPodNs, "Running", 10*time.Second)).To(BeTrue())

			hlsPod, err := InstallHLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			hlsPodName, hlsPodNs = hlsPod.Name, hlsPod.Namespace
			Expect(WaitPodCondition(client, ctx, hlsPodName, hlsPodNs, "Running", 10*time.Second)).To(BeTrue())
		})

		It("测试内存达到阈值时的驱逐行为", func() {
			var bePodName, bePodNs string
			var lsPodName, lsPodNs string
			var hlsPodName, hlsPodNs string

			defer func() {
				if client == nil {
					return
				}
				if bePodName != "" {
					_ = DeletePod(client, ctx, bePodName, bePodNs, 30*time.Second)
				}
				if lsPodName != "" {
					_ = DeletePod(client, ctx, lsPodName, lsPodNs, 30*time.Second)
				}
				if hlsPodName != "" {
					_ = DeletePod(client, ctx, hlsPodName, hlsPodNs, 30*time.Second)
				}
			}()

			err := StartStressProcess(client, ctx, stressPod.Name, stressPod.Namespace, StressConfig{
				VM:      1,
				VMBytes: evictionStressMemLoad,
			})
			Expect(err).NotTo(HaveOccurred())
			time.Sleep(10 * time.Second)

			bePod, err := InstallBEPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			bePodName, bePodNs = bePod.Name, bePod.Namespace
			Expect(WaitPodDelete(client, ctx, bePodName, bePodNs, 10*time.Second)).To(BeTrue())

			lsPod, err := InstallLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			lsPodName, lsPodNs = lsPod.Name, lsPod.Namespace
			Expect(WaitPodCondition(client, ctx, lsPodName, lsPodNs, "Running", 10*time.Second)).To(BeTrue())

			hlsPod, err := InstallHLSPod(client, ctx, "", "")
			Expect(err).NotTo(HaveOccurred())
			hlsPodName, hlsPodNs = hlsPod.Name, hlsPod.Namespace
			Expect(WaitPodCondition(client, ctx, hlsPodName, hlsPodNs, "Running", 10*time.Second)).To(BeTrue())
		})
	})
})
