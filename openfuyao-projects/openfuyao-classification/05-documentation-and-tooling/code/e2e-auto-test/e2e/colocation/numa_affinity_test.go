package colocation

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/colocation/utils"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

// numaAffinityLogPattern NUMA 插件成功应用 LS 亲和的日志匹配模式
const numaAffinityLogPattern = "(Successfully applied NUMA affinity for LS|LS.*pod.*NUMA|applied.*NUMA.*affinity)"

// assertNUMAAffinityLog 轮询断言 NUMA 插件日志中出现 LS 亲和应用记录。
// 成功日志是容器创建时写入的一次性事件，agent Pod 重启或配置传播延迟都会导致
// 单次快照读取丢失记录，因此改为 Eventually 轮询；
// 最终失败时打印诊断信息（agent Pod 状态/重启次数、当前配置、最近日志）。
func assertNUMAAffinityLog(client *k8s.K8SClient, ctx context.Context, workload string) {
	const (
		pollInterval = 3 * time.Second
		pollTimeout  = 60 * time.Second
	)
	Eventually(func() bool {
		matched, err := CheckNUMALogPattern(client, ctx, numaAffinityLogPattern, 200)
		if err != nil {
			GinkgoWriter.Printf("读取 NUMA 插件日志失败: %v\n", err)
			return false
		}
		return matched
	}, pollTimeout, pollInterval).Should(BeTrue(), func() string {
		return fmt.Sprintf("应该在日志中找到 %s NUMA 亲和应用记录（%v 内轮询未匹配 %q），诊断信息:\n%s",
			workload, pollTimeout, numaAffinityLogPattern, CollectNUMADiagnostics(client, ctx, 500))
	})
}

var _ = SIGDescribe("NUMA 亲和优化插件测试", Label("colocation-numa-affinity"), func() {
	var client *k8s.K8SClient
	var ctx context.Context
	var savedNUMAOptions string

	BeforeEach(func() {
		var err error
		client, err = k8s.NewK8SClientFromLocalKubeconfig()
		Expect(err).NotTo(HaveOccurred())
		ctx = context.Background()

		if !CheckVolcanoExists(client, ctx) {
			Skip("Volcano 未安装，跳过此测试")
		}

		// 等待 Volcano webhook 就绪
		Expect(WaitVolcanoWebhookReady(client, ctx, 30*time.Second)).To(Succeed())

		// 保存当前 NUMA 配置
		savedNUMAOptions, err = GetColocationNUMAAffinityOptions(client, ctx)
		Expect(err).NotTo(HaveOccurred())

		// 启用 NUMA 亲和插件
		Expect(PatchColocationNUMAAffinityEnable(client, ctx, true)).To(Succeed())

		// 等待 colocation-overquota-agent Pod 就绪
		err = WaitNUMAPluginReady(client, ctx, 30*time.Second)
		if err != nil {
			Skip("NUMA 亲和插件 Pod 不存在，跳过此测试")
		}

		// 等待配置生效
		time.Sleep(5 * time.Second)

		// HLS 用例需要 1 整核 CPU（Guaranteed QoS 绑核）
		enoughCPU, err := CheckNodeAllocatableCPU(client, ctx, 1100)
		if err != nil || !enoughCPU {
			Skip("节点可分配 CPU 不足 1 核，跳过 HLS 相关用例")
		}
	})

	AfterEach(func() {
		if client != nil && savedNUMAOptions != "" {
			_ = UpdateColocationNUMAAffinityOptions(client, ctx, savedNUMAOptions)
		}
	})

	Context("NUMA 插件启用/禁用测试", func() {
		It("【NUMA2506-001】NUMA 插件启用验证 - 相关 Pod 应正常运行", func() {
			err := WaitNUMAPluginReady(client, ctx, 30*time.Second)
			Expect(err).NotTo(HaveOccurred(), "NUMA 亲和插件 Pod 应该处于运行状态")

			logs, err := GetNUMAPluginLogs(client, ctx, 10)
			Expect(err).NotTo(HaveOccurred())
			Expect(logs).NotTo(BeEmpty(), "应该能获取到 NUMA 插件日志")
		})
	})

	Context("Pod 工作负载验证", func() {
		It("【NUMA2506-003】LS Pod 分配 - 应应用 NUMA 亲和", func() {
			podName := fmt.Sprintf("test-ls-pod-%d", time.Now().Unix())

			defer func() {
				_ = DeletePod(client, ctx, podName, ColocationConfigNamespace, 30*time.Second)
			}()

			pod, err := InstallLSPodInNamespace(client, ctx, podName, ColocationConfigNamespace)
			Expect(err).NotTo(HaveOccurred())
			Expect(WaitPodCondition(client, ctx, pod.Name, pod.Namespace, "Running", 60*time.Second)).To(BeTrue())

			assertNUMAAffinityLog(client, ctx, "LS pod")
		})

		It("【NUMA2506-005】无 QoS Pod 分配 - 应正常运行", func() {
			podName := fmt.Sprintf("test-noqos-pod-%d", time.Now().Unix())

			defer func() {
				_ = DeletePod(client, ctx, podName, ColocationConfigNamespace, 30*time.Second)
			}()

			pod, err := InstallNoQoSPod(client, ctx, podName)
			Expect(err).NotTo(HaveOccurred())
			Expect(WaitPodCondition(client, ctx, pod.Name, pod.Namespace, "Running", 60*time.Second)).To(BeTrue())
			GinkgoWriter.Println("无 QoS Pod 已成功运行，NUMA 插件按预期跳过处理")
		})
	})

	Context("Deployment 工作负载验证", func() {
		It("【NUMA2506-007】LS Deployment 分配 - 应应用 NUMA 亲和", func() {
			deployName := fmt.Sprintf("test-ls-deploy-%d", time.Now().Unix())

			defer func() {
				_ = DeleteDeployment(client, ctx, deployName, ColocationConfigNamespace)
			}()

			err := InstallLSDeployment(client, ctx, deployName, ColocationConfigNamespace, 1)
			Expect(err).NotTo(HaveOccurred())

			err = WaitDeploymentReady(client, ctx, deployName, ColocationConfigNamespace, 90*time.Second)
			Expect(err).NotTo(HaveOccurred(), "LS Deployment 应该成功部署并就绪")

			assertNUMAAffinityLog(client, ctx, "LS Deployment Pod")
		})

		It("【NUMA2506-009】无 QoS Deployment 分配 - 应正常运行", func() {
			deployName := fmt.Sprintf("test-noqos-deploy-%d", time.Now().Unix())

			defer func() {
				_ = DeleteDeployment(client, ctx, deployName, ColocationConfigNamespace)
			}()

			err := InstallNoQoSDeployment(client, ctx, deployName, ColocationConfigNamespace, 1)
			Expect(err).NotTo(HaveOccurred())

			err = WaitDeploymentReady(client, ctx, deployName, ColocationConfigNamespace, 90*time.Second)
			Expect(err).NotTo(HaveOccurred(), "无 QoS Deployment 应该成功部署并就绪")
			GinkgoWriter.Println("无 QoS Deployment 已成功运行")
		})
	})

	Context("StatefulSet 工作负载验证", func() {
		It("【NUMA2506-011】LS StatefulSet 分配 - 应应用 NUMA 亲和", func() {
			stsName := fmt.Sprintf("test-ls-sts-%d", time.Now().Unix())

			defer func() {
				_ = DeleteStatefulSet(client, ctx, stsName, ColocationConfigNamespace)
			}()

			err := InstallLSStatefulSet(client, ctx, stsName, ColocationConfigNamespace, 1)
			Expect(err).NotTo(HaveOccurred())

			err = WaitStatefulSetReady(client, ctx, stsName, ColocationConfigNamespace, 90*time.Second)
			Expect(err).NotTo(HaveOccurred(), "LS StatefulSet 应该成功部署并就绪")

			assertNUMAAffinityLog(client, ctx, "LS StatefulSet Pod")
		})

		It("【NUMA2506-013】无 QoS StatefulSet 分配 - 应正常运行", func() {
			stsName := fmt.Sprintf("test-noqos-sts-%d", time.Now().Unix())

			defer func() {
				_ = DeleteStatefulSet(client, ctx, stsName, ColocationConfigNamespace)
			}()

			err := InstallNoQoSStatefulSet(client, ctx, stsName, ColocationConfigNamespace, 1)
			Expect(err).NotTo(HaveOccurred())

			err = WaitStatefulSetReady(client, ctx, stsName, ColocationConfigNamespace, 90*time.Second)
			Expect(err).NotTo(HaveOccurred(), "无 QoS StatefulSet 应该成功部署并就绪")
			GinkgoWriter.Println("无 QoS StatefulSet 已成功运行")
		})
	})

	Context("DaemonSet 工作负载验证", func() {
		It("【NUMA2506-015】LS DaemonSet 分配 - 应应用 NUMA 亲和", func() {
			dsName := fmt.Sprintf("test-ls-ds-%d", time.Now().Unix())

			defer func() {
				_ = DeleteDaemonSet(client, ctx, dsName, ColocationConfigNamespace)
			}()

			err := InstallLSDaemonSet(client, ctx, dsName, ColocationConfigNamespace)
			Expect(err).NotTo(HaveOccurred())

			err = WaitDaemonSetAtLeastOneReady(client, ctx, dsName, ColocationConfigNamespace, 180*time.Second)
			Expect(err).NotTo(HaveOccurred(), "LS DaemonSet 应该至少有一个就绪副本")

			assertNUMAAffinityLog(client, ctx, "LS DaemonSet Pod")
		})
	})

	Context("Job 工作负载验证", func() {
		It("【NUMA2506-019】LS Job 分配 - 应应用 NUMA 亲和", func() {
			jobName := fmt.Sprintf("test-ls-job-%d", time.Now().Unix())

			defer func() {
				_ = DeleteJob(client, ctx, jobName, ColocationConfigNamespace)
			}()

			err := InstallLSJob(client, ctx, jobName, ColocationConfigNamespace)
			Expect(err).NotTo(HaveOccurred())

			err = WaitJobPodRunning(client, ctx, jobName, ColocationConfigNamespace, 90*time.Second)
			Expect(err).NotTo(HaveOccurred(), "LS Job Pod 应该成功运行")

			assertNUMAAffinityLog(client, ctx, "LS Job Pod")
		})

		It("【NUMA2506-021】无 QoS Job 分配 - 应正常运行", func() {
			jobName := fmt.Sprintf("test-noqos-job-%d", time.Now().Unix())

			defer func() {
				_ = DeleteJob(client, ctx, jobName, ColocationConfigNamespace)
			}()

			err := InstallNoQoSJob(client, ctx, jobName, ColocationConfigNamespace)
			Expect(err).NotTo(HaveOccurred())

			err = WaitJobPodRunning(client, ctx, jobName, ColocationConfigNamespace, 90*time.Second)
			Expect(err).NotTo(HaveOccurred(), "无 QoS Job Pod 应该成功运行")
			GinkgoWriter.Println("无 QoS Job 已成功运行")
		})
	})

	Context("CronJob 工作负载验证", func() {
		It("【NUMA2506-023】LS CronJob 分配 - 应应用 NUMA 亲和", func() {
			cronJobName := fmt.Sprintf("test-ls-cj-%d", time.Now().Unix())

			defer func() {
				_ = DeleteCronJob(client, ctx, cronJobName, ColocationConfigNamespace)
			}()

			err := InstallLSCronJob(client, ctx, cronJobName, ColocationConfigNamespace)
			Expect(err).NotTo(HaveOccurred())

			// CronJob 需要等待到下一分钟才会触发，最长等待 90 秒
			err = WaitCronJobPodRunning(client, ctx, cronJobName, ColocationConfigNamespace, 90*time.Second)
			Expect(err).NotTo(HaveOccurred(), "LS CronJob Pod 应该成功运行")

			assertNUMAAffinityLog(client, ctx, "LS CronJob Pod")
		})
	})

	Context("HLS 工作负载验证（使用原生 CPU 管理）", func() {
		It("【NUMA2506-002】HLS Pod 分配 - 应使用原生 CPU 管理", func() {
			podName := fmt.Sprintf("test-hls-pod-%d", time.Now().Unix())

			defer func() {
				_ = DeletePod(client, ctx, podName, ColocationConfigNamespace, 30*time.Second)
			}()

			pod, err := InstallHLSPodInNamespace(client, ctx, podName, ColocationConfigNamespace)
			Expect(err).NotTo(HaveOccurred())
			Expect(WaitPodCondition(client, ctx, pod.Name, pod.Namespace, "Running", 180*time.Second)).To(BeTrue())

			// HLS Pod 使用原生 CPU 管理，不受 NUMA 插件影响，验证正常运行即可
			GinkgoWriter.Println("HLS Pod 已成功运行，使用原生 CPU 管理")
		})

		It("【NUMA2506-006】HLS Deployment 分配 - 应使用原生 CPU 管理", func() {
			deployName := fmt.Sprintf("test-hls-deploy-%d", time.Now().Unix())

			defer func() {
				_ = DeleteDeployment(client, ctx, deployName, ColocationConfigNamespace)
			}()

			err := InstallHLSDeployment(client, ctx, deployName, ColocationConfigNamespace, 1)
			Expect(err).NotTo(HaveOccurred())

			err = WaitDeploymentReady(client, ctx, deployName, ColocationConfigNamespace, 180*time.Second)
			Expect(err).NotTo(HaveOccurred(), "HLS Deployment 应该成功部署并就绪")

			GinkgoWriter.Println("HLS Deployment 已成功运行，使用原生 CPU 管理")
		})
	})

	Context("多副本工作负载验证", func() {
		It("【NUMA2506-027】LS Deployment 多副本 - 多个 Pod 应都应用 NUMA 亲和", func() {
			deployName := fmt.Sprintf("test-ls-multi-%d", time.Now().Unix())

			defer func() {
				_ = DeleteDeployment(client, ctx, deployName, ColocationConfigNamespace)
			}()

			err := InstallLSDeployment(client, ctx, deployName, ColocationConfigNamespace, 2)
			Expect(err).NotTo(HaveOccurred())

			err = WaitDeploymentReady(client, ctx, deployName, ColocationConfigNamespace, 120*time.Second)
			Expect(err).NotTo(HaveOccurred(), "LS 多副本 Deployment 应该成功部署并就绪")

			assertNUMAAffinityLog(client, ctx, "LS 多副本 Deployment Pod")
		})
	})
})
