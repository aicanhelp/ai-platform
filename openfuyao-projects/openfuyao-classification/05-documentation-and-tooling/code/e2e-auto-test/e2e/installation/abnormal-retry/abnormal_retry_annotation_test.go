package abnormal_retry

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

/*
  异常重试处理 — Retry 注解清除 NeedSkip / ManualInterventionRequired（FIT_6 ~ FIT_9）

  业务集群用例：在管理集群上创建业务集群（标签 with-management-cluster）
  管理集群用例：在引导节点上创建管理集群（标签 post-init）

  CSV 来源：v26.09_Installation_异常重试处理特性级集成测试用例.csv
  合并说明：FIT_6/FIT_8（业务/管理集群清除 NeedSkip）与 FIT_7/FIT_9（业务/管理集群清除 ManualInterventionRequired）
           逻辑相同、集群类型不同，按业务/管理集群分 It 实现，共享 helper。
*/

var _ = SIGDescribe("异常重试-Retry注解清除",
	Label("abnormal-retry", "online"), func() {

		var (
			arCtx          *AbnormalRetryCtx
			clusterName    string
			configPath     string
			nodeConfigPath string
			isMgmtCluster  bool
		)

		BeforeEach(func() {
			arCtx = setupAbnormalRetryFixtures()
			clusterName = ""
			configPath = ""
			nodeConfigPath = ""
			isMgmtCluster = false
		})

		AfterEach(func() {
			if arCtx != nil && clusterName != "" {
				if isMgmtCluster {
					cleanupAbnormalRetryMgmtCluster(arCtx, clusterName, configPath, nodeConfigPath)
				} else {
					cleanupAbnormalRetryCluster(arCtx, clusterName, mgmtKubeconfigPath, configPath, nodeConfigPath)
				}
			}
		})

		Describe("Retry注解清除NeedSkip并重置RetryCount", func() {

			Describe("业务集群", Label("with-management-cluster"), func() {
				/*
				 用例名称：【异常重试处理_FIT_6】Retry注解清除NeedSkip并重置RetryCount（业务集群）
				 用例步骤：1）业务集群中存在 NeedSkip=true 的 worker 节点（RetryCount 已达阈值）
				         2）记录当前 needSkip 和 retryCount 值
				         3）节点故障已修复（patch BKENode 正确密码）
				         4）在 BKECluster 资源上添加 RetryAnnotationKey 注解
				         5）等待控制器 Reconcile 处理 retry 注解
				         6）检查 status.needSkip=false, status.retryCount=0
				         7）检查节点重新进入部署流程（state != InitFailed）
				         8）等待集群最终达到 Healthy/Ready
				 预期结果：1）needSkip=false, retryCount=0
				         2）节点重新进入部署流程
				         3）集群最终达到 Healthy/Ready
				*/
				It("【异常重试处理_FIT_6】Retry注解清除NeedSkip并重置RetryCount（业务集群）",
					Label("required_nodes:2", "异常重试处理_FIT_6", "P1", "v26.09"),
					SpecTimeout(specTimeout),
					func(ctx SpecContext) {
						nodes := requireMinWorkloadNodes(2)

						clusterName = uniqueClusterName("abnretry-fit6")
						master := nodes[0]
						master.Role = []string{"master/node", "etcd"}
						worker := nodes[1]
						worker.Role = []string{"node"}

						By("1. 部署 1M1W 业务集群，worker 注入错误密码，等待 NeedSkip=true")
						originalWorker := injectWrongPassword(&worker)
						configPath, nodeConfigPath = uploadClusterConfig(arCtx, clusterName, []config.NodeInfo{master, worker})
						createClusterInBackground(arCtx, configPath, nodeConfigPath, mgmtKubeconfigPath)

						workerNodeName := waitBKENodeExists(arCtx, clusterName, worker.IP, mgmtKubeconfigPath, retryWaitTimeout)
						waitBKENodeNeedSkip(arCtx, clusterName, workerNodeName, true, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("2. 记录当前 needSkip 和 retryCount 值")
						initialRetryCount, _ := getBKENodeRetryCount(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
						GinkgoWriter.Printf("FIT_6 初始状态: worker=%s needSkip=true retryCount=%d\n",
							workerNodeName, initialRetryCount)

						By("3. 节点故障已修复：patch BKENode 正确密码")
						Expect(patchBKENodePassword(arCtx, clusterName, workerNodeName, originalWorker, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
							"patch BKENode 正确密码应成功")

						By("4. 在 BKECluster 资源上添加 RetryAnnotationKey 注解（指定 worker IP）")
						Expect(addRetryAnnotation(arCtx, clusterName, worker.IP, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
							"添加 retry 注解应成功")

						By("5. 等待控制器处理 retry 注解，检查 needSkip=false")
						waitBKENodeNeedSkip(arCtx, clusterName, workerNodeName, false, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("6. 检查 retryCount=0 或字段消失")
						Eventually(func() bool {
							count, exists := getBKENodeRetryCount(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
							GinkgoWriter.Printf("[%s] retryCount=%d exists=%v (期望 0 或不存在)\n",
								workerNodeName, count, exists)
							return !exists || count == 0
						}, clusterHealthyTimeout, pollInterval).Should(BeTrue(),
							"retryCount 应为 0 或字段消失")

						By("7. 检查节点重新进入部署流程（state != InitFailed）")
						Eventually(func() bool {
							state := getBKENodeState(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
							GinkgoWriter.Printf("[%s] state=%s (期望 != InitFailed)\n", workerNodeName, state)
							return state != bkeNodeStateInitFailed
						}, clusterHealthyTimeout, pollInterval).Should(BeTrue(),
							"节点应离开 InitFailed 状态，重新进入部署流程")

						By("8. 等待集群最终达到 Healthy/Ready")
						waitClusterHealthyReady(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)

						GinkgoWriter.Printf("FIT_6 最终状态: worker=%s needSkip=false retryCount=0 集群=Healthy/Ready\n",
							workerNodeName)
					})
			})

			Describe("管理集群", Label("post-init"), func() {
				/*
				 用例名称：【异常重试处理_FIT_8】Retry注解清除NeedSkip并重置RetryCount（管理集群）
				 用例步骤：1）部署 1M1W 管理集群，worker 注入错误密码，等待 NeedSkip=true
				         2）记录当前 needSkip 和 retryCount 值
				         3）patch 引导节点 BKENode 正确密码
				         4）在引导节点 BKECluster 上添加 retry 注解
				         5）等待引导节点 BC 恢复 Healthy/Ready（worker 重新加入）
				         6）patch 管理集群自管 BC 上的 BKENode 密码 + 添加 retry 注解到自管 BC
				            （自管 BC 的 BKENode 是一次性快照，密码不会从引导节点同步）
				         7）检查 needSkip=false, retryCount=0, state != InitFailed
				         8）等待管理集群完全健康（引导节点 BC + 自管 BC）
				*/
				It("【异常重试处理_FIT_8】Retry注解清除NeedSkip并重置RetryCount（管理集群）",
					Label("required_nodes:2", "异常重试处理_FIT_8", "P1", "v26.09"),
					SpecTimeout(specTimeout),
					func(ctx SpecContext) {
						nodes := requireMinWorkloadNodes(2)

						clusterName = uniqueClusterName("abnretry-fit8")
						isMgmtCluster = true
						master := nodes[0]
						master.Role = []string{"master/node", "etcd"}
						worker := nodes[1]
						worker.Role = []string{"node"}

						By("1. 部署 1M1W 管理集群，worker 注入错误密码，等待 NeedSkip=true")
						originalWorker := injectWrongPassword(&worker)
						configPath, nodeConfigPath = uploadMgmtClusterConfig(arCtx, clusterName, []config.NodeInfo{master, worker})
						createClusterInBackground(arCtx, configPath, nodeConfigPath, mgmtKubeconfigPath)

						workerNodeName := waitBKENodeExists(arCtx, clusterName, worker.IP, mgmtKubeconfigPath, retryWaitTimeout)
						waitBKENodeNeedSkip(arCtx, clusterName, workerNodeName, true, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("2. 记录当前 needSkip 和 retryCount 值")
						initialRetryCount, _ := getBKENodeRetryCount(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
						GinkgoWriter.Printf("FIT_8 初始状态: worker=%s needSkip=true retryCount=%d\n",
							workerNodeName, initialRetryCount)

						By("3. 节点故障已修复：patch 引导节点 BKENode 正确密码")
						Expect(patchBKENodePassword(arCtx, clusterName, workerNodeName, originalWorker, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
							"patch BKENode 正确密码应成功")

						By("4. 在引导节点 BKECluster 资源上添加 RetryAnnotationKey 注解（指定 worker IP）")
						Expect(addRetryAnnotation(arCtx, clusterName, worker.IP, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
							"添加 retry 注解应成功")

						By("5. 等待引导节点 BC 恢复 Healthy/Ready（worker 重新加入后自管 BC 注册）")
						waitClusterHealthyReady(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("6. Patch 管理集群自管 BC 上的 BKENode 密码 + 添加 retry 注解")
						patchMgmtClusterBKENodesAndRetry(arCtx, clusterName, []config.NodeInfo{originalWorker})

						By("7. 检查 needSkip=false, retryCount=0, state != InitFailed")
						waitBKENodeNeedSkip(arCtx, clusterName, workerNodeName, false, mgmtKubeconfigPath, clusterHealthyTimeout)
						Eventually(func() bool {
							count, exists := getBKENodeRetryCount(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
							GinkgoWriter.Printf("[%s] retryCount=%d exists=%v (期望 0 或不存在)\n",
								workerNodeName, count, exists)
							return !exists || count == 0
						}, clusterHealthyTimeout, pollInterval).Should(BeTrue(),
							"retryCount 应为 0 或字段消失")
						Eventually(func() bool {
							state := getBKENodeState(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
							GinkgoWriter.Printf("[%s] state=%s (期望 != InitFailed)\n", workerNodeName, state)
							return state != bkeNodeStateInitFailed
						}, clusterHealthyTimeout, pollInterval).Should(BeTrue(),
							"节点应离开 InitFailed 状态，重新进入部署流程")

						By("8. 等待管理集群完全健康（引导节点 BC + 自管 BC）")
						waitMgmtClusterFullyHealthy(arCtx, clusterName, clusterHealthyTimeout)

						GinkgoWriter.Printf("FIT_8 最终状态: worker=%s needSkip=false retryCount=0 管理集群=Healthy/Ready\n",
							workerNodeName)
					})
			})
		})

		Describe("Retry注解清除ManualInterventionRequired条件", func() {

			Describe("业务集群", Label("with-management-cluster"), func() {
				/*
				 用例名称：【异常重试处理_FIT_7】Retry注解清除ManualInterventionRequired条件（业务集群）
				 用例步骤：1）业务集群已进入 *Failed 终态且已触发 ManualInterventionRequired=True
				         2）确认集群当前有 ManualInterventionRequired=True 条件
				         3）已手动修复集群问题（patch BKENode 正确密码）
				         4）在 BKECluster 资源上添加 RetryAnnotationKey 注解
				         5）等待控制器 Reconcile 处理 retry 注解
				         6）检查 ManualInterventionRequired 条件 status=False, message=Manual intervention resolved
				         7）等待集群最终达到 Healthy/Ready
				 预期结果：1）ManualInterventionRequired 条件 status=False
				         2）集群最终达到 Healthy/Ready
				*/
				It("【异常重试处理_FIT_7】Retry注解清除ManualInterventionRequired条件（业务集群）",
					Label("required_nodes:1", "异常重试处理_FIT_7", "P1", "v26.09"),
					SpecTimeout(specTimeout),
					func(ctx SpecContext) {
						nodes := requireMinWorkloadNodes(1)

						clusterName = uniqueClusterName("abnretry-fit7")
						master := nodes[0]
						master.Role = []string{"master/node", "etcd"}

						By("1. 部署 1Master 业务集群，master 注入错误密码，等待集群 *Failed + ManualInterventionRequired=True")
						originalMaster := injectWrongPassword(&master)
						configPath, nodeConfigPath = uploadClusterConfig(arCtx, clusterName, []config.NodeInfo{master})
						createClusterInBackground(arCtx, configPath, nodeConfigPath, mgmtKubeconfigPath)

						masterNodeName := waitBKENodeExists(arCtx, clusterName, master.IP, mgmtKubeconfigPath, retryWaitTimeout)
						waitBKENodeRetryCountEquals(arCtx, clusterName, masterNodeName, maxRetryCount, mgmtKubeconfigPath, clusterHealthyTimeout)
						waitClusterStatusContains(arCtx, clusterName, failedSuffix, mgmtKubeconfigPath, clusterHealthyTimeout)
						waitManualInterventionRequired(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("2. 确认集群当前有 ManualInterventionRequired=True 条件")
						Expect(hasManualInterventionRequired(arCtx, clusterName, mgmtKubeconfigPath)).To(BeTrue(),
							"集群应存在 ManualInterventionRequired=True 条件")
						reason := getManualInterventionRequiredReason(arCtx, clusterName, mgmtKubeconfigPath)
						GinkgoWriter.Printf("FIT_7 确认: ManualInterventionRequired=True reason=%s\n", reason)

						By("3. 已手动修复集群问题：patch BKENode 正确密码")
						Expect(patchBKENodePassword(arCtx, clusterName, masterNodeName, originalMaster, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
							"patch BKENode 正确密码应成功")

						By("4. 在 BKECluster 资源上添加 RetryAnnotationKey 注解（指定 master IP）")
						Expect(addRetryAnnotation(arCtx, clusterName, master.IP, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
							"添加 retry 注解应成功")

						By("5. 等待控制器处理 retry 注解，检查 ManualInterventionRequired status=False")
						waitManualInterventionRequiredCleared(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("6. 检查 ManualInterventionRequired message 为 Manual intervention resolved")
						msg := getManualInterventionRequiredMessage(arCtx, clusterName, mgmtKubeconfigPath)
						Expect(msg).To(ContainSubstring("resolved"),
							"ManualInterventionRequired message 应包含 resolved, 实际: %s", msg)

						By("7. 等待集群最终达到 Healthy/Ready")
						waitClusterHealthyReady(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)

						GinkgoWriter.Printf("FIT_7 最终状态: 集群=Healthy/Ready ManualInterventionRequired=false\n")
					})
			})

			Describe("管理集群", Label("post-init"), func() {
				/*
				 用例名称：【异常重试处理_FIT_9】Retry注解清除ManualInterventionRequired条件（管理集群）
				 用例步骤：1）部署 1Master 管理集群，master 注入错误密码，等待集群 *Failed + ManualInterventionRequired=True
				         2）确认 ManualInterventionRequired=True
				         3）patch 引导节点 BKENode 正确密码
				         4）在引导节点 BKECluster 上添加 retry 注解
				         5）等待 ManualInterventionRequired 清除
				         6）等待引导节点 BC 恢复 Healthy/Ready
				         7）patch 管理集群自管 BC 上的 BKENode 密码 + 添加 retry 注解到自管 BC
				            （自管 BC 的 BKENode 是一次性快照，密码不会从引导节点同步）
				         8）等待管理集群完全健康（引导节点 BC + 自管 BC）
				*/
				It("【异常重试处理_FIT_9】Retry注解清除ManualInterventionRequired条件（管理集群）",
					Label("required_nodes:1", "异常重试处理_FIT_9", "P1", "v26.09"),
					SpecTimeout(specTimeout),
					func(ctx SpecContext) {
						nodes := requireMinWorkloadNodes(1)

						clusterName = uniqueClusterName("abnretry-fit9")
						isMgmtCluster = true
						master := nodes[0]
						master.Role = []string{"master/node", "etcd"}

						By("1. 部署 1Master 管理集群，master 注入错误密码，等待集群 *Failed + ManualInterventionRequired=True")
						originalMaster := injectWrongPassword(&master)
						configPath, nodeConfigPath = uploadMgmtClusterConfig(arCtx, clusterName, []config.NodeInfo{master})
						createClusterInBackground(arCtx, configPath, nodeConfigPath, mgmtKubeconfigPath)

						masterNodeName := waitBKENodeExists(arCtx, clusterName, master.IP, mgmtKubeconfigPath, retryWaitTimeout)
						waitBKENodeRetryCountEquals(arCtx, clusterName, masterNodeName, maxRetryCount, mgmtKubeconfigPath, clusterHealthyTimeout)
						waitClusterStatusContains(arCtx, clusterName, failedSuffix, mgmtKubeconfigPath, clusterHealthyTimeout)
						waitManualInterventionRequired(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("2. 确认集群当前有 ManualInterventionRequired=True 条件")
						Expect(hasManualInterventionRequired(arCtx, clusterName, mgmtKubeconfigPath)).To(BeTrue(),
							"集群应存在 ManualInterventionRequired=True 条件")
						reason := getManualInterventionRequiredReason(arCtx, clusterName, mgmtKubeconfigPath)
						GinkgoWriter.Printf("FIT_9 确认: ManualInterventionRequired=True reason=%s\n", reason)

						By("3. 已手动修复集群问题：patch 引导节点 BKENode 正确密码")
						Expect(patchBKENodePassword(arCtx, clusterName, masterNodeName, originalMaster, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
							"patch BKENode 正确密码应成功")

						By("4. 在引导节点 BKECluster 资源上添加 RetryAnnotationKey 注解（指定 master IP）")
						Expect(addRetryAnnotation(arCtx, clusterName, master.IP, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
							"添加 retry 注解应成功")

						By("5. 等待控制器处理 retry 注解，检查 ManualInterventionRequired status=False")
						waitManualInterventionRequiredCleared(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("6. 检查 ManualInterventionRequired message 为 Manual intervention resolved")
						msg := getManualInterventionRequiredMessage(arCtx, clusterName, mgmtKubeconfigPath)
						Expect(msg).To(ContainSubstring("resolved"),
							"ManualInterventionRequired message 应包含 resolved, 实际: %s", msg)

						By("7. 等待引导节点 BC 恢复 Healthy/Ready（自管 BC 注册后 patch 管理集群 BKENode）")
						waitClusterHealthyReady(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)

						By("8. Patch 管理集群自管 BC 上的 BKENode 密码 + 添加 retry 注解")
						patchMgmtClusterBKENodesAndRetry(arCtx, clusterName, []config.NodeInfo{originalMaster})

						By("9. 等待管理集群完全健康（引导节点 BC + 自管 BC）")
						waitMgmtClusterFullyHealthy(arCtx, clusterName, clusterHealthyTimeout)

						GinkgoWriter.Printf("FIT_9 最终状态: 管理集群=Healthy/Ready ManualInterventionRequired=false\n")
					})
			})
		})
	})
