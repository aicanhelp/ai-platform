package abnormal_retry

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

/*
  异常重试处理 — RetryCount 递增、重置与集群失败终态（FIT_1 ~ FIT_5）

  前置：引导节点同机 1Master 管理集群（标签 with-management-cluster）
  运行面：引导节点默认 kubeconfig（管理集群 API），与 concurrent 模块一致

  CSV 来源：v26.09_Installation_异常重试处理特性级集成测试用例.csv
  合并说明：FIT_4（Master BKENode retryCount/needSkip）与 FIT_5（集群 Failed + ManualInterventionRequired）共享同一前置场景，合并为一个 It。
*/

var _ = SIGDescribe("异常重试-RetryCount与失败终态",
	Label("abnormal-retry", "with-management-cluster", "online"), func() {

		var (
			arCtx          *AbnormalRetryCtx
			clusterName    string
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			arCtx = setupAbnormalRetryFixtures()
			clusterName = ""
			configPath = ""
			nodeConfigPath = ""
		})

		AfterEach(func() {
			if arCtx != nil && clusterName != "" {
				cleanupAbnormalRetryCluster(arCtx, clusterName, mgmtKubeconfigPath, configPath, nodeConfigPath)
			}
		})

		Describe("Worker节点异常重试", func() {
			/*
			 用例名称：【异常重试处理_FIT_1】Worker节点推送BKEAgent失败后RetryCount递增
			 用例步骤：1）部署 1M1W 业务集群，worker 节点 BKEAgent 推送失败（SSH 不可达/密码错误）
			         2）等待控制器 Reconcile 重试
			         3）检查 status.retryCount > 0（每次失败后递增 1）
			         4）检查 status.state = InitFailed
			         5）等待达到重试次数上限，检查 worker 节点 needSkip
			 预期结果：1）retryCount 大于 0，每次失败后递增 1
			         2）state 为 InitFailed
			         3）worker 节点有 needSkip
			*/
			It("【异常重试处理_FIT_1】Worker节点推送BKEAgent失败后RetryCount递增",
				Label("required_nodes:2", "异常重试处理_FIT_1", "P0", "v26.09"),
				SpecTimeout(specTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(2)

					clusterName = uniqueClusterName("abnretry-fit1")
					master := nodes[0]
					master.Role = []string{"master/node", "etcd"}
					worker := nodes[1]
					worker.Role = []string{"node"}

					By("1. 部署 1M1W 业务集群，worker 节点注入错误密码（SSH 不可达）")
					injectWrongPassword(&worker)
					configPath, nodeConfigPath = uploadClusterConfig(arCtx, clusterName, []config.NodeInfo{master, worker})
					createClusterInBackground(arCtx, configPath, nodeConfigPath, mgmtKubeconfigPath)

					By("2. 等待控制器 Reconcile 重试，检查 retryCount > 0")
					workerNodeName := waitBKENodeExists(arCtx, clusterName, worker.IP, mgmtKubeconfigPath, retryWaitTimeout)
					waitBKENodeRetryCountGreaterThan(arCtx, clusterName, workerNodeName, 0, mgmtKubeconfigPath, retryWaitTimeout)

					By("3. 检查 status.state 为 InitFailed")
					waitBKENodeState(arCtx, clusterName, workerNodeName, bkeNodeStateInitFailed, mgmtKubeconfigPath, retryWaitTimeout)

					By("4. 等待达到重试次数上限，检查 worker 节点 needSkip")
					waitBKENodeNeedSkip(arCtx, clusterName, workerNodeName, true, mgmtKubeconfigPath, clusterHealthyTimeout)

					retryCount, _ := getBKENodeRetryCount(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
					GinkgoWriter.Printf("FIT_1 最终状态: worker=%s retryCount=%d needSkip=true state=%s\n",
						workerNodeName, retryCount, getBKENodeState(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath))
				})

			/*
			 用例名称：【异常重试处理_FIT_2】Worker节点Ping失败也递增RetryCount
			 用例步骤：1）在 worker 节点上占用 bkeagent 健康监听端口（58080），使 bkeagent 无法启动
			         2）部署 1M1W 业务集群（正确密码），BKEAgent 通过 SSH 推送成功（SSH 走 22 端口不受影响）
			         3）bkeagent 因端口冲突无法正常启动 → 无法连接 K8s API 监听 Command CRD
			         4）控制器设置 NodeAgentPushedFlag 后执行 pingAgent，创建 Ping Command CRD
			         5）bkeagent 收不到 Ping 命令，等待 ping 超时（~5min），retryCount 递增
			         6）检查 status.retryCount > 0
			         7）检查 status.state = InitFailed
			 预期结果：1）retryCount 大于 0（ping 失败后递增）
			         2）state 为 InitFailed

			 方案说明（参考 health_mechanism_test.go 端口占用方案）：
			 - nc -l 58080 占用 bkeagent 健康监听端口 → bkeagent 无法启动监听
			 - SSH 推送不受影响（SSH 走 22 端口），PushedFlag 由 controller 直接设置
			 - ping 超时 = max(5min, N×5s)，硬编码在 agent.go，无注解可调
			 - 已通过 bke.bocloud.com/node-boot-wait-timeout=1m 注解加速 push 阶段
			 - DeferCleanup 确保即使用例失败也一定释放端口，再由 AfterEach 清理集群
			*/
			It("【异常重试处理_FIT_2】Worker节点Ping失败也递增RetryCount",
				Label("required_nodes:2", "异常重试处理_FIT_2", "P0", "v26.09"),
				SpecTimeout(specTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(2)

					clusterName = uniqueClusterName("abnretry-fit2")
					master := nodes[0]
					master.Role = []string{"master/node", "etcd"}
					worker := nodes[1]
					worker.Role = []string{"node"}

					By("1. 在 worker 节点上占用 bkeagent 健康监听端口（58080）")
					occupyPid, err := occupyBkeagentPort(worker)
					Expect(err).NotTo(HaveOccurred(), "占用 58080 端口应成功")
					DeferCleanup(func() {
						releaseBkeagentPort(worker, occupyPid)
					})

					By("2. 部署 1M1W 业务集群（正确密码），BKEAgent 通过 SSH 推送成功")
					configPath, nodeConfigPath = uploadClusterConfig(arCtx, clusterName, []config.NodeInfo{master, worker})
					createClusterInBackground(arCtx, configPath, nodeConfigPath, mgmtKubeconfigPath)

					workerNodeName := waitBKENodeExists(arCtx, clusterName, worker.IP, mgmtKubeconfigPath, retryWaitTimeout)

					By("3. 等待控制器 pingAgent 超时（~5min），检查 retryCount > 0")
					waitBKENodeRetryCountGreaterThan(arCtx, clusterName, workerNodeName, 0, mgmtKubeconfigPath, clusterHealthyTimeout)

					By("4. 检查 status.state 为 InitFailed")
					waitBKENodeState(arCtx, clusterName, workerNodeName, bkeNodeStateInitFailed, mgmtKubeconfigPath, retryWaitTimeout*2)

					retryCount, _ := getBKENodeRetryCount(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
					GinkgoWriter.Printf("FIT_2 最终状态: worker=%s retryCount=%d state=%s\n",
						workerNodeName, retryCount, getBKENodeState(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath))
				})

			/*
			 用例名称：【异常重试处理_FIT_3】Worker节点成功后RetryCount重置为零
			 用例步骤：1）部署 1M1W 业务集群，worker 节点曾推送失败 1-2 次（retryCount > 0，未达阈值）
			         2）记录当前 retryCount 值（应 > 0）
			         3）恢复节点（patch BKENode 正确密码），等待控制器下一次 Reconcile
			         4）检查 status.retryCount 字段消失
			         5）检查 status.state = Ready, status.message = BKEAgent is ready
			         6）检查 stateCode 包含 NodeAgentPushedFlag(1) 和 NodeAgentReadyFlag(2)
			 预期结果：1）无 status.retryCount 字段
			         2）state 为 Ready，message 为 BKEAgent is ready
			         3）stateCode 包含 NodeAgentPushedFlag(1) 和 NodeAgentReadyFlag(2) 标志位
			*/
			It("【异常重试处理_FIT_3】Worker节点成功后RetryCount重置为零",
				Label("required_nodes:2", "异常重试处理_FIT_3", "P0", "v26.09"),
				SpecTimeout(specTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(2)

					clusterName = uniqueClusterName("abnretry-fit3")
					master := nodes[0]
					master.Role = []string{"master/node", "etcd"}
					worker := nodes[1]
					worker.Role = []string{"node"}

					By("1. 部署 1M1W 业务集群，worker 节点注入错误密码（曾推送失败 1-2 次）")
					originalWorker := injectWrongPassword(&worker)
					configPath, nodeConfigPath = uploadClusterConfig(arCtx, clusterName, []config.NodeInfo{master, worker})
					createClusterInBackground(arCtx, configPath, nodeConfigPath, mgmtKubeconfigPath)

					workerNodeName := waitBKENodeExists(arCtx, clusterName, worker.IP, mgmtKubeconfigPath, retryWaitTimeout)

					By("2. 记录当前 retryCount 值（应 > 0）")
					waitBKENodeRetryCountGreaterThan(arCtx, clusterName, workerNodeName, 0, mgmtKubeconfigPath, retryWaitTimeout)
					initialRetryCount, _ := getBKENodeRetryCount(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
					GinkgoWriter.Printf("FIT_3 初始 retryCount=%d (worker=%s)\n", initialRetryCount, workerNodeName)

					By("3. 恢复节点：patch BKENode 正确密码，等待控制器下一次 Reconcile")
					Expect(patchBKENodePassword(arCtx, clusterName, workerNodeName, originalWorker, mgmtKubeconfigPath)).NotTo(HaveOccurred(),
						"patch BKENode 正确密码应成功")

					By("4. 检查 status.retryCount 字段消失")
					waitBKENodeRetryCountAbsent(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath, clusterHealthyTimeout)

					By("5. 检查 stateCode 包含 NodeAgentPushedFlag(1) 和 NodeAgentReadyFlag(2)")
					Eventually(func() bool {
						stateCode := getBKENodeStateCode(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath)
						GinkgoWriter.Printf("[%s] stateCode=%d (期望包含 Pushed=%d 和 Ready=%d)\n",
							workerNodeName, stateCode, nodeAgentPushedFlag, nodeAgentReadyFlag)
						return hasStateCodeFlag(stateCode, nodeAgentPushedFlag) && hasStateCodeFlag(stateCode, nodeAgentReadyFlag)
					}, clusterHealthyTimeout, pollInterval).Should(BeTrue(),
						"stateCode 应包含 NodeAgentPushedFlag(1) 和 NodeAgentReadyFlag(2)")

					GinkgoWriter.Printf("FIT_3 最终状态: worker=%s state=%s message=%s stateCode=%d\n",
						workerNodeName,
						getBKENodeState(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath),
						getBKENodeMessage(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath),
						getBKENodeStateCode(arCtx, clusterName, workerNodeName, mgmtKubeconfigPath))
				})
		})

		Describe("Master节点异常与集群失败终态", func() {
			/*
			 用例名称：【异常重试处理_FIT_4_5】Master节点推送失败触发集群Failed终态与人工干预通知
			           （FIT_4 + FIT_5 合并：共享 Master 推送失败前置场景）
			 用例步骤：1）部署 1Master 业务集群，master 节点 BKEAgent 推送持续失败（SSH 端口不可达）
			         2）等待控制器 Reconcile 重试至少 3 次
			         3）检查 BKENode status.retryCount = 3，无 needSkip 字段（master 不设 NeedSkip）
			         4）检查集群 ClusterStatus 进入 *Failed 终态（如 InitializationFailed）
			         5）检查 BKECluster conditions 中 ManualInterventionRequired=True, reason=RetryExhausted
			         6）检查 Events 中存在 Warning 事件，reason=RetryExhausted
			 预期结果：1）retryCount = 3，无 needSkip（master 节点不设置 NeedSkip）
			         2）clusterStatus 以 Failed 结尾（如 InitializationFailed）
			         3）conditions 中存在 ManualInterventionRequired=True, reason=RetryExhausted
			         4）Events 中存在 Warning 事件，reason=RetryExhausted
			*/
			It("【异常重试处理_FIT_4_5】Master节点推送失败触发集群Failed终态与人工干预通知",
				Label("required_nodes:1", "异常重试处理_FIT_4", "异常重试处理_FIT_5", "P0", "v26.09"),
				SpecTimeout(specTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(1)

					clusterName = uniqueClusterName("abnretry-fit45")
					master := nodes[0]
					master.Role = []string{"master/node", "etcd"}

					By("1. 部署 1Master 业务集群，master 节点注入错误密码（SSH 不可达）")
					injectWrongPassword(&master)
					configPath, nodeConfigPath = uploadClusterConfig(arCtx, clusterName, []config.NodeInfo{master})
					createClusterInBackground(arCtx, configPath, nodeConfigPath, mgmtKubeconfigPath)

					By("2. 等待控制器 Reconcile 重试至少 3 次，检查 retryCount = 3")
					masterNodeName := waitBKENodeExists(arCtx, clusterName, master.IP, mgmtKubeconfigPath, retryWaitTimeout)
					waitBKENodeRetryCountEquals(arCtx, clusterName, masterNodeName, maxRetryCount, mgmtKubeconfigPath, clusterHealthyTimeout)

					By("3. 检查 BKENode 无 needSkip 字段（master 节点不设置 NeedSkip）")
					Expect(getBKENodeNeedSkip(arCtx, clusterName, masterNodeName, mgmtKubeconfigPath)).To(BeFalse(),
						"master 节点不应设置 needSkip")

					By("4. 检查集群 ClusterStatus 进入 *Failed 终态（如 InitializationFailed）")
					waitClusterStatusContains(arCtx, clusterName, failedSuffix, mgmtKubeconfigPath, clusterHealthyTimeout)
					clusterStatus := getBKEClusterStatus(arCtx, clusterName, mgmtKubeconfigPath)
					GinkgoWriter.Printf("FIT_4_5 集群 clusterStatus=%s\n", clusterStatus)

					By("5. 检查 BKECluster conditions 中 ManualInterventionRequired=True, reason=RetryExhausted")
					waitManualInterventionRequired(arCtx, clusterName, mgmtKubeconfigPath, clusterHealthyTimeout)
					Expect(hasManualInterventionRequired(arCtx, clusterName, mgmtKubeconfigPath)).To(BeTrue(),
						"集群应存在 ManualInterventionRequired=True 条件")
					reason := getManualInterventionRequiredReason(arCtx, clusterName, mgmtKubeconfigPath)
					Expect(reason).To(Equal(reasonRetryExhausted),
						"ManualInterventionRequired 的 reason 应为 RetryExhausted")

					By("6. 检查 Events 中存在 Warning 事件，reason=RetryExhausted")
					Eventually(func() bool {
						return hasRetryExhaustedEvent(arCtx, clusterName, mgmtKubeconfigPath)
					}, retryWaitTimeout, pollInterval).Should(BeTrue(),
						"Events 中应存在 reason=RetryExhausted 的事件")

					events := getClusterEvents(arCtx, clusterName, mgmtKubeconfigPath)
					GinkgoWriter.Printf("FIT_4_5 Events:\n%s\n", events)

					retryCount, _ := getBKENodeRetryCount(arCtx, clusterName, masterNodeName, mgmtKubeconfigPath)
					GinkgoWriter.Printf("FIT_4_5 最终状态: master=%s retryCount=%d needSkip=false clusterStatus=%s ManualInterventionRequired=true reason=%s\n",
						masterNodeName, retryCount, clusterStatus, reason)

					By("7. 等待 1 分钟，确保使用错误密码连接节点后 SSH 恢复可达")
					// 使用错误密码连接节点会触发 SSH 连接失败，需要等待一段时间让 SSH 服务恢复可达
					time.Sleep(1 * time.Minute)
				})
		})
	})
