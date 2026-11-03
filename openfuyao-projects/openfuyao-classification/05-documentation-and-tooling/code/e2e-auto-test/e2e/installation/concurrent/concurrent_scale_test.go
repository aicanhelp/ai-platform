package concurrent

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

/*
  安装部署并发场景 — 扩缩容与交叉场景（CONC-08 ~ CONC-11）

  前置：引导节点同机 1Master 管理集群（mgmt-1master Job，标签 with-management-cluster）
  扩缩容验收仅断言终态（附录 B A3）
*/

var _ = SIGDescribe("安装部署并发场景-扩缩容",
	Label("concurrent", "mgmt-1master", "with-management-cluster", "online"), func() {

		BeforeEach(func() {
			setupConcurrentTestFixtures()
		})

		Describe("扩缩容并发", func() {
			/*
			 用例名称：CONC-08-同一业务集群并发扩容 2 个不同 Worker
			 用例步骤：1）创建 1M1W 业务集群 wl-scale（Master=TEST_NODE2，Worker=TEST_NODE3）并 Healthy/Ready，预期结果 1
			         2）确认当前 BKENode 数为 2，预期结果 1
			         3）2 个 goroutine 并发 ScaleOutNodeWithKubeconfig 扩容 TEST_NODE4、TEST_NODE5，预期结果 2
			         4）Eventually 轮询 Healthy/Ready，BKENode 共 4 个，含 node4、node5，预期结果 1、2、3
			 预期结果：1）并发扩容完成后，集群最终 Healthy/Ready
			         2）节点集合包含原 2 节点 + 新增 2 Worker，无重复 IP
			         3）不断言扩容过程中的中间态或操作顺序
			*/
			It("CONC-08-同一业务集群并发扩容 2 个不同 Worker",
				Label("required_nodes:4", "concurrent", "scale", "scale-out", "P0", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(4)

					clusterName := uniqueClusterName("conc-wl-scale")
					job := createAndWait1M1WCluster(clusterName, []config.NodeInfo{nodes[0], nodes[1]})
					registerScaleWorkloadClusterCleanupJobs([]WorkloadClusterJob{job})

					node4 := nodes[2]
					node4.Role = []string{"node"}
					node5 := nodes[3]
					node5.Role = []string{"node"}

					By("并发 ScaleOutNodeWithKubeconfig")
					runConcurrentOperations(
						func() error {
							return clusterManager.ScaleOutNodeWithKubeconfig(clusterName, node4, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.ScaleOutNodeWithKubeconfig(clusterName, node5, mgmtKubeconfigPath)
						},
					)

					By("Eventually Healthy/Ready，4 个 BKENode 含 node4、node5")
					eventuallyClusterScaleSettled(
						clusterName,
						[]string{nodes[0].IP, nodes[1].IP, nodes[2].IP, nodes[3].IP},
						nil,
						installationItTimeout,
					)
				})

			/*
			 用例名称：CONC-09-同一业务集群并发缩容 2 个不同 Worker
			 用例步骤：1）创建 1M3W 业务集群 wl-scale（占用 TEST_NODE2~5）并 Healthy/Ready，预期结果 1
			         2）确认 BKENode 数为 4 且 Healthy/Ready，预期结果 1
			         3）2 个 goroutine 并发 ScaleInNodeWithKubeconfig 缩容 2 个 Worker，预期结果 2
			         4）Eventually 轮询 Healthy/Ready，BKENode 数为 2，被缩容节点 IP 已移除，预期结果 1、2、3
			 预期结果：1）并发缩容完成后集群 Healthy/Ready
			         2）最终保留 1 Master + 1 Worker，共 2 个 BKENode
			         3）不校验缩容过程中的瞬时状态
			*/
			It("CONC-09-同一业务集群并发缩容 2 个不同 Worker",
				Label("required_nodes:4", "concurrent", "scale", "scale-in", "P1", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(4)

					clusterName := uniqueClusterName("conc-wl-scale")
					master := nodes[0]
					master.Role = []string{"master/node", "etcd"}
					w1, w2, w3 := nodes[1], nodes[2], nodes[3]
					w1.Role = []string{"node"}
					w2.Role = []string{"node"}
					w3.Role = []string{"node"}

					configPath, nodeConfigPath, err := uploadWorkloadConfig(
						clusterName, []config.NodeInfo{master, w1, w2, w3})
					Expect(err).NotTo(HaveOccurred())
					job := WorkloadClusterJob{ClusterName: clusterName, ConfigPath: configPath, NodeConfigPath: nodeConfigPath}
					registerScaleWorkloadClusterCleanupJobs([]WorkloadClusterJob{job})

					Expect(clusterManager.CreateClusterWithKubeconfig(
						configPath, nodeConfigPath, mgmtKubeconfigPath)).To(Succeed())
					waitClusterHealthyReady(clusterName, installationItTimeout)
					Expect(countBKENodes(clusterName)).To(Equal(4))

					By("并发缩容 w2、w3，保留 1 Master + 1 Worker")
					runConcurrentOperations(
						func() error {
							return clusterManager.ScaleInNodeWithKubeconfig(clusterName, w2.IP, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.ScaleInNodeWithKubeconfig(clusterName, w3.IP, mgmtKubeconfigPath)
						},
					)

					By("Eventually 2 个 BKENode，被缩容节点已移除")
					eventuallyClusterScaleSettled(
						clusterName,
						[]string{master.IP, w1.IP},
						[]string{w2.IP, w3.IP},
						installationItTimeout,
					)
				})

			/*
			 用例名称：CONC-10-两业务集群并发缩容与扩容（不同集群、不同节点）
			 用例步骤：1）2 个 goroutine 并发创建 wl-shrink（1M1W，TEST_NODE2/3）与 wl-grow（1M1W，TEST_NODE4/5）并 Healthy/Ready，预期结果 1
			         2）goroutine-1 对 wl-shrink 缩容 TEST_NODE3；goroutine-2 对 wl-grow 扩容 TEST_NODE6，预期结果 2
			         3）Eventually 两集群分别收敛：wl-shrink 为 1M；wl-grow 为 1M2W 含 TEST_NODE6，预期结果 1、2、3
			 预期结果：1）不同业务集群上的缩容与扩容可并发执行，最终均 Healthy/Ready
			         2）wl-shrink 仅保留 Master（TEST_NODE2）；wl-grow 含 Master/原 Worker/新 Worker（TEST_NODE4/5/6）
			         3）节点 IP 分属两集群、互不重叠
			*/
			It("CONC-10-两业务集群并发缩容与扩容（不同集群、不同节点）",
				Label("skip-temporarily", "concurrent", "scale", "scale-race", "P1", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(5)

					jobs := make([]WorkloadClusterJob, 0, 2)
					registerScaleWorkloadClusterCleanup(&jobs)

					clusterShrink := uniqueClusterName("conc-wl-shrink")
					clusterGrow := uniqueClusterName("conc-wl-grow")

					shrinkMaster := nodes[0]
					shrinkMaster.Role = []string{"master/node", "etcd"}
					shrinkWorker := nodes[1]
					shrinkWorker.Role = []string{"node"}
					growMaster := nodes[2]
					growMaster.Role = []string{"master/node", "etcd"}
					growWorker := nodes[3]
					growWorker.Role = []string{"node"}

					By("为 wl-shrink、wl-grow 生成 1M1W 配置")
					configShrink, nodeShrink, err := uploadWorkloadConfig(
						clusterShrink, []config.NodeInfo{shrinkMaster, shrinkWorker})
					Expect(err).NotTo(HaveOccurred())
					configGrow, nodeGrow, err := uploadWorkloadConfig(
						clusterGrow, []config.NodeInfo{growMaster, growWorker})
					Expect(err).NotTo(HaveOccurred())
					jobs = append(jobs,
						WorkloadClusterJob{ClusterName: clusterShrink, ConfigPath: configShrink, NodeConfigPath: nodeShrink},
						WorkloadClusterJob{ClusterName: clusterGrow, ConfigPath: configGrow, NodeConfigPath: nodeGrow},
					)

					By("并发 CreateClusterWithKubeconfig 创建 wl-shrink 与 wl-grow")
					runConcurrentOperations(
						func() error {
							return clusterManager.CreateClusterWithKubeconfig(configShrink, nodeShrink, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.CreateClusterWithKubeconfig(configGrow, nodeGrow, mgmtKubeconfigPath)
						},
					)

					By("Eventually 两集群 Healthy/Ready")
					waitClustersHealthyReady([]string{clusterShrink, clusterGrow}, installationItTimeout)

					scaleOutWorker := nodes[4]
					scaleOutWorker.Role = []string{"node"}

					By("goroutine-1 缩容 wl-shrink（TEST_NODE3），goroutine-2 扩容 wl-grow（TEST_NODE6）")
					runConcurrentOperations(
						func() error {
							return clusterManager.ScaleInNodeWithKubeconfig(clusterShrink, nodes[1].IP, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.ScaleOutNodeWithKubeconfig(clusterGrow, scaleOutWorker, mgmtKubeconfigPath)
						},
					)

					By("Eventually wl-shrink 收敛为 1M（仅 TEST_NODE2）")
					eventuallyClusterScaleSettled(
						clusterShrink,
						[]string{nodes[0].IP},
						[]string{nodes[1].IP},
						installationItTimeout,
					)

					By("Eventually wl-grow 收敛为 1M2W（TEST_NODE4/5/6）")
					eventuallyClusterScaleSettled(
						clusterGrow,
						[]string{nodes[2].IP, nodes[3].IP, scaleOutWorker.IP},
						nil,
						installationItTimeout,
					)

					assertNodeIPsDisjoint(getBKENodeIPs(clusterShrink), getBKENodeIPs(clusterGrow))
				})
		})

		Describe("创建与扩容竞态", func() {
			/*
			 用例名称：CONC-11-业务集群 A 创建过程中对业务集群 B 并发扩容
			 用例步骤：1）创建 1M1W 集群 B（wl-b，TEST_NODE2 Master、TEST_NODE3 Worker）并 Healthy/Ready，预期结果 1
			         2）为集群 A（wl-a，TEST_NODE4 1Master）生成配置并上传，预期结果 2
			         3）goroutine-1 调用 CreateClusterInBackgroundWithKubeconfig 创建 A；goroutine-2 对 B 扩容 TEST_NODE5，预期结果 2、3
			         4）Eventually：A、B 均 Healthy/Ready，B 的 BKENode 含 TEST_NODE5，预期结果 1、2
			 预期结果：1）不同业务集群上的「创建」与「扩容」可并发执行，最终均成功
			         2）A、B 终态均为 Healthy/Ready
			*/
			It("CONC-11-业务集群 A 创建过程中对业务集群 B 并发扩容",
				Label("required_nodes:4", "concurrent", "create-scale-race", "P2", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(4)

					jobs := make([]WorkloadClusterJob, 0, 2)
					registerScaleWorkloadClusterCleanup(&jobs)

					clusterB := uniqueClusterName("conc-wl-b")
					jobB := createAndWait1M1WCluster(clusterB, []config.NodeInfo{nodes[0], nodes[1]})
					jobs = append(jobs, jobB)

					clusterA := uniqueClusterName("conc-wl-a")
					masterA := nodes[2]
					masterA.Role = []string{"master/node", "etcd"}
					configA, nodeA, err := uploadWorkloadConfig(clusterA, []config.NodeInfo{masterA})
					Expect(err).NotTo(HaveOccurred())
					jobs = append(jobs, WorkloadClusterJob{
						ClusterName: clusterA, ConfigPath: configA, NodeConfigPath: nodeA,
					})

					node5 := nodes[3]
					node5.Role = []string{"node"}

					By("goroutine-1 创建集群 A，goroutine-2 对 B 扩容 node5")
					runConcurrentOperations(
						func() error {
							return clusterManager.CreateClusterInBackgroundWithKubeconfig(configA, nodeA, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.ScaleOutNodeWithKubeconfig(clusterB, node5, mgmtKubeconfigPath)
						},
					)

					By("Eventually A 创建完成、B 扩容收敛且含 node5")
					eventuallyClusterScaleSettled(
						clusterA,
						[]string{masterA.IP},
						nil,
						installationItTimeout,
					)
					eventuallyClusterScaleSettled(
						clusterB,
						[]string{nodes[0].IP, nodes[1].IP, node5.IP},
						nil,
						installationItTimeout,
					)
				})
		})
	})
