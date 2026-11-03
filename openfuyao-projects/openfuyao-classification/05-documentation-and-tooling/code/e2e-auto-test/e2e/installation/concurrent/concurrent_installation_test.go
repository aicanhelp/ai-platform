package concurrent

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

/*
  安装部署并发场景 — 集群创建与删除（CONC-01 ~ CONC-07）

  前置：引导节点同机 1Master 管理集群（mgmt-1master Job，标签 with-management-cluster）
  运行面：引导节点默认 kubeconfig（管理集群 API），与 multi_image_repo_test.go 一致
*/

var _ = SIGDescribe("安装部署并发场景",
	Label("concurrent", "mgmt-1master", "with-management-cluster", "online"), func() {

		BeforeEach(func() {
			setupConcurrentTestFixtures()
		})

		Describe("集群创建", func() {
			/*
			 用例名称：CONC-01-管理集群上 2 个 1Master 业务集群并发创建成功
			 用例步骤：1）为 wl-a、wl-b 生成 1Master 拓扑 BKECluster + BKENode 配置并上传，预期结果 1、2
			         2）2 个 goroutine 并发调用 CreateClusterWithKubeconfig（声明式 CR apply），预期结果 1
			         3）Eventually 轮询两集群直至 state=Healthy 且 clusterStatus=Ready，预期结果 1
			         4）校验管理集群上 2 个业务 BC、BKENode IP 分别为 TEST_NODE2、TEST_NODE3，预期结果 2、3
			 预期结果：1）两个 1Master 业务集群均创建成功并达 Healthy/Ready
			         2）可见 2 个业务 BC，节点 IP 互不重叠
			         3）节点 IP 均不等于引导节点 GUIDE_NODE_HOST
			*/
			It("CONC-01-管理集群上 2 个 1Master 业务集群并发创建成功",
				Label("required_nodes:2", "concurrent", "multi-cluster-create", "1master", "P0", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(2)

					clusterA := uniqueClusterName("conc-wl-a")
					clusterB := uniqueClusterName("conc-wl-b")
					jobs := []WorkloadClusterJob{}

					registerWorkloadClusterCleanup(&jobs)

					By("为 wl-a、wl-b 生成 1Master 配置并上传")
					masterA := nodes[0]
					masterA.Role = []string{"master/node", "etcd"}
					configA, nodeA, err := uploadWorkloadConfig(clusterA, []config.NodeInfo{masterA})
					Expect(err).NotTo(HaveOccurred())

					masterB := nodes[1]
					masterB.Role = []string{"master/node", "etcd"}
					configB, nodeB, err := uploadWorkloadConfig(clusterB, []config.NodeInfo{masterB})
					Expect(err).NotTo(HaveOccurred())

					jobs = []WorkloadClusterJob{
						{ClusterName: clusterA, ConfigPath: configA, NodeConfigPath: nodeA},
						{ClusterName: clusterB, ConfigPath: configB, NodeConfigPath: nodeB},
					}

					By("并发 CreateClusterWithKubeconfig（声明式 CR apply）")
					runConcurrentOperations(
						func() error {
							return clusterManager.CreateClusterWithKubeconfig(configA, nodeA, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.CreateClusterWithKubeconfig(configB, nodeB, mgmtKubeconfigPath)
						},
					)

					By("Eventually 轮询两集群 Healthy/Ready")
					waitClustersHealthyReady([]string{clusterA, clusterB}, installationItTimeout)

					By("校验 BC 存在且节点 IP 互斥、未占用引导节点")
					assertClusterBCExists(clusterA)
					assertClusterBCExists(clusterB)

					ipsA := getBKENodeIPs(clusterA)
					ipsB := getBKENodeIPs(clusterB)
					Expect(ipsA).To(ContainElement(nodes[0].IP))
					Expect(ipsB).To(ContainElement(nodes[1].IP))
					assertNodeIPsDisjoint(ipsA, ipsB)
					assertIPsNotGuideNode(ipsA, ipsB)
				})

			/*
			 用例名称：CONC-02-管理集群上 2 个 1M1W 业务集群并发创建成功
			 用例步骤：1）PartitionWorkloadNodes 切分 2 组互斥节点（各 1 Master + 1 Worker）并生成配置，预期结果 1、2
			         2）2 个 goroutine 并发 CreateClusterWithKubeconfig，预期结果 1
			         3）Eventually 轮询两集群 Healthy/Ready，预期结果 1
			         4）校验各集群 BKENode 数 = 2，节点 IP 集合无交集，预期结果 2、3
			 预期结果：1）2 个 1M1W 业务集群均 Healthy/Ready
			         2）共 4 个 BKENode 分布在 4 ԰不同业务节点上
			         3）两组节点 IP 无交集
			*/
			It("CONC-02-管理集群上 2 个 1M1W 业务集群并发创建成功",
				Label("required_nodes:4", "concurrent", "multi-cluster-create", "1m1w", "P0", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					allNodes := loadWorkloadNodes()

					groups, err := PartitionWorkloadNodes(allNodes, guideNodeIP, 1, 1, 2)
					Expect(err).NotTo(HaveOccurred(), "L2 环境需要至少 4 SO业务节点")

					clusterA := uniqueClusterName("conc-wl-a")
					clusterB := uniqueClusterName("conc-wl-b")
					jobs := []WorkloadClusterJob{}
					registerWorkloadClusterCleanup(&jobs)

					By("为 wl-a、wl-b 生成 1M1W 配置")
					configA, nodeA, err := uploadWorkloadConfig(clusterA, groups[0])
					Expect(err).NotTo(HaveOccurred())
					configB, nodeB, err := uploadWorkloadConfig(clusterB, groups[1])
					Expect(err).NotTo(HaveOccurred())
					jobs = []WorkloadClusterJob{
						{ClusterName: clusterA, ConfigPath: configA, NodeConfigPath: nodeA},
						{ClusterName: clusterB, ConfigPath: configB, NodeConfigPath: nodeB},
					}

					By("并发 CreateClusterWithKubeconfig")
					runConcurrentOperations(
						func() error {
							return clusterManager.CreateClusterWithKubeconfig(configA, nodeA, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.CreateClusterWithKubeconfig(configB, nodeB, mgmtKubeconfigPath)
						},
					)

					By("Eventually 轮询 Healthy/Ready")
					waitClustersHealthyReady([]string{clusterA, clusterB}, installationItTimeout)

					By("校验各集群 2 个 BKENode，节点 IP 互斥")
					ipsA := getBKENodeIPs(clusterA)
					ipsB := getBKENodeIPs(clusterB)
					Expect(ipsA).To(HaveLen(2))
					Expect(ipsB).To(HaveLen(2))
					assertNodeIPsDisjoint(ipsA, ipsB)
				})

			/*
			 用例名称：CONC-03-管理集群上 3 个业务集群并发创建（节点完全隔离）
			 用例步骤：1）PartitionWorkloadNodes 切分 3 组 1M1W 节点并为 wl-a/b/c 生成配置，预期结果 1、2
			         2）3 个 goroutine 并发 CreateClusterWithKubeconfig，预期结果 1
			         3）Eventually 轮询 3 个集群 Healthy/Ready，预期结果 1
			         4）汇总 3 组 BKENode IP，断言共 6 个节点且无重复，预期结果 2、3
			 预期结果：1）3 个业务集群全部创建成功并达 Healthy/Ready
			         2）6 个节点各归属唯一集群
			         3）无 IP 冲突
			*/
			It("CONC-03-管理集群上 3 个业务集群并发创建（节点完全隔离）",
				Label("skip-temporarily", "concurrent", "multi-cluster-create", "1m1w", "P1", "online"),
				SpecTimeout(concurrentItTimeout),
				func(ctx SpecContext) {
					allNodes := loadWorkloadNodes()

					groups, err := PartitionWorkloadNodes(allNodes, guideNodeIP, 1, 1, 3)
					Expect(err).NotTo(HaveOccurred(), "L2 环境需要至少 6 台业务节点")

					names := []string{
						uniqueClusterName("conc-wl-a"),
						uniqueClusterName("conc-wl-b"),
						uniqueClusterName("conc-wl-c"),
					}
					jobs := make([]WorkloadClusterJob, 0, 3)
					registerWorkloadClusterCleanup(&jobs)

					type createTask struct {
						name, configPath, nodeConfigPath string
					}
					tasks := make([]createTask, 3)
					for i, name := range names {
						cp, np, e := uploadWorkloadConfig(name, groups[i])
						Expect(e).NotTo(HaveOccurred())
						tasks[i] = createTask{name, cp, np}
						jobs = append(jobs, WorkloadClusterJob{ClusterName: name, ConfigPath: cp, NodeConfigPath: np})
					}

					By("3 goroutine 并发 CreateClusterWithKubeconfig")
					ops := make([]func() error, 0, len(tasks))
					for _, t := range tasks {
						task := t
						ops = append(ops, func() error {
							return clusterManager.CreateClusterWithKubeconfig(task.configPath, task.nodeConfigPath, mgmtKubeconfigPath)
						})
					}
					runConcurrentOperations(ops...)

					By("Eventually 3 集群 Healthy/Ready")
					waitClustersHealthyReady(names, concurrentItTimeout)

					By("汇总 BKENode IP，断言 6 个节点无重复")
					allIPs := []string{}
					ipSets := make([][]string, 3)
					for i, name := range names {
						ipSets[i] = getBKENodeIPs(name)
						allIPs = append(allIPs, ipSets[i]...)
					}
					Expect(allIPs).To(HaveLen(6))
					assertNodeIPsDisjoint(ipSets[0], ipSets[1], ipSets[2])
				})

			/*
			 用例名称：CONC-04-管理集群上并发 bke cluster create CLI（不同业务集群配置）
			 用例步骤：1）为 wl-a、wl-b 生成 1Master 配置（-f 集群 YAML、-n 节点 YAML）并上传，预期结果 1、2
			         2）2 个 goroutine 并发 CreateClusterInBackgroundWithKubeconfig（bke cluster create），预期结果 1
			         3）Eventually 轮询两集群 Healthy/Ready，预期结果 1
			         4）校验管理集群上 2 个业务 BC，节点 IP 互斥且未占用引导节点，预期结果 2、3
			 预期结果：1）通过 BKE CLI 并发创建的两业务集群均达 Healthy/Ready
			         2）CLI 编排链路与 CR 路径（CONC-01）均可承受并发
			         3）节点 IP 互斥且未占用引导节点
			*/
			It("CONC-04-管理集群上并发 bke cluster create CLI（不同业务集群配置）",
				Label("required_nodes:2", "concurrent", "cluster-create", "1master", "P0", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(2)

					clusterA := uniqueClusterName("conc-wl-a")
					clusterB := uniqueClusterName("conc-wl-b")
					jobs := []WorkloadClusterJob{}
					registerWorkloadClusterCleanup(&jobs)

					masterA := nodes[0]
					masterA.Role = []string{"master/node", "etcd"}
					configA, nodeA, err := uploadWorkloadConfig(clusterA, []config.NodeInfo{masterA})
					Expect(err).NotTo(HaveOccurred())

					masterB := nodes[1]
					masterB.Role = []string{"master/node", "etcd"}
					configB, nodeB, err := uploadWorkloadConfig(clusterB, []config.NodeInfo{masterB})
					Expect(err).NotTo(HaveOccurred())
					jobs = []WorkloadClusterJob{
						{ClusterName: clusterA, ConfigPath: configA, NodeConfigPath: nodeA},
						{ClusterName: clusterB, ConfigPath: configB, NodeConfigPath: nodeB},
					}

					By("并发 CreateClusterInBackgroundWithKubeconfig（BKE CLI）")
					runConcurrentOperations(
						func() error {
							return clusterManager.CreateClusterInBackgroundWithKubeconfig(configA, nodeA, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.CreateClusterInBackgroundWithKubeconfig(configB, nodeB, mgmtKubeconfigPath)
						},
					)

					By("Eventually 轮询 Healthy/Ready")
					waitClustersHealthyReady([]string{clusterA, clusterB}, installationItTimeout)

					ipsA := getBKENodeIPs(clusterA)
					ipsB := getBKENodeIPs(clusterB)
					assertNodeIPsDisjoint(ipsA, ipsB)
					assertIPsNotGuideNode(ipsA, ipsB)
				})
		})

		Describe("集群删除与竞态", func() {
			/*
			 用例名称：CONC-05-管理集群上 2 个已就绪业务集群并发删除成功
			 用例步骤：1）BeforeEach 创建 wl-a、wl-b（各 1Master）并达 Healthy/Ready，预期结果 1
			         2）2 个 goroutine 并发 DeleteClusterWithKubeconfig，预期结果 1、2
			         3）Eventually 轮询 ClusterExistsWithKubeconfig 均为 false，预期结果 1
			 预期结果：1）两个业务集群均被成功删除，对应 BC 不再存在
			         2）删除过程无 hang
			*/
			It("CONC-05-管理集群上 2 个已就绪业务集群并发删除成功",
				Label("required_nodes:2", "concurrent", "multi-cluster-delete", "delete", "P1", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(2)

					clusterA := uniqueClusterName("conc-wl-a")
					clusterB := uniqueClusterName("conc-wl-b")
					jobA := createAndWait1MasterCluster(clusterA, nodes[0])
					jobB := createAndWait1MasterCluster(clusterB, nodes[1])

					By("并发 DeleteClusterWithKubeconfig")
					deleteClustersConcurrent([]string{clusterA, clusterB})

					By("Eventually 确认 BC 已不存在")
					waitClustersDeleted([]string{clusterA, clusterB}, uninstallTimeout)

					_ = clusterManager.CleanupConfig(jobA.ConfigPath)
					_ = clusterManager.CleanupConfig(jobA.NodeConfigPath)
					_ = clusterManager.CleanupConfig(jobB.ConfigPath)
					_ = clusterManager.CleanupConfig(jobB.NodeConfigPath)
				})

			/*
			 用例名称：CONC-06-业务集群 A 创建过程中对业务集群 B 并发删除
			 用例步骤：1）BeforeEach 创建集群 B（wl-b，1Master，TEST_NODE3）并达 Healthy/Ready，预期结果 1
			         2）为集群 A（wl-a，TEST_NODE2）生成 1Master 配置并上传，预期结果 2
			         3）goroutine-1 调用 CreateClusterInBackgroundWithKubeconfig 创建 A；goroutine-2 删除 B，预期结果 2、3
			         4）Eventually：A 达 Healthy/Ready；B 的 ClusterExistsWithKubeconfig 为 false，预期结果 1、2
			 预期结果：1）创建与删除交叉进行互不影响：A 最终创建成功
			         2）B 最终被删除
			*/
			It("CONC-06-业务集群 A 创建过程中对业务集群 B 并发删除",
				Label("required_nodes:2", "concurrent", "create-delete-race", "P1", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(2)

					clusterB := uniqueClusterName("conc-wl-b")
					jobB := createAndWait1MasterCluster(clusterB, nodes[1])
					DeferCleanup(func() {
						if clusterManager.ClusterExistsWithKubeconfig(clusterB, mgmtKubeconfigPath) {
							cleanupWorkloadClusters([]WorkloadClusterJob{jobB})
						} else {
							_ = clusterManager.CleanupConfig(jobB.ConfigPath)
							_ = clusterManager.CleanupConfig(jobB.NodeConfigPath)
						}
					})

					clusterA := uniqueClusterName("conc-wl-a")
					masterA := nodes[0]
					masterA.Role = []string{"master/node", "etcd"}
					configA, nodeA, err := uploadWorkloadConfig(clusterA, []config.NodeInfo{masterA})
					Expect(err).NotTo(HaveOccurred())
					DeferCleanup(func() {
						if clusterManager.ClusterExistsWithKubeconfig(clusterA, mgmtKubeconfigPath) {
							cleanupWorkloadClusters([]WorkloadClusterJob{
								{ClusterName: clusterA, ConfigPath: configA, NodeConfigPath: nodeA},
							})
						} else {
							_ = clusterManager.CleanupConfig(configA)
							_ = clusterManager.CleanupConfig(nodeA)
						}
					})

					By("goroutine-1 创建集群 A，goroutine-2 删除集群 B")
					runConcurrentOperations(
						func() error {
							return clusterManager.CreateClusterInBackgroundWithKubeconfig(configA, nodeA, mgmtKubeconfigPath)
						},
						func() error {
							return clusterManager.DeleteClusterWithKubeconfig(clusterB, mgmtKubeconfigPath)
						},
					)

					By("A 达 Healthy/Ready，B 已删除")
					waitClusterHealthyReady(clusterA, installationItTimeout)
					waitClusterDeleted(clusterB)
				})

			/*
			 用例名称：CONC-07-两业务集群并发删除后，节点可被新业务集群复用创建
			 用例步骤：1）创建 wl-a、wl-b（各 1Master），分别占用 TEST_NODE2、TEST_NODE3 并 Healthy/Ready，预期结果 1
			         2）2 个 goroutine 并发 DeleteClusterWithKubeconfig 删除 wl-a、wl-b，预期结果 1
			         3）Eventually 等待两集群 BC 均不存在，预期结果 1、2
			         4）使用 TEST_NODE2 创建新业务集群 wl-reuse（1Master）并 Healthy/Ready，预期结果 2、3
			 预期结果：1）并发删除完成后，原节点资源已释放
			         2）新业务集群可在原节点上成功创建并达 Healthy/Ready
			         3）验证删除清理与节点复用链路正确
			*/
			It("CONC-07-两业务集群并发删除后，节点可被新业务集群复用创建",
				Label("required_nodes:2", "concurrent", "cleanup-reuse", "P1", "online"),
				SpecTimeout(installationItTimeout),
				func(ctx SpecContext) {
					nodes := requireMinWorkloadNodes(2)

					clusterA := uniqueClusterName("conc-wl-a")
					clusterB := uniqueClusterName("conc-wl-b")
					jobA := createAndWait1MasterCluster(clusterA, nodes[0])
					jobB := createAndWait1MasterCluster(clusterB, nodes[1])

					By("并发删除 wl-a、wl-b")
					deleteClustersConcurrent([]string{clusterA, clusterB})
					waitClustersDeleted([]string{clusterA, clusterB}, uninstallTimeout)

					_ = clusterManager.CleanupConfig(jobA.ConfigPath)
					_ = clusterManager.CleanupConfig(jobA.NodeConfigPath)
					_ = clusterManager.CleanupConfig(jobB.ConfigPath)
					_ = clusterManager.CleanupConfig(jobB.NodeConfigPath)

					By("使用 TEST_NODE2 创建集群 wl-reuse")
					clusterReuse := uniqueClusterName("conc-wl-reuse")
					jobReuse := createAndWait1MasterCluster(clusterReuse, nodes[0])
					registerWorkloadClusterCleanupJobs([]WorkloadClusterJob{jobReuse})

					ips := getBKENodeIPs(clusterReuse)
					Expect(ips).To(ContainElement(nodes[0].IP))
				})
		})
	})
