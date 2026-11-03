package system_integration

import (
	"context"
	"os"
	"strconv"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/system-integration/utils"
	"k8s.io/client-go/dynamic"
)

var (
	sshClient     *executor.SSHExecutor
	nodesshClient *executor.SSHExecutor
	ctx           context.Context
	cancel        context.CancelFunc
	dynamicClient dynamic.Interface
)

var _ = Describe("anomaly scene: guide installation reboot", Label("guide-reboot", "pre-init", "skip-temporarily", "online"), func() {

	BeforeEach(func() {
		// context
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

		// ssh client
		var err error
		port, _ := strconv.Atoi(os.Getenv("TEST_NODE1_PORT"))
		sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE1_IP"), port, os.Getenv("TEST_NODE1_USER"), os.Getenv("TEST_NODE1_PASSWORD"))
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client should be created successfully")
	})

	AfterEach(func() {
		err := utils.DoBkeReset(sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "bke reset command should be succeeded")
	})

	It("reboot during guide_installation", func() {

		var err error

		// step1: do reboot after some time
		GinkgoWriter.Println("=====step1: do reboot after some time=====")
		err = utils.SystemReboot(sshClient, 4)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "shutdown command should be exexuted successfully")
		GinkgoWriter.Println("system will restart in 4mins...")

		// step2: do bke init at background
		GinkgoWriter.Println("=====step2: do bke init at background=====")
		err = utils.DoGuideInstallationAtBackground(sshClient, "--otherRepo cr.openfuyao.cn/openfuyao/bke-online-installed:1.2.2", "--confirm")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "guide installation should be successful")
		GinkgoWriter.Println("start guide installation at background...")

		time.Sleep(5 * time.Minute)

		// step3: rebuild sshclient after system restarted
		GinkgoWriter.Println("=====step3: rebuild sshclient after system restarted=====")
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		defer cancel()

		port, _ := strconv.Atoi(os.Getenv("TEST_NODE1_PORT"))
		sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE1_IP"), port, os.Getenv("TEST_NODE1_USER"), os.Getenv("TEST_NODE1_PASSWORD"))
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client should be created successfully")
		GinkgoWriter.Println("ssh client recreated successfully")

		// step4: do bke init after system restarted
		GinkgoWriter.Println("=====step4: do bke init after system restarted======")
		err = utils.DoGuideInstallationAtBackground(sshClient, "--otherRepo cr.openfuyao.cn/openfuyao/bke-online-installed:1.2.2", "--confirm")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "guide installation should be successful")
		GinkgoWriter.Println("guide installation start...")

		time.Sleep(10 * time.Minute)

		// step5: guide installation verification
		GinkgoWriter.Println("=====step5: guide installation verification=====")
		checkRes, err := utils.InstallationCheck(sshClient, ctx)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(checkRes).NotTo(BeTrue(), "guide installation should not be successful")
		GinkgoWriter.Println("guide installation failed as expected")
	})

	It("reboot after guide installation", func() {

		var err error

		// step1: do bke init
		GinkgoWriter.Println("=====step1: do bke init=====")
		err = utils.DoGuideInstallation(sshClient, "--otherRepo cr.openfuyao.cn/openfuyao/bke-online-installed:1.2.2", "--confirm")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "guide installation should be successful")
		GinkgoWriter.Println("guide installation succeeded")

		// step2: system restart
		GinkgoWriter.Println("=====step2: system restart=====")
		err = utils.SystemReboot(sshClient, 0)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "system restart should be succeeded")
		time.Sleep(8 * time.Minute)
		GinkgoWriter.Println("system has been restarted")

		// step3: rebuild ssh client
		GinkgoWriter.Println("=====step3: rebuild sshclient after system restarted=====")
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		defer cancel()

		port, _ := strconv.Atoi(os.Getenv("TEST_NODE1_PORT"))
		sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE1_IP"), port, os.Getenv("TEST_NODE1_USER"), os.Getenv("TEST_NODE1_PASSWORD"))
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client should be created successfully")
		GinkgoWriter.Println("ssh client recreated successfully")

		// step5: guide installation verification
		GinkgoWriter.Println("=====guide installation verification=====")
		checkRes, err := utils.InstallationCheck(sshClient, ctx)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(checkRes).To(BeTrue(), "guide installation should not be healthy after system started")
		GinkgoWriter.Println("guide installation is healthy as expected")
	})
})

var _ = Describe("anomaly scene: cluster installation reboot", Label("cluster-reboot", "post-init", "skip-temporarily", "online"), func() {

	BeforeEach(func() {
		// 默认引导节点已初始化
		// guide ssh client
		var err error
		port, _ := strconv.Atoi(os.Getenv("TEST_NODE1_PORT"))
		sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE1_IP"), port, os.Getenv("TEST_NODE1_USER"), os.Getenv("TEST_NODE1_PASSWORD"))
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client should be created successfully")

		// guide dynamic client
		dynamicClient, err = k8s.NewK8SDynamicClientViaSSH(sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "dynamic client should be create successfully")

	})

	AfterEach(func() {

		var err error

		// cluster reset
		GinkgoWriter.Println("start reset cluster....")
		err = utils.ClusterReset(sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "cluster reset should be succeeded")

		// cluster reset check
		Eventually(func() bool {
			checkRes, err := utils.ClusterResetCheck(sshClient, "bke-cluster", "bke-cluster")
			if err != nil {
				GinkgoWriter.Println(err)
			}
			return checkRes
		}, 10*time.Minute, 1*time.Second).Should(BeTrue(), "cluster reset should be succeeded")
		GinkgoWriter.Println("cluster reset succeeded")

		// remove cluster config
		err = utils.RemoveClusterConfig(sshClient, "1master-cluster.yaml", "/tmp/")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		err = utils.RemoveClusterConfig(sshClient, "bke-nodes.yaml", "/tmp/")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "remove cluster config should be succeeded")
		GinkgoWriter.Println("cluster config remove succeeded")
	})

	It("reboot in EnsureBKEAgent phase", func() {

		var err error

		// step1: copy cluster config
		GinkgoWriter.Println("=====step1: copy cluster config=====")
		err = utils.CopyClusterConfig(sshClient, "1master-cluster.yaml", "/tmp")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "copy command should be succeeded")
		GinkgoWriter.Println("copy command succeeded")

		// step2: edit cluster config
		GinkgoWriter.Println("=====step2: edit cluster config=====")
		err = utils.ClusterConfigdeletion(sshClient, []string{"cluster-api", "openfuyao-system-controller"}, "/tmp/1master-cluster.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "deletion command should be succeeded")

		nodes := []utils.NodeInfo{}

		var node1 utils.NodeInfo
		node1.Hostname = os.Getenv("TEST_NODE2_HOSTNAME")
		node1.IP = os.Getenv("TEST_NODE2_IP")
		node1.Password = os.Getenv("TEST_NODE2_PASSWORD")
		node1.Port = os.Getenv("TEST_NODE2_PORT")
		node1.Role = []string{"master/node", "etcd"}
		node1.Username = os.Getenv("TEST_NODE2_USER")
		nodes = append(nodes, node1)

		port, _ := strconv.Atoi(nodes[len(nodes)-1].Port)
		nodesshClient, err = executor.NewSSHExecutor(nodes[len(nodes)-1].IP, port, nodes[len(nodes)-1].Username, nodes[len(nodes)-1].Password)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "node ssh Client should be created successfully")

		// err = utils.CleanClusterConfigNodes(sshClient, "/tmp/1master-cluster.yaml")
		// if err != nil {
		// 	GinkgoWriter.Println(err)
		// }
		// Expect(err).NotTo(HaveOccurred(), "clean cluster config nodes should be succeeded")

		err = utils.GenerateBKENodeConfig(sshClient, nodes, "/tmp/bke-nodes.yaml", "bke-cluster")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "generate BKENode config should be succeeded")
		GinkgoWriter.Println("cluster config edit completed")

		// step3: cluster create
		GinkgoWriter.Println("=====step3: cluster create=====")
		GinkgoWriter.Println("start creating cluster...")
		err = utils.ClusterCreateAtBackground(sshClient, "/tmp/1master-cluster.yaml", "/tmp/bke-nodes.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		time.Sleep(5 * time.Second)
		Expect(err).NotTo(HaveOccurred(), "cluster create should be successfully")

		start_time := time.Now()
		timeout := 3 * time.Minute
		deploy_flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("bc deploy out of time")
				break
			}
			deploy_flag = utils.BkeClusterDeployCheck(sshClient, "bke-cluster", "bke-cluster")
			if deploy_flag == true {
				break
			}
		}
		Expect(deploy_flag).To(BeTrue(), "bc should be created successfully")
		GinkgoWriter.Println("bc has been created...")

		// step4: cluster node restart in EnsureBKEAgent phase
		GinkgoWriter.Println("=====step4: cluster node restart in EnsureBKEAgent phase=====")
		start_time = time.Now()
		timeout = 10 * time.Minute
		flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("phase check out of time")
				break
			}
			flag, err = utils.BkeClusterPhaseCheck(dynamicClient, "EnsureBKEAgent", "bke-cluster", "bke-cluster")
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "phase check should be successful")
			if flag {
				GinkgoWriter.Println("phase: EnsureBKEAgent is running")
				// restart cluster node
				err = utils.SystemReboot(nodesshClient, 0)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "node reboot should be successfully")
				GinkgoWriter.Printf("node: %s has been restarted\n", nodes[len(nodes)-1].IP)
				break
			}
		}
		Expect(flag).To(BeTrue(), "cluster node should be restarted in EnsureBKEAgent phase")

		// step5: cluster health check
		GinkgoWriter.Println("=====step5: cluster health check")
		Eventually(
			func() bool {
				status, err := utils.ClusterHealthCheck(sshClient, "bke-cluster", "bke-cluster")
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "cluster health check should be success")
				return status
			}, 10*time.Minute, 5*time.Second).Should(BeTrue(), "cluster should be healthy")
		GinkgoWriter.Println("cluster health check pass")
	})

	It("reboot in EnsureNodesEnv phase", func() {

		var err error

		// step1: copy cluster config
		GinkgoWriter.Println("=====step1: copy cluster config=====")
		err = utils.CopyClusterConfig(sshClient, "1master-cluster.yaml", "/tmp")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "copy command should be succeeded")
		GinkgoWriter.Println("copy command succeeded")

		// step2: edit cluster config
		GinkgoWriter.Println("=====step2: edit cluster config=====")
		err = utils.ClusterConfigdeletion(sshClient, []string{"cluster-api", "openfuyao-system-controller"}, "/tmp/1master-cluster.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "deletion command should be succeeded")

		nodes := []utils.NodeInfo{}

		var node1 utils.NodeInfo
		node1.Hostname = os.Getenv("TEST_NODE2_HOSTNAME")
		node1.IP = os.Getenv("TEST_NODE2_IP")
		node1.Password = os.Getenv("TEST_NODE2_PASSWORD")
		node1.Port = os.Getenv("TEST_NODE2_PORT")
		node1.Role = []string{"master/node", "etcd"}
		node1.Username = os.Getenv("TEST_NODE2_USER")
		nodes = append(nodes, node1)

		port, _ := strconv.Atoi(nodes[len(nodes)-1].Port)
		nodesshClient, err = executor.NewSSHExecutor(nodes[len(nodes)-1].IP, port, nodes[len(nodes)-1].Username, nodes[len(nodes)-1].Password)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "node ssh Client should be created successfully")

		// err = utils.CleanClusterConfigNodes(sshClient, "/tmp/1master-cluster.yaml")
		// if err != nil {
		// 	GinkgoWriter.Println(err)
		// }
		// Expect(err).NotTo(HaveOccurred(), "clean cluster config nodes should be succeeded")

		err = utils.GenerateBKENodeConfig(sshClient, nodes, "/tmp/bke-nodes.yaml", "bke-cluster")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "generate BKENode config should be succeeded")
		GinkgoWriter.Println("cluster config edit completed")

		// step3: cluster create
		GinkgoWriter.Println("=====step3: cluster create=====")
		GinkgoWriter.Println("start creating cluster...")
		err = utils.ClusterCreateAtBackground(sshClient, "/tmp/1master-cluster.yaml", "/tmp/bke-nodes.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		time.Sleep(5 * time.Second)
		Expect(err).NotTo(HaveOccurred(), "cluster create should be successfully")

		start_time := time.Now()
		timeout := 3 * time.Minute
		deploy_flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("bc deploy out of time")
				break
			}
			deploy_flag = utils.BkeClusterDeployCheck(sshClient, "bke-cluster", "bke-cluster")
			if deploy_flag == true {
				break
			}
		}
		Expect(deploy_flag).To(BeTrue(), "bc should be created successfully")
		GinkgoWriter.Println("bc has been created...")

		// step4: cluster node restart in EnsureNodesEnv phase
		GinkgoWriter.Println("=====step4: cluster node restart in EnsureNodesEnv phase=====")
		start_time = time.Now()
		timeout = 10 * time.Minute
		flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("phase check out of time")
				break
			}
			flag, err = utils.BkeClusterPhaseCheck(dynamicClient, "EnsureNodesEnv", "bke-cluster", "bke-cluster")
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "phase check should be successful")
			if flag {
				GinkgoWriter.Println("phase: EnsureNodesEnv is running")
				// restart cluster node
				err = utils.SystemReboot(nodesshClient, 0)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "node reboot should be successfully")
				GinkgoWriter.Printf("node: %s has been restarted\n", nodes[len(nodes)-1].IP)
				break
			}
		}
		Expect(flag).To(BeTrue(), "cluster node should be restarted in EnsureNodesEnv phase")

		// step5: cluster health check
		GinkgoWriter.Println("=====step5: cluster health check=====")
		Eventually(
			func() bool {
				status, err := utils.ClusterHealthCheck(sshClient, "bke-cluster", "bke-cluster")
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "cluster health check should be success")
				return status
			}, 10*time.Minute, 5*time.Second).Should(BeTrue(), "cluster should be healthy")
		GinkgoWriter.Println("cluster health check pass")
	})

	It("reboot in EnsureMasterInit phase", func() {

		var err error

		// step1: copy cluster config
		GinkgoWriter.Println("=====step1: copy cluster config=====")
		err = utils.CopyClusterConfig(sshClient, "1master-cluster.yaml", "/tmp")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "copy command should be succeeded")
		GinkgoWriter.Println("copy command succeeded")

		// step2: edit cluster config
		GinkgoWriter.Println("=====step2: edit cluster config=====")
		err = utils.ClusterConfigdeletion(sshClient, []string{"cluster-api", "openfuyao-system-controller"}, "/tmp/1master-cluster.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "deletion command should be succeeded")

		nodes := []utils.NodeInfo{}

		var node1 utils.NodeInfo
		node1.Hostname = os.Getenv("TEST_NODE2_HOSTNAME")
		node1.IP = os.Getenv("TEST_NODE2_IP")
		node1.Password = os.Getenv("TEST_NODE2_PASSWORD")
		node1.Port = os.Getenv("TEST_NODE2_PORT")
		node1.Role = []string{"master/node", "etcd"}
		node1.Username = os.Getenv("TEST_NODE2_USER")
		nodes = append(nodes, node1)

		port, _ := strconv.Atoi(nodes[len(nodes)-1].Port)
		nodesshClient, err = executor.NewSSHExecutor(nodes[len(nodes)-1].IP, port, nodes[len(nodes)-1].Username, nodes[len(nodes)-1].Password)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "node ssh Client should be created successfully")

		// err = utils.CleanClusterConfigNodes(sshClient, "/tmp/1master-cluster.yaml")
		// if err != nil {
		// 	GinkgoWriter.Println(err)
		// }
		// Expect(err).NotTo(HaveOccurred(), "clean cluster config nodes should be succeeded")

		err = utils.GenerateBKENodeConfig(sshClient, nodes, "/tmp/bke-nodes.yaml", "bke-cluster")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "generate BKENode config should be succeeded")
		GinkgoWriter.Println("cluster config edit completed")

		// step3: cluster create
		GinkgoWriter.Println("=====step3: cluster create=====")
		GinkgoWriter.Println("start creating cluster...")
		err = utils.ClusterCreateAtBackground(sshClient, "/tmp/1master-cluster.yaml", "/tmp/bke-nodes.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		time.Sleep(5 * time.Second)
		Expect(err).NotTo(HaveOccurred(), "cluster create should be successfully")

		start_time := time.Now()
		timeout := 3 * time.Minute
		deploy_flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("bc deploy out of time")
				break
			}
			deploy_flag = utils.BkeClusterDeployCheck(sshClient, "bke-cluster", "bke-cluster")
			if deploy_flag == true {
				break
			}
		}
		Expect(deploy_flag).To(BeTrue(), "bc should be created successfully")
		GinkgoWriter.Println("bc has been created...")

		// step4: cluster node restart in EnsureMasterInit phase
		GinkgoWriter.Println("=====step4: cluster node restart in EnsureMasterInit phase=====")
		start_time = time.Now()
		timeout = 10 * time.Minute
		flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("phase check out of time")
				break
			}
			flag, err = utils.BkeClusterPhaseCheck(dynamicClient, "EnsureMasterInit", "bke-cluster", "bke-cluster")
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "phase check should be successful")
			if flag {
				GinkgoWriter.Println("phase: EnsureMasterInit is running")
				// restart cluster node
				err = utils.SystemReboot(nodesshClient, 0)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "node reboot should be successfully")
				GinkgoWriter.Printf("node: %s has been restarted\n", nodes[len(nodes)-1].IP)
				break
			}
		}
		Expect(flag).To(BeTrue(), "cluster node should be restarted in EnsureMasterInit phase")

		// step5: cluster health check
		GinkgoWriter.Println("=====step5: cluster health check")
		Eventually(
			func() bool {
				status, err := utils.ClusterHealthCheck(sshClient, "bke-cluster", "bke-cluster")
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "cluster health check should be success")
				return status
			}, 10*time.Minute, 5*time.Second).Should(BeTrue(), "cluster should be healthy")
		GinkgoWriter.Println("cluster health check pass")
	})

	It("reboot in EnsureMasterJoin phase", func() {

		var err error

		// step1: copy cluster config
		GinkgoWriter.Println("=====step1: copy cluster config=====")
		err = utils.CopyClusterConfig(sshClient, "1master-cluster.yaml", "/tmp")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "copy command should be succeeded")
		GinkgoWriter.Println("copy command succeeded")

		// step2: edit cluster config
		GinkgoWriter.Println("=====step2: edit cluster config=====")
		err = utils.ClusterConfigdeletion(sshClient, []string{"cluster-api", "openfuyao-system-controller"}, "/tmp/1master-cluster.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "deletion command should be succeeded")

		nodes := []utils.NodeInfo{}

		var node1 utils.NodeInfo
		node1.Hostname = os.Getenv("TEST_NODE2_HOSTNAME")
		node1.IP = os.Getenv("TEST_NODE2_IP")
		node1.Password = os.Getenv("TEST_NODE2_PASSWORD")
		node1.Port = os.Getenv("TEST_NODE2_PORT")
		node1.Role = []string{"master/node", "etcd"}
		node1.Username = os.Getenv("TEST_NODE2_USER")
		nodes = append(nodes, node1)

		var node2 utils.NodeInfo
		node2.Hostname = os.Getenv("TEST_NODE3_HOSTNAME")
		node2.IP = os.Getenv("TEST_NODE3_IP")
		node2.Password = os.Getenv("TEST_NODE3_PASSWORD")
		node2.Port = os.Getenv("TEST_NODE3_PORT")
		node2.Role = []string{"master/node", "etcd"}
		node2.Username = os.Getenv("TEST_NODE3_USER")
		nodes = append(nodes, node2)

		var node3 utils.NodeInfo
		node3.Hostname = os.Getenv("TEST_NODE4_HOSTNAME")
		node3.IP = os.Getenv("TEST_NODE4_IP")
		node3.Password = os.Getenv("TEST_NODE4_PASSWORD")
		node3.Port = os.Getenv("TEST_NODE4_PORT")
		node3.Role = []string{"master/node", "etcd"}
		node3.Username = os.Getenv("TEST_NODE4_USER")
		nodes = append(nodes, node3)

		port, _ := strconv.Atoi(nodes[len(nodes)-1].Port)
		nodesshClient, err = executor.NewSSHExecutor(nodes[len(nodes)-1].IP, port, nodes[len(nodes)-1].Username, nodes[len(nodes)-1].Password)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "node ssh Client should be created successfully")

		// err = utils.CleanClusterConfigNodes(sshClient, "/tmp/1master-cluster.yaml")
		// if err != nil {
		// 	GinkgoWriter.Println(err)
		// }
		// Expect(err).NotTo(HaveOccurred(), "clean cluster config nodes should be succeeded")

		err = utils.GenerateBKENodeConfig(sshClient, nodes, "/tmp/bke-nodes.yaml", "bke-cluster")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "generate BKENode config should be succeeded")
		GinkgoWriter.Println("cluster config edit completed")

		// step3: cluster create
		GinkgoWriter.Println("=====step3: cluster create=====")
		GinkgoWriter.Println("start creating cluster...")
		err = utils.ClusterCreateAtBackground(sshClient, "/tmp/1master-cluster.yaml", "/tmp/bke-nodes.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		time.Sleep(5 * time.Second)
		Expect(err).NotTo(HaveOccurred(), "cluster create should be successfully")

		start_time := time.Now()
		timeout := 3 * time.Minute
		deploy_flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("bc deploy out of time")
				break
			}
			deploy_flag = utils.BkeClusterDeployCheck(sshClient, "bke-cluster", "bke-cluster")
			if deploy_flag == true {
				break
			}
		}
		Expect(deploy_flag).To(BeTrue(), "bc should be created successfully")
		GinkgoWriter.Println("bc has been created...")

		// step4: cluster node restart in EnsureMasterJoin phase
		GinkgoWriter.Println("=====step4: cluster node restart in EnsureMasterJoin phase=====")
		start_time = time.Now()
		timeout = 10 * time.Minute
		flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("phase check out of time")
				break
			}
			flag, err = utils.BkeClusterPhaseCheck(dynamicClient, "EnsureMasterJoin", "bke-cluster", "bke-cluster")
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "phase check should be successful")
			if flag {
				GinkgoWriter.Println("phase: EnsureMasterJoin is running")
				// restart cluster node
				err = utils.SystemReboot(nodesshClient, 0)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "node reboot should be successfully")
				GinkgoWriter.Printf("node: %s has been restarted\n", nodes[len(nodes)-1].IP)
				break
			}
		}
		Expect(flag).To(BeTrue(), "cluster node should be restarted in EnsureMasterJoin phase")

		// step5: cluster health check
		GinkgoWriter.Println("=====step5: cluster health check")
		Eventually(
			func() bool {
				status, err := utils.ClusterHealthCheck(sshClient, "bke-cluster", "bke-cluster")
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "cluster health check should be success")
				return status
			}, 10*time.Minute, 5*time.Second).Should(BeTrue(), "cluster should be healthy")
		GinkgoWriter.Println("cluster health check pass")
	})

	It("reboot in EnsureWorkerJoin phase", func() {

		var err error

		// step1: copy cluster config
		GinkgoWriter.Println("=====step1: copy cluster config=====")
		err = utils.CopyClusterConfig(sshClient, "1master1node-cluster.yaml", "/tmp")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "copy command should be succeeded")
		GinkgoWriter.Println("copy command succeeded")

		// step2: edit cluster config
		GinkgoWriter.Println("=====step2: edit cluster config=====")
		err = utils.ClusterConfigdeletion(sshClient, []string{"cluster-api", "openfuyao-system-controller"}, "/tmp/1master1node-cluster.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "deletion command should be succeeded")

		nodes := []utils.NodeInfo{}

		var node1 utils.NodeInfo
		node1.Hostname = os.Getenv("TEST_NODE2_HOSTNAME")
		node1.IP = os.Getenv("TEST_NODE2_IP")
		node1.Password = os.Getenv("TEST_NODE2_PASSWORD")
		node1.Port = os.Getenv("TEST_NODE2_PORT")
		node1.Role = []string{"master/node", "etcd"}
		node1.Username = os.Getenv("TEST_NODE2_USER")
		nodes = append(nodes, node1)

		var node2 utils.NodeInfo
		node2.Hostname = os.Getenv("TEST_NODE5_HOSTNAME")
		node2.IP = os.Getenv("TEST_NODE5_IP")
		node2.Password = os.Getenv("TEST_NODE5_PASSWORD")
		node2.Port = os.Getenv("TEST_NODE5_PORT")
		node2.Role = []string{"node"}
		node2.Username = os.Getenv("TEST_NODE5_USER")
		nodes = append(nodes, node2)

		port, _ := strconv.Atoi(nodes[len(nodes)-1].Port)
		nodesshClient, err = executor.NewSSHExecutor(nodes[len(nodes)-1].IP, port, nodes[len(nodes)-1].Username, nodes[len(nodes)-1].Password)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "node ssh Client should be created successfully")

		// err = utils.CleanClusterConfigNodes(sshClient, "/tmp/1master1node-cluster.yaml")
		// if err != nil {
		// 	GinkgoWriter.Println(err)
		// }
		// Expect(err).NotTo(HaveOccurred(), "clean cluster config nodes should be succeeded")

		err = utils.GenerateBKENodeConfig(sshClient, nodes, "/tmp/bke-nodes.yaml", "bke-cluster")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "generate BKENode config should be succeeded")
		GinkgoWriter.Println("cluster config edit completed")

		// step3: cluster create
		GinkgoWriter.Println("=====step3: cluster create=====")
		GinkgoWriter.Println("start creating cluster...")
		err = utils.ClusterCreateAtBackground(sshClient, "/tmp/1master1node-cluster.yaml", "/tmp/bke-nodes.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		time.Sleep(5 * time.Second)
		Expect(err).NotTo(HaveOccurred(), "cluster create should be successfully")

		start_time := time.Now()
		timeout := 3 * time.Minute
		deploy_flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("bc deploy out of time")
				break
			}
			deploy_flag = utils.BkeClusterDeployCheck(sshClient, "bke-cluster", "bke-cluster")
			if deploy_flag == true {
				break
			}
		}
		Expect(deploy_flag).To(BeTrue(), "bc should be created successfully")
		GinkgoWriter.Println("bc has been created...")

		// step4: cluster node restart in EnsureWorkerJoin phase
		GinkgoWriter.Println("=====step4: cluster node restart in EnsureWorkerJoin phase=====")
		start_time = time.Now()
		timeout = 10 * time.Minute
		flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("phase check out of time")
				break
			}
			flag, err = utils.BkeClusterPhaseCheck(dynamicClient, "EnsureWorkerJoin", "bke-cluster", "bke-cluster")
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "phase check should be successful")
			if flag {
				GinkgoWriter.Println("phase: EnsureWorkerJoin is running")
				// restart cluster node
				err = utils.SystemReboot(nodesshClient, 0)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "node reboot should be successfully")
				GinkgoWriter.Printf("node: %s has been restarted\n", nodes[len(nodes)-1].IP)
				break
			}
		}
		Expect(flag).To(BeTrue(), "cluster node should be restarted in EnsureWorkerJoin phase")

		// step5: cluster health check
		GinkgoWriter.Println("=====step5: cluster health check")
		time.Sleep(7 * time.Minute)
		status, err := utils.ClusterHealthCheck(sshClient, "bke-cluster", "bke-cluster")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "cluster health check should be success")
		Expect(status).NotTo(BeTrue(), "cluster should not be created correctly")
		GinkgoWriter.Println("cluster health check not pass as expected")

	})

	It("reboot after cluster created", func() {

		var err error

		// step1: copy cluster config
		GinkgoWriter.Println("=====step1: copy cluster config=====")
		err = utils.CopyClusterConfig(sshClient, "1master-cluster.yaml", "/tmp")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "copy command should be succeeded")
		GinkgoWriter.Println("copy command succeeded")

		// step2: edit cluster config
		GinkgoWriter.Println("=====step2: edit cluster config=====")
		err = utils.ClusterConfigdeletion(sshClient, []string{"cluster-api", "openfuyao-system-controller"}, "/tmp/1master-cluster.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "deletion command should be succeeded")

		nodes := []utils.NodeInfo{}

		var node1 utils.NodeInfo
		node1.Hostname = os.Getenv("TEST_NODE2_HOSTNAME")
		node1.IP = os.Getenv("TEST_NODE2_IP")
		node1.Password = os.Getenv("TEST_NODE2_PASSWORD")
		node1.Port = os.Getenv("TEST_NODE2_PORT")
		node1.Role = []string{"master/node", "etcd"}
		node1.Username = os.Getenv("TEST_NODE2_USER")
		nodes = append(nodes, node1)

		// err = utils.CleanClusterConfigNodes(sshClient, "/tmp/1master-cluster.yaml")
		// if err != nil {
		// 	GinkgoWriter.Println(err)
		// }
		// Expect(err).NotTo(HaveOccurred(), "clean cluster config nodes should be succeeded")

		err = utils.GenerateBKENodeConfig(sshClient, nodes, "/tmp/bke-nodes.yaml", "bke-cluster")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "generate BKENode config should be succeeded")
		GinkgoWriter.Println("cluster config edit completed")

		// step3: cluster create
		GinkgoWriter.Println("=====step3: cluster create=====")
		GinkgoWriter.Println("start creating cluster...")
		err = utils.ClusterCreateAtBackground(sshClient, "/tmp/1master-cluster.yaml", "/tmp/bke-nodes.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		time.Sleep(5 * time.Second)
		Expect(err).NotTo(HaveOccurred(), "cluster create should be successfully")

		start_time := time.Now()
		timeout := 3 * time.Minute
		deploy_flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("bc deploy out of time")
				break
			}
			deploy_flag = utils.BkeClusterDeployCheck(sshClient, "bke-cluster", "bke-cluster")
			if deploy_flag == true {
				break
			}
		}
		Expect(deploy_flag).To(BeTrue(), "bc should be created successfully")
		GinkgoWriter.Println("bc has been created...")

		// step4: cluster health check
		GinkgoWriter.Println("=====step4: cluster health check")
		Eventually(
			func() bool {
				status, err := utils.ClusterHealthCheck(sshClient, "bke-cluster", "bke-cluster")
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "cluster health check should be success")
				return status
			}, 10*time.Minute, 5*time.Second).Should(BeTrue(), "cluster should be healthy")
		GinkgoWriter.Println("cluster health check pass")

		// step5: restart master node after cluster created
		GinkgoWriter.Println("=====step5: restart master node after cluster created=====")
		port, _ := strconv.Atoi(nodes[len(nodes)-1].Port)
		nodesshClient, err = executor.NewSSHExecutor(nodes[len(nodes)-1].IP, port, nodes[len(nodes)-1].Username, nodes[len(nodes)-1].Password)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "node ssh Client should be created successfully")

		err = utils.SystemReboot(nodesshClient, 0)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "node reboot should be successfully")
		GinkgoWriter.Printf("node: %s has been restarted\n", nodes[len(nodes)-1].IP)

		time.Sleep(2 * time.Minute)

		// step6: cluster health check
		GinkgoWriter.Println("=====step6: cluster health check")
		Eventually(
			func() bool {
				status, err := utils.ClusterHealthCheck(sshClient, "bke-cluster", "bke-cluster")
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "cluster health check should be success")
				return status
			}, 10*time.Minute, 5*time.Second).Should(BeTrue(), "cluster should be healthy")
		GinkgoWriter.Println("cluster health check pass")
	})

	It("reboot guide during cluster creating", Label("pre-init", "skip-temporarily"), func() {

		var err error

		// step1: copy cluster config
		GinkgoWriter.Println("=====step1: copy cluster config=====")
		err = utils.CopyClusterConfig(sshClient, "1master-cluster.yaml", "/tmp")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "copy command should be succeeded")
		GinkgoWriter.Println("copy command succeeded")

		// step2: edit cluster config
		GinkgoWriter.Println("=====step2: edit cluster config=====")
		err = utils.ClusterConfigdeletion(sshClient, []string{"cluster-api", "openfuyao-system-controller"}, "/tmp/1master-cluster.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "deletion command should be succeeded")

		nodes := []utils.NodeInfo{}

		var node1 utils.NodeInfo
		node1.Hostname = os.Getenv("TEST_NODE2_HOSTNAME")
		node1.IP = os.Getenv("TEST_NODE2_IP")
		node1.Password = os.Getenv("TEST_NODE2_PASSWORD")
		node1.Port = os.Getenv("TEST_NODE2_PORT")
		node1.Role = []string{"master/node", "etcd"}
		node1.Username = os.Getenv("TEST_NODE2_USER")
		nodes = append(nodes, node1)

		// err = utils.CleanClusterConfigNodes(sshClient, "/tmp/1master-cluster.yaml")
		// if err != nil {
		// 	GinkgoWriter.Println(err)
		// }
		// Expect(err).NotTo(HaveOccurred(), "clean cluster config nodes should be succeeded")

		err = utils.GenerateBKENodeConfig(sshClient, nodes, "/tmp/bke-nodes.yaml", "bke-cluster")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "generate BKENode config should be succeeded")
		GinkgoWriter.Println("cluster config edit completed")

		// step3: cluster create
		GinkgoWriter.Println("=====step3: cluster create=====")
		GinkgoWriter.Println("start creating cluster...")
		err = utils.ClusterCreateAtBackground(sshClient, "/tmp/1master-cluster.yaml", "/tmp/bke-nodes.yaml")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		time.Sleep(5 * time.Second)
		Expect(err).NotTo(HaveOccurred(), "cluster create should be successfully")

		start_time := time.Now()
		timeout := 3 * time.Minute
		deploy_flag := false

		for {
			if time.Since(start_time) > timeout {
				GinkgoWriter.Println("bc deploy out of time")
				break
			}
			deploy_flag = utils.BkeClusterDeployCheck(sshClient, "bke-cluster", "bke-cluster")
			if deploy_flag == true {
				break
			}
		}
		Expect(deploy_flag).To(BeTrue(), "bc should be created successfully")
		GinkgoWriter.Println("bc has been created...")

		time.Sleep(1 * time.Minute)

		// step4: guide reboot
		GinkgoWriter.Println("=====step4: guide reboot=====")
		err = utils.SystemReboot(sshClient, 0)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "restart command should be succeeded")
		GinkgoWriter.Println("guide has been restarted")

		time.Sleep(2 * time.Minute)

		// step5: rebuild guide ssh client
		GinkgoWriter.Println("=====step5: rebuild guide ssh client=====")
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		defer cancel()

		port, _ := strconv.Atoi(os.Getenv("GUIDE_TEST_PORT"))
		sshClient, err = executor.NewSSHExecutor(os.Getenv("GUIDE_TEST_IP"), port, os.Getenv("GUIDE_TEST_USER"), os.Getenv("GUIDE_TEST_PASSWORD"))
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client should be created successfully")
		GinkgoWriter.Println("ssh client recreated successfully")

		time.Sleep(1 * time.Minute)

		// step6: cluster health check
		GinkgoWriter.Println("=====step6: cluster health check")
		Eventually(
			func() bool {
				status, err := utils.ClusterHealthCheck(sshClient, "bke-cluster", "bke-cluster")
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "cluster health check should be success")
				return status
			}, 10*time.Minute, 5*time.Second).Should(BeTrue(), "cluster should be healthy")
		GinkgoWriter.Println("cluster health check pass")
	})

})
