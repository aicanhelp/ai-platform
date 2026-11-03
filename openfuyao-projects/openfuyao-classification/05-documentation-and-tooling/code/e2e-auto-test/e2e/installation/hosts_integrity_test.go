package installation

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

func newHostsCheckKubeClient() (dynamic.Interface, error) {
	kubeconfig := fmt.Sprintf("%s/.kube/config", homedir.HomeDir())
	restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		return nil, err
	}
	return dynamic.NewForConfig(restConfig)
}

func normalizeHostsOutput(content string) string {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	return strings.TrimSpace(content)
}

func hostsContentPreserved(before, after string) bool {
	before = normalizeHostsOutput(before)
	after = normalizeHostsOutput(after)
	if before == "" || after == "" {
		return false
	}

	beforeLines := strings.Split(before, "\n")
	afterLines := strings.Split(after, "\n")
	if len(afterLines) < len(beforeLines) {
		return false
	}

	for i, line := range beforeLines {
		if afterLines[i] != line {
			return false
		}
	}

	return true
}

var _ = SIGDescribe("安装部署后端补充验证", Label("installation", "backend", "post-init",
	"online"), Ordered, ContinueOnFailure, func() {
	var (
		guideExecutor   *executor.SSHExecutor
		kubectlExecutor *executor.SSHExecutor
		guideConfig     *config.GuideNodeConfig
		clusterMgr      *utils.ClusterManager
		nodes           []config.NodeInfo
		clusterNodes    []config.NodeInfo
		mgmtClusterName string
		hostsBefore     map[string]string
		testNS          = "default"
		testPodNode1    = "test-pod-node1"
		testPodPeer     = "test-pod-node1-peer"
		testPodNode2    = "test-pod-node2"
		nginxSvc        = "nginx-svc"
	)

	runKubectlNode := func(cmd string) *executor.ExecResult {
		res, err := kubectlExecutor.Exec(cmd)
		Expect(err).NotTo(HaveOccurred(), "management command should execute: %s", cmd)
		Expect(res).NotTo(BeNil())
		Expect(res.ExitCode).To(Equal(0), "management command should exit 0: %s\nstdout:%s\nstderr:%s", cmd, res.Stdout, res.Stderr)
		return res
	}

	readHostsOnNode := func(node config.NodeInfo) string {
		result, err := utils.ExecuteCommandOnNode(node, "sudo cat /etc/hosts")
		Expect(err).NotTo(HaveOccurred(), "should read /etc/hosts on %s", node.Hostname)
		Expect(result.ExitCode).To(Equal(0), result.Stderr)
		return normalizeHostsOutput(result.Stdout)
	}

	ensureNetworkFixtures := func() {
		nodeNames := strings.Fields(strings.TrimSpace(
			runKubectlNode("kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{\" \"}{end}'").Stdout))
		Expect(len(nodeNames)).To(BeNumerically(">=", 2),
			"at least 2 Kubernetes nodes are required for network checks")

		manifest := fmt.Sprintf(`apiVersion: v1
kind: Pod
metadata:
  name: %[2]s
  namespace: %[1]s
  labels:
    app: test-pod
    role: node1
spec:
  nodeName: %[6]s
  tolerations:
  - operator: Exists
  containers:
  - name: netshoot
    image: nicolaka/netshoot:latest
    command: ["sleep", "3600"]
---
apiVersion: v1
kind: Pod
metadata:
  name: %[3]s
  namespace: %[1]s
  labels:
    app: test-pod
    role: node1-peer
spec:
  nodeName: %[6]s
  tolerations:
  - operator: Exists
  containers:
  - name: netshoot
    image: nicolaka/netshoot:latest
    command: ["sleep", "3600"]
---
apiVersion: v1
kind: Pod
metadata:
  name: %[4]s
  namespace: %[1]s
  labels:
    app: test-pod
    role: node2
spec:
  nodeName: %[7]s
  tolerations:
  - operator: Exists
  containers:
  - name: netshoot
    image: nicolaka/netshoot:latest
    command: ["sleep", "3600"]
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  namespace: %[1]s
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      tolerations:
      - operator: Exists
      containers:
      - name: nginx
        image: hub.oepkgs.net/openfuyao/nginx:1.23.0-alpine
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: %[5]s
  namespace: %[1]s
spec:
  selector:
    app: nginx
  ports:
  - port: 80
    targetPort: 80
`, testNS, testPodNode1, testPodPeer, testPodNode2, nginxSvc, nodeNames[0], nodeNames[1])

		fixturePath := "/tmp/e2e-network-fixtures.yaml"
		runKubectlNode(fmt.Sprintf("cat > %s <<'EOF'\n%sEOF", fixturePath, manifest))
		runKubectlNode(fmt.Sprintf("kubectl delete pod -n %s %s %s %s --ignore-not-found=true", testNS, testPodNode1, testPodPeer, testPodNode2))
		// 预拉取镜像，避免 Pod 创建时因镜像拉取超时导致 kubectl wait 失败
		for _, node := range clusterNodes[:2] {
			utils.ExecuteCommandOnNode(node, "sudo crictl pull docker.io/nicolaka/netshoot:latest")
			utils.ExecuteCommandOnNode(node, "sudo crictl pull hub.oepkgs.net/openfuyao/nginx:1.23.0-alpine")
		}
		runKubectlNode(fmt.Sprintf("kubectl apply -f %s", fixturePath))
		podStatus := runKubectlNode(fmt.Sprintf("kubectl get pod -n %s -o wide", testNS))
		fmt.Fprintf(GinkgoWriter, "=== 网络测试 Pod 状态 ===\n%s\n", podStatus.Stdout)
		events := runKubectlNode(fmt.Sprintf("kubectl get events -n %s --sort-by='.lastTimestamp'", testNS))
		fmt.Fprintf(GinkgoWriter, "=== 网络测试 Pod 事件 ===\n%s\n", events.Stdout)
		runKubectlNode(fmt.Sprintf("kubectl wait --for=condition=Ready -n %s pod/%s pod/%s pod/%s --timeout=180s", testNS, testPodNode1, testPodPeer, testPodNode2))
		runKubectlNode(fmt.Sprintf("kubectl rollout status deployment/nginx -n %s --timeout=180s", testNS))
	}

	BeforeAll(func() {
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST must be set")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD must be set")

		var err error
		guideExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "guide node SSH should connect")

		dynamicClient, err := newHostsCheckKubeClient()
		Expect(err).NotTo(HaveOccurred(), "kube client should initialize")
		clusterMgr = utils.NewClusterManager(guideExecutor, dynamicClient)
		nodes = config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 3),
			"at least 3 test nodes are required for a 3-master management cluster")

		clusterNodes = make([]config.NodeInfo, 0, 3)
		hostsBefore = make(map[string]string, 3)
		for i := 0; i < 3; i++ {
			node := nodes[i]
			node.Role = []string{"master/node", "etcd"}
			clusterNodes = append(clusterNodes, node)
			hostsBefore[node.Hostname] = readHostsOnNode(node)
		}

		mgmtClusterName = fmt.Sprintf("mgmt-3m-%d", time.Now().Unix())
		clusterConfig := config.NewBKEClusterConfigForMgmt(mgmtClusterName, clusterNodes)
		clusterConfig.ExcludeAddons = []string{"openfuyao-system-controller"}
		configPath, nodeConfigPath, err := clusterMgr.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred(), "management cluster config should be generated and uploaded")

		By("create 3-master management cluster")
		err = clusterMgr.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "management cluster creation should start")

		By("wait for management cluster ready")
		Eventually(func() bool {
			phase, state, clusterStatus, err := clusterMgr.GetClusterFullStatusWithKubeconfig(mgmtClusterName, "")
			if err != nil {
				GinkgoWriter.Printf("get management cluster %s status failed: %v\n", mgmtClusterName, err)
				return false
			}
			GinkgoWriter.Printf("[mgmt %s] phase=%s state=%s clusterStatus=%s\n", mgmtClusterName, phase, state, clusterStatus)
			return clusterStatus == "Ready"
		}, 30*time.Minute, 30*time.Second).Should(BeTrue(), "3-master management cluster should become Ready")

		kubectlPort, portErr := strconv.Atoi(clusterNodes[0].Port)
		if portErr != nil {
			kubectlPort = 22
		}
		kubectlExecutor, err = executor.NewSSHExecutor(clusterNodes[0].IP, kubectlPort,
			clusterNodes[0].Username, clusterNodes[0].Password)
		Expect(err).NotTo(HaveOccurred(), "management node SSH should connect")

		By("verify kube-system DNS and network components")
		runKubectlNode("kubectl -n kube-system get pod -l k8s-app=kube-dns -o wide")
		runKubectlNode("kubectl -n kube-system get svc kube-dns -o wide")
		runKubectlNode("kubectl -n kube-system get endpoints kube-dns -o wide")
		runKubectlNode("kubectl -n kube-system get pod -l k8s-app=calico-node -o wide")
	})

	AfterAll(func() {
		if kubectlExecutor != nil {
			_, _ = kubectlExecutor.Exec(fmt.Sprintf(
				"kubectl delete pod -n %s %s %s %s --ignore-not-found=true", testNS,
				testPodNode1, testPodPeer, testPodNode2))
			_, _ = kubectlExecutor.Exec(fmt.Sprintf(
				"kubectl delete deploy nginx -n %s --ignore-not-found=true", testNS))
			_, _ = kubectlExecutor.Exec(fmt.Sprintf(
				"kubectl delete svc %s -n %s --ignore-not-found=true", nginxSvc, testNS))
			_ = kubectlExecutor.Close()
		}
		if guideExecutor != nil {
			if mgmtClusterName != "" && clusterMgr != nil &&
				clusterMgr.ClusterExistsWithKubeconfig(mgmtClusterName, "") {
				if err := clusterMgr.DeleteClusterWithKubeconfig(mgmtClusterName, ""); err != nil {
					GinkgoWriter.Printf("delete management cluster failed: %v\n", err)
				}
				Eventually(func() bool {
					return !clusterMgr.ClusterExistsWithKubeconfig(mgmtClusterName, "")
				}, 20*time.Minute, 30*time.Second).Should(BeTrue(), "management cluster should be deleted")
			}
			_ = guideExecutor.Close()
		}
	})

	It("etc/hosts文件检查", Label("required_nodes:3"), SpecTimeout(30*time.Minute), func(ctx SpecContext) {
		for _, node := range clusterNodes {
			after := readHostsOnNode(node)
			Expect(hostsContentPreserved(hostsBefore[node.Hostname], after)).To(
				BeTrue(), "node %s /etc/hosts original content should stay unchanged", node.Hostname)
			Expect(after).To(ContainSubstring("localhost"),
				"node %s /etc/hosts should contain localhost", node.Hostname)
			Expect(after).To(ContainSubstring(node.IP),
				"node %s /etc/hosts should contain node IP", node.Hostname)
			Expect(after).To(ContainSubstring(node.Hostname),
				"node %s /etc/hosts should contain node hostname", node.Hostname)
		}
	})

	It("kubeapi废弃参数检查", Label("required_nodes:3"),
		SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			cmd := "kubectl logs -n kube-system -l k8s-app=kube-dns --tail=5 2>&1 | tee /tmp/kubectl-logs.out"
			res := runKubectlNode(cmd)
			out := strings.ToLower(res.Stdout + res.Stderr)
			Expect(out).NotTo(ContainSubstring("has been deprecated"))
			Expect(out).NotTo(ContainSubstring("unknown flag"))
		})

	It("http访问apiserver端口", Label("required_nodes:3"),
		SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			host := strings.TrimSpace(clusterNodes[0].IP)
			GinkgoWriter.Printf("apiserver http test host=%s\n", host)
			cmd := fmt.Sprintf("curl -skv http://%s:6443 2>&1 || true", host)
			res := runKubectlNode(cmd)
			out := strings.ToLower(res.Stdout + res.Stderr)
			Expect(out).To(ContainSubstring("client sent an http request to an https server"))
		})

	It("检查yaml及配置文件是否存在明文密码", Label("required_nodes:3"),
		SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("下载扫描工具 kubescape")
			runKubectlNode(`curl -s https://raw.githubusercontent.com/kubescape/kubescape/master/install.sh | /bin/bash`)
			runKubectlNode(`kubescape version`)
			By("执行 kubescape scan")
			runKubectlNode(`kubescape scan control C-0012 --format json --output /tmp/kubescape-result.json 2>&1; true`)
			By("检查执行结果")
			res := runKubectlNode(`grep -qE '"status":\s*"passed"' /tmp/kubescape-result.json && echo "Application credentials in configuration files 0" || echo "Application credentials in configuration files 1"`)
			Expect(res.Stdout).To(ContainSubstring("Application credentials in configuration files 0"))
		})

	It("同节点pod间通信", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		ensureNetworkFixtures()
		peerIP := strings.TrimSpace(strings.Trim(runKubectlNode(
			fmt.Sprintf("kubectl get pod -n %s %s -o jsonpath='{.status.podIP}'", testNS, testPodPeer)).Stdout, "'"))
		res := runKubectlNode(fmt.Sprintf("kubectl exec -n %s %s -- ping -c 3 %s", testNS, testPodNode1, peerIP))
		out := res.Stdout + res.Stderr
		Expect(out).To(ContainSubstring("3 packets transmitted"))
		Expect(out).To(ContainSubstring("0% packet loss"))
	})

	It("跨节点pod间通信", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		ensureNetworkFixtures()
		sourceNode := strings.TrimSpace(strings.Trim(
			runKubectlNode(fmt.Sprintf("kubectl get pod -n %s %s -o jsonpath='{.spec.nodeName}'", testNS, testPodNode1)).Stdout, "'"))
		targetNode := strings.TrimSpace(strings.Trim(
			runKubectlNode(fmt.Sprintf("kubectl get pod -n %s %s -o jsonpath='{.spec.nodeName}'", testNS, testPodNode2)).Stdout, "'"))
		Expect(sourceNode).NotTo(Equal(targetNode),
			"test pods should be scheduled on different nodes")
		targetIP := strings.TrimSpace(strings.Trim(runKubectlNode(
			fmt.Sprintf("kubectl get pod -n %s %s -o jsonpath='{.status.podIP}'", testNS, testPodNode2)).Stdout, "'"))
		res := runKubectlNode(fmt.Sprintf("kubectl exec -n %s %s -- ping -c 3 %s", testNS, testPodNode1, targetIP))
		out := res.Stdout + res.Stderr
		Expect(out).To(ContainSubstring("3 packets transmitted"))
		Expect(out).To(ContainSubstring("0% packet loss"))
	})

	It("pod和node间通信", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		ensureNetworkFixtures()
		nodeIP := strings.TrimSpace(clusterNodes[1].IP)
		res := runKubectlNode(fmt.Sprintf("kubectl exec -n %s %s -- ping -c 3 %s", testNS, testPodNode1, nodeIP))
		out := res.Stdout + res.Stderr
		Expect(out).To(ContainSubstring("3 packets transmitted"))
		Expect(out).To(ContainSubstring("0% packet loss"))
	})

	It("DNS解析测试", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		ensureNetworkFixtures()
		for _, dnsName := range []string{
			"kubernetes.default.svc.cluster.local",
			fmt.Sprintf("%s.%s.svc.cluster.local", nginxSvc, testNS),
		} {
			res := runKubectlNode(fmt.Sprintf("kubectl exec -n %s %s -- nslookup %s", testNS, testPodNode1, dnsName))
			out := strings.ToLower(res.Stdout + res.Stderr)
			Expect(out).To(ContainSubstring(strings.ToLower(dnsName)))
			Expect(out).To(ContainSubstring("address:"))
			Expect(out).NotTo(ContainSubstring("nxdomain"))
		}
	})

	It("pod-clusterip service通信", Label("required_nodes:3"),
		SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			ensureNetworkFixtures()
			serviceIP := strings.TrimSpace(strings.Trim(
				runKubectlNode(fmt.Sprintf("kubectl get svc -n %s %s -o jsonpath='{.spec.clusterIP}'", testNS, nginxSvc)).Stdout, "'"))
			for _, target := range []string{
				fmt.Sprintf("%s.%s.svc.cluster.local", nginxSvc, testNS),
				serviceIP,
			} {
				res := runKubectlNode(fmt.Sprintf("kubectl exec -n %s %s -- wget -q -O- http://%s | head -5", testNS, testPodNode1, target))
				Expect(res.Stdout).To(ContainSubstring("Welcome to nginx"))
			}
		})

	It("service端点检查", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		ensureNetworkFixtures()
		endpointIPs := strings.TrimSpace(strings.Trim(
			runKubectlNode(fmt.Sprintf("kubectl get endpoints -n %s %s -o jsonpath='{.subsets[*].addresses[*].ip}'", testNS, nginxSvc)).Stdout, "'"))
		Expect(strings.Fields(endpointIPs)).To(HaveLen(2),
			"nginx-svc should have two endpoint IPs")
		serviceIP := strings.TrimSpace(strings.Trim(
			runKubectlNode(fmt.Sprintf("kubectl get svc -n %s %s -o jsonpath='{.spec.clusterIP}'", testNS, nginxSvc)).Stdout, "'"))
		res := runKubectlNode(fmt.Sprintf("iptables -t nat -L KUBE-SERVICES -n | grep %s || ipvsadm -Ln | grep %s", serviceIP, serviceIP))
		Expect(res.Stdout + res.Stderr).To(ContainSubstring(serviceIP))
	})
})
