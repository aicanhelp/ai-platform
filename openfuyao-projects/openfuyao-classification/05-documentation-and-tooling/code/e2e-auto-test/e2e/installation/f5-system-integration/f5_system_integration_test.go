package f5_system_integration

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	installation "gitcode.com/openFuyao/e2e-auto-test/e2e/installation"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

const (
	f5SITInstallationItTimeout = 45 * time.Minute
	f5SITRecoveryTimeout       = 20 * time.Minute
	f5SITResourceDeleteTimeout = 10 * time.Minute
	f5SITCleanupTimeout        = 30 * time.Minute
	f5SITResetTimeout          = 20 * time.Minute
	f5SITResetStateDir         = "/var/lib/f5-sit"
	f5SITResetLogDir           = "/var/log/f5-sit"
	f5SITResetLogPath          = f5SITResetLogDir + "/bke-reset.log"
	f5SITResetExitPath         = f5SITResetStateDir + "/bke-reset.exit"
	f5SITResetScriptPath       = f5SITResetStateDir + "/bke-reset.sh"
	f5SITResetUnit             = "f5-sit-bke-reset.service"
	f5SITPollInterval          = 15 * time.Second
	f5SITAuditRulesPath        = "/etc/audit/rules.d/bke-sit1.rules"
	f5SITKeepEnvOnFailureEnv   = "F5_SIT_KEEP_ENV_ON_FAILURE"
	f5SITCertSourcePath        = "/etc/openFuyao/certs/ca.crt"
	f5SITCertTargetDir         = "/etc/openFuyao/certs"
)

var f5SITAuditKeys = []string{
	"bke_sit1_kubelet_cfg",
	"bke_sit1_containerd_bin",
	"bke_sit1_etcd_wal",
}

var _ = SIGDescribe("F5 3Master 管理集群系统级集成测试", Label("f5-lb", "3m-mgmt", "kylin-v10", "skip-temporarily"), Ordered, func() {
	var (
		guideSSH              *executor.SSHExecutor
		localExecutor         *executor.LocalExecutor
		clusterManager        *utils.ClusterManager
		managementNodes       []config.NodeInfo
		loadBalancerNode      config.NodeInfo
		clusterName           string
		clusterConfigPath     string
		nodeConfigPath        string
		kubeconfigPath        string
		preserveEnvironment   bool
		skipPreserveOnFailure bool
	)

	BeforeAll(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "必须配置 GUIDE_NODE_HOST")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "必须配置 GUIDE_NODE_PASSWORD")

		var err error
		guideSSH, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "应该成功连接引导节点")

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功加载引导集群 kubeconfig")

		dynamicClient, err := dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功创建引导集群动态客户端")

		clusterManager = utils.NewClusterManager(guideSSH, dynamicClient)
		localExecutor = executor.NewLocalExecutor(30 * time.Second)

		nodes := config.LoadTestNodesFromEnv()
		if len(nodes) < 4 {
			Skip(fmt.Sprintf("F5 SIT 需要 3 个管理集群 Master 节点和 1 个负载均衡节点，当前仅配置了 %d 个节点", len(nodes)))
		}

		managementNodes = make([]config.NodeInfo, 3)
		copy(managementNodes, nodes[:3])
		for i := range managementNodes {
			managementNodes[i].Role = []string{"master/node", "etcd"}
		}
		loadBalancerNode = nodes[3]
		Expect(strings.TrimSpace(loadBalancerNode.IP)).NotTo(BeEmpty(), "负载均衡节点 IP 不能为空")
		Expect(strings.TrimSpace(loadBalancerNode.Password)).NotTo(BeEmpty(), "负载均衡节点密码不能为空")

		masterIPs := make([]string, 0, len(managementNodes))
		for _, node := range managementNodes {
			masterIPs = append(masterIPs, node.IP)
		}
		By(fmt.Sprintf("在负载均衡节点 %s 配置 HAProxy，后端管理集群 Master 为 %s", loadBalancerNode.IP, strings.Join(masterIPs, ",")))
		err = utils.SetupHAProxyTCPProxyOnNode(loadBalancerNode, masterIPs, "36443")
		Expect(err).NotTo(HaveOccurred(), "应该成功配置 F5 负载均衡模拟服务")

		By("在开始测试前同步 CA 证书到全部管理集群 Master")
		Expect(syncF5SITCertsToMasters(guideSSH, managementNodes)).To(Succeed())
	})

	BeforeEach(func() {
		skipPreserveOnFailure = false
		if clusterName != "" && kubeconfigPath != "" {
			By(fmt.Sprintf("复用上一用例恢复健康的 F5 3 Master 管理集群：%s", clusterName))
			Expect(verifyF5SITManagementNodesReady(localExecutor, kubeconfigPath)).To(Succeed())
			Expect(verifyF5SITBKEClusterHealth(clusterManager, clusterName, kubeconfigPath)).To(Succeed())
			return
		}

		clusterName = fmt.Sprintf("test-f5-sit-3m-%d", time.Now().UnixNano())
		clusterConfigPath = ""
		nodeConfigPath = ""
		kubeconfigPath = ""

		By("在当前用例部署前清理可能残留的同名引导侧资源")
		Expect(preCleanF5SITBootstrapResources(guideSSH, clusterManager, clusterName)).To(Succeed())

		clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, managementNodes)
		clusterConfig.ExcludeAddons = []string{"openfuyao-system-controller"}
		clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
			Host: loadBalancerNode.IP,
			Port: "36443",
		}
		clusterConfig.CustomExtra = &config.ClusterCustomExtra{
			ExtraLoadBalanceIP: loadBalancerNode.IP,
		}

		var err error
		clusterConfigPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功生成当前用例的3 Master管理集群配置")

		By(fmt.Sprintf("为当前用例独立部署 F5 3 Master 高可用管理集群：%s", clusterName))
		err = clusterManager.CreateClusterInBackgroundWithKubeconfig(clusterConfigPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "应该成功触发当前用例的管理集群部署")

		kubeconfigPath, err = installation.WaitAndVerifyManagementCluster(clusterManager, clusterName)
		Expect(err).NotTo(HaveOccurred(), "当前用例的管理集群应该进入 Healthy 和 Ready 状态")
		Expect(verifyF5SITManagementNodesReady(localExecutor, kubeconfigPath)).To(Succeed())
	})

	It("F5管理集群_SIT_1：验证3 Master同时重启后审计规则、关键目录与集群健康状态", SpecTimeout(f5SITInstallationItTimeout), func(ctx SpecContext) {
		By("步骤 1：在每台管理集群 Master 上写入并加载 kubelet 与 containerd 持久化审计规则")
		for _, node := range managementNodes {
			result, err := executeF5SITNodeCommand(node, f5SITAuditSetupCommand())
			Expect(err).NotTo(HaveOccurred(), "节点 %s 应该成功配置持久化审计规则；命令输出：%s", node.IP, f5SITCommandOutput(result))
		}

		By("步骤 2：同时重启三台管理集群 Master，并从发起重启时开始计时")
		bootIDsBefore, err := getF5SITNodeBootIDs(managementNodes)
		Expect(err).NotTo(HaveOccurred(), "重启前应该成功获取三台 Master 的 boot ID")

		rebootStartedAt := time.Now()
		rebootResults := executeF5SITCommandOnNodes(managementNodes, "nohup sh -c 'sleep 2; systemctl reboot' >/dev/null 2>&1 &")
		for _, nodeResult := range rebootResults {
			Expect(nodeResult.err).NotTo(HaveOccurred(), "节点 %s 应该成功触发重启；命令输出：%s", nodeResult.node.IP, f5SITCommandOutput(nodeResult.result))
		}

		By("步骤 3：在同一个 20 分钟超时内等待 auditd、关键路径、节点、etcd Pod 和 BKECluster 恢复健康")
		recoveryDeadline := rebootStartedAt.Add(f5SITRecoveryTimeout)
		By("步骤 3.1：等待三台 Master 的 boot ID 全部变化，确认节点已真正完成重启")
		err = waitForF5SITNodesRebooted(recoveryDeadline, managementNodes, bootIDsBefore)
		Expect(err).NotTo(HaveOccurred(), "三台 Master 应该在20分钟总时限内完成真实重启")

		By("步骤 3.2：重连三台 Master，并检查审计规则、关键路径、etcd Pod 与 BKECluster 健康状态")
		err = waitForF5SITRecovery(
			recoveryDeadline,
			managementNodes,
			localExecutor,
			clusterManager,
			clusterName,
			kubeconfigPath,
		)
		if err != nil {
			collectF5SITAuditDiagnostics(managementNodes)
		}
		Expect(err).NotTo(HaveOccurred(), "管理集群应该在三台 Master 发起重启后的 20 分钟内恢复")
	})

	It("F5管理集群_SIT_2：验证3Master reset 后 containerd 目录清空并可再次部署", SpecTimeout(f5SITInstallationItTimeout), func(ctx SpecContext) {
		By("步骤 1：在三台管理集群 Master 上并发执行带 Debug 日志的 bke reset")
		resetStartResults := executeF5SITCommandOnNodes(managementNodes, f5SITResetStartCommand())
		for _, nodeResult := range resetStartResults {
			Expect(nodeResult.err).NotTo(HaveOccurred(), "节点 %s 应该成功触发后台 bke reset；命令输出：%s", nodeResult.node.IP, f5SITCommandOutput(nodeResult.result))
		}

		Expect(waitForF5SITResetCompletion(managementNodes, f5SITResetTimeout)).To(Succeed())

		By("步骤 2：从引导集群删除对应的 BKECluster 和全部 BKENode 资源")
		namespace := "bke-" + clusterName
		deleteNodesCommand := fmt.Sprintf(
			"kubectl delete bn -n %s --all --ignore-not-found=true --wait=false",
			namespace,
		)
		result, err := guideSSH.Exec(deleteNodesCommand)
		Expect(err).NotTo(HaveOccurred(), "应该成功删除全部 BKENode 资源；命令输出：%s", f5SITCommandOutput(result))

		deleteClusterCommand := fmt.Sprintf(
			"kubectl delete bc bke-%s -n %s --ignore-not-found=true --wait=false",
			clusterName,
			namespace,
		)
		result, err = guideSSH.Exec(deleteClusterCommand)
		Expect(err).NotTo(HaveOccurred(), "应该成功删除 BKECluster 资源；命令输出：%s", f5SITCommandOutput(result))

		deleteNamespaceCommand := fmt.Sprintf(
			"kubectl delete namespace %s --ignore-not-found=true --wait=false",
			namespace,
		)
		result, err = guideSSH.Exec(deleteNamespaceCommand)
		Expect(err).NotTo(HaveOccurred(), "应该成功触发集群命名空间删除；命令输出：%s", f5SITCommandOutput(result))

		Eventually(func() bool {
			deleted, output := f5SITBootstrapResourcesFullyDeleted(guideSSH, clusterName)
			if !deleted {
				GinkgoWriter.Printf("SIT_2 引导侧资源仍在删除中：%s\n", output)
			}
			return deleted
		}, f5SITResourceDeleteTimeout, 10*time.Second).Should(BeTrue(), "BKECluster、BKENode 与命名空间应该完全删除后才能再次部署")

		By("步骤 3：检查每台已 reset 节点上的 /var/lib/containerd 不存在或内容为空")
		containerdResults := executeF5SITCommandOnNodes(
			managementNodes,
			`[ ! -e /var/lib/containerd ] || [ -z "$(ls -A /var/lib/containerd 2>/dev/null)" ]`,
		)
		for _, nodeResult := range containerdResults {
			Expect(nodeResult.err).NotTo(HaveOccurred(), "节点 %s 上的 /var/lib/containerd 应该不存在或内容为空；命令输出：%s", nodeResult.node.IP, f5SITCommandOutput(nodeResult.result))
		}

		By("步骤 4：使用相同的 F5 与3Master 配置重新部署管理集群")
		err = clusterManager.CreateClusterInBackgroundWithKubeconfig(clusterConfigPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "应该成功触发第二次管理集群部署")

		kubeconfigPath, err = installation.WaitAndVerifyManagementCluster(clusterManager, clusterName)
		if err != nil {
			portConflict, detail := detectF5SITKubeAPIServerPortConflict(managementNodes)
			GinkgoWriter.Printf("SIT_2 重部署失败后检查 6443 端口占用：occupied=%t\n%s\n", portConflict, detail)
			if portConflict {
				skipPreserveOnFailure = true
				GinkgoWriter.Printf("检测到 kube-apiserver 可能因 6443 端口占用启动失败，本轮失败不保留现场，将继续执行清理\n")
			}
		}
		Expect(err).NotTo(HaveOccurred(), "重新部署的管理集群应该进入 Healthy 和 Ready 状态")
		Expect(verifyF5SITManagementNodesReady(localExecutor, kubeconfigPath)).To(Succeed())
		Expect(verifyF5SITBKEClusterHealth(clusterManager, clusterName, kubeconfigPath)).To(Succeed())
	})

	AfterEach(func() {
		if !CurrentSpecReport().State.Is(types.SpecStateFailureStates) {
			return
		}
		if skipPreserveOnFailure {
			GinkgoWriter.Printf("检测到用例失败，但已标记为不保留 F5 SIT 现场，将继续执行清理：cluster=%s, lb=%s\n", clusterName, loadBalancerNode.IP)
			return
		}
		if f5SITEnvEnabled(f5SITKeepEnvOnFailureEnv) {
			preserveEnvironment = true
			GinkgoWriter.Printf("检测到用例失败且 %s=1，保留 F5 SIT 现场：cluster=%s, clusterConfig=%s, nodeConfig=%s, kubeconfig=%s, lb=%s\n",
				f5SITKeepEnvOnFailureEnv,
				clusterName,
				clusterConfigPath,
				nodeConfigPath,
				kubeconfigPath,
				loadBalancerNode.IP,
			)
			if guideSSH != nil {
				_ = guideSSH.Close()
			}
		}
	})
	AfterAll(func() {
		if preserveEnvironment {
			GinkgoWriter.Printf("存在失败用例需要保留现场，跳过集群与共享 HAProxy 清理：cluster=%s, lb=%s\n", clusterName, loadBalancerNode.IP)
			if guideSSH != nil {
				_ = guideSSH.Close()
			}
			return
		}

		cleanupFailures := make([]string, 0)
		for _, node := range managementNodes {
			if _, err := executeF5SITNodeCommand(node, f5SITAuditCleanupCommand()); err != nil {
				cleanupFailures = append(cleanupFailures, fmt.Sprintf("清理节点 %s 的 SIT 审计规则失败：%v", node.IP, err))
			}
		}

		if clusterManager != nil && clusterName != "" {
			namespace := "bke-" + clusterName
			if kubeconfigPath != "" {
				if err := clusterManager.DeleteManagementClusterSelfBC(kubeconfigPath, clusterName); err != nil {
					GinkgoWriter.Printf("管理集群自管 BC 无法注解删除，改用三节点 bke reset 清理：%v\n", err)
					resetStartFailed := false
					for _, nodeResult := range executeF5SITCommandOnNodes(managementNodes, f5SITResetStartCommand()) {
						if nodeResult.err != nil {
							resetStartFailed = true
							cleanupFailures = append(cleanupFailures, fmt.Sprintf("节点 %s 触发清理 reset 失败：%v；命令输出：%s", nodeResult.node.IP, nodeResult.err, f5SITCommandOutput(nodeResult.result)))
						}
					}
					if !resetStartFailed {
						if err := waitForF5SITResetCompletion(managementNodes, f5SITResetTimeout); err != nil {
							cleanupFailures = append(cleanupFailures, fmt.Sprintf("等待清理 reset 完成失败：%v", err))
						} else if err := waitForF5SITManagementNodesCleaned(managementNodes, 10*time.Minute); err != nil {
							cleanupFailures = append(cleanupFailures, fmt.Sprintf("等待 reset 后管理集群节点清理完成失败：%v", err))
						}
					}
				} else {
					if err := waitForF5SITManagementNodesCleaned(managementNodes, 10*time.Minute); err != nil {
						cleanupFailures = append(cleanupFailures, fmt.Sprintf("等待管理集群节点清理完成失败：%v", err))
					}
				}
			}

			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				if err := clusterManager.DeleteClusterWithKubeconfig(clusterName, ""); err != nil {
					cleanupFailures = append(cleanupFailures, fmt.Sprintf("触发引导集群 BC 删除失败：%v", err))
				}
			}
			if guideSSH != nil {
				cleanupCommands := []string{
					fmt.Sprintf("kubectl delete bn -n %s --all --ignore-not-found=true --wait=false", namespace),
					fmt.Sprintf("kubectl delete namespace %s --ignore-not-found=true --wait=false", namespace),
				}
				for _, command := range cleanupCommands {
					result, err := guideSSH.Exec(command)
					if err != nil {
						cleanupFailures = append(cleanupFailures, fmt.Sprintf("执行引导侧清理命令失败：%v；命令输出：%s", err, f5SITCommandOutput(result)))
					}
				}
			}
		}

		if guideSSH != nil && clusterName != "" {
			if err := f5SITWaitForBootstrapResourcesDeleted(guideSSH, clusterName, f5SITCleanupTimeout); err != nil {
				cleanupFailures = append(cleanupFailures, fmt.Sprintf("等待引导侧 BC、BN 与 namespace 完全删除失败：%v", err))
			}
		}
		if clusterManager != nil {
			for _, path := range []string{clusterConfigPath, nodeConfigPath, kubeconfigPath} {
				if path != "" {
					_ = clusterManager.CleanupConfig(path)
				}
			}
		}
		if len(cleanupFailures) > 0 && f5SITEnvEnabled(f5SITKeepEnvOnFailureEnv) {
			preserveEnvironment = true
		}
		Expect(cleanupFailures).To(BeEmpty(), "F5 SIT 当前用例清理必须完全成功：%s", strings.Join(cleanupFailures, "; "))
		if strings.TrimSpace(loadBalancerNode.IP) != "" {
			Expect(utils.CleanupHAProxyOnNode(loadBalancerNode)).To(Succeed(), "应该成功清理共享 HAProxy")
		}
		if guideSSH != nil {
			_ = guideSSH.Close()
		}
	})
})

func detectF5SITKubeAPIServerPortConflict(nodes []config.NodeInfo) (bool, string) {
	command := `echo "### listeners on 6443"
(ss -lntp 2>/dev/null || netstat -lntp 2>/dev/null || true) | grep -E '(:|\.)6443([[:space:]]|$)' || true
echo "### kube-apiserver containers"
crictl ps -a --name kube-apiserver 2>/dev/null || true
echo "### recent kubelet kube-apiserver errors"
journalctl -u kubelet --since '30 min ago' --no-pager 2>/dev/null | grep -Ei 'kube-apiserver|6443|address already in use|bind: address already' | tail -n 80 || true`

	portConflict := false
	details := make([]string, 0, len(nodes))
	for _, nodeResult := range executeF5SITCommandOnNodes(nodes, command) {
		output := f5SITCommandOutput(nodeResult.result)
		if f5SITOutputIndicatesPort6443Conflict(output) {
			portConflict = true
		}
		if nodeResult.err != nil {
			details = append(details, fmt.Sprintf("节点 %s 诊断命令执行失败：%v；命令输出：%s", nodeResult.node.IP, nodeResult.err, output))
			continue
		}
		details = append(details, fmt.Sprintf("节点 %s 诊断输出：\n%s", nodeResult.node.IP, output))
	}
	return portConflict, strings.Join(details, "\n---\n")
}

func f5SITOutputIndicatesPort6443Conflict(output string) bool {
	lowerOutput := strings.ToLower(output)
	if !strings.Contains(lowerOutput, "address already in use") {
		return false
	}
	return strings.Contains(lowerOutput, "6443") || strings.Contains(lowerOutput, "kube-apiserver")
}
func f5SITEnvEnabled(name string) bool {
	value := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	return value == "1" || value == "true" || value == "yes" || value == "y"
}

func preCleanF5SITBootstrapResources(guideSSH *executor.SSHExecutor, clusterManager *utils.ClusterManager, clusterName string) error {
	if guideSSH == nil || clusterManager == nil || clusterName == "" {
		return nil
	}

	namespace := "bke-" + clusterName
	cleanupCommands := []string{
		fmt.Sprintf("kubectl delete bn -n %s --all --ignore-not-found=true --wait=false", namespace),
		fmt.Sprintf("kubectl delete bc bke-%s -n %s --ignore-not-found=true --wait=false", clusterName, namespace),
		fmt.Sprintf("kubectl delete namespace %s --ignore-not-found=true --wait=false", namespace),
	}
	for _, command := range cleanupCommands {
		result, err := guideSSH.Exec(command)
		if err != nil {
			return fmt.Errorf("执行预清理命令失败：%w；命令输出：%s", err, f5SITCommandOutput(result))
		}
		if result != nil && result.ExitCode != 0 {
			return fmt.Errorf("执行预清理命令失败：%s", f5SITCommandOutput(result))
		}
	}

	if err := clusterManager.CleanupPathRecursive(fmt.Sprintf("/tmp/bke-create-%s.log", clusterName)); err != nil {
		GinkgoWriter.Printf("清理旧的创建日志失败：%v\n", err)
	}
	return nil
}

func syncF5SITCertsToMasters(guideSSH *executor.SSHExecutor, nodes []config.NodeInfo) error {
	if guideSSH == nil {
		return fmt.Errorf("guideSSH 不能为空")
	}
	if len(nodes) == 0 {
		return nil
	}

	result, err := guideSSH.Exec(fmt.Sprintf("test -f %s", f5SITCertSourcePath))
	if err != nil {
		return fmt.Errorf("检查源证书失败：%w", err)
	}
	if result == nil || result.ExitCode != 0 {
		return fmt.Errorf("源证书不存在：%s", f5SITCommandOutput(result))
	}

	for _, node := range nodes {
		cmd := fmt.Sprintf("sshpass -p '%s' scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null %s root@%s:%s/", node.Password, f5SITCertSourcePath, node.IP, f5SITCertTargetDir)
		result, err := guideSSH.Exec(cmd)
		if err != nil {
			return fmt.Errorf("同步证书到节点 %s 失败：%w；命令输出：%s", node.IP, err, f5SITCommandOutput(result))
		}
		if result != nil && result.ExitCode != 0 {
			return fmt.Errorf("同步证书到节点 %s 失败：%s", node.IP, f5SITCommandOutput(result))
		}
	}
	return nil
}

type f5SITNodeCommandResult struct {
	node   config.NodeInfo
	result *executor.ExecResult
	err    error
}

func f5SITAuditSetupCommand() string {
	return fmt.Sprintf(`%s
install -d -m 0750 /etc/audit/rules.d
cat > %s <<'EOF'
-w /var/lib/kubelet/config.yaml -p wa -k bke_sit1_kubelet_cfg
-w /usr/bin/containerd -p wa -k bke_sit1_containerd_bin
-w /var/lib/etcd/member/wal -p wa -k bke_sit1_etcd_wal
EOF
chmod 0640 %s
augenrules --load
systemctl is-active --quiet auditd
rules="$(auditctl -l)"
for key in %s; do printf '%%s\n' "$rules" | grep -Fq "$key" || exit 1; done`,
		f5SITAuditUnloadCommand(),
		f5SITAuditRulesPath,
		f5SITAuditRulesPath,
		strings.Join(f5SITAuditKeys, " "),
	)
}

func f5SITAuditUnloadCommand() string {
	return `remove_f5_sit_watch() {
  path="$1"
  key="$2"
  if auditctl -l | grep -Fq -- "$key"; then
    auditctl -W "$path" -p wa -k "$key" || exit 1
  fi
}
remove_f5_sit_watch /var/lib/kubelet/config.yaml bke_sit1_kubelet_cfg
remove_f5_sit_watch /usr/bin/containerd bke_sit1_containerd_bin
remove_f5_sit_watch /var/lib/etcd/member/wal bke_sit1_etcd_wal
rules="$(auditctl -l)"
for key in bke_sit1_kubelet_cfg bke_sit1_containerd_bin bke_sit1_etcd_wal; do
  if printf '%s\n' "$rules" | grep -Fq -- "$key"; then
    echo "failed to unload audit rule: $key" >&2
    exit 1
  fi
done`
}

func f5SITAuditCleanupCommand() string {
	return fmt.Sprintf(`%s
rm -f %s
augenrules --load
rules="$(auditctl -l)"
for key in %s; do
  if printf '%%s\n' "$rules" | grep -Fq "$key"; then
    echo "audit rule still loaded: $key" >&2
    exit 1
  fi
done`,
		f5SITAuditUnloadCommand(),
		f5SITAuditRulesPath,
		strings.Join(f5SITAuditKeys, " "),
	)
}

func f5SITResetStartCommand() string {
	return fmt.Sprintf(`install -d -m 0700 %s %s
if pgrep -f '[b]ke reset --all --mount' >/dev/null 2>&1; then
  echo "a previous bke reset process is still running" >&2
  pgrep -af '[b]ke reset --all --mount' >&2 || true
  exit 20
fi
if command -v systemd-run >/dev/null 2>&1; then
  systemctl stop %s >/dev/null 2>&1 || true
  systemctl reset-failed %s >/dev/null 2>&1 || true
fi
rm -f %s %s %s
cat > %s <<'EOF'
#!/bin/sh
umask 077
: > %s
export DEBUG=1
bke_path="$(command -v bke 2>/dev/null || true)"
if [ -z "$bke_path" ]; then
  echo "bke command not found in PATH=$PATH" >> %s
  code=127
else
  timeout 15m "$bke_path" reset --all --mount --confirm >> %s 2>&1
  code=$?
fi
tmp="%s.tmp.$$"
printf '%%s\n' "$code" > "$tmp"
mv -f "$tmp" %s
exit "$code"
EOF
chmod 0700 %s
if command -v systemd-run >/dev/null 2>&1; then
  systemd-run --unit=%s --no-block /bin/sh %s
else
  command -v setsid >/dev/null 2>&1 || { echo "systemd-run and setsid are unavailable" >&2; exit 1; }
  nohup setsid /bin/sh %s </dev/null >/dev/null 2>&1 &
  echo "fallback reset pid=$!"
fi`,
		f5SITResetStateDir,
		f5SITResetLogDir,
		f5SITResetUnit,
		f5SITResetUnit,
		f5SITResetLogPath,
		f5SITResetExitPath,
		f5SITResetScriptPath,
		f5SITResetScriptPath,
		f5SITResetLogPath,
		f5SITResetLogPath,
		f5SITResetLogPath,
		f5SITResetExitPath,
		f5SITResetExitPath,
		f5SITResetScriptPath,
		f5SITResetUnit,
		f5SITResetScriptPath,
		f5SITResetScriptPath,
	)
}

func waitForF5SITResetCompletion(nodes []config.NodeInfo, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	pending := make(map[string]config.NodeInfo, len(nodes))
	lastOutputs := make(map[string]string, len(nodes))
	for _, node := range nodes {
		pending[node.IP] = node
	}

	for len(pending) > 0 {
		for ip, node := range pending {
			result, err := executeF5SITNodeCommand(node, f5SITResetStatusCommand())
			output := f5SITCommandOutput(result)
			if strings.TrimSpace(output) != "" {
				lastOutputs[ip] = output
			}
			if err == nil {
				GinkgoWriter.Printf("节点 %s 上的 bke reset 已完成：\n%s\n", ip, output)
				delete(pending, ip)
				continue
			}
			if result != nil && result.ExitCode == 10 {
				GinkgoWriter.Printf("节点 %s 上的 bke reset 仍在执行：\n%s\n", ip, output)
				continue
			}
			if time.Now().Before(deadline) && result == nil {
				lastOutputs[ip] = err.Error()
				GinkgoWriter.Printf("节点 %s 暂时无法读取 bke reset 状态：%v\n", ip, err)
				continue
			}
			return fmt.Errorf("节点 %s 上的 bke reset 执行失败：%w；命令输出：%s", ip, err, output)
		}
		if len(pending) == 0 {
			return nil
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("等待 bke reset 完成超时，剩余节点：%s；最近输出：%s", f5SITPendingNodeIPs(pending), f5SITResetLastOutputs(lastOutputs))
		}
		time.Sleep(5 * time.Second)
	}
	return nil
}

func f5SITPendingNodeIPs(pending map[string]config.NodeInfo) string {
	ips := make([]string, 0, len(pending))
	for ip := range pending {
		ips = append(ips, ip)
	}
	return strings.Join(ips, ",")
}

func f5SITResetLastOutputs(outputs map[string]string) string {
	parts := make([]string, 0, len(outputs))
	for ip, output := range outputs {
		parts = append(parts, fmt.Sprintf("[%s] %s", ip, output))
	}
	return strings.Join(parts, "\n")
}

func f5SITResetStatusCommand() string {
	return fmt.Sprintf(`if [ -s %s ]; then
  code="$(tr -d '[:space:]' < %s 2>/dev/null || true)"
  if ! printf '%%s\n' "$code" | grep -Eq '^[0-9]+$'; then
    echo "bke reset exit code is not numeric: $code"
    tail -n 120 %s 2>/dev/null || true
    exit 12
  fi
  log="$(tail -n 200 %s 2>/dev/null || true)"
  echo "bke reset exit code: $code"
  printf '%%s\n' "$log"
  exit "$code"
fi
log="$(tail -n 200 %s 2>/dev/null || true)"
if printf '%%s\n' "$log" | grep -Fq 'BKE reset completed'; then
  echo "bke reset completed; wrapper exit code was not preserved"
  printf '%%s\n' "$log"
  exit 0
fi
echo "bke reset has not completed; waiting"
printf '%%s\n' "$log"
exit 10`,
		f5SITResetExitPath,
		f5SITResetExitPath,
		f5SITResetLogPath,
		f5SITResetLogPath,
		f5SITResetLogPath,
	)
}
func executeF5SITNodeCommand(node config.NodeInfo, command string) (*executor.ExecResult, error) {
	port, err := strconv.Atoi(node.Port)
	if err != nil || port == 0 {
		port = 22
	}

	sshExecutor, err := executor.NewSSHExecutor(node.IP, port, node.Username, node.Password)
	if err != nil {
		return nil, fmt.Errorf("连接节点 %s 失败：%w", node.IP, err)
	}
	defer sshExecutor.Close()

	result, err := sshExecutor.Exec(command)
	if err != nil {
		return result, fmt.Errorf("在节点 %s 上执行命令失败：%w", node.IP, err)
	}
	if result == nil {
		return nil, fmt.Errorf("在节点 %s 上执行命令未返回结果", node.IP)
	}
	if result.ExitCode != 0 {
		return result, fmt.Errorf("在节点 %s 上执行命令失败，退出码为 %d", node.IP, result.ExitCode)
	}
	return result, nil
}

func executeF5SITCommandOnNodes(nodes []config.NodeInfo, command string) []f5SITNodeCommandResult {
	results := make([]f5SITNodeCommandResult, len(nodes))
	var waitGroup sync.WaitGroup
	for i := range nodes {
		waitGroup.Add(1)
		go func(index int) {
			defer waitGroup.Done()
			results[index].node = nodes[index]
			results[index].result, results[index].err = executeF5SITNodeCommand(nodes[index], command)
		}(i)
	}
	waitGroup.Wait()
	return results
}

func getF5SITNodeBootIDs(nodes []config.NodeInfo) (map[string]string, error) {
	results := executeF5SITCommandOnNodes(nodes, "cat /proc/sys/kernel/random/boot_id")
	bootIDs := make(map[string]string, len(nodes))
	for _, nodeResult := range results {
		if nodeResult.err != nil {
			return nil, fmt.Errorf("read boot ID from node %s: %w; output: %s", nodeResult.node.IP, nodeResult.err, f5SITCommandOutput(nodeResult.result))
		}
		bootID := strings.TrimSpace(nodeResult.result.Stdout)
		if bootID == "" {
			return nil, fmt.Errorf("node %s returned an empty boot ID", nodeResult.node.IP)
		}
		bootIDs[nodeResult.node.IP] = bootID
	}
	return bootIDs, nil
}

func waitForF5SITNodesRebooted(deadline time.Time, nodes []config.NodeInfo, oldBootIDs map[string]string) error {
	pending := make(map[string]config.NodeInfo, len(nodes))
	lastErrors := make(map[string]string, len(nodes))
	for _, node := range nodes {
		if strings.TrimSpace(oldBootIDs[node.IP]) == "" {
			return fmt.Errorf("missing pre-reboot boot ID for node %s", node.IP)
		}
		pending[node.IP] = node
	}

	for len(pending) > 0 {
		pendingNodes := make([]config.NodeInfo, 0, len(pending))
		for _, node := range pending {
			pendingNodes = append(pendingNodes, node)
		}

		results := executeF5SITCommandOnNodes(pendingNodes, "cat /proc/sys/kernel/random/boot_id")
		for _, nodeResult := range results {
			if nodeResult.err != nil {
				// Connection failures are expected while the node is going down or booting.
				lastErrors[nodeResult.node.IP] = nodeResult.err.Error()
				continue
			}

			bootID := strings.TrimSpace(nodeResult.result.Stdout)
			if bootID == "" {
				lastErrors[nodeResult.node.IP] = "node is reachable but returned an empty boot ID"
				continue
			}
			if bootID == oldBootIDs[nodeResult.node.IP] {
				lastErrors[nodeResult.node.IP] = "node still has its pre-reboot boot ID"
				continue
			}

			GinkgoWriter.Printf("node %s completed reboot, boot ID: %s -> %s\n", nodeResult.node.IP, oldBootIDs[nodeResult.node.IP], bootID)
			delete(pending, nodeResult.node.IP)
			delete(lastErrors, nodeResult.node.IP)
		}

		if len(pending) == 0 {
			return nil
		}
		if !time.Now().Before(deadline) {
			parts := make([]string, 0, len(pending))
			for ip := range pending {
				parts = append(parts, fmt.Sprintf("%s (%s)", ip, lastErrors[ip]))
			}
			return fmt.Errorf("timed out waiting for node boot IDs to change: %s", strings.Join(parts, ", "))
		}

		GinkgoWriter.Printf("still waiting for nodes to complete reboot: %s\n", f5SITPendingNodeIPs(pending))
		time.Sleep(5 * time.Second)
	}
	return nil
}

func waitForF5SITRecovery(
	deadline time.Time,
	nodes []config.NodeInfo,
	localExecutor *executor.LocalExecutor,
	clusterManager *utils.ClusterManager,
	clusterName string,
	kubeconfigPath string,
) error {
	var lastErr error
	for {
		lastErr = verifyF5SITRecovery(nodes, localExecutor, clusterManager, clusterName, kubeconfigPath)
		if lastErr == nil {
			return nil
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("超过集群恢复截止时间：%w", lastErr)
		}
		GinkgoWriter.Printf("F5 SIT 集群仍在恢复中：%v\n", lastErr)
		time.Sleep(f5SITPollInterval)
	}
}

func verifyF5SITRecovery(
	nodes []config.NodeInfo,
	localExecutor *executor.LocalExecutor,
	clusterManager *utils.ClusterManager,
	clusterName string,
	kubeconfigPath string,
) error {
	verifyNodeCommand := fmt.Sprintf(`systemctl is-active --quiet auditd
test -f %s
rules="$(auditctl -l)"
for key in %s; do printf '%%s\n' "$rules" | grep -Fq "$key" || exit 1; done
test -e /var/lib/kubelet/config.yaml
test -e /usr/bin/containerd
test -d /var/lib/etcd/member/wal
test -n "$(find /var/lib/etcd/member/wal -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)"`,
		f5SITAuditRulesPath,
		strings.Join(f5SITAuditKeys, " "),
	)

	for _, node := range nodes {
		result, err := executeF5SITNodeCommand(node, verifyNodeCommand)
		if err != nil {
			return fmt.Errorf("节点 %s 的 auditd 和关键路径尚未恢复：%w；命令输出：%s", node.IP, err, f5SITCommandOutput(result))
		}
	}
	if err := verifyF5SITManagementNodesReady(localExecutor, kubeconfigPath); err != nil {
		return err
	}
	if err := verifyF5SITEtcdPodsHealthy(localExecutor, kubeconfigPath); err != nil {
		return err
	}
	return verifyF5SITBKEClusterHealth(clusterManager, clusterName, kubeconfigPath)
}

func verifyF5SITManagementNodesReady(localExecutor *executor.LocalExecutor, kubeconfigPath string) error {
	command := fmt.Sprintf("KUBECONFIG=%s kubectl get nodes --no-headers", kubeconfigPath)
	result, err := localExecutor.Exec(command)
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("获取管理集群节点失败：%v；命令输出：%s", err, f5SITCommandOutput(result))
	}

	lines := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	if len(lines) != 3 {
		return fmt.Errorf("预期存在 3 个管理集群节点，实际为 %d：%s", len(lines), strings.TrimSpace(result.Stdout))
	}
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			return fmt.Errorf("kubectl 节点输出格式不符合预期：%s", line)
		}
		if fields[1] != "Ready" {
			return fmt.Errorf("节点 %s 未处于 Ready 状态：%s", fields[0], line)
		}
		if !strings.Contains(fields[2], "control-plane") && !strings.Contains(fields[2], "master") {
			return fmt.Errorf("节点 %s 不具备 control-plane 角色：%s", fields[0], line)
		}
	}
	return nil
}

func verifyF5SITEtcdPodsHealthy(localExecutor *executor.LocalExecutor, kubeconfigPath string) error {
	listCommand := fmt.Sprintf(`KUBECONFIG=%s kubectl get pods -n kube-system --no-headers`, kubeconfigPath)
	result, err := localExecutor.Exec(listCommand)
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("获取 kube-system etcd Pod 失败：%v；命令输出：%s", err, f5SITCommandOutput(result))
	}

	etcdPods := make([]string, 0, 3)
	for _, line := range strings.Split(strings.TrimSpace(result.Stdout), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 || !strings.HasPrefix(fields[0], "etcd-") {
			continue
		}
		etcdPods = append(etcdPods, line)
		if fields[1] != "1/1" || !strings.EqualFold(fields[2], "Running") {
			return fmt.Errorf("etcd Pod 未处于 1/1 Running：%s", line)
		}
	}
	if len(etcdPods) != 3 {
		return fmt.Errorf("预期 kube-system 中存在 3 个 etcd Pod，实际为 %d：%s", len(etcdPods), strings.TrimSpace(result.Stdout))
	}
	return nil
}
func verifyF5SITPodsHealthy(localExecutor *executor.LocalExecutor, kubeconfigPath string) error {
	namespaces := []string{"kube-system", "cluster-system"}
	for _, ns := range namespaces {
		if err := verifyF5SITPodsHealthyInNamespace(localExecutor, kubeconfigPath, ns); err != nil {
			return err
		}
	}
	return nil
}

func verifyF5SITPodsHealthyInNamespace(localExecutor *executor.LocalExecutor, kubeconfigPath, namespace string) error {
	listCommand := fmt.Sprintf(`KUBECONFIG=%s kubectl get pods -n %s --no-headers`, kubeconfigPath, namespace)
	result, err := localExecutor.Exec(listCommand)
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("获取命名空间 %s 的 Pod 失败：%v；命令输出：%s", namespace, err, f5SITCommandOutput(result))
	}

	lines := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	if len(lines) == 0 || (len(lines) == 1 && strings.TrimSpace(lines[0]) == "") {
		return fmt.Errorf("命名空间 %s 未返回任何 Pod", namespace)
	}

	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}
		name := fields[0]
		status := fields[2]
		if strings.EqualFold(status, "CrashLoopBackOff") || strings.EqualFold(status, "Pending") || strings.EqualFold(status, "ContainerCreating") || strings.EqualFold(status, "ImagePullBackOff") || strings.EqualFold(status, "ErrImagePull") || strings.EqualFold(status, "Terminating") {
			return fmt.Errorf("命名空间 %s 存在状态异常的 Pod：%s", namespace, line)
		}
		if name == "" {
			return fmt.Errorf("命名空间 %s 存在无法解析的 Pod 行：%s", namespace, line)
		}
	}
	return nil
}

func verifyF5SITBKEClusterHealth(clusterManager *utils.ClusterManager, clusterName, kubeconfigPath string) error {
	phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
	if err != nil {
		return fmt.Errorf("获取引导集群中的 BKECluster 状态失败：%w", err)
	}
	if clusterStatus != "Ready" {
		return fmt.Errorf("引导集群中的 BKECluster 未就绪：phase=%s state=%s clusterStatus=%s", phase, state, clusterStatus)
	}

	phase, state, clusterStatus, err = clusterManager.GetManagementClusterSelfFullStatus(kubeconfigPath, clusterName)
	if err != nil {
		return fmt.Errorf("获取管理集群自管 BKECluster 状态失败：%w", err)
	}
	if clusterStatus != "Ready" {
		return fmt.Errorf("管理集群自管 BKECluster 未就绪：phase=%s state=%s clusterStatus=%s", phase, state, clusterStatus)
	}
	return nil
}

func waitForF5SITManagementNodesCleaned(nodes []config.NodeInfo, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	lastOutputs := make(map[string]string, len(nodes))
	command := `if systemctl is-active --quiet kubelet; then
  echo "kubelet is still active"
  exit 1
fi
if [ -e /var/lib/containerd ] && [ -n "$(ls -A /var/lib/containerd 2>/dev/null)" ]; then
  echo "/var/lib/containerd is not empty"
  exit 1
fi
echo "management node cleanup completed"`

	for {
		allCleaned := true
		for _, nodeResult := range executeF5SITCommandOnNodes(nodes, command) {
			output := f5SITCommandOutput(nodeResult.result)
			if nodeResult.err != nil {
				allCleaned = false
				if strings.TrimSpace(output) == "" {
					output = nodeResult.err.Error()
				}
				lastOutputs[nodeResult.node.IP] = output
			}
		}
		if allCleaned {
			return nil
		}
		if !time.Now().Before(deadline) {
			return fmt.Errorf("等待节点清理超时：%s", f5SITResetLastOutputs(lastOutputs))
		}
		GinkgoWriter.Printf("管理集群节点仍在清理中：%s\n", f5SITResetLastOutputs(lastOutputs))
		time.Sleep(10 * time.Second)
	}
}
func f5SITWaitForBootstrapResourcesDeleted(guideSSH *executor.SSHExecutor, clusterName string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastOutput string
	for {
		deleted, output := f5SITBootstrapResourcesFullyDeleted(guideSSH, clusterName)
		if deleted {
			return nil
		}
		lastOutput = output
		if !time.Now().Before(deadline) {
			return fmt.Errorf("等待超时，残留资源：%s", lastOutput)
		}
		GinkgoWriter.Printf("F5 SIT 引导集群资源仍在清理中：%s\n", lastOutput)
		time.Sleep(10 * time.Second)
	}
}

func f5SITBootstrapResourcesFullyDeleted(guideSSH *executor.SSHExecutor, clusterName string) (bool, string) {
	namespace := "bke-" + clusterName
	command := fmt.Sprintf(`bc="$(kubectl get bc bke-%s -n %s -o name --ignore-not-found 2>/dev/null || true)"
bn="$(kubectl get bn -n %s -o name --ignore-not-found 2>/dev/null || true)"
ns="$(kubectl get namespace %s -o name --ignore-not-found 2>/dev/null || true)"
printf 'namespace=%%s bc=%%s bn=%%s\n' "$ns" "$bc" "$bn"
[ -z "$bc$bn$ns" ]`, clusterName, namespace, namespace, namespace)
	result, err := guideSSH.Exec(command)
	output := f5SITCommandOutput(result)
	return err == nil && result != nil && result.ExitCode == 0, output
}
func f5SITBootstrapResourcesDeleted(guideSSH *executor.SSHExecutor, clusterName string) bool {
	namespace := "bke-" + clusterName
	command := fmt.Sprintf(`bc="$(kubectl get bc bke-%s -n %s -o name --ignore-not-found 2>/dev/null || true)"
bn="$(kubectl get bn -n %s -o name --ignore-not-found 2>/dev/null || true)"
ns_phase="$(kubectl get namespace %s -o jsonpath='{.status.phase}' 2>/dev/null || true)"
[ -z "$bc$bn" ] && [ "$ns_phase" != "Terminating" ]`, clusterName, namespace, namespace, namespace)
	result, err := guideSSH.Exec(command)
	return err == nil && result != nil && result.ExitCode == 0
}

func collectF5SITAuditDiagnostics(nodes []config.NodeInfo) {
	command := fmt.Sprintf(
		"systemctl status auditd --no-pager 2>&1; auditctl -l 2>&1; for key in %s; do ausearch -k \"$key\" -i 2>/dev/null | tail -n 30; done",
		strings.Join(f5SITAuditKeys, " "),
	)
	for _, nodeResult := range executeF5SITCommandOnNodes(nodes, command) {
		GinkgoWriter.Printf("节点 %s 的审计诊断信息：\n%s\n", nodeResult.node.IP, f5SITCommandOutput(nodeResult.result))
	}
}

func f5SITCommandOutput(result *executor.ExecResult) string {
	if result == nil {
		return "<无命令输出>"
	}
	output := strings.TrimSpace(strings.Join([]string{result.Stdout, result.Stderr}, "\n"))
	if len(output) > 4000 {
		return output[len(output)-4000:]
	}
	return output
}
