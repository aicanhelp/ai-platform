package abnormal_retry

import (
	"fmt"
	"path/filepath"
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

const (
	// specTimeout 单个 It 用例的总超时（含创建+等待+清理），参考 concurrent 模块 installationItTimeout=45min。
	specTimeout = 45 * time.Minute
	// clusterHealthyTimeout 等待集群达到 Healthy/Ready 的超时时间。
	clusterHealthyTimeout = 15 * time.Minute
	// retryWaitTimeout 用于等待单次 BKENode 状态变更（retryCount/state 出现）。
	retryWaitTimeout = 10 * time.Minute
	// uninstallTimeout 集群清理超时，与 declarative-upgrade / concurrent 模块一致。
	uninstallTimeout = 15 * time.Minute
	pollInterval     = 30 * time.Second
	fastPollInterval = 15 * time.Second

	// maxRetryCount BKE 控制器推送 BKEAgent 的最大重试次数（达到后 worker 标记 NeedSkip 或集群进入 Failed）
	maxRetryCount = 3

	// RetryAnnotationKey BKECluster 上的 retry 注解 key，触发控制器清除 NeedSkip / ManualInterventionRequired
	RetryAnnotationKey = "bke.bocloud.com/retry"

	// NodeBootWaitTimeoutAnnotationKey BKECluster 注解，控制等待 NodeAgentPushedFlag 可见的超时时间
	// 默认 10m，测试中缩短为 1m 以加速 push 阶段（ping 超时本身硬编码 5min，无法通过注解调整）
	NodeBootWaitTimeoutAnnotationKey = "bke.bocloud.com/node-boot-wait-timeout"
	// FastNodeBootWaitTimeout 加速用短超时
	FastNodeBootWaitTimeout = "1m"

	// BKENode state 常量
	bkeNodeStateInitFailed = "InitFailed"
	bkeNodeStateReady      = "Ready"

	// stateCode 标志位
	nodeAgentPushedFlag = 1 // NodeAgentPushedFlag
	nodeAgentReadyFlag  = 2 // NodeAgentReadyFlag

	// 集群终态后缀
	failedSuffix = "Failed"

	// ManualInterventionRequired condition 字段
	conditionManualInterventionRequired = "ManualInterventionRequired"
	reasonRetryExhausted                = "RetryExhausted"

	// mgmtKubeconfigPath 管理集群与引导节点同机共部署时，引导节点默认 kubeconfig 即管理集群 API
	mgmtKubeconfigPath = ""

	failedStatesMsgTmpl = "集群进入失败态: state=%s, clusterStatus=%s，不再等待 Healthy"
)

// AbnormalRetryCtx 异常重试测试上下文，持有各执行器与管理器。
type AbnormalRetryCtx struct {
	SSHExecutor    *executor.SSHExecutor
	LocalExecutor  *executor.LocalExecutor
	ClusterManager *utils.ClusterManager
	GuideConfig    *config.GuideNodeConfig
}

// setupAbnormalRetryFixtures 初始化 SSH、本地执行器与 ClusterManager。
func setupAbnormalRetryFixtures() *AbnormalRetryCtx {
	guideConfig := config.LoadGuideNodeFromEnv()
	Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
	Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")

	var err error
	sshExecutor, err := executor.NewSSHExecutor(
		guideConfig.Host,
		guideConfig.Port,
		guideConfig.Username,
		guideConfig.Password,
	)
	Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")

	localExecutor := executor.NewLocalExecutor(30 * time.Second)

	kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
	restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")

	dynamicClient, err := dynamic.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")

	clusterManager := utils.NewClusterManager(sshExecutor, dynamicClient)

	return &AbnormalRetryCtx{
		SSHExecutor:    sshExecutor,
		LocalExecutor:  localExecutor,
		ClusterManager: clusterManager,
		GuideConfig:    guideConfig,
	}
}

// ==================== BKENode 状态查询 ====================

// getBKENodeNameByIP 通过 IP 在引导侧集群命名空间查找 BKENode 名称。
func getBKENodeNameByIP(ctx *AbnormalRetryCtx, clusterName, nodeIP, kubeconfigPath string) string {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(`%skubectl get bkenode -n %s -o jsonpath='{range .items[*]}{.metadata.name},{.spec.ip}{"\n"}{end}'`,
		kubeconfigArg, namespace)
	result, err := ctx.SSHExecutor.Exec(cmd)
	Expect(err).NotTo(HaveOccurred(), "查询 BKENode 列表应该成功")
	Expect(result.ExitCode).To(BeZero(), "查询 BKENode 列表应该成功: %s", result.Stderr)
	return parseBKENodeNameByIP(result.Stdout, nodeIP)
}

// getBKENodeNameByIPInNamespace 在指定命名空间内按 IP 查找 BKENode 名称（查询失败返回空，由调用方决定重试/跳过）。
func getBKENodeNameByIPInNamespace(ctx *AbnormalRetryCtx, namespace, nodeIP, kubeconfigPath string) string {
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(`%skubectl get bkenode -n %s -o jsonpath='{range .items[*]}{.metadata.name},{.spec.ip}{"\n"}{end}'`,
		kubeconfigArg, namespace)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || result == nil || result.ExitCode != 0 {
		return ""
	}
	return parseBKENodeNameByIP(result.Stdout, nodeIP)
}

func parseBKENodeNameByIP(stdout, nodeIP string) string {
	for _, line := range strings.Split(strings.Trim(stdout, "'"), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ",", 2)
		if len(parts) == 2 && parts[1] == nodeIP {
			return parts[0]
		}
	}
	return ""
}

// getBKENodeField 获取 BKENode 的指定 jsonpath 字段值。
func getBKENodeField(ctx *AbnormalRetryCtx, clusterName, nodeName, field, kubeconfigPath string) (string, error) {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(`%skubectl get bkenode %s -n %s -o jsonpath='{%s}'`,
		kubeconfigArg, nodeName, namespace, field)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil {
		return "", fmt.Errorf("查询 BKENode %s 字段 %s 失败: %w", nodeName, field, err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("查询 BKENode %s 字段 %s 失败: %s", nodeName, field, result.Stderr)
	}
	return strings.Trim(strings.TrimSpace(result.Stdout), "'"), nil
}

// getBKENodeRetryCount 获取 BKENode 的 status.retryCount 值，返回 (count, exists)。
func getBKENodeRetryCount(ctx *AbnormalRetryCtx, clusterName, nodeName, kubeconfigPath string) (int, bool) {
	val, err := getBKENodeField(ctx, clusterName, nodeName, ".status.retryCount", kubeconfigPath)
	if err != nil || val == "" {
		return 0, false
	}
	count, err := strconv.Atoi(val)
	if err != nil {
		return 0, false
	}
	return count, true
}

// getBKENodeNeedSkip 获取 BKENode 的 status.needSkip 值。
func getBKENodeNeedSkip(ctx *AbnormalRetryCtx, clusterName, nodeName, kubeconfigPath string) bool {
	val, err := getBKENodeField(ctx, clusterName, nodeName, ".status.needSkip", kubeconfigPath)
	if err != nil {
		return false
	}
	return val == "true"
}

// getBKENodeState 获取 BKENode 的 status.state 值。
func getBKENodeState(ctx *AbnormalRetryCtx, clusterName, nodeName, kubeconfigPath string) string {
	val, _ := getBKENodeField(ctx, clusterName, nodeName, ".status.state", kubeconfigPath)
	return val
}

// getBKENodeStateCode 获取 BKENode 的 status.stateCode 值。
func getBKENodeStateCode(ctx *AbnormalRetryCtx, clusterName, nodeName, kubeconfigPath string) int {
	val, _ := getBKENodeField(ctx, clusterName, nodeName, ".status.stateCode", kubeconfigPath)
	if val == "" {
		return 0
	}
	code, _ := strconv.Atoi(val)
	return code
}

// getBKENodeMessage 获取 BKENode 的 status.message 值。
func getBKENodeMessage(ctx *AbnormalRetryCtx, clusterName, nodeName, kubeconfigPath string) string {
	val, _ := getBKENodeField(ctx, clusterName, nodeName, ".status.message", kubeconfigPath)
	return val
}

// ==================== BKECluster 状态查询 ====================

// getBKEClusterStatus 获取 BKECluster 的 status.clusterStatus 值。
func getBKEClusterStatus(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string) string {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(`%skubectl get bc %s -n %s -o jsonpath='{.status.clusterStatus}'`,
		kubeconfigArg, bcName, namespace)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || result.ExitCode != 0 {
		return ""
	}
	return strings.Trim(strings.TrimSpace(result.Stdout), "'")
}

// hasManualInterventionRequired 检查 BKECluster 的 conditions 中是否存在 ManualInterventionRequired=True。
func hasManualInterventionRequired(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string) bool {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(
		`%skubectl get bc %s -n %s -o jsonpath='{.status.conditions[?(@.type=="ManualInterventionRequired")].status}'`,
		kubeconfigArg, bcName, namespace)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || result.ExitCode != 0 {
		return false
	}
	return strings.Trim(strings.TrimSpace(result.Stdout), "'") == "True"
}

// getManualInterventionRequiredReason 获取 ManualInterventionRequired condition 的 reason。
func getManualInterventionRequiredReason(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string) string {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(
		`%skubectl get bc %s -n %s -o jsonpath='{.status.conditions[?(@.type=="ManualInterventionRequired")].reason}'`,
		kubeconfigArg, bcName, namespace)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || result.ExitCode != 0 {
		return ""
	}
	return strings.Trim(strings.TrimSpace(result.Stdout), "'")
}

// getManualInterventionRequiredMessage 获取 ManualInterventionRequired condition 的 message。
func getManualInterventionRequiredMessage(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string) string {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(
		`%skubectl get bc %s -n %s -o jsonpath='{.status.conditions[?(@.type=="ManualInterventionRequired")].message}'`,
		kubeconfigArg, bcName, namespace)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || result.ExitCode != 0 {
		return ""
	}
	return strings.Trim(strings.TrimSpace(result.Stdout), "'")
}

// hasRetryExhaustedEvent 检查 BKECluster 的 Events 中是否存在 reason=RetryExhausted 的 Warning 事件。
func hasRetryExhaustedEvent(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string) bool {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(`%skubectl get events -n %s --field-selector reason=%s --no-headers 2>/dev/null | head -5`,
		kubeconfigArg, namespace, reasonRetryExhausted)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || result.ExitCode != 0 {
		return false
	}
	return strings.TrimSpace(result.Stdout) != ""
}

// getClusterEvents 获取 BKECluster 的 Events 输出。
func getClusterEvents(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string) string {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(`%skubectl get events -n %s --sort-by=.lastTimestamp --no-headers 2>/dev/null | tail -20`,
		kubeconfigArg, namespace)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || result.ExitCode != 0 {
		return ""
	}
	return result.Stdout
}

// ==================== Retry 注解 ====================

// addRetryAnnotation 在 BKECluster 资源上添加 RetryAnnotationKey 注解。
// nodeIPs 为空字符串时重试所有节点；为逗号分隔的 IP 列表时仅重试指定节点。
// 参考 cluster-api-provider-bke controllers/capbke/bkecluster_controller.go handleRetryLogic。
func addRetryAnnotation(ctx *AbnormalRetryCtx, clusterName, nodeIPs, kubeconfigPath string) error {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf(`%skubectl annotate bc %s -n %s %s=%s --overwrite`,
		kubeconfigArg, bcName, namespace, RetryAnnotationKey, shQuote(nodeIPs))
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("添加 retry 注解失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("添加 retry 注解失败: %s", result.Stderr)
	}
	GinkgoWriter.Printf("已添加 retry 注解: bc=%s/%s, nodeIPs=%q\n", namespace, bcName, nodeIPs)
	return nil
}

// ==================== 故障注入与恢复 ====================

// injectWrongPassword 将节点密码修改为错误值，用于模拟 BKEAgent 推送失败（SSH 不可达）。
// 返回修改前的原始节点副本（含正确密码），供后续恢复使用。
func injectWrongPassword(node *config.NodeInfo) config.NodeInfo {
	original := *node
	node.Password = "wrong-password-" + node.Password
	GinkgoWriter.Printf("已注入错误密码到节点 %s (%s)\n", node.Hostname, node.IP)
	return original
}

// patchBKENodePassword 更新 BKENode 的 spec.password 为正确值，用于恢复 SSH 可达性。
// patch 后会通过 SSH 验证密码可用性，并等待 fail2ban 限制解除（如有）。
func patchBKENodePassword(ctx *AbnormalRetryCtx, clusterName, nodeName string, node config.NodeInfo, kubeconfigPath string) error {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	patchJSON := fmt.Sprintf(`{"spec":{"password":%s}}`, jsonQuote(node.Password))
	cmd := fmt.Sprintf(`%skubectl patch bkenode %s -n %s --type=merge -p '%s'`,
		kubeconfigArg, nodeName, namespace, patchJSON)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("patch BKENode %s 密码失败: %w", nodeName, err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("patch BKENode %s 密码失败: %s", nodeName, result.Stderr)
	}
	GinkgoWriter.Printf("已恢复节点 %s (%s) 的正确密码\n", nodeName, node.IP)

	By(fmt.Sprintf("验证节点 %s SSH 可达性（等待 fail2ban 限制解除）", node.IP))
	Eventually(func() bool {
		sshResult, sshErr := utils.ExecuteCommandOnNode(node, "echo ok")
		if sshErr != nil {
			GinkgoWriter.Printf("SSH 验证失败: %v (stderr: %s)\n", sshErr, func() string {
				if sshResult != nil {
					return sshResult.Stderr
				}
				return ""
			}())
			return false
		}
		return strings.TrimSpace(sshResult.Stdout) == "ok"
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(),
		fmt.Sprintf("节点 %s SSH 应在 2 分钟内恢复可达（可能被 fail2ban 临时限制）", node.IP))
	GinkgoWriter.Printf("节点 %s SSH 验证通过\n", node.IP)
	return nil
}

// patchMgmtClusterBKENodePassword 更新管理集群自管 BC 上的 BKENode spec.password。
// 自管 BC 继承引导侧逻辑名（sig-installation#224）：namespace = bke-{bootstrapClusterName}；
// BN 优先继承引导侧 BKENode.metadata.name（E2E 里通常就是 hostname，如 w1），
// 不再固定为 bke-cluster-<hostname> 或 <自管BC名>-<hostname>。
// 这些 BKENode 是由 EnsureClusterAPIManagerManifest 阶段一次性快照创建的，
// patch 引导节点 BKENode 密码不会同步到管理集群，需要单独 patch。
// mgmtKubeconfigPath 为管理集群 kubeconfig 文件路径（在引导节点上执行 kubectl）。
func patchMgmtClusterBKENodePassword(ctx *AbnormalRetryCtx, node config.NodeInfo, mgmtKubeconfigPath, bootstrapClusterName string) error {
	if mgmtKubeconfigPath == "" {
		return fmt.Errorf("管理集群 kubeconfig 路径不能为空")
	}
	if strings.TrimSpace(bootstrapClusterName) == "" {
		return fmt.Errorf("bootstrapClusterName 不能为空")
	}
	selfNS := utils.ManagementClusterSelfBCNamespace(bootstrapClusterName)
	mgmtNodeName := getBKENodeNameByIPInNamespace(ctx, selfNS, node.IP, mgmtKubeconfigPath)
	if mgmtNodeName == "" {
		// 兼容：尚未列出时先按继承名（hostname）再按旧拼接名尝试
		for _, candidate := range []string{
			node.Hostname,
			utils.ManagementClusterSelfBCName(bootstrapClusterName) + "-" + node.Hostname,
		} {
			checkCmd := fmt.Sprintf(`KUBECONFIG=%s kubectl get bkenode %s -n %s --no-headers 2>/dev/null`,
				mgmtKubeconfigPath, candidate, selfNS)
			if result, err := ctx.SSHExecutor.Exec(checkCmd); err == nil && result != nil && result.ExitCode == 0 {
				mgmtNodeName = candidate
				break
			}
		}
	}
	if mgmtNodeName == "" {
		GinkgoWriter.Printf("管理集群自管 ns=%s 上尚未找到 IP=%s 的 BKENode，跳过 patch\n", selfNS, node.IP)
		return nil
	}
	kubeconfigArg := fmt.Sprintf("KUBECONFIG=%s ", mgmtKubeconfigPath)
	patchJSON := fmt.Sprintf(`{"spec":{"password":%s}}`, jsonQuote(node.Password))
	cmd := fmt.Sprintf(`%skubectl patch bkenode %s -n %s --type=merge -p '%s'`,
		kubeconfigArg, mgmtNodeName, selfNS, patchJSON)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if result != nil && result.ExitCode != 0 {
		if strings.Contains(result.Stderr, "not found") {
			GinkgoWriter.Printf("管理集群 BKENode %s 尚未创建（自管 BC 可能还未注册），跳过 patch\n", mgmtNodeName)
			return nil
		}
		return fmt.Errorf("patch 管理集群 BKENode %s 密码失败: exit=%d stderr=%s", mgmtNodeName, result.ExitCode, result.Stderr)
	}
	if err != nil {
		return fmt.Errorf("patch 管理集群 BKENode %s 密码失败: %w", mgmtNodeName, err)
	}
	GinkgoWriter.Printf("已恢复管理集群 BKENode %s (%s) 的正确密码\n", mgmtNodeName, node.IP)
	return nil
}

// occupyBkeagentPort 在远程节点上占用 bkeagent 健康监听端口（58080），使 bkeagent 无法启动监听。
// bkeagent 进程因端口冲突无法正常启动 → 无法连接 K8s API 监听 Command CRD → 控制器 pingAgent 超时（~5min），
// 从而达到模拟 ping bkeagent 失败的效果。
// 参考 health_mechanism_test.go 中 bkeagent 端口占用方案。
// 返回占用进程的 PID，用于后续释放。
func occupyBkeagentPort(node config.NodeInfo) (string, error) {
	By(fmt.Sprintf("预检查：停止节点 %s 上可能已有的 bkeagent（避免抢占 58080）", node.IP))
	stopCmd := "systemctl stop bkeagent 2>/dev/null || true; pkill -9 -x bkeagent >/dev/null 2>&1 || true"
	_, _ = utils.ExecuteCommandOnNode(node, stopCmd)

	By(fmt.Sprintf("等待确认 58080 端口已未被监听（节点 %s）", node.IP))
	Eventually(func() bool {
		checkPortCmd := "((ss -ltn 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E '(:|\\.)58080\\b' >/dev/null 2>&1) && echo listening || echo not"
		res, _ := utils.ExecuteCommandOnNode(node, checkPortCmd)
		if res == nil {
			return false
		}
		return strings.TrimSpace(res.Stdout) == "not"
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "58080 端口应在 1 分钟内释放/停止监听")

	By(fmt.Sprintf("在节点 %s 上占用 58080 端口（nc while 循环持续占用）", node.IP))
	occupyCmd := "nohup sh -c 'while true; do nc -l 58080 >/dev/null 2>&1; done' >/dev/null 2>&1 & echo $!"
	occupyResult, err := utils.ExecuteCommandOnNode(node, occupyCmd)
	if err != nil {
		return "", fmt.Errorf("在节点 %s 上占用 58080 端口失败: %w", node.IP, err)
	}
	occupyPid := strings.TrimSpace(occupyResult.Stdout)
	if occupyPid == "" {
		return "", fmt.Errorf("未获取到占用 58080 端口的进程 PID（节点 %s）", node.IP)
	}

	By(fmt.Sprintf("等待确认 58080 端口已被占用（节点 %s, pid=%s）", node.IP, occupyPid))
	Eventually(func() bool {
		checkPortCmd := "((ss -ltn 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E '(:|\\.)58080\\b' >/dev/null 2>&1) && echo listening || echo not"
		res, _ := utils.ExecuteCommandOnNode(node, checkPortCmd)
		if res == nil {
			return false
		}
		return strings.TrimSpace(res.Stdout) == "listening"
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "58080 端口应在 1 分钟内开始被监听")

	GinkgoWriter.Printf("已在节点 %s 上占用 58080 端口 (pid=%s)\n", node.IP, occupyPid)
	return occupyPid, nil
}

// releaseBkeagentPort 释放节点上被占用的 bkeagent 监听端口（58080）。
// 通过 kill 占用进程 PID 和清理残留 nc 监听来释放端口。
func releaseBkeagentPort(node config.NodeInfo, occupyPid string) {
	By(fmt.Sprintf("释放节点 %s 上的 58080 端口占用 (pid=%s)", node.IP, occupyPid))
	stopCmd := fmt.Sprintf(
		"kill -9 %s >/dev/null 2>&1 || true; pkill -9 -f 'nc.*58080' >/dev/null 2>&1 || true; pkill -9 -f 'nc -l 58080' >/dev/null 2>&1 || true",
		occupyPid)
	_, _ = utils.ExecuteCommandOnNode(node, stopCmd)

	Eventually(func() bool {
		checkPortCmd := "((ss -ltn 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E '(:|\\.)58080\\b' >/dev/null 2>&1) && echo listening || echo not"
		res, _ := utils.ExecuteCommandOnNode(node, checkPortCmd)
		if res == nil {
			return false
		}
		return strings.TrimSpace(res.Stdout) == "not"
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "58080 端口应在 2 分钟内释放")

	GinkgoWriter.Printf("已释放节点 %s 上的 58080 端口\n", node.IP)
}

// ==================== 轮询等待 ====================

// waitBKENodeRetryCountGreaterThan 等待 BKENode 的 retryCount 大于指定值。
func waitBKENodeRetryCountGreaterThan(ctx *AbnormalRetryCtx, clusterName, nodeName string, minCount int, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		count, exists := getBKENodeRetryCount(ctx, clusterName, nodeName, kubeconfigPath)
		if !exists {
			GinkgoWriter.Printf("[%s] retryCount 字段尚不存在\n", nodeName)
			return false
		}
		GinkgoWriter.Printf("[%s] retryCount=%d (期望 > %d)\n", nodeName, count, minCount)
		return count > minCount
	}, timeout, fastPollInterval).Should(BeTrue(),
		fmt.Sprintf("BKENode %s 的 retryCount 应大于 %d", nodeName, minCount))
}

// waitBKENodeState 等待 BKENode 的 state 达到期望值。
func waitBKENodeState(ctx *AbnormalRetryCtx, clusterName, nodeName, expectedState, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		state := getBKENodeState(ctx, clusterName, nodeName, kubeconfigPath)
		GinkgoWriter.Printf("[%s] state=%s (期望 %s)\n", nodeName, state, expectedState)
		return state == expectedState
	}, timeout, fastPollInterval).Should(BeTrue(),
		fmt.Sprintf("BKENode %s 的 state 应为 %s", nodeName, expectedState))
}

// waitBKENodeNeedSkip 等待 BKENode 的 needSkip 达到期望值。
func waitBKENodeNeedSkip(ctx *AbnormalRetryCtx, clusterName, nodeName string, expected bool, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		got := getBKENodeNeedSkip(ctx, clusterName, nodeName, kubeconfigPath)
		GinkgoWriter.Printf("[%s] needSkip=%v (期望 %v)\n", nodeName, got, expected)
		return got == expected
	}, timeout, fastPollInterval).Should(BeTrue(),
		fmt.Sprintf("BKENode %s 的 needSkip 应为 %v", nodeName, expected))
}

// waitBKENodeRetryCountEquals 等待 BKENode 的 retryCount 等于指定值。
func waitBKENodeRetryCountEquals(ctx *AbnormalRetryCtx, clusterName, nodeName string, expected int, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		count, exists := getBKENodeRetryCount(ctx, clusterName, nodeName, kubeconfigPath)
		GinkgoWriter.Printf("[%s] retryCount=%d (exists=%v, 期望 %d)\n", nodeName, count, exists, expected)
		return exists && count == expected
	}, timeout, fastPollInterval).Should(BeTrue(),
		fmt.Sprintf("BKENode %s 的 retryCount 应等于 %d", nodeName, expected))
}

// waitBKENodeRetryCountAbsent 等待 BKENode 的 retryCount 字段消失（成功后重置）。
func waitBKENodeRetryCountAbsent(ctx *AbnormalRetryCtx, clusterName, nodeName, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		_, exists := getBKENodeRetryCount(ctx, clusterName, nodeName, kubeconfigPath)
		GinkgoWriter.Printf("[%s] retryCount exists=%v (期望 false)\n", nodeName, exists)
		return !exists
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("BKENode %s 的 retryCount 字段应消失", nodeName))
}

// waitClusterStatusContains 等待集群 clusterStatus 包含指定后缀（如 Failed）。
func waitClusterStatusContains(ctx *AbnormalRetryCtx, clusterName, suffix, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		_, state, clusterStatus, err := ctx.ClusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[%s] 获取集群状态失败: %v\n", clusterName, err)
			return false
		}
		GinkgoWriter.Printf("[%s] state=%s, clusterStatus=%s (期望包含 %s)\n", clusterName, state, clusterStatus, suffix)
		return strings.HasSuffix(clusterStatus, suffix)
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("集群 %s 的 clusterStatus 应包含后缀 %s", clusterName, suffix))
}

// waitManualInterventionRequired 等待 BKECluster 出现 ManualInterventionRequired=True 条件。
func waitManualInterventionRequired(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		got := hasManualInterventionRequired(ctx, clusterName, kubeconfigPath)
		GinkgoWriter.Printf("[%s] ManualInterventionRequired=%v (期望 true)\n", clusterName, got)
		return got
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("集群 %s 应出现 ManualInterventionRequired=True 条件", clusterName))
}

// waitManualInterventionRequiredCleared 等待 BKECluster 的 ManualInterventionRequired 条件被清除（status=False）。
func waitManualInterventionRequiredCleared(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		got := hasManualInterventionRequired(ctx, clusterName, kubeconfigPath)
		msg := getManualInterventionRequiredMessage(ctx, clusterName, kubeconfigPath)
		GinkgoWriter.Printf("[%s] ManualInterventionRequired=%v, message=%s (期望 false)\n", clusterName, got, msg)
		return !got
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("集群 %s 的 ManualInterventionRequired 条件应被清除", clusterName))
}

// waitClusterStatusEquals 等待集群 clusterStatus 等于指定值。
func waitClusterStatusEquals(ctx *AbnormalRetryCtx, clusterName, expected, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		_, _, clusterStatus, err := ctx.ClusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if err != nil {
			return false
		}
		GinkgoWriter.Printf("[%s] clusterStatus=%s (期望 %s)\n", clusterName, clusterStatus, expected)
		return clusterStatus == expected
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("集群 %s 的 clusterStatus 应为 %s", clusterName, expected))
}

// waitClusterHealthyReady 等待集群达到 state=Healthy 且 clusterStatus=Ready。
// 参考 concurrent 模块 waitClusterHealthyReady 和 diagnosis_handle_test.go 中的判断标准。
func waitClusterHealthyReady(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		phase, state, clusterStatus, err := ctx.ClusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[%s] 获取状态失败: %v\n", clusterName, err)
			return false
		}
		GinkgoWriter.Printf("[%s] phase=%s, state=%s, clusterStatus=%s (期望 Ready)\n",
			clusterName, phase, state, clusterStatus)
		return clusterStatus == "Ready"
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("集群 %s 应达到 Ready", clusterName))
}

// waitMgmtClusterFullyHealthy 等待管理集群完全健康（引导节点 BC + 管理集群自管 BC 均 Healthy/Ready）。
// 参考 cluster_test_helpers.go WaitAndVerifyManagementCluster 和 declarative-upgrade waitDeclarativeUpgradeMgmtClusterFullyHealthy。
// 管理集群创建后存在两个 BC：引导侧 BC（bke-{clusterName}）与自管 BC（同名，继承引导侧逻辑名，见 #224）。
func waitMgmtClusterFullyHealthy(ctx *AbnormalRetryCtx, clusterName string, timeout time.Duration) {
	By("等待引导节点上管理集群 BC 变为 Ready")
	Eventually(func() bool {
		phase, state, clusterStatus, err := ctx.ClusterManager.GetClusterFullStatusWithKubeconfig(clusterName, mgmtKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[bootstrap] 获取集群状态失败: %v\n", err)
			return false
		}
		GinkgoWriter.Printf("[bootstrap] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		return clusterStatus == "Ready"
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("引导节点上管理集群 %s 应达到 Ready", clusterName))

	mgmtKubeconfigPath := getMgmtClusterKubeconfig(ctx, clusterName)

	By("等待管理集群自管 BC 变为 Ready")
	Eventually(func() bool {
		phase, state, clusterStatus, err := ctx.ClusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, clusterName)
		if err != nil {
			GinkgoWriter.Printf("[mgmt-self] 获取自管 BC 状态失败: %v\n", err)
			return false
		}
		GinkgoWriter.Printf("[mgmt-self] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		return clusterStatus == "Ready"
	}, timeout, pollInterval).Should(BeTrue(),
		"管理集群自管 BC 应达到 Ready")

	_, _ = ctx.SSHExecutor.Exec(fmt.Sprintf("rm -f %s", mgmtKubeconfigPath))
}

// getMgmtClusterKubeconfig 获取管理集群 kubeconfig 并返回临时文件路径（落在引导节点上）。
// 必须经 SSH 在引导节点读取 Secret/~/.bke；本机 LocalExecutor 无 kubectl 时会失败。
// 调用方负责在完成后清理临时文件。
func getMgmtClusterKubeconfig(ctx *AbnormalRetryCtx, clusterName string) string {
	By("获取管理集群 kubeconfig（引导节点）")
	checker := utils.NewClusterCheckerWithParentKubeconfig(ctx.SSHExecutor, clusterName, mgmtKubeconfigPath)
	var savedPath string
	Eventually(func() bool {
		path, err := checker.SaveKubeconfigToFile()
		if err != nil {
			GinkgoWriter.Printf("获取管理集群 kubeconfig 失败: %v\n", err)
			return false
		}
		savedPath = path
		return true
	}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "应该成功获取管理集群 kubeconfig")
	GinkgoWriter.Printf("管理集群 kubeconfig: %s\n", savedPath)
	return savedPath
}

// patchMgmtClusterBKENodesAndRetry 在管理集群自管 BC 上 patch BKENode 密码并添加 retry 注解。
// 用于管理集群场景（FIT_8/FIT_9）：patch 引导节点密码后，需同步 patch 管理集群自身的 BKENode 密码，
// 否则 EnsureClusterManage 阶段因快照密码错误而纳管失败。
// 同时补写 node-boot-wait-timeout（自管 BC 不继承引导侧注解，默认等 flag 10m）。
// 流程：等待引导节点 BC Healthy → 获取管理集群 kubeconfig → 缩短 boot-wait → patch 密码 → retry 注解。
func patchMgmtClusterBKENodesAndRetry(ctx *AbnormalRetryCtx, clusterName string, nodes []config.NodeInfo) {
	By("等待引导节点上管理集群 BC 变为 Ready（确保自管 BC 已注册）")
	Eventually(func() bool {
		_, state, clusterStatus, err := ctx.ClusterManager.GetClusterFullStatusWithKubeconfig(clusterName, mgmtKubeconfigPath)
		if err != nil {
			return false
		}
		GinkgoWriter.Printf("[bootstrap] state=%s, clusterStatus=%s\n", state, clusterStatus)
		return clusterStatus == "Ready"
	}, clusterHealthyTimeout, pollInterval).Should(BeTrue(),
		"引导节点上管理集群应达到 Ready 后再 patch 自管 BC BKENode")

	mgmtKubeconfigPath := getMgmtClusterKubeconfig(ctx, clusterName)
	defer func() {
		_, _ = ctx.SSHExecutor.Exec(fmt.Sprintf("rm -f %s", mgmtKubeconfigPath))
	}()

	mgmtBcName := utils.ManagementClusterSelfBCName(clusterName)
	mgmtBcNamespace := utils.ManagementClusterSelfBCNamespace(clusterName)
	kubeconfigArg := fmt.Sprintf("KUBECONFIG=%s ", mgmtKubeconfigPath)

	// 自管 BC 不继承引导侧注解；不设则等 NodeAgentPushedFlag 默认 10m。
	By(fmt.Sprintf("在自管 BC 上设置 %s=%s（缩短等 flag 超时）", NodeBootWaitTimeoutAnnotationKey, FastNodeBootWaitTimeout))
	bootWaitCmd := fmt.Sprintf(`%skubectl annotate bc %s -n %s %s=%s --overwrite`,
		kubeconfigArg, mgmtBcName, mgmtBcNamespace, NodeBootWaitTimeoutAnnotationKey, FastNodeBootWaitTimeout)
	bootWaitResult, err := ctx.SSHExecutor.Exec(bootWaitCmd)
	Expect(err).NotTo(HaveOccurred(), "设置自管 BC node-boot-wait-timeout 应成功")
	Expect(bootWaitResult.ExitCode).To(BeZero(), "设置自管 BC node-boot-wait-timeout 应成功: %s", bootWaitResult.Stderr)

	By("Patch 管理集群自管 BC 上的 BKENode 密码")
	for _, node := range nodes {
		err := patchMgmtClusterBKENodePassword(ctx, node, mgmtKubeconfigPath, clusterName)
		Expect(err).NotTo(HaveOccurred(), "patch 管理集群自管 BKENode %s (%s) 密码应成功", node.Hostname, node.IP)
	}

	By("在管理集群自管 BC 上添加 RetryAnnotationKey 注解")
	nodeIPs := make([]string, 0, len(nodes))
	for _, n := range nodes {
		nodeIPs = append(nodeIPs, n.IP)
	}
	retryValue := strings.Join(nodeIPs, ",")
	cmd := fmt.Sprintf(`%skubectl annotate bc %s -n %s %s=%s --overwrite`,
		kubeconfigArg, mgmtBcName, mgmtBcNamespace, RetryAnnotationKey, shQuote(retryValue))
	result, err := ctx.SSHExecutor.Exec(cmd)
	Expect(err).NotTo(HaveOccurred(), "添加 retry 注解到自管 BC 应成功")
	Expect(result.ExitCode).To(BeZero(), "添加 retry 注解到自管 BC 应成功: %s", result.Stderr)
	GinkgoWriter.Printf("已添加 retry 注解到自管 BC: nodeIPs=%q\n", retryValue)
}

// waitBKENodeExists 等待指定 IP 的 BKENode 出现并返回其名称。
func waitBKENodeExists(ctx *AbnormalRetryCtx, clusterName, nodeIP, kubeconfigPath string, timeout time.Duration) string {
	var nodeName string
	Eventually(func() bool {
		nodeName = getBKENodeNameByIP(ctx, clusterName, nodeIP, kubeconfigPath)
		return nodeName != ""
	}, timeout, fastPollInterval).Should(BeTrue(), fmt.Sprintf("IP 为 %s 的 BKENode 应存在", nodeIP))
	return nodeName
}

// waitBKENodePushed 等待 BKENode 的 stateCode 包含 NodeAgentPushedFlag。
func waitBKENodePushed(ctx *AbnormalRetryCtx, clusterName, nodeName, kubeconfigPath string, timeout time.Duration) {
	Eventually(func() bool {
		stateCode := getBKENodeStateCode(ctx, clusterName, nodeName, kubeconfigPath)
		GinkgoWriter.Printf("[%s] stateCode=%d (期望包含 NodeAgentPushedFlag=%d)\n", nodeName, stateCode, nodeAgentPushedFlag)
		return hasStateCodeFlag(stateCode, nodeAgentPushedFlag)
	}, timeout, fastPollInterval).Should(BeTrue(),
		fmt.Sprintf("BKENode %s 的 BKEAgent 应已推送 (stateCode 含 NodeAgentPushedFlag)", nodeName))
}

// uploadMgmtClusterConfig 生成并上传管理集群配置文件（保留 cluster-api addon，含加速注解）。
func uploadMgmtClusterConfig(ctx *AbnormalRetryCtx, clusterName string, nodes []config.NodeInfo) (configPath, nodeConfigPath string) {
	cfg := config.NewDefaultBKEClusterConfig(clusterName, nodes)
	cfg.ExcludeAddons = []string{"openfuyao-system-controller"}
	cfg.Annotations = map[string]string{
		NodeBootWaitTimeoutAnnotationKey: FastNodeBootWaitTimeout,
	}
	var err error
	configPath, nodeConfigPath, err = ctx.ClusterManager.GetConfigGenerator().GenerateAndUpload(cfg)
	Expect(err).NotTo(HaveOccurred(), "生成管理集群配置文件应成功")
	return configPath, nodeConfigPath
}

// ==================== 集群生命周期 ====================

// uploadClusterConfig 生成并上传集群配置文件（含加速注解）。
func uploadClusterConfig(ctx *AbnormalRetryCtx, clusterName string, nodes []config.NodeInfo) (configPath, nodeConfigPath string) {
	cfg := config.NewDefaultBKEClusterConfig(clusterName, nodes)
	cfg.Annotations = map[string]string{
		NodeBootWaitTimeoutAnnotationKey: FastNodeBootWaitTimeout,
	}
	var err error
	configPath, nodeConfigPath, err = ctx.ClusterManager.GetConfigGenerator().GenerateAndUpload(cfg)
	Expect(err).NotTo(HaveOccurred(), "生成集群配置文件应成功")
	return configPath, nodeConfigPath
}

// createClusterInBackground 在后台创建集群（适用于预期可能失败的场景）。
func createClusterInBackground(ctx *AbnormalRetryCtx, configPath, nodeConfigPath, kubeconfigPath string) {
	err := ctx.ClusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "后台创建集群命令应成功执行")
}

// createClusterApply 通过 CR apply 创建集群（适用于管理集群上创建业务集群）。
func createClusterApply(ctx *AbnormalRetryCtx, configPath, nodeConfigPath, kubeconfigPath string) {
	err := ctx.ClusterManager.CreateClusterWithKubeconfig(configPath, nodeConfigPath, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "apply BC/BN 应成功")
}

// cleanupAbnormalRetryCluster 清理业务集群与配置文件（仅删除引导节点上的 BC）。
func cleanupAbnormalRetryCluster(ctx *AbnormalRetryCtx, clusterName, kubeconfigPath, configPath, nodeConfigPath string) {
	By("清理测试集群")
	if clusterName != "" && ctx.ClusterManager.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath) {
		err := ctx.ClusterManager.DeleteClusterWithKubeconfig(clusterName, kubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("触发集群删除失败: %v\n", err)
		}
		Eventually(func() bool {
			return !ctx.ClusterManager.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath)
		}, uninstallTimeout, 60*time.Second).Should(BeTrue(), "集群应被完全删除")
	}
	if configPath != "" {
		_ = ctx.ClusterManager.CleanupConfig(configPath)
	}
	if nodeConfigPath != "" {
		_ = ctx.ClusterManager.CleanupConfig(nodeConfigPath)
	}
}

// cleanupAbnormalRetryMgmtCluster 清理管理集群（引导 BC 先 reset、短窗等待、节点 reset、BC 收尾）。
func cleanupAbnormalRetryMgmtCluster(ctx *AbnormalRetryCtx, clusterName, configPath, nodeConfigPath string) {
	mgmtNodes := loadAbnormalRetryMgmtNodes()
	var configPaths []string
	if configPath != "" {
		configPaths = append(configPaths, configPath)
	}
	if nodeConfigPath != "" {
		configPaths = append(configPaths, nodeConfigPath)
	}
	ctx.ClusterManager.CleanupManagementCluster(utils.ManagementClusterCleanupOptions{
		ClusterName:        clusterName,
		BootstrapKubeconfig: mgmtKubeconfigPath,
		MgmtNodes:          mgmtNodes,
		LocalExecutor:      ctx.LocalExecutor,
		ConfigPaths:        configPaths,
	})
}

func loadAbnormalRetryMgmtNodes() []config.NodeInfo {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 2 {
		return nil
	}
	mgmtNodes := make([]config.NodeInfo, 2)
	copy(mgmtNodes, nodes[:2])
	return mgmtNodes
}

// uniqueClusterName 生成唯一的集群名称。
func uniqueClusterName(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().Unix())
}

// loadWorkloadNodes 加载测试节点。
func loadWorkloadNodes() []config.NodeInfo {
	nodes := config.LoadTestNodesFromEnv()
	Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要配置 TEST_NODE2 业务节点")
	return nodes
}

// requireMinWorkloadNodes 确保至少有指定数量的测试节点。
func requireMinWorkloadNodes(min int) []config.NodeInfo {
	nodes := loadWorkloadNodes()
	Expect(len(nodes)).To(BeNumerically(">=", min),
		"当前环境业务节点不足，需要至少 %d 台（从 TEST_NODE2 起）", min)
	return nodes
}

// ==================== 辅助函数 ====================

// isClusterFailedState 判断集群是否处于终态失败。
func isClusterFailedState(state, clusterStatus string) bool {
	return state == "DeployFailed" || strings.HasSuffix(clusterStatus, failedSuffix)
}

// failOnClusterFailure 集群进入终态失败时立即 Fail。
func failOnClusterFailure(state, clusterStatus string) {
	if isClusterFailedState(state, clusterStatus) {
		Fail(fmt.Sprintf(failedStatesMsgTmpl, state, clusterStatus))
	}
}

// shQuote 对字符串进行 shell 单引号转义。
func shQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}

// jsonQuote 对字符串进行 JSON 字符串引用。
func jsonQuote(value string) string {
	var sb strings.Builder
	sb.WriteByte('"')
	for _, r := range value {
		switch r {
		case '"':
			sb.WriteString(`\"`)
		case '\\':
			sb.WriteString(`\\`)
		case '\n':
			sb.WriteString(`\n`)
		case '\r':
			sb.WriteString(`\r`)
		case '\t':
			sb.WriteString(`\t`)
		default:
			sb.WriteRune(r)
		}
	}
	sb.WriteByte('"')
	return sb.String()
}

// hasStateCodeFlag 检查 stateCode 是否包含指定标志位。
func hasStateCodeFlag(stateCode, flag int) bool {
	return stateCode&flag != 0
}
