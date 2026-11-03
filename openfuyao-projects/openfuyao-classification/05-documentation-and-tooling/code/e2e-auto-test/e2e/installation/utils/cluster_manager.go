package utils

import (
	"context"
	"fmt"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"

	. "github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

const (
	// DefaultBootstraoClusterKubeconfig 引导集群的kubeconfig
	DefaultBootstraoClusterKubeconfig = "/etc/rancher/k3s/k3s.yaml"
)

// BKEClusterGVR BKECluster 资源的 GroupVersionResource
var BKEClusterGVR = schema.GroupVersionResource{
	Group:    "bke.bocloud.com",
	Version:  "v1beta1",
	Resource: "bkeclusters",
}

// BKENodeGVR BKENode 资源的 GroupVersionResource
var BKENodeGVR = schema.GroupVersionResource{
	Group:    "bke.bocloud.com",
	Version:  "v1beta1",
	Resource: "bkenodes",
}

// ClusterManager BKE 集群管理器
type ClusterManager struct {
	executor        *executor.SSHExecutor
	dynamicClient   dynamic.Interface
	configGenerator *ClusterConfigGenerator
}

// NewClusterManager 创建集群管理器
func NewClusterManager(exec *executor.SSHExecutor, dynamicClient dynamic.Interface) *ClusterManager {
	return &ClusterManager{
		executor:        exec,
		dynamicClient:   dynamicClient,
		configGenerator: NewClusterConfigGenerator(exec),
	}
}

// SSHExecutor 返回创建 ClusterManager 时使用的 SSH 执行器。
func (m *ClusterManager) SSHExecutor() *executor.SSHExecutor {
	return m.executor
}

func (m *ClusterManager) CreateBinaryConfig(kubeletConfigPath, containerdConfigPath string) error {
	cmd := fmt.Sprintf("kubectl apply -f %s -f %s", kubeletConfigPath, containerdConfigPath)
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("新增自定义配置命令失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("新增自定义配置失败: %s", result.Stderr)
	}
	return nil
}

// CreateCluster 创建集群
// 引导节点和管理集群共部署时，默认使用/root/.kube/config，也就是管理集群的，如果要使用引导节点安装业务集群，需要传入对应的kubeconfig
// 拆分后需要同时传入集群配置和节点配置文件
func (m *ClusterManager) CreateCluster(clusterConfigPath, nodeConfigPath string) error {
	return m.CreateClusterWithConfig(clusterConfigPath, nodeConfigPath, "")
}

func (m *ClusterManager) CreateClusterWithConfig(clusterConfigPath, nodeConfigPath, kubeconfigPath string) error {
	return m.CreateClusterWithImmutableOS(clusterConfigPath, nodeConfigPath, kubeconfigPath, "")
}

// CreateClusterWithImmutableOS 通过 bke cluster create 创建集群，并可选传入 --immutable-os 参数
// immutableOS 为空时不传 --immutable-os；非空时（如 "kubeos"）会附加 --immutable-os=<immutableOS>
func (m *ClusterManager) CreateClusterWithImmutableOS(clusterConfigPath, nodeConfigPath, kubeconfigPath, immutableOS string) error {
	cmd := fmt.Sprintf("bke cluster create -f %s -n %s", clusterConfigPath, nodeConfigPath)
	if kubeconfigPath != "" {
		cmd = fmt.Sprintf("bke --kubeconfig %s cluster create -f %s -n %s", kubeconfigPath, clusterConfigPath, nodeConfigPath)
	}
	if immutableOS != "" {
		cmd = fmt.Sprintf("%s --immutable-os=%s", cmd, immutableOS)
	}
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("执行创建集群命令失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("创建集群失败: %s", result.Stderr)
	}
	return nil
}

// CreateClusterInBackground 在后台创建集群（使用 nohup，适用于预期可能失败的场景）
// 拆分后需要同时传入集群配置和节点配置文件
func (m *ClusterManager) CreateClusterInBackground(clusterConfigPath, nodeConfigPath string) error {
	return m.CreateClusterInBackgroundWithKubeconfig(clusterConfigPath, nodeConfigPath, "")
}

// CreateClusterInBackgroundWithKubeconfig 指定kubeconfig在后台创建集群（使用 nohup，适用于预期可能失败的场景）
// 拆分后需要同时传入集群配置和节点配置文件
func (m *ClusterManager) CreateClusterInBackgroundWithKubeconfig(clusterConfigPath, nodeConfigPath, kubeconfigPath string) error {
	return m.CreateClusterInBackgroundWithImmutableOS(clusterConfigPath, nodeConfigPath, kubeconfigPath, "")
}

// CreateClusterInBackgroundWithImmutableOS 指定kubeconfig在后台创建集群，并可选传入 --immutable-os 参数
// immutableOS 为空时不传 --immutable-os；非空时（如 "kubeos"）会附加 --immutable-os=<immutableOS>
func (m *ClusterManager) CreateClusterInBackgroundWithImmutableOS(clusterConfigPath, nodeConfigPath, kubeconfigPath, immutableOS string) error {
	clusterName := extractClusterNameFromConfigPath(clusterConfigPath)
	logPath := fmt.Sprintf("/tmp/bke-create-%s.log", clusterName)
	GinkgoWriter.Printf("[ClusterManager] 后台创建集群 %s，日志文件路径: %s\n", clusterName, logPath)
	cmd := fmt.Sprintf("nohup bke cluster create -f %s -n %s > %s 2>&1 &", clusterConfigPath, nodeConfigPath, logPath)
	if kubeconfigPath != "" {
		cmd = fmt.Sprintf("nohup bke --kubeconfig %s cluster create -f %s -n %s > %s 2>&1 &", kubeconfigPath, clusterConfigPath, nodeConfigPath, logPath)
	}
	if immutableOS != "" {
		cmd = fmt.Sprintf("nohup bke cluster create -f %s -n %s --immutable-os=%s > %s 2>&1 &", clusterConfigPath, nodeConfigPath, immutableOS, logPath)
		if kubeconfigPath != "" {
			cmd = fmt.Sprintf("nohup bke --kubeconfig %s cluster create -f %s -n %s --immutable-os=%s > %s 2>&1 &", kubeconfigPath, clusterConfigPath, nodeConfigPath, immutableOS, logPath)
		}
	}
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("执行后台创建集群命令失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("后台创建集群失败: %s", result.Stderr)
	}
	return nil
}

// CreateClusterWithKubeconfig creates a cluster using the specified kubeconfig (creates workload cluster on management cluster)
// 拆分后流程：
// 1. Create namespace first (if not exists)
// 2. Apply BKENode YAML first (must be created before BKECluster)
// 3. Apply BKECluster YAML
func (m *ClusterManager) CreateClusterWithKubeconfig(clusterConfigPath, nodeConfigPath, kubeconfigPath string) error {
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}

	// Extract namespace from cluster config file
	getNamespaceCmd := fmt.Sprintf("yq eval '.metadata.namespace' %s", clusterConfigPath)
	result, err := m.executor.Exec(getNamespaceCmd)
	if err != nil {
		return fmt.Errorf("failed to read namespace: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("failed to read namespace: %s", result.Stderr)
	}
	namespace := strings.TrimSpace(result.Stdout)
	if namespace == "" {
		return fmt.Errorf("unable to get namespace from config file")
	}

	// 1. Create namespace (if not exists)
	createNsCmd := fmt.Sprintf("%skubectl create namespace %s --dry-run=client -o yaml | %skubectl apply -f -", kubeconfigArg, namespace, kubeconfigArg)
	result, err = m.executor.Exec(createNsCmd)
	if err != nil {
		verifyCmd := fmt.Sprintf("%skubectl cluster-info --request-timeout=5s", kubeconfigArg)
		verifyResult, verifyErr := m.executor.Exec(verifyCmd)
		if verifyErr != nil || verifyResult.ExitCode != 0 {
			return fmt.Errorf("failed to create namespace (kubeconfig may be invalid): %s, cluster-info check failed: %s", result.Stderr, verifyResult.Stderr)
		}
		return fmt.Errorf("failed to create namespace: %w, stdout: %s, stderr: %s", err, result.Stdout, result.Stderr)
	}
	if result.ExitCode != 0 {
		if strings.Contains(result.Stderr, "AlreadyExists") {
			// Namespace already exists, this is fine
		} else {
			verifyCmd := fmt.Sprintf("%skubectl cluster-info --request-timeout=5s", kubeconfigArg)
			verifyResult, verifyErr := m.executor.Exec(verifyCmd)
			if verifyErr != nil || verifyResult.ExitCode != 0 {
				return fmt.Errorf("failed to create namespace (kubeconfig may be invalid): %s, cluster-info check failed: %s", result.Stderr, verifyResult.Stderr)
			}
			return fmt.Errorf("failed to create namespace: %s, stdout: %s", result.Stderr, result.Stdout)
		}
	}

	// 2. Apply BKENode resources first (must be created before BKECluster)
	applyNodesCmd := fmt.Sprintf("%skubectl apply -f %s", kubeconfigArg, nodeConfigPath)
	result, err = m.executor.Exec(applyNodesCmd)
	if err != nil {
		return fmt.Errorf("failed to apply node config: %w, stdout: %s, stderr: %s", err, result.Stdout, result.Stderr)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("failed to apply node config: %s", result.Stderr)
	}

	// 3. Apply BKECluster resource
	applyClusterCmd := fmt.Sprintf("%skubectl apply -f %s", kubeconfigArg, clusterConfigPath)
	result, err = m.executor.Exec(applyClusterCmd)
	if err != nil {
		return fmt.Errorf("failed to apply cluster config: %w, stdout: %s, stderr: %s", err, result.Stdout, result.Stderr)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("failed to apply cluster config: %s", result.Stderr)
	}
	return nil
}

// DeleteCluster 删除集群（使用默认的 kubeconfig）
// 1. 设置 annotation bke.bocloud.com/ignore-target-cluster-delete=false（允许删除）
// 2. 设置 spec.reset=true 触发集群销毁
// BKE 控制器会销毁集群，销毁完成后 BC 会自动被删除
func (m *ClusterManager) DeleteCluster(clusterName string) error {
	return m.DeleteClusterWithKubeconfig(clusterName, "")
}

// shouldProceedClusterDelete 判断 BC 是否已进入可继续删除流程的状态。
// Ready 走 reset 异步销毁；失败/升级失败等终态不再等待 Ready，直接进入删除流程。
func shouldProceedClusterDelete(state, clusterStatus string) bool {
	if clusterStatus == "Ready" {
		return true
	}
	switch state {
	case "DeployFailed", "Managing", "Unhealthy", "UpgradeFailed":
		return true
	}
	switch clusterStatus {
	case "InitializationFailed", "UpgradeFailed", "ManageFailed":
		return true
	}
	return false
}

// DeleteClusterWithKubeconfig 使用指定的 kubeconfig 删除集群。
// 删除流程：
// 1. 检查 BC 是否存在，不存在则直接返回成功。
// 2. 等待 BC 进入可删除状态，最多等待 10 分钟；如果一直未就绪，也继续进入删除流程。
// 3. 设置 bke.bocloud.com/ignore-target-cluster-delete=false，允许删除目标集群。
// 4. 设置 bke.bocloud.com/ignore-namespace-delete=false，允许删除命名空间。
// 5. 设置 spec.reset=true 触发集群销毁。
// 6. 如果集群不是 Healthy + Ready 状态，补充删除 BC 资源。
// BKE 控制器会销毁集群，销毁完成后 BC 会自动被删除。
func (m *ClusterManager) DeleteClusterWithKubeconfig(clusterName string, kubeconfigPath string) error {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}

	// 检查集群是否存在。
	if !m.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath) {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	pollInterval := 30 * time.Second

	// 等待 BC 状态变为 Healthy + Ready 后再删除；失败态或初始化失败则直接继续删除流程。
	var finalState, finalClusterStatus string
	for {
		_, state, clusterStatus, err := m.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if err != nil {
			// 如果获取状态失败，可能是 BC 已经被删除，直接返回成功。
			if !m.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath) {
				return nil
			}
		} else {
			finalState = state
			finalClusterStatus = clusterStatus
			if shouldProceedClusterDelete(state, clusterStatus) {
				break
			}
		}

		select {
		case <-ctx.Done():
			break
		case <-time.After(pollInterval):
			continue
		}
		break
	}

	// 设置 annotation 允许删除目标集群。
	annotateCmd := fmt.Sprintf(`%skubectl annotate bc %s -n %s bke.bocloud.com/ignore-target-cluster-delete="false" --overwrite`, kubeconfigArg, bcName, namespace)
	result, err := m.executor.Exec(annotateCmd)
	if kubectlCommandFailed(result, err) {
		if kubectlOutputHasNotFound(result) {
			return nil
		}
		return formatKubectlCommandFailure("设置删除注解失败", result, err)
	}

	// 设置 annotation 允许删除命名空间。
	annotateNamespaceCmd := fmt.Sprintf(`%skubectl annotate bc %s -n %s bke.bocloud.com/ignore-namespace-delete="false" --overwrite`, kubeconfigArg, bcName, namespace)
	result, err = m.executor.Exec(annotateNamespaceCmd)
	if kubectlCommandFailed(result, err) {
		if kubectlOutputHasNotFound(result) {
			return nil
		}
		return formatKubectlCommandFailure("设置命名空间删除注解失败", result, err)
	}

	// 设置 spec.reset=true 触发集群销毁。
	patchCmd := fmt.Sprintf(`%skubectl patch bc %s -n %s --type=merge -p '{"spec":{"reset":true}}'`, kubeconfigArg, bcName, namespace)
	result, err = m.executor.Exec(patchCmd)
	if kubectlCommandFailed(result, err) {
		if kubectlOutputHasNotFound(result) {
			return nil
		}
		return formatKubectlCommandFailure("触发集群删除失败", result, err)
	}

	// 集群状态正常时，由控制器根据 spec.reset=true 完成销毁，跳过直接删除 BC。
	if finalState == "Healthy" && finalClusterStatus == "Ready" {
		return nil
	}

	// 非正常状态下补充删除 BC 资源（异步删除，避免 finalizer 阻塞 SSH 命令）。
	deleteCmd := fmt.Sprintf(`%skubectl delete bc %s -n %s --wait=false`, kubeconfigArg, bcName, namespace)
	result, err = m.executor.Exec(deleteCmd)
	if kubectlCommandFailed(result, err) {
		if kubectlOutputHasNotFound(result) {
			return nil
		}
		return formatKubectlCommandFailure("删除 BC 资源失败", result, err)
	}
	return nil
}

// kubectlCommandFailed 判断 kubectl 命令是否执行失败。
// SSHExecutor 在远端命令返回非 0 时会同时返回 result 和 err，因此不能只通过 err 判断。
func kubectlCommandFailed(result *executor.ExecResult, err error) bool {
	return err != nil || result == nil || result.ExitCode != 0
}

// kubectlOutputHasNotFound 判断 kubectl 输出是否表示资源已不存在。
func kubectlOutputHasNotFound(result *executor.ExecResult) bool {
	if result == nil {
		return false
	}
	output := strings.ToLower(result.Stdout + "\n" + result.Stderr)
	return strings.Contains(output, "notfound") || strings.Contains(output, "not found")
}

// formatKubectlCommandFailure 格式化 kubectl 失败信息，保留 exit code、stdout 和 stderr 便于排查。
func formatKubectlCommandFailure(action string, result *executor.ExecResult, err error) error {
	if result == nil {
		return fmt.Errorf("%s: %w", action, err)
	}
	if err != nil {
		return fmt.Errorf("%s: %w, exit=%d stdout=%s stderr=%s", action, err, result.ExitCode, strings.TrimSpace(result.Stdout), strings.TrimSpace(result.Stderr))
	}
	return fmt.Errorf("%s: exit=%d stdout=%s stderr=%s", action, result.ExitCode, strings.TrimSpace(result.Stdout), strings.TrimSpace(result.Stderr))
}
// DeleteManagementClusterSelfBC 删除管理集群自管 BC。
// bootstrapClusterName 为引导侧管理集群逻辑名（与 Create*ManagementCluster 返回的 ClusterName 一致）。
func (m *ClusterManager) DeleteManagementClusterSelfBC(mgmtKubeconfigPath, bootstrapClusterName string) error {
	if strings.TrimSpace(bootstrapClusterName) == "" {
		return fmt.Errorf("bootstrapClusterName 不能为空")
	}
	return m.DeleteClusterWithKubeconfig(bootstrapClusterName, mgmtKubeconfigPath)
}

// ForceResetManagementCluster 使用 bke reset 强制清理管理集群
// 在等待删除超时后调用，用于强制清理管理集群
// nodes: 管理集群的节点列表，需要在每个节点上执行清理命令
// localExecutor: 本地执行器（保留参数用于兼容）
func (m *ClusterManager) ForceResetManagementCluster(nodes []config.NodeInfo, localExecutor *executor.LocalExecutor) error {
	if len(nodes) == 0 {
		return fmt.Errorf("管理集群节点列表为空")
	}
	if localExecutor == nil {
		return fmt.Errorf("localExecutor 不能为空")
	}

	var lastErr error
	// 在每个管理节点上执行清理命令
	for i, node := range nodes {
		nodeErr := m.resetNode(node, localExecutor, i+1, len(nodes))
		if nodeErr != nil {
			lastErr = nodeErr
			// 继续执行其他节点，不立即返回错误
		}
	}

	return lastErr
}

// resetNode 在单个节点上执行清理命令
func (m *ClusterManager) resetNode(node config.NodeInfo, localExecutor *executor.LocalExecutor, nodeIndex, totalNodes int) error {
	nodeLabel := fmt.Sprintf("节点 %d/%d (%s)", nodeIndex, totalNodes, node.IP)
	_ = localExecutor

	// Step 1: 下载 bke 工具。没有 bke 时后续 reset 无法可靠执行，直接返回错误。
	downloadCmd := "curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh | bash"
	result, err := ExecuteCommandOnNode(node, downloadCmd)
	if err != nil {
		return fmt.Errorf("%s: 下载 bke 工具失败: %w, %s", nodeLabel, err, formatResetNodeResult(result))
	}
	if result == nil {
		return fmt.Errorf("%s: 下载 bke 工具失败: nil command result", nodeLabel)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("%s: 下载 bke 工具失败 (exit code %d): %s", nodeLabel, result.ExitCode, result.Stderr)
	}

	// Step 2: 执行 bke reset --all --mount，自动输入 y 确认。
	resetCmd := "echo \"y\" | bke reset --all --mount"
	result, err = ExecuteCommandOnNode(node, resetCmd)
	if err != nil || result == nil || result.ExitCode != 0 {
		GinkgoWriter.Printf("[WARN] %s: bke reset failed: err=%v, %s\n", nodeLabel, err, formatResetNodeResult(result))
	}

	// Step 3: 清理 /bke 目录。
	rmBkeCmd := "rm -rf /bke"
	result, err = ExecuteCommandOnNode(node, rmBkeCmd)
	if err != nil || result == nil || result.ExitCode != 0 {
		GinkgoWriter.Printf("[WARN] %s: remove /bke failed: err=%v, %s\n", nodeLabel, err, formatResetNodeResult(result))
	}

	// Step 4: 删除 bke 二进制文件。
	rmBkeBinCmd := "rm -f /usr/bin/bke"
	result, err = ExecuteCommandOnNode(node, rmBkeBinCmd)
	if err != nil || result == nil || result.ExitCode != 0 {
		GinkgoWriter.Printf("[WARN] %s: remove /usr/bin/bke failed: err=%v, %s\n", nodeLabel, err, formatResetNodeResult(result))
	}

	return nil
}

func formatResetNodeResult(result *executor.ExecResult) string {
	if result == nil {
		return "result=<nil>"
	}
	return fmt.Sprintf("exit=%d stdout=%s stderr=%s", result.ExitCode, strings.TrimSpace(result.Stdout), strings.TrimSpace(result.Stderr))
}

// WaitForClusterDeleted 等待集群完全删除（BC 资源不存在）
func (m *ClusterManager) WaitForClusterDeleted(ctx context.Context, clusterName string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	pollInterval := 1 * time.Minute

	for time.Now().Before(deadline) {
		if !m.ClusterExists(clusterName) {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pollInterval):
			continue
		}
	}

	return fmt.Errorf("等待集群删除超时")
}

// CleanupBKECluster 强制清理 BKECluster 资源（仅在必要时使用）
func (m *ClusterManager) CleanupBKECluster(clusterName string, force bool) error {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName

	// 先检查是否存在
	if !m.ClusterExists(clusterName) {
		return nil
	}

	cmd := fmt.Sprintf("kubectl delete bc %s -n %s", bcName, namespace)
	if force {
		cmd += " --force --grace-period=0"
	}
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("删除 BC 资源失败: %w", err)
	}
	if result.ExitCode != 0 {
		if strings.Contains(result.Stderr, "not found") {
			return nil
		}
		return fmt.Errorf("删除 BC 资源失败: %s", result.Stderr)
	}
	return nil
}

// CleanupClusterNamespaceWithKubeconfig 删除 bke-<clusterName> 命名空间，清理 BKENode/BKECluster 等残留 CR。
// BC 被异步删除或控制器未完成 EnsureDeleteOrReset 时，命名空间与 BN 可能仍挂在引导/管理面 API 上。
func (m *ClusterManager) CleanupClusterNamespaceWithKubeconfig(clusterName, kubeconfigPath string) error {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}

	checkCmd := fmt.Sprintf(`%skubectl get namespace %s --no-headers 2>/dev/null`, kubeconfigArg, namespace)
	checkResult, _ := m.executor.Exec(checkCmd)
	if checkResult == nil || checkResult.ExitCode != 0 || strings.TrimSpace(checkResult.Stdout) == "" {
		return nil
	}

	deleteCmd := fmt.Sprintf(`%skubectl delete namespace %s --ignore-not-found=true --wait=false`, kubeconfigArg, namespace)
	result, err := m.executor.Exec(deleteCmd)
	if kubectlCommandFailed(result, err) {
		if kubectlOutputHasNotFound(result) {
			return nil
		}
		return formatKubectlCommandFailure("删除集群命名空间失败", result, err)
	}
	GinkgoWriter.Printf("已触发删除命名空间 %s（含 BKENode 残留）\n", namespace)
	return nil
}

// GetClusterStatus 获取集群状态（返回 BC 的 clusterHealthState：Healthy/Unhealthy）
// 使用默认的 kubeconfig（引导集群）
func (m *ClusterManager) GetClusterStatus(clusterName string) (string, error) {
	return m.GetClusterStatusWithKubeconfig(clusterName, "")
}

// GetClusterStatusWithKubeconfig 使用指定的 kubeconfig 获取集群状态
func (m *ClusterManager) GetClusterStatusWithKubeconfig(clusterName, kubeconfigPath string) (string, error) {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf("%skubectl get bc %s -n %s -o jsonpath='{.status.clusterHealthState}'", kubeconfigArg, bcName, namespace)
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return "", err
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("获取集群状态失败: %s", result.Stderr)
	}
	state := strings.TrimSpace(strings.Trim(result.Stdout, "'"))
	if state == "" {
		return "Unknown", nil
	}
	return state, nil
}

// GetClusterPhase 获取集群阶段（返回 BC 的 phase 字段，使用默认的 kubeconfig）
func (m *ClusterManager) GetClusterPhase(clusterName string) (string, error) {
	return m.GetClusterPhaseWithKubeconfig(clusterName, "")
}

// GetClusterPhaseWithKubeconfig 使用指定的 kubeconfig 获取集群阶段
func (m *ClusterManager) GetClusterPhaseWithKubeconfig(clusterName, kubeconfigPath string) (string, error) {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf("%skubectl get bc %s -n %s -o jsonpath='{.status.phase}'", kubeconfigArg, bcName, namespace)
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return "", err
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("获取集群阶段失败: %s", result.Stderr)
	}
	phase := strings.TrimSpace(strings.Trim(result.Stdout, "'"))
	if phase == "" {
		return "Unknown", nil
	}
	return phase, nil
}

// GetClusterFullStatus 获取集群完整状态信息（使用默认的 kubeconfig）
func (m *ClusterManager) GetClusterFullStatus(clusterName string) (phase, state, clusterStatus string, err error) {
	return m.GetClusterFullStatusWithKubeconfig(clusterName, "")
}

// GetClusterFullStatusWithKubeconfig 使用指定的 kubeconfig 获取集群完整状态信息
func (m *ClusterManager) GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath string) (phase, state, clusterStatus string, err error) {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	// 使用 custom-columns 一次性获取多个字段
	cmd := fmt.Sprintf("%skubectl get bc %s -n %s -o custom-columns=PHASE:.status.phase,STATE:.status.clusterHealthState,STATUS:.status.clusterStatus --no-headers", kubeconfigArg, bcName, namespace)
	result, err := m.executor.Exec(cmd)
	if result == nil {
		// executors return a nil result when the command fails to start (e.g. SSH dropped)
		return "", "", "", fmt.Errorf("获取集群状态失败: %v", err)
	}
	if err != nil || result.ExitCode != 0 {
		// a non-nil result with err means the command ran but exited non-zero
		return "", "", "", fmt.Errorf("获取集群状态失败(exit %d): %s, err: %v",
			result.ExitCode, strings.TrimSpace(result.Stderr), err)
	}

	// 解析输出（格式：PHASE STATE STATUS）
	fields := strings.Fields(strings.TrimSpace(result.Stdout))
	if len(fields) >= 3 {
		phase = fields[0]
		state = fields[1]
		clusterStatus = fields[2]
	}
	// 处理 <none> 值
	if phase == "<none>" {
		phase = ""
	}
	if state == "<none>" {
		state = ""
	}
	if clusterStatus == "<none>" {
		clusterStatus = ""
	}
	return phase, state, clusterStatus, nil
}

// ManagementClusterSelfBCName 返回管理集群自管 BC 的资源名。
// bootstrapClusterName 为引导侧管理集群逻辑名（与 Create*ManagementCluster 返回的 ClusterName 一致，不含 bke- 前缀）。
// 合入 sig-installation#224 后，自管 BC 继承引导侧 name/ns，不再固定为 bke-cluster。
func ManagementClusterSelfBCName(bootstrapClusterName string) string {
	return ClusterName(bootstrapClusterName)
}

// ManagementClusterSelfBCNamespace 返回管理集群自管 BC 所在命名空间（与资源名一致）。
func ManagementClusterSelfBCNamespace(bootstrapClusterName string) string {
	return ClusterNamespace(bootstrapClusterName)
}

// GetManagementClusterSelfFullStatus 通过管理集群 kubeconfig 获取自管 BC 完整状态。
// bootstrapClusterName 为引导侧管理集群逻辑名，用于推导自管 BC name/ns。
func (m *ClusterManager) GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, bootstrapClusterName string) (phase, state, clusterStatus string, err error) {
	if mgmtKubeconfigPath == "" {
		return "", "", "", fmt.Errorf("mgmtKubeconfigPath 不能为空")
	}
	if strings.TrimSpace(bootstrapClusterName) == "" {
		return "", "", "", fmt.Errorf("bootstrapClusterName 不能为空")
	}
	bcName := ManagementClusterSelfBCName(bootstrapClusterName)
	namespace := ManagementClusterSelfBCNamespace(bootstrapClusterName)
	kubeconfigArg := fmt.Sprintf("KUBECONFIG=%s ", mgmtKubeconfigPath)
	cmd := fmt.Sprintf("%skubectl get bc %s -n %s -o custom-columns=PHASE:.status.phase,STATE:.status.clusterHealthState,STATUS:.status.clusterStatus --no-headers",
		kubeconfigArg, bcName, namespace)
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return "", "", "", err
	}
	if result.ExitCode != 0 {
		return "", "", "", fmt.Errorf("获取管理集群自管 BC 状态失败: %s", result.Stderr)
	}

	fields := strings.Fields(strings.TrimSpace(result.Stdout))
	if len(fields) >= 3 {
		phase = fields[0]
		state = fields[1]
		clusterStatus = fields[2]
	}
	if phase == "<none>" {
		phase = ""
	}
	if state == "<none>" {
		state = ""
	}
	if clusterStatus == "<none>" {
		clusterStatus = ""
	}
	return phase, state, clusterStatus, nil
}

// IsManagementClusterSelfHealthy 判断管理集群自管 BC 是否健康（clusterStatus 为 Ready）
func (m *ClusterManager) IsManagementClusterSelfHealthy(mgmtKubeconfigPath, bootstrapClusterName string) (bool, error) {
	_, _, clusterStatus, err := m.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, bootstrapClusterName)
	if err != nil {
		return false, err
	}
	return clusterStatus == "Ready", nil
}

// ClusterExists 检查集群是否存在（使用默认的 kubeconfig）
func (m *ClusterManager) ClusterExists(clusterName string) bool {
	return m.ClusterExistsWithKubeconfig(clusterName, "")
}

// ClusterExistsWithKubeconfig 使用指定的 kubeconfig 检查集群是否存在
func (m *ClusterManager) ClusterExistsWithKubeconfig(clusterName, kubeconfigPath string) bool {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}
	cmd := fmt.Sprintf("%skubectl get bc %s -n %s --no-headers 2>/dev/null", kubeconfigArg, bcName, namespace)
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return false
	}
	return result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != ""
}

// WaitForClusterReady 等待集群就绪（clusterStatus=Ready）
func (m *ClusterManager) WaitForClusterReady(ctx context.Context, clusterName string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	pollInterval := 30 * time.Second

	for time.Now().Before(deadline) {
		_, _, clusterStatus, err := m.GetClusterFullStatusWithKubeconfig(clusterName, "")
		if err == nil && clusterStatus == "Ready" {
			return nil
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pollInterval):
			continue
		}
	}

	return fmt.Errorf("等待集群就绪超时")
}

// ScaleOutNode 扩容节点（使用默认 kubeconfig）
func (m *ClusterManager) ScaleOutNode(clusterName string, node config.NodeInfo) error {
	return m.ScaleOutNodeWithKubeconfig(clusterName, node, "")
}

// ScaleOutNodeWithKubeconfig 扩容节点（拆分后直接创建 BKENode 资源，控制器 Watch 到 BKENode 创建事件后自动触发扩容）
func (m *ClusterManager) ScaleOutNodeWithKubeconfig(clusterName string, node config.NodeInfo, kubeconfigPath string) error {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}

	// 生成 BKENode YAML
	nodeYAML := GenerateSingleBKENodeYAML(clusterName, node)

	// 写入临时文件
	tempPath := fmt.Sprintf("/tmp/bke-scaleout-%s-%s.yaml", clusterName, node.Hostname)
	writeCmd := fmt.Sprintf("cat > %s << 'BKENODE_EOF'\n%sBKENODE_EOF", tempPath, nodeYAML)
	result, err := m.executor.Exec(writeCmd)
	if err != nil {
		return fmt.Errorf("写入扩容节点配置失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("写入扩容节点配置失败: %s", result.Stderr)
	}

	// 应用 BKENode 资源（直接创建，无需 pause/unpause）
	applyCmd := fmt.Sprintf("%skubectl apply -f %s", kubeconfigArg, tempPath)
	result, err = m.executor.Exec(applyCmd)
	if err != nil {
		return fmt.Errorf("扩容节点失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("扩容节点失败: %s", result.Stderr)
	}

	// 清理临时文件
	m.executor.Exec(fmt.Sprintf("rm -f %s", tempPath))

	_ = namespace // 用于调试日志
	return nil
}

// ScaleInNode 缩容节点（使用默认 kubeconfig）
func (m *ClusterManager) ScaleInNode(clusterName, nodeIP string) error {
	return m.ScaleInNodeWithKubeconfig(clusterName, nodeIP, "")
}

// ScaleInNodeWithKubeconfig 缩容节点（拆分后直接删除 BKENode 资源，控制器 Watch 到 BKENode 删除事件后自动触发缩容）
func (m *ClusterManager) ScaleInNodeWithKubeconfig(clusterName, nodeIP, kubeconfigPath string) error {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}

	// 通过 IP 查找 BKENode 名称
	findCmd := fmt.Sprintf(`%skubectl get bkenode -n %s -o jsonpath='{range .items[*]}{.metadata.name},{.spec.ip}{"\n"}{end}'`, kubeconfigArg, namespace)
	result, err := m.executor.Exec(findCmd)
	if err != nil {
		return fmt.Errorf("获取 BKENode 列表失败: %w", err)
	}

	nodeName := ""
	for _, line := range strings.Split(strings.Trim(result.Stdout, "'"), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ",", 2)
		if len(parts) == 2 && parts[1] == nodeIP {
			nodeName = parts[0]
			break
		}
	}

	if nodeName == "" {
		return fmt.Errorf("在集群中未找到 IP 为 %s 的 BKENode", nodeIP)
	}

	// 直接删除 BKENode 资源（无需 pause/unpause）
	deleteCmd := fmt.Sprintf("%skubectl delete bkenode %s -n %s", kubeconfigArg, nodeName, namespace)
	result, err = m.executor.Exec(deleteCmd)
	if err != nil {
		return fmt.Errorf("删除 BKENode 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("删除 BKENode 失败: %s", result.Stderr)
	}

	return nil
}

// GetBKENodeState 获取指定 IP 的 BKENode 状态（status.state，使用默认 kubeconfig）
func (m *ClusterManager) GetBKENodeState(clusterName, nodeIP string) (string, error) {
	return m.GetBKENodeStateWithKubeconfig(clusterName, nodeIP, "")
}

// GetBKENodeStateWithKubeconfig 获取指定 IP 的 BKENode 状态（status.state）
func (m *ClusterManager) GetBKENodeStateWithKubeconfig(clusterName, nodeIP, kubeconfigPath string) (string, error) {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}

	cmd := fmt.Sprintf(`%skubectl get bkenode -n %s -o jsonpath='{.items[?(@.spec.ip=="%s")].status.state}'`,
		kubeconfigArg, namespace, nodeIP)
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return "", fmt.Errorf("获取 BKENode 状态失败: %w", err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("获取 BKENode 状态失败: %s", result.Stderr)
	}

	return strings.TrimSpace(strings.Trim(result.Stdout, "'")), nil
}

// GetClusterList 获取集群列表
func (m *ClusterManager) GetClusterList() ([]string, error) {
	cmd := "bke cluster list --no-headers 2>/dev/null | awk '{print $1}'"
	result, err := m.executor.Exec(cmd)
	if err != nil {
		return nil, fmt.Errorf("获取集群列表失败: %w", err)
	}

	clusters := []string{}
	for _, line := range splitLines(result.Stdout) {
		if line != "" {
			clusters = append(clusters, line)
		}
	}
	return clusters, nil
}

// PrepareAndCreateCluster 生成配置并创建集群
// 返回 (clusterConfigPath, nodeConfigPath, error)
func (m *ClusterManager) PrepareAndCreateCluster(cfg *config.BKEClusterConfig) (string, string, error) {
	// 生成并上传配置文件
	clusterConfigPath, nodeConfigPath, err := m.configGenerator.GenerateAndUpload(cfg)
	if err != nil {
		return "", "", fmt.Errorf("生成集群配置失败: %w", err)
	}

	// 创建集群（后台创建，使用引导集群的默认 kubeconfig）
	if err := m.CreateClusterInBackgroundWithKubeconfig(clusterConfigPath, nodeConfigPath, DefaultBootstraoClusterKubeconfig); err != nil {
		return clusterConfigPath, nodeConfigPath, fmt.Errorf("创建集群失败: %w", err)
	}

	return clusterConfigPath, nodeConfigPath, nil
}

// GetConfigGenerator 获取配置生成器
func (m *ClusterManager) GetConfigGenerator() *ClusterConfigGenerator {
	return m.configGenerator
}

// CleanupConfig 清理配置文件
func (m *ClusterManager) CleanupConfig(configPath string) error {
	return m.configGenerator.CleanupConfig(configPath)
}

// CleanupPathRecursive 递归清理路径（目录请优先使用此方法）
func (m *ClusterManager) CleanupPathRecursive(path string) error {
	return m.configGenerator.CleanupPathRecursive(path)
}

// GetDynamicClient 获取动态客户端
func (m *ClusterManager) GetDynamicClient() dynamic.Interface {
	return m.dynamicClient
}

// AddNewAddons 添加新的 addon
func (m *ClusterManager) AddNewAddons(clusterName, namespace string, newAddon config.AddonConfig) error {
	clusterName = ClusterName(clusterName)
	namespace = ClusterNamespace(namespace)

	unstructuredObj, err := m.dynamicClient.Resource(BKEClusterGVR).Namespace(namespace).Get(
		context.TODO(),
		clusterName,
		metav1.GetOptions{},
	)
	if err != nil {
		return fmt.Errorf("获取 bkecluster 失败：%w", err)
	}

	addons, found, err := unstructured.NestedSlice(unstructuredObj.Object, "spec", "clusterConfig", "addons")
	if err != nil {
		return fmt.Errorf("获取 addons 失败：%w", err)
	}
	if !found {
		// 如果 addons 字段不存在，初始化为空数组
		addons = make([]interface{}, 0)
	}

	for _, addon := range addons {
		addonMap, ok := addon.(map[string]interface{})
		if !ok {
			continue
		}
		if name, _, _ := unstructured.NestedString(addonMap, "name"); name == newAddon.Name {
			return fmt.Errorf("addon %s 已存在，如需更新请使用 PatchBkeClusterAddonVersion", newAddon.Name)
		}
	}

	newAddonMap := make(map[string]interface{})
	newAddonMap["name"] = newAddon.Name
	newAddonMap["version"] = newAddon.Version
	if newAddon.Type != "" {
		newAddonMap["type"] = newAddon.Type
	}
	if newAddon.ReleaseName != "" {
		newAddonMap["releaseName"] = newAddon.ReleaseName
	}
	if newAddon.Namespace != "" {
		newAddonMap["namespace"] = newAddon.Namespace
	}
	if newAddon.Block {
		newAddonMap["block"] = newAddon.Block
	}
	addons = append(addons, newAddonMap)
	if err := unstructured.SetNestedSlice(unstructuredObj.Object, addons, "spec", "clusterConfig", "addons"); err != nil {
		return fmt.Errorf("设置 addons 失败：%w", err)
	}

	_, err = m.dynamicClient.Resource(BKEClusterGVR).Namespace(namespace).Update(
		context.TODO(),
		unstructuredObj,
		metav1.UpdateOptions{},
	)
	return err
}

// PatchBkeClusterAddonVersion 修改bc addon version
func (m *ClusterManager) PatchBkeClusterAddonVersion(clusterName, namespace, addonName, version string) error {
	clusterName = ClusterName(clusterName)
	namespace = ClusterNamespace(namespace)

	unstructuredObj, err := m.dynamicClient.Resource(BKEClusterGVR).Namespace(namespace).Get(
		context.TODO(),
		clusterName,
		metav1.GetOptions{},
	)
	if err != nil {
		return fmt.Errorf("获取 bkecluster 失败：%w", err)
	}

	addons, found, err := unstructured.NestedSlice(unstructuredObj.Object, "spec", "clusterConfig", "addons")
	if err != nil || !found {
		return fmt.Errorf("获取 addons 失败：%w", err)
	}

	for i, addon := range addons {
		addonMap, ok := addon.(map[string]interface{})
		if !ok {
			continue
		}
		if name, _, _ := unstructured.NestedString(addonMap, "name"); name == addonName {
			addonMap["version"] = version
			addons[i] = addonMap
			break
		}
	}

	if err := unstructured.SetNestedSlice(unstructuredObj.Object, addons, "spec", "clusterConfig", "addons"); err != nil {
		return fmt.Errorf("设置 addons 失败：%w", err)
	}

	_, err = m.dynamicClient.Resource(BKEClusterGVR).Namespace(namespace).Update(
		context.TODO(),
		unstructuredObj,
		metav1.UpdateOptions{},
	)
	return err
}

// RemoveBkeClusterAddonByName 删除一个addon
func (m *ClusterManager) RemoveBkeClusterAddonByName(clusterName, namespace, addonName string) error {
	clusterName = ClusterName(clusterName)
	namespace = ClusterNamespace(namespace)

	unstructuredObj, err := m.dynamicClient.Resource(BKEClusterGVR).Namespace(namespace).Get(context.TODO(), clusterName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("获取bkecluster失败: %w", err)
	}
	addons, found, err := unstructured.NestedSlice(unstructuredObj.Object, "spec", "clusterConfig", "addons")
	if err != nil {
		return fmt.Errorf("获取addons失败: %w", err)
	}
	if !found {
		return fmt.Errorf("addons未找到")
	}

	newAddons := make([]interface{}, 0, len(addons))
	for _, item := range addons {
		itemMap, ok := item.(map[string]interface{})
		if !ok {
			// 格式异常，保留原样（或跳过）
			newAddons = append(newAddons, item)
			continue
		}
		nameVal, _, _ := unstructured.NestedString(itemMap, "name")
		if nameVal == addonName {
			continue
		}
		newAddons = append(newAddons, item)
	}

	// 如果长度没变，说明没有对应addon，直接返回
	if len(newAddons) == len(addons) {
		return nil
	}

	err = unstructured.SetNestedSlice(unstructuredObj.Object, newAddons, "spec", "clusterConfig", "addons")
	if err != nil {
		return fmt.Errorf("failed to set addons: %v", err)
	}

	_, err = m.dynamicClient.Resource(BKEClusterGVR).Namespace(namespace).Update(context.TODO(), unstructuredObj, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update BKECluster: %v", err)
	}

	return nil
}

// GetBkeCluster 获取BKE集群
func (m *ClusterManager) GetBkeCluster(clusterName, namespace string) (*unstructured.Unstructured, error) {
	clusterName = ClusterName(clusterName)
	namespace = ClusterNamespace(namespace)
	return m.dynamicClient.Resource(BKEClusterGVR).Namespace(namespace).Get(context.TODO(), clusterName, metav1.GetOptions{})
}

// splitLines 分割字符串为行
func splitLines(s string) []string {
	return strings.Split(strings.TrimSpace(s), "\n")
}

// ClusterName 集群名称
func ClusterName(clusterName string) string {
	return "bke-" + clusterName
}

// ClusterNamespace 集群命名空间
func ClusterNamespace(clusterName string) string {
	return "bke-" + clusterName
}

func extractClusterNameFromConfigPath(configPath string) string {
	base := configPath
	if idx := strings.LastIndex(base, "/"); idx >= 0 {
		base = base[idx+1:]
	}
	base = strings.TrimSuffix(base, ".yaml")
	base = strings.TrimPrefix(base, "bke-cluster-")
	return base
}
