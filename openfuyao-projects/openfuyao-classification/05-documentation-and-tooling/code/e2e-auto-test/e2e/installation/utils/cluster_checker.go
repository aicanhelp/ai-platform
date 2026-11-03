package utils

import (
	"context"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

// ClusterChecker 集群状态检查器
type ClusterChecker struct {
	clientSet   *kubernetes.Clientset
	executor    executor.Executor
	clusterName string
	// parentKubeconfigPath is the kubeconfig of the parent cluster where the BKE secrets/CRs live.
	// Empty means using executor's default kubeconfig context (e.g. ~/.kube/config).
	parentKubeconfigPath string
}

// NewClusterChecker 创建集群检查器
func NewClusterChecker(exec executor.Executor, clusterName string) *ClusterChecker {
	return NewClusterCheckerWithParentKubeconfig(exec, clusterName, "")
}

// NewClusterCheckerWithParentKubeconfig 创建集群检查器，并指定父集群 kubeconfig
func NewClusterCheckerWithParentKubeconfig(exec executor.Executor, clusterName, parentKubeconfigPath string) *ClusterChecker {
	return &ClusterChecker{
		executor:             exec,
		clusterName:          clusterName,
		parentKubeconfigPath: parentKubeconfigPath,
	}
}

// GetKubeconfig 获取业务集群的 kubeconfig（从默认集群，通常是引导集群）
func (c *ClusterChecker) GetKubeconfig() (string, error) {
	return c.GetKubeconfigWithKubeconfig(c.parentKubeconfigPath)
}

// GetKubeconfigWithKubeconfig 使用指定的 kubeconfig 获取业务集群的 kubeconfig
// 如果 kubeconfigPath 为空，则从默认集群（引导集群）获取
// 如果 kubeconfigPath 不为空，则从指定的集群（如管理集群）获取业务集群的 kubeconfig
func (c *ClusterChecker) GetKubeconfigWithKubeconfig(kubeconfigPath string) (string, error) {
	bcName := "bke-" + c.clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}

	// Try to get kubeconfig from secret
	// Always use "value" field, not "ha" field, because "ha" field contains domain endpoint
	// that may not be resolvable from bootstrap node (not in /etc/hosts)
	secretName := bcName + "-kubeconfig"
	secretCmd := fmt.Sprintf("%skubectl get secret %s -n %s -o jsonpath='{.data.value}'", kubeconfigArg, secretName, bcName)
	result, err := c.executor.Exec(secretCmd)
	if err == nil && result.ExitCode == 0 && result.Stdout != "" {
		decoded, err := base64.StdEncoding.DecodeString(strings.TrimSpace(result.Stdout))
		if err == nil && strings.Contains(string(decoded), "apiVersion:") {
			return string(decoded), nil
		}
	}
	secretErr := ""
	if result != nil {
		secretErr = strings.TrimSpace(result.Stderr)
		if secretErr == "" {
			secretErr = strings.TrimSpace(result.Stdout)
		}
	}
	if err != nil && secretErr == "" {
		secretErr = err.Error()
	}

	// 尝试文件路径 (BKE 默认存放路径)
	// 注意：文件路径方式通常只在引导集群上有效
	triedPaths := []string{}
	if kubeconfigPath == "" {
		paths := []string{
			fmt.Sprintf("/root/.bke/%s/admin.conf", bcName),
			fmt.Sprintf("/root/.bke/%s/kubeconfig", bcName),
			fmt.Sprintf("~/.bke/%s/admin.conf", bcName),
		}

		for _, path := range paths {
			triedPaths = append(triedPaths, path)
			cmd := fmt.Sprintf("cat %s", path)
			result, err := c.executor.Exec(cmd)
			if err == nil && result.ExitCode == 0 {
				if strings.Contains(result.Stdout, "apiVersion:") {
					return result.Stdout, nil
				}
			}
		}
	}

	return "", fmt.Errorf("无法获取业务集群 %s 的 kubeconfig (secret %s/%s: %s; files: %v)",
		c.clusterName, bcName, secretName, secretErr, triedPaths)
}

// SaveKubeconfigToFile 保存 kubeconfig 到临时文件（从默认集群获取）
func (c *ClusterChecker) SaveKubeconfigToFile() (string, error) {
	return c.SaveKubeconfigToFileWithKubeconfig(c.parentKubeconfigPath)
}

// SaveKubeconfigToFileWithKubeconfig 使用指定的 kubeconfig 获取业务集群的 kubeconfig 并保存到临时文件
// 如果 parentKubeconfigPath 为空，则从默认集群（引导集群）获取
// 如果 parentKubeconfigPath 不为空，则从指定的集群（如管理集群）获取业务集群的 kubeconfig
func (c *ClusterChecker) SaveKubeconfigToFileWithKubeconfig(parentKubeconfigPath string) (string, error) {
	kubeconfig, err := c.GetKubeconfigWithKubeconfig(parentKubeconfigPath)
	if err != nil {
		return "", err
	}

	kubeconfigPath := fmt.Sprintf("/tmp/kubeconfig-%s", c.clusterName)
	// 使用 base64 编码传输，避免特殊字符问题
	encoded := base64.StdEncoding.EncodeToString([]byte(kubeconfig))
	// 使用 cat <<EOF 方式写入，比 echo 更安全，防止 shell 长度限制或参数解析问题
	cmd := fmt.Sprintf("cat <<EOF | base64 -d > %s\n%s\nEOF", kubeconfigPath, encoded)

	result, err := c.executor.Exec(cmd)
	if err != nil {
		return "", fmt.Errorf("保存kubeconfig失败: %w", err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("保存kubeconfig失败: %s", result.Stderr)
	}
	return kubeconfigPath, nil
}

// execKubectl 执行 kubectl 命令
func (c *ClusterChecker) execKubectl(kubeconfigPath, args string) (*executor.ExecResult, error) {
	// 使用环境变量方式，与 GetKubeconfigWithKubeconfig 保持一致
	cmd := fmt.Sprintf("KUBECONFIG=%s kubectl %s", kubeconfigPath, args)
	result, err := c.executor.Exec(cmd)
	if err != nil {
		// 即使 stderr 为空，也输出 stdout 和 exit code 用于调试
		if result != nil {
			if result.Stderr != "" {
				return result, fmt.Errorf("执行 kubectl 失败: %w, exit_code: %d, stderr: %s, stdout: %s", err, result.ExitCode, result.Stderr, result.Stdout)
			}
			return result, fmt.Errorf("执行 kubectl 失败: %w, exit_code: %d, stdout: %s", err, result.ExitCode, result.Stdout)
		}
		return result, fmt.Errorf("执行 kubectl 失败: %w", err)
	}
	// 即使 err == nil，也要检查 exit code
	if result != nil && result.ExitCode != 0 {
		if result.Stderr != "" {
			return result, fmt.Errorf("kubectl 命令执行失败，exit_code: %d, stderr: %s, stdout: %s", result.ExitCode, result.Stderr, result.Stdout)
		}
		return result, fmt.Errorf("kubectl 命令执行失败，exit_code: %d, stdout: %s", result.ExitCode, result.Stdout)
	}
	return result, nil
}

// GetNodeCount 获取集群节点数量（从默认集群获取 kubeconfig）
func (c *ClusterChecker) GetNodeCount() (int, error) {
	return c.GetNodeCountWithKubeconfig(c.parentKubeconfigPath)
}

// GetNodeCountWithKubeconfig 获取集群节点数量（使用指定的父集群 kubeconfig 获取目标集群的 kubeconfig）
func (c *ClusterChecker) GetNodeCountWithKubeconfig(parentKubeconfigPath string) (int, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFileWithKubeconfig(parentKubeconfigPath)
	if err != nil {
		return 0, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	result, err := c.execKubectl(kubeconfigPath, "get nodes --no-headers | wc -l")
	if err != nil {
		return 0, err
	}

	count, _ := strconv.Atoi(strings.TrimSpace(result.Stdout))
	return count, nil
}

// GetReadyNodeCount 获取 Ready 状态的节点数量（从默认集群获取 kubeconfig）
func (c *ClusterChecker) GetReadyNodeCount() (int, error) {
	return c.GetReadyNodeCountWithKubeconfig(c.parentKubeconfigPath)
}

// GetReadyNodeCountWithKubeconfig 获取 Ready 状态的节点数量（使用指定的父集群 kubeconfig 获取目标集群的 kubeconfig）
func (c *ClusterChecker) GetReadyNodeCountWithKubeconfig(parentKubeconfigPath string) (int, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFileWithKubeconfig(parentKubeconfigPath)
	if err != nil {
		return 0, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	result, err := c.execKubectl(kubeconfigPath, "get nodes --no-headers | grep ' Ready' | wc -l")
	if err != nil {
		return 0, err
	}

	count, _ := strconv.Atoi(strings.TrimSpace(result.Stdout))
	return count, nil
}

// GetNodes 获取节点列表
func (c *ClusterChecker) GetNodes() ([]string, error) {
	return c.GetNodesWithKubeconfig(c.parentKubeconfigPath)
}

// GetNodesWithKubeconfig 获取节点列表（使用指定的父集群 kubeconfig 获取目标集群的 kubeconfig）
func (c *ClusterChecker) GetNodesWithKubeconfig(parentKubeconfigPath string) ([]string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFileWithKubeconfig(parentKubeconfigPath)
	if err != nil {
		return nil, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	result, err := c.execKubectl(kubeconfigPath, "get nodes -o jsonpath='{.items[*].metadata.name}'")
	if err != nil {
		return nil, err
	}

	nodes := strings.Fields(strings.Trim(result.Stdout, "'"))
	return nodes, nil
}

// GetNodeStatus 获取指定节点的状态
func (c *ClusterChecker) GetNodeStatus(nodeName string) (string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return "", err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	result, err := c.execKubectl(kubeconfigPath,
		fmt.Sprintf("get node %s -o jsonpath='{.status.conditions[?(@.type==\"Ready\")].status}'", nodeName))
	if err != nil {
		return "", err
	}

	return strings.Trim(result.Stdout, "'"), nil
}

// GetPodCount 获取 Pod 数量
func (c *ClusterChecker) GetPodCount(namespace string) (int, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return 0, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	nsArg := "-A"
	if namespace != "" {
		nsArg = fmt.Sprintf("-n %s", namespace)
	}

	result, err := c.execKubectl(kubeconfigPath, fmt.Sprintf("get pods %s --no-headers | wc -l", nsArg))
	if err != nil {
		return 0, err
	}

	count, _ := strconv.Atoi(strings.TrimSpace(result.Stdout))
	return count, nil
}

// GetRunningPodCount 获取 Running 状态的 Pod 数量
func (c *ClusterChecker) GetRunningPodCount(namespace string) (int, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return 0, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	nsArg := "-A"
	if namespace != "" {
		nsArg = fmt.Sprintf("-n %s", namespace)
	}

	result, err := c.execKubectl(kubeconfigPath,
		fmt.Sprintf("get pods %s --no-headers --field-selector=status.phase=Running | wc -l", nsArg))
	if err != nil {
		return 0, err
	}

	count, _ := strconv.Atoi(strings.TrimSpace(result.Stdout))
	return count, nil
}

// GetNamespaceCount 获取命名空间数量
func (c *ClusterChecker) GetNamespaceCount() (int, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return 0, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	result, err := c.execKubectl(kubeconfigPath, "get namespaces --no-headers | wc -l")
	if err != nil {
		return 0, err
	}

	count, _ := strconv.Atoi(strings.TrimSpace(result.Stdout))
	return count, nil
}

// GetClusterResourceStatus 获取集群资源状态概览
func (c *ClusterChecker) GetClusterResourceStatus() (*config.ClusterResourceStatus, error) {
	status := &config.ClusterResourceStatus{}

	nodeCount, err := c.GetNodeCount()
	if err != nil {
		return nil, fmt.Errorf("获取节点数量失败: %w", err)
	}
	status.NodeCount = nodeCount

	readyNodeCount, err := c.GetReadyNodeCount()
	if err != nil {
		return nil, fmt.Errorf("获取Ready节点数量失败: %w", err)
	}
	status.ReadyNodeCount = readyNodeCount

	podCount, err := c.GetPodCount("")
	if err != nil {
		return nil, fmt.Errorf("获取Pod数量失败: %w", err)
	}
	status.PodCount = podCount

	runningPodCount, err := c.GetRunningPodCount("")
	if err != nil {
		return nil, fmt.Errorf("获取Running Pod数量失败: %w", err)
	}
	status.RunningPodCount = runningPodCount

	namespaceCount, err := c.GetNamespaceCount()
	if err != nil {
		return nil, fmt.Errorf("获取命名空间数量失败: %w", err)
	}
	status.NamespaceCount = namespaceCount

	return status, nil
}

// CheckSystemPods 检查系统组件 Pod 状态
func (c *ClusterChecker) CheckSystemPods() (bool, string, error) {
	return c.CheckNamespacePodsReady("kube-system")
}

// CheckNamespacePodsReady 检查指定命名空间下的所有 Pod 是否都处于 Running 或 Completed 状态（从默认集群获取 kubeconfig）
func (c *ClusterChecker) CheckNamespacePodsReady(namespace string) (bool, string, error) {
	return c.CheckNamespacePodsReadyWithKubeconfig(namespace, "")
}

// CheckNamespacePodsReadyWithKubeconfig 检查指定命名空间下的所有 Pod 是否都处于 Running 或 Completed 状态（使用指定的父集群 kubeconfig 获取目标集群的 kubeconfig）
func (c *ClusterChecker) CheckNamespacePodsReadyWithKubeconfig(namespace, parentKubeconfigPath string) (bool, string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFileWithKubeconfig(parentKubeconfigPath)
	if err != nil {
		return false, "", err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	result, err := c.execKubectl(kubeconfigPath,
		fmt.Sprintf("get pods -n %s --no-headers", namespace))
	if err != nil {
		return false, "", err
	}

	output := result.Stdout
	lines := strings.Split(strings.TrimSpace(output), "\n")

	allReady := true
	notReadyPods := []string{}

	for _, line := range lines {
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		podName := fields[0]
		status := fields[2]
		// 只要有一个不是 Running 或 Completed，就认为没准备好
		if status != "Running" && status != "Completed" {
			allReady = false
			notReadyPods = append(notReadyPods, fmt.Sprintf("%s(%s)", podName, status))
		}
	}

	// 如果没有任何 pod，也认为没准备好（可能还没开始创建）
	if len(lines) == 1 && lines[0] == "" {
		return false, "no pods found", nil
	}

	return allReady, strings.Join(notReadyPods, ", "), nil
}

// GetNamespacePodStatuses 获取指定命名空间下所有 Pod 的状态列表
func (c *ClusterChecker) GetNamespacePodStatuses(namespace string) (map[string]string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return nil, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)
	result, err := c.execKubectl(kubeconfigPath,
		fmt.Sprintf("get pods -n %s --no-headers", namespace))
	if err != nil {
		return nil, err
	}
	output := result.Stdout
	lines := strings.Split(strings.TrimSpace(output), "\n")
	podStatuses := make(map[string]string)
	for _, line := range lines {
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		podName := fields[0]
		status := fields[2]
		podStatuses[podName] = status
	}
	return podStatuses, nil
}

// GetPodResourceRequest 获取 Pod 的资源请求值（如 CPU、内存）
func (c *ClusterChecker) GetPodResourceRequest(namespace, selector, resourceType string) (string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return "", err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)
	args := fmt.Sprintf("get pods -n %s -l %s -o jsonpath='{.items[0].spec.containers[-1].resources.requests.%s}'", namespace, selector, resourceType)
	result, err := c.execKubectl(kubeconfigPath, args)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(result.Stdout), nil
}

// CheckCoreDNS 检查 CoreDNS 是否正常运行
func (c *ClusterChecker) CheckCoreDNS() (bool, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	result, err := c.execKubectl(kubeconfigPath,
		"get pods -n kube-system -l k8s-app=kube-dns --no-headers | grep Running | wc -l")
	if err != nil {
		return false, err
	}

	count, _ := strconv.Atoi(strings.TrimSpace(result.Stdout))
	return count > 0, nil
}

// CheckCalico 检查 Calico 网络是否正常
func (c *ClusterChecker) CheckCalico() (bool, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	// 检查 calico-node DaemonSet
	result, err := c.execKubectl(kubeconfigPath,
		"get pods -n kube-system -l k8s-app=calico-node --no-headers | grep Running | wc -l")
	if err != nil {
		return false, err
	}

	count, _ := strconv.Atoi(strings.TrimSpace(result.Stdout))
	return count > 0, nil
}

// GetClusterVersion 获取集群版本
func (c *ClusterChecker) GetClusterVersion() (string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return "", err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	result, err := c.execKubectl(kubeconfigPath, "version --short 2>/dev/null | grep 'Server Version' || kubectl --kubeconfig "+kubeconfigPath+" version -o json | grep gitVersion | head -1")
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(result.Stdout), nil
}

// VerifyDNSResolution 验证 DNS 解析是否正常工作
func (c *ClusterChecker) VerifyDNSResolution() (bool, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	// 创建一个临时 Pod 测试 DNS
	testCmd := `run dns-test --image=busybox:1.28 --rm -it --restart=Never -- nslookup kubernetes.default 2>&1 | grep -q "Address" && echo "success" || echo "failed"`
	result, err := c.execKubectl(kubeconfigPath, testCmd)
	if err != nil {
		// DNS 测试失败不一定是错误，可能只是超时
		return false, nil
	}

	return strings.Contains(result.Stdout, "success"), nil
}

// cleanupKubeconfig 清理临时 kubeconfig 文件
func (c *ClusterChecker) cleanupKubeconfig(path string) {
	c.executor.Exec(fmt.Sprintf("rm -f %s", path))
}

// WaitForAllNodesReady 等待所有节点 Ready
func (c *ClusterChecker) WaitForAllNodesReady(expectedCount int) (bool, error) {
	readyCount, err := c.GetReadyNodeCount()
	if err != nil {
		return false, err
	}
	return readyCount == expectedCount, nil
}

// WaitForSystemPodsReady 等待系统 Pod 就绪
func (c *ClusterChecker) WaitForSystemPodsReady() (bool, error) {
	allRunning, notRunning, err := c.CheckSystemPods()
	if err != nil {
		return false, err
	}
	if !allRunning {
		return false, fmt.Errorf("以下Pod未就绪: %s", notRunning)
	}
	return true, nil
}

// CheckNodeHasLabel 检查集群中是否有节点包含指定标签
func (c *ClusterChecker) CheckNodeHasLabel(labelKey, labelValue string) (bool, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	selector := fmt.Sprintf("%s=%s", labelKey, labelValue)
	args := fmt.Sprintf("get nodes -l %s --no-headers", selector)
	result, err := c.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, err
	}

	output := strings.TrimSpace(result.Stdout)
	return output != "", nil
}

// GetPodNodeName 获取符合 label selector 的 Pod 所在的节点名称
func (c *ClusterChecker) GetPodNodeName(namespace, selector string) (string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return "", err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	nsArg := "-A"
	if namespace != "" {
		nsArg = fmt.Sprintf("-n %s", namespace)
	}

	args := fmt.Sprintf("get pods %s -l %s -o jsonpath='{.items[*].spec.nodeName}'", nsArg, selector)
	result, err := c.execKubectl(kubeconfigPath, args)
	if err != nil {
		return "", err
	}

	return strings.TrimSpace(result.Stdout), nil
}

// GetNodeLabels 获取指定节点的所有标签
func (c *ClusterChecker) GetNodeLabels(nodeName string) (map[string]string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return nil, err
	}

	args := fmt.Sprintf("get node %s -o jsonpath='{.metadata.labels}'", nodeName)
	result, err := c.execKubectl(kubeconfigPath, args)
	if err != nil {
		return nil, err
	}

	// 简单返回原始输出，实际解析可以更复杂
	labels := make(map[string]string)
	// 这里只返回原始字符串作为调试信息
	labels["_raw"] = strings.TrimSpace(strings.Trim(result.Stdout, "'"))
	return labels, nil
}

// CheckAllNodesHaveLabel 检查集群中所有节点是否都有指定标签
func (c *ClusterChecker) CheckAllNodesHaveLabel(labelKey string) (bool, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}

	// 获取所有节点数量
	totalResult, err := c.execKubectl(kubeconfigPath, "get nodes --no-headers | wc -l")
	if err != nil {
		return false, err
	}
	totalCount, _ := strconv.Atoi(strings.TrimSpace(totalResult.Stdout))

	// 获取包含指定标签的节点数量
	labeledArgs := fmt.Sprintf("get nodes -l %s --no-headers | wc -l", labelKey)
	labeledResult, err := c.execKubectl(kubeconfigPath, labeledArgs)
	if err != nil {
		return false, err
	}
	labeledCount, _ := strconv.Atoi(strings.TrimSpace(labeledResult.Stdout))

	return totalCount > 0 && totalCount == labeledCount, nil
}

// verifySingleStaticPodKubeconfig 验证单个静态 Pod 的 kubeconfig 配置
func (c *ClusterChecker) verifySingleStaticPodKubeconfig(kubeconfigPath, podName, componentName string) (bool, string, error) {
	// 获取 Pod 的 command 参数，查找 --authorization-kubeconfig
	args := fmt.Sprintf("get pod %s -n kube-system -o jsonpath='{.spec.containers[0].command[*]}'", podName)
	result, err := c.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, "", fmt.Errorf("获取 Pod command 失败: %w", err)
	}

	commandArgs := strings.Fields(result.Stdout)
	var authKubeconfigPath string
	for i, arg := range commandArgs {
		if arg == "--authorization-kubeconfig" && i+1 < len(commandArgs) {
			authKubeconfigPath = commandArgs[i+1]
			break
		}
	}

	if authKubeconfigPath == "" {
		return false, fmt.Sprintf("Pod %s 的 command 中未找到 --authorization-kubeconfig 参数", podName), nil
	}

	// 检查 Pod 的 volume mounts，验证是否挂载了对应的路径
	// 获取 volume mounts 的详细信息：name 和 mountPath
	args = fmt.Sprintf("get pod %s -n kube-system -o jsonpath='{range .spec.containers[0].volumeMounts[*]}{.name}:{.mountPath}{\\n}{end}'", podName)
	result, err = c.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, "", fmt.Errorf("获取 Pod volume mounts 失败: %w", err)
	}

	mountLines := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	found := false
	var matchedMountPath string
	for _, line := range mountLines {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) != 2 {
			continue
		}
		mountPath := parts[1]

		// 检查挂载路径是否匹配 authorization-kubeconfig 指定的路径
		// 可能挂载的是文件或目录，需要精确匹配或路径包含
		if mountPath == authKubeconfigPath {
			found = true
			matchedMountPath = mountPath
			break
		}
		// 如果挂载的是目录，检查文件路径是否在目录下
		if strings.HasPrefix(authKubeconfigPath, mountPath+"/") || strings.HasPrefix(authKubeconfigPath, mountPath) {
			found = true
			matchedMountPath = mountPath
			break
		}
	}

	if !found {
		return false, fmt.Sprintf("Pod %s 的 --authorization-kubeconfig 指向 %s，但未找到对应的 volume mount (已检查的挂载路径: %v)", podName, authKubeconfigPath, mountLines), nil
	}

	return true, fmt.Sprintf("Pod %s: --authorization-kubeconfig=%s 已挂载到 %s", podName, authKubeconfigPath, matchedMountPath), nil
}

// VerifyStaticPodKubeconfig 验证静态 Pod 的 kubeconfig 配置
// 通过检查所有 master 节点上的静态 Pod 来验证
// componentName: 组件名称，如 "kube-controller-manager" 或 "kube-scheduler"
// parentKubeconfigPath: 父集群的 kubeconfig 路径（如果业务集群在管理集群上创建，需要传入管理集群的 kubeconfig）
func (c *ClusterChecker) VerifyStaticPodKubeconfig(componentName string, parentKubeconfigPath string) (bool, string, error) {
	kubeconfigPath, err := c.SaveKubeconfigToFileWithKubeconfig(parentKubeconfigPath)
	if err != nil {
		return false, "", err
	}
	defer c.cleanupKubeconfig(kubeconfigPath)

	// 获取所有 master 节点（通过标签查找）
	masterNodes := []string{}
	// 尝试多种可能的标签
	labels := []string{
		"node-role.kubernetes.io/master",
		"node-role.kubernetes.io/control-plane",
	}

	for _, label := range labels {
		args := fmt.Sprintf("get nodes -l %s --no-headers -o custom-columns=NAME:.metadata.name", label)
		result, err := c.execKubectl(kubeconfigPath, args)
		if err == nil && result.Stdout != "" {
			nodes := strings.Fields(result.Stdout)
			if len(nodes) > 0 {
				masterNodes = nodes
				break
			}
		}
	}

	// 如果通过标签找不到，尝试查找所有匹配的静态 Pod（静态 Pod 名称格式：kube-controller-manager-<节点名称>）
	// 然后从 Pod 名称中提取节点名称
	if len(masterNodes) == 0 {
		args := fmt.Sprintf("get pods -n kube-system --no-headers -o custom-columns=NAME:.metadata.name | grep '^%s-'", componentName)
		result, err := c.execKubectl(kubeconfigPath, args)
		if err == nil && result.Stdout != "" {
			podNames := strings.Fields(result.Stdout)
			// 从 Pod 名称中提取节点名称（格式：kube-controller-manager-<节点名称>）
			for _, podName := range podNames {
				// 移除组件名称前缀和连字符
				if strings.HasPrefix(podName, componentName+"-") {
					nodeName := strings.TrimPrefix(podName, componentName+"-")
					masterNodes = append(masterNodes, nodeName)
				}
			}
		}
	}

	if len(masterNodes) == 0 {
		return false, fmt.Sprintf("未找到 master 节点，无法验证 %s 配置", componentName), nil
	}

	// 验证每个 master 节点上的静态 Pod
	allValid := true
	errors := []string{}
	successes := []string{}

	for _, nodeName := range masterNodes {
		// 静态 Pod 名称格式：kube-controller-manager-<节点名称> 或 kube-scheduler-<节点名称>
		expectedPodName := fmt.Sprintf("%s-%s", componentName, nodeName)

		// 检查 Pod 是否存在
		args := fmt.Sprintf("get pod %s -n kube-system --no-headers -o custom-columns=NAME:.metadata.name", expectedPodName)
		result, err := c.execKubectl(kubeconfigPath, args)
		if err != nil || strings.TrimSpace(result.Stdout) == "" {
			allValid = false
			errors = append(errors, fmt.Sprintf("节点 %s: 未找到 Pod %s", nodeName, expectedPodName))
			continue
		}

		// 验证该 Pod 的配置
		valid, msg, err := c.verifySingleStaticPodKubeconfig(kubeconfigPath, expectedPodName, componentName)
		if err != nil {
			allValid = false
			errors = append(errors, fmt.Sprintf("节点 %s: %v", nodeName, err))
			continue
		}

		if !valid {
			allValid = false
			errors = append(errors, fmt.Sprintf("节点 %s: %s", nodeName, msg))
		} else {
			successes = append(successes, fmt.Sprintf("节点 %s: %s", nodeName, msg))
		}
	}

	if allValid {
		return true, fmt.Sprintf("所有 %d 个 master 节点的 %s 配置正确: %s", len(masterNodes), componentName, strings.Join(successes, "; ")), nil
	}

	return false, fmt.Sprintf("部分 master 节点的 %s 配置不正确: %s", componentName, strings.Join(errors, "; ")), nil
}

// VerifyKubeControllerManagerKubeconfig 验证 kube-controller-manager 是否挂载了 kubeconfig
func (c *ClusterChecker) VerifyKubeControllerManagerKubeconfig(parentKubeconfigPath string) (bool, string, error) {
	return c.VerifyStaticPodKubeconfig("kube-controller-manager", parentKubeconfigPath)
}

// VerifyKubeSchedulerKubeconfig 验证 kube-scheduler 是否挂载了 kubeconfig
func (c *ClusterChecker) VerifyKubeSchedulerKubeconfig(parentKubeconfigPath string) (bool, string, error) {
	return c.VerifyStaticPodKubeconfig("kube-scheduler", parentKubeconfigPath)
}

// VerifyComponentKubeconfigs 验证所有组件的 kubeconfig 配置
// parentKubeconfigPath: 父集群的 kubeconfig 路径（如果业务集群在管理集群上创建，需要传入管理集群的 kubeconfig；如果为空，则从默认集群获取）
// 返回验证结果和详细信息
func (c *ClusterChecker) VerifyComponentKubeconfigs(parentKubeconfigPath string) (map[string]bool, map[string]string, error) {
	results := make(map[string]bool)
	messages := make(map[string]string)

	// 验证 kube-controller-manager
	ok, msg, err := c.VerifyKubeControllerManagerKubeconfig(parentKubeconfigPath)
	if err != nil {
		return nil, nil, fmt.Errorf("验证 kube-controller-manager kubeconfig 失败: %w", err)
	}
	results["kube-controller-manager"] = ok
	messages["kube-controller-manager"] = msg

	// 验证 kube-scheduler
	ok, msg, err = c.VerifyKubeSchedulerKubeconfig(parentKubeconfigPath)
	if err != nil {
		return nil, nil, fmt.Errorf("验证 kube-scheduler kubeconfig 失败: %w", err)
	}
	results["kube-scheduler"] = ok
	messages["kube-scheduler"] = msg

	return results, messages, nil
}

// SetClientSet 设置 kubernetes.Clientset
func (c *ClusterChecker) SetClientSet(kubeconfigPath string) error {
	if c.clientSet != nil {
		return nil
	}

	kubeconfigStr, err := c.GetKubeconfigWithKubeconfig(kubeconfigPath)
	if err != nil {
		return err
	}
	kubeConfig, err := clientcmd.RESTConfigFromKubeConfig([]byte(kubeconfigStr))
	if err != nil {
		return err
	}
	clientSet, err := kubernetes.NewForConfig(kubeConfig)
	if err != nil {
		panic(err)
	}
	c.clientSet = clientSet
	return nil
}

// IsNamespaceHasPods 检查命名空间下是否有 Pod
func (c *ClusterChecker) IsNamespaceHasPods(namespace string) (bool, error) {
	return c.IsNamespaceHasPodsWithKubeconfig(namespace, "")
}

// IsNamespaceHasPodsWithKubeconfig 检查命名空间下是否有 Pod
func (c *ClusterChecker) IsNamespaceHasPodsWithKubeconfig(namespace string, parentKubeconfigPath string) (bool, error) {
	if err := c.SetClientSet(parentKubeconfigPath); err != nil {
		return false, err
	}
	podList, err := c.clientSet.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})
	return len(podList.Items) > 0, err
}

// IsNamespaceHasPodByName 检查命名空间下是否有指定 Pod
func (c *ClusterChecker) IsNamespaceHasPodByName(namespace, podNamePrefix string) (bool, error) {
	return c.IsNamespaceHasPodByNameWithKubeconfig(namespace, podNamePrefix, "")
}

// IsNamespaceHasPodByNameWithKubeconfig 检查命名空间下是否有指定 Pod
func (c *ClusterChecker) IsNamespaceHasPodByNameWithKubeconfig(namespace, podNamePrefix, parentKubeconfigPath string) (bool, error) {
	if err := c.SetClientSet(parentKubeconfigPath); err != nil {
		return false, err
	}
	pods, err := c.clientSet.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return false, err
	}

	for _, pod := range pods.Items {
		if strings.HasPrefix(pod.Name, podNamePrefix) {
			return true, nil
		}
	}
	return false, nil
}

// GetDeploymentImages 获取 Deployment 的镜像列表
func (c *ClusterChecker) GetDeploymentImages(namespace, deploymentName string) ([]string, error) {
	return c.GetDeploymentImagesWithKubeconfig(namespace, deploymentName, "")
}

// GetDeploymentImagesWithKubeconfig 获取 Deployment 的镜像列表
func (c *ClusterChecker) GetDeploymentImagesWithKubeconfig(namespace, deploymentName, parentKubeconfigPath string) ([]string, error) {
	if err := c.SetClientSet(parentKubeconfigPath); err != nil {
		return []string{}, err
	}

	deployment, err := c.clientSet.AppsV1().Deployments(namespace).Get(context.TODO(), deploymentName, metav1.GetOptions{})
	if err != nil {
		return []string{}, err
	}

	return getImagesFromDeployment(deployment), nil
}

func getImagesFromDeployment(deployment *appsv1.Deployment) []string {
	var images []string
	for _, c := range deployment.Spec.Template.Spec.Containers {
		images = append(images, c.Image)
	}
	for _, c := range deployment.Spec.Template.Spec.InitContainers {
		images = append(images, c.Image)
	}
	return images
}

// GetImageTag 获取镜像的 tag
func GetImageTag(image string) string {
	parts := strings.Split(image, ":")
	if len(parts) == 2 {
		return parts[1]
	}
	// 没有显式 tag 时默认为 latest
	return "latest"
}
