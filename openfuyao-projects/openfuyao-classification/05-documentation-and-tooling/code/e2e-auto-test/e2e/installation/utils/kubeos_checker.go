package utils

import (
	"encoding/base64"
	"fmt"
	"strings"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// KubeOS 相关常量
const (
	// ImmutableOSAnnotationKey BKECluster 上标识不可变 OS 模式的 annotation key
	// 注意：bkeadm 实际写入的 key 是 openfuyao.io/immutable-os（非设计文档中的 bke.bocloud.com/immutable-os）
	ImmutableOSAnnotationKey = "openfuyao.io/immutable-os"
	// ImmutableOSAnnotationKeyLegacy 设计文档中定义的 key（bkeadm 早期版本可能使用）
	ImmutableOSAnnotationKeyLegacy = "bke.bocloud.com/immutable-os"
	// ImmutableOSAnnotationValue KubeOS 模式对应的 annotation value
	ImmutableOSAnnotationValue = "kubeos"

	// KubeOSNodeOSLabel KubeOS worker 节点 OS 标签 key
	KubeOSNodeOSLabel = "node.openfuyao.io/os"
	// KubeOSNodeOSLabelValue KubeOS worker 节点 OS 标签值
	KubeOSNodeOSLabelValue = "kubeos"
	// KubeOSNodeRoleLabel KubeOS worker 节点角色标签 key
	KubeOSNodeRoleLabel = "node.openfuyao.io/role"
	// KubeOSNodeRoleLabelValue KubeOS worker 节点角色标签值
	KubeOSNodeRoleLabelValue = "worker"
	// KubeOSUpgradeSelectorLabel KubeOS 升级节点选择器标签 key
	KubeOSUpgradeSelectorLabel = "upgrade.openeuler.org/node-selector"
	// KubeOSUpgradeSelectorLabelValue KubeOS 升级节点选择器标签值
	KubeOSUpgradeSelectorLabelValue = "default-worker-pool"

	// KubeOSAddonNamespace KubeOS addon 安装的命名空间
	KubeOSAddonNamespace = "upgrade-system"
	// OSOperatorDeploymentName os-operator Deployment 名称（chart 中为 upgrade-operator）
	OSOperatorDeploymentName = "upgrade-operator"
	// OSProxyDaemonSetName os-proxy DaemonSet 名称（chart 中为 upgrade-proxy）
	OSProxyDaemonSetName = "upgrade-proxy"
	// OSOperatorPodLabel os-operator Pod 的 label selector（chart 中 control-plane=upgrade-operator）
	OSOperatorPodLabel = "control-plane=upgrade-operator"
	// OSProxyPodLabel os-proxy Pod 的 label selector（chart 中 control-plane=upgrade-proxy）
	OSProxyPodLabel = "control-plane=upgrade-proxy"
)

// KubeOSChecker KubeOS 适配相关检查器
type KubeOSChecker struct {
	clusterChecker *ClusterChecker
	clusterManager *ClusterManager
}

// NewKubeOSChecker 创建 KubeOS 检查器
func NewKubeOSChecker(clusterChecker *ClusterChecker, clusterManager *ClusterManager) *KubeOSChecker {
	return &KubeOSChecker{
		clusterChecker: clusterChecker,
		clusterManager: clusterManager,
	}
}

// GetImmutableOSAnnotation 获取 BKECluster 上的 immutable-os annotation
// 返回 annotation 的值（如 "kubeos"），如果不存在则返回空字符串
// 兼容 openfuyao.io/immutable-os（bkeadm 实际使用）和 bke.bocloud.com/immutable-os（设计文档）
func (k *KubeOSChecker) GetImmutableOSAnnotation(clusterName string) (string, error) {
	annotations, err := k.GetBKEClusterAnnotations(clusterName)
	if err != nil {
		return "", err
	}
	// 优先查找 bkeadm 实际使用的 key
	if val, ok := annotations[ImmutableOSAnnotationKey]; ok {
		return val, nil
	}
	// 兼容设计文档中的 key
	if val, ok := annotations[ImmutableOSAnnotationKeyLegacy]; ok {
		return val, nil
	}
	return "", nil
}

// IsImmutableOSMode 判断集群是否为 KubeOS 不可变 OS 模式
func (k *KubeOSChecker) IsImmutableOSMode(clusterName string) (bool, error) {
	value, err := k.GetImmutableOSAnnotation(clusterName)
	if err != nil {
		return false, err
	}
	return value == ImmutableOSAnnotationValue, nil
}

// GetKubeOSWorkerNodes 获取集群中带有 node.openfuyao.io/os=kubeos 标签的节点列表
func (k *KubeOSChecker) GetKubeOSWorkerNodes() ([]string, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return nil, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get nodes -l %s=%s -o jsonpath='{.items[*].metadata.name}'", KubeOSNodeOSLabel, KubeOSNodeOSLabelValue)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return nil, err
	}

	nodes := strings.Fields(strings.Trim(strings.TrimSpace(result.Stdout), "'"))
	return nodes, nil
}

// GetKubeOSMasterNodes 获取集群中不带 node.openfuyao.io/os=kubeos 标签的节点列表（即 master 节点）
func (k *KubeOSChecker) GetKubeOSMasterNodes() ([]string, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return nil, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	// 获取 control-plane 节点
	args := "get nodes -l node-role.kubernetes.io/control-plane -o jsonpath='{.items[*].metadata.name}'"
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return nil, err
	}

	nodes := strings.Fields(strings.Trim(strings.TrimSpace(result.Stdout), "'"))
	return nodes, nil
}

// CheckNodeHasKubeOSLabels 检查指定节点是否带有 KubeOS 相关标签
// 返回缺失的标签 key 列表，全部存在则返回空切片
func (k *KubeOSChecker) CheckNodeHasKubeOSLabels(nodeName string) ([]string, error) {
	labels, err := k.GetNodeLabelsMap(nodeName)
	if err != nil {
		return nil, err
	}

	missing := []string{}
	if labels[KubeOSNodeOSLabel] != KubeOSNodeOSLabelValue {
		missing = append(missing, KubeOSNodeOSLabel)
	}
	if labels[KubeOSUpgradeSelectorLabel] != KubeOSUpgradeSelectorLabelValue {
		missing = append(missing, KubeOSUpgradeSelectorLabel)
	}
	return missing, nil
}

// GetNodeLabelsMap 获取指定节点的所有标签，返回 map[string]string
// 使用 kubectl -o json + jq 解析，避免 jsonpath 输出 map[] 格式难以解析的问题
func (k *KubeOSChecker) GetNodeLabelsMap(nodeName string) (map[string]string, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return nil, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	// 用 jq 提取 labels 为 key\tvalue 格式，每行一个
	args := fmt.Sprintf("get node %s -o json | jq -r '.metadata.labels // {} | to_entries[] | \"\\(.key)\\t\\(.value)\"'", nodeName)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return nil, err
	}

	labels := make(map[string]string)
	for _, line := range strings.Split(strings.TrimSpace(result.Stdout), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line == "null" {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) == 2 {
			labels[parts[0]] = parts[1]
		}
	}
	return labels, nil
}

// CheckMasterNotKubeOSLabeled 检查 master 节点不带 KubeOS 标签
// 返回带 KubeOS 标签的 master 节点列表，应为空
func (k *KubeOSChecker) CheckMasterNotKubeOSLabeled() ([]string, error) {
	masterNodes, err := k.GetKubeOSMasterNodes()
	if err != nil {
		return nil, err
	}

	labeledMasters := []string{}
	for _, master := range masterNodes {
		labels, err := k.GetNodeLabelsMap(master)
		if err != nil {
			continue
		}
		if labels[KubeOSNodeOSLabel] == KubeOSNodeOSLabelValue {
			labeledMasters = append(labeledMasters, master)
		}
	}
	return labeledMasters, nil
}

// CheckReadOnlyDirsNotWritten 检查 KubeOS worker 上只读目录未被安装器写入
// 通过检查安装日志中是否出现 "read-only file system" 来判断
// installLogPath 为安装日志路径（如 /tmp/bke-create-<cluster>.log）
func (k *KubeOSChecker) CheckReadOnlyDirsNotWritten(installLogPath string) (bool, error) {
	cmd := fmt.Sprintf("grep -i 'read-only file system' %s 2>/dev/null || true", installLogPath)
	result, err := k.clusterManager.executor.Exec(cmd)
	if err != nil {
		return false, fmt.Errorf("检查安装日志失败: %w", err)
	}
	// 没有匹配到 read-only file system 即为通过
	return strings.TrimSpace(result.Stdout) == "", nil
}

// CheckBinaryOnNode 检查节点上指定路径的二进制是否存在
// binaryPath 如 /usr/bin/kubelet、/usr/local/bin/bkeagent
func (k *KubeOSChecker) CheckBinaryOnNode(node config.NodeInfo, binaryPath string) (bool, error) {
	cmd := fmt.Sprintf("test -f %s && echo exists || echo notfound", binaryPath)
	result, err := ExecuteCommandOnNode(node, cmd)
	if err != nil {
		return false, fmt.Errorf("在节点 %s 上检查 %s 失败: %w", node.IP, binaryPath, err)
	}
	return strings.TrimSpace(result.Stdout) == "exists", nil
}

// CheckDirWritableOnNode 检查节点上指定目录是否可写
func (k *KubeOSChecker) CheckDirWritableOnNode(node config.NodeInfo, dir string) (bool, error) {
	// 尝试在目录下创建一个临时文件来验证可写性
	cmd := fmt.Sprintf("touch %s/.kubeos-write-test 2>/dev/null && rm -f %s/.kubeos-write-test && echo writable || echo readonly", dir, dir)
	result, err := ExecuteCommandOnNode(node, cmd)
	if err != nil {
		return false, fmt.Errorf("在节点 %s 上检查目录 %s 可写性失败: %w", node.IP, dir, err)
	}
	return strings.TrimSpace(result.Stdout) == "writable", nil
}

// CheckServiceActiveOnNode 检查节点上指定 systemd 服务是否 active
func (k *KubeOSChecker) CheckServiceActiveOnNode(node config.NodeInfo, serviceName string) (bool, error) {
	cmd := fmt.Sprintf("systemctl is-active %s 2>/dev/null || echo inactive", serviceName)
	result, err := ExecuteCommandOnNode(node, cmd)
	if err != nil {
		return false, fmt.Errorf("在节点 %s 上检查服务 %s 状态失败: %w", node.IP, serviceName, err)
	}
	return strings.TrimSpace(result.Stdout) == "active", nil
}

// CheckOSOperatorReady 检查 os-operator Deployment 是否 Ready（readyReplicas >= 1）
func (k *KubeOSChecker) CheckOSOperatorReady() (bool, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get deployment %s -n %s -o jsonpath='{.status.readyReplicas}'", OSOperatorDeploymentName, KubeOSAddonNamespace)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, err
	}
	readyReplicas := strings.TrimSpace(strings.Trim(result.Stdout, "'"))
	return readyReplicas != "" && readyReplicas != "0", nil
}

// GetOSProxyDaemonSetStatus 获取 os-proxy DaemonSet 的 desiredNumberScheduled 和 numberReady
func (k *KubeOSChecker) GetOSProxyDaemonSetStatus() (desired, ready int, err error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return 0, 0, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get daemonset %s -n %s -o jsonpath='{.status.desiredNumberScheduled},{.status.numberReady}'", OSProxyDaemonSetName, KubeOSAddonNamespace)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return 0, 0, err
	}

	output := strings.Trim(strings.TrimSpace(result.Stdout), "'")
	parts := strings.Split(output, ",")
	if len(parts) < 2 {
		return 0, 0, fmt.Errorf("解析 os-proxy DaemonSet 状态失败: %s", output)
	}
	desired = parseIntOrDefault(parts[0], 0)
	ready = parseIntOrDefault(parts[1], 0)
	return desired, ready, nil
}

// CheckOSProxyScheduledOnKubeOSWorkers 检查 os-proxy Pod 是否只调度到 KubeOS worker 节点
// 返回调度到非 KubeOS 节点的 Pod 列表（应为空）和 KubeOS worker 上的 Pod 数量
func (k *KubeOSChecker) CheckOSProxyScheduledOnKubeOSWorkers() (nonKubeOSPods []string, kubeOSPodCount int, err error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return nil, 0, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	// 获取所有 os-proxy Pod 所在节点
	args := fmt.Sprintf("get pods -n %s -l %s -o jsonpath='{range .items[*]}{.spec.nodeName}{\"\\n\"}{end}'", KubeOSAddonNamespace, OSProxyPodLabel)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return nil, 0, err
	}

	kubeOSWorkers, err := k.GetKubeOSWorkerNodes()
	if err != nil {
		return nil, 0, err
	}
	kubeOSWorkerSet := make(map[string]bool)
	for _, w := range kubeOSWorkers {
		kubeOSWorkerSet[w] = true
	}

	podNodes := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	for _, node := range podNodes {
		node = strings.TrimSpace(node)
		if node == "" {
			continue
		}
		if kubeOSWorkerSet[node] {
			kubeOSPodCount++
		} else {
			nonKubeOSPods = append(nonKubeOSPods, node)
		}
	}
	return nonKubeOSPods, kubeOSPodCount, nil
}

// CheckOSProxyNotOnMaster 检查 os-proxy 不调度到 master 节点
// 返回调度到 master 的 Pod 列表（应为空）
func (k *KubeOSChecker) CheckOSProxyNotOnMaster() ([]string, error) {
	masterNodes, err := k.GetKubeOSMasterNodes()
	if err != nil {
		return nil, err
	}
	masterSet := make(map[string]bool)
	for _, m := range masterNodes {
		masterSet[m] = true
	}

	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return nil, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get pods -n %s -l %s -o jsonpath='{range .items[*]}{.spec.nodeName}{\"\\n\"}{end}'", KubeOSAddonNamespace, OSProxyPodLabel)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return nil, err
	}

	masterPods := []string{}
	podNodes := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	for _, node := range podNodes {
		node = strings.TrimSpace(node)
		if node == "" {
			continue
		}
		if masterSet[node] {
			masterPods = append(masterPods, node)
		}
	}
	return masterPods, nil
}

// CheckPodScheduledToWorker 检查指定 label selector 的 Pod 是否调度到 KubeOS worker
// 返回 Pod 所在节点名和是否调度到 KubeOS worker
func (k *KubeOSChecker) CheckPodScheduledToWorker(selector string) (nodeName string, onKubeOS bool, err error) {
	nodeName, err = k.clusterChecker.GetPodNodeName("", selector)
	if err != nil {
		return "", false, err
	}
	if nodeName == "" {
		return "", false, nil
	}

	kubeOSWorkers, err := k.GetKubeOSWorkerNodes()
	if err != nil {
		return nodeName, false, err
	}
	for _, w := range kubeOSWorkers {
		if w == nodeName {
			return nodeName, true, nil
		}
	}
	return nodeName, false, nil
}

// CheckPodRunning 检查指定 namespace 和 label selector 的 Pod 是否处于 Running 状态
func (k *KubeOSChecker) CheckPodRunning(namespace, selector string) (bool, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	nsArg := "-A"
	if namespace != "" {
		nsArg = fmt.Sprintf("-n %s", namespace)
	}
	args := fmt.Sprintf("get pods %s -l %s -o jsonpath='{.items[*].status.phase}'", nsArg, selector)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, err
	}

	phases := strings.Fields(strings.Trim(strings.TrimSpace(result.Stdout), "'"))
	if len(phases) == 0 {
		return false, nil
	}
	for _, phase := range phases {
		if phase != "Running" {
			return false, nil
		}
	}
	return true, nil
}

// GetOSProxyPodNodeName 获取某个 KubeOS worker 上的 os-proxy Pod 名称
func (k *KubeOSChecker) GetOSProxyPodOnNode(nodeName string) (string, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return "", err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get pods -n %s -l %s --field-selector spec.nodeName=%s -o jsonpath='{.items[0].metadata.name}'", KubeOSAddonNamespace, OSProxyPodLabel, nodeName)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(strings.Trim(result.Stdout, "'")), nil
}

// DeleteOSProxyPodOnNode 删除指定节点上的 os-proxy Pod（用于 DFR 测试）
// 在单次 kubeconfig 生命周期内完成获取 Pod 名和删除，避免 kubeconfig 被提前清理
func (k *KubeOSChecker) DeleteOSProxyPodOnNode(nodeName string) error {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	// 先获取 Pod 名
	args := fmt.Sprintf("get pods -n %s -l %s --field-selector spec.nodeName=%s -o jsonpath='{.items[0].metadata.name}'", KubeOSAddonNamespace, OSProxyPodLabel, nodeName)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return fmt.Errorf("获取节点 %s 上的 os-proxy Pod 失败: %w", nodeName, err)
	}
	podName := strings.TrimSpace(strings.Trim(result.Stdout, "'"))
	if podName == "" {
		return fmt.Errorf("节点 %s 上未找到 os-proxy Pod", nodeName)
	}

	// 删除 Pod（复用同一个 kubeconfig）
	args = fmt.Sprintf("delete pod %s -n %s", podName, KubeOSAddonNamespace)
	_, err = k.clusterChecker.execKubectl(kubeconfigPath, args)
	return err
}

// GetOSProxyPodLogs 获取指定节点上 os-proxy Pod 的最近日志
func (k *KubeOSChecker) GetOSProxyPodLogs(nodeName string) (string, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return "", err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get pods -n %s -l %s --field-selector spec.nodeName=%s -o jsonpath='{.items[0].metadata.name}'", KubeOSAddonNamespace, OSProxyPodLabel, nodeName)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return "", err
	}
	podName := strings.TrimSpace(strings.Trim(result.Stdout, "'"))
	if podName == "" {
		return "", fmt.Errorf("节点 %s 上未找到 os-proxy Pod", nodeName)
	}

	args = fmt.Sprintf("logs %s -n %s --tail=50", podName, KubeOSAddonNamespace)
	result, err = k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return "", err
	}
	return result.Stdout, nil
}

// GetOSProxyPodEvents 获取指定节点上 os-proxy Pod 的 Event
func (k *KubeOSChecker) GetOSProxyPodEvents(nodeName string) (string, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return "", err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get pods -n %s -l %s --field-selector spec.nodeName=%s -o jsonpath='{.items[0].metadata.name}'", KubeOSAddonNamespace, OSProxyPodLabel, nodeName)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return "", err
	}
	podName := strings.TrimSpace(strings.Trim(result.Stdout, "'"))
	if podName == "" {
		return "", fmt.Errorf("节点 %s 上未找到 os-proxy Pod", nodeName)
	}

	args = fmt.Sprintf("describe pod %s -n %s | tail -20", podName, KubeOSAddonNamespace)
	result, err = k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return "", err
	}
	return result.Stdout, nil
}

// StopOSAgentOnNode 在 KubeOS worker 上停止 os-agent 进程
// os-agent 不是标准 systemd 服务，需要直接 kill 进程并移除 socket
// 注意：不能在 SSH 命令行中直接出现 os-agent 字符串，否则 pgrep -f 会匹配到 SSH 自身
// 解决方案：将脚本写到临时文件再执行，SSH 命令行不含 os-agent
func (k *KubeOSChecker) StopOSAgentOnNode(node config.NodeInfo) error {
	// 用 base64 编码脚本内容，避免 SSH 命令行中出现 os-agent 字符串
	// 脚本逻辑：systemctl stop → pgrep -x 精确匹配进程名 → kill → 移除 socket
	script := `#!/bin/bash
systemctl stop os-agent 2>/dev/null
systemctl stop upgrade-agent 2>/dev/null
for pid in $(pgrep -x os-agent 2>/dev/null); do kill $pid 2>/dev/null; done
rm -f /var/run/os-agent/*.sock /var/run/os-agent/socket 2>/dev/null
echo done
`
	// 将脚本 base64 编码后在远程节点解码执行
	encoded := base64.StdEncoding.EncodeToString([]byte(script))
	cmd := fmt.Sprintf("echo %s | base64 -d | bash", encoded)
	result, err := ExecuteCommandOnNode(node, cmd)
	if err != nil {
		// 即使 SSH 报错，也可能已经成功 kill 了进程，检查 stdout 是否有 done
		if result != nil && strings.Contains(result.Stdout, "done") {
			return nil
		}
		return fmt.Errorf("在节点 %s 上停止 os-agent 失败: %w", node.IP, err)
	}
	if !strings.Contains(result.Stdout, "done") {
		return fmt.Errorf("在节点 %s 上停止 os-agent 可能失败: stdout=%s stderr=%s", node.IP, result.Stdout, result.Stderr)
	}
	return nil
}

// StartOSAgentOnNode 在 KubeOS worker 上启动 os-agent 进程
// os-agent 二进制位于 /opt/kubeOS/bin/os-agent，通过 nohup 后台启动
// 同样用 base64 编码脚本，避免 SSH 命令行中出现 os-agent 被 pgrep 误杀
func (k *KubeOSChecker) StartOSAgentOnNode(node config.NodeInfo) error {
	script := `#!/bin/bash
systemctl start os-agent 2>/dev/null
systemctl start upgrade-agent 2>/dev/null
# 如果 systemctl 无效，直接启动二进制
if ! pgrep -x os-agent >/dev/null 2>&1; then
  nohup /opt/kubeOS/bin/os-agent > /tmp/os-agent.log 2>&1 &
fi
echo done
`
	encoded := base64.StdEncoding.EncodeToString([]byte(script))
	cmd := fmt.Sprintf("echo %s | base64 -d | bash", encoded)
	result, err := ExecuteCommandOnNode(node, cmd)
	if err != nil {
		if result != nil && strings.Contains(result.Stdout, "done") {
			return nil
		}
		return fmt.Errorf("在节点 %s 上启动 os-agent 失败: %w", node.IP, err)
	}
	if !strings.Contains(result.Stdout, "done") {
		return fmt.Errorf("在节点 %s 上启动 os-agent 可能失败: stdout=%s stderr=%s", node.IP, result.Stdout, result.Stderr)
	}
	return nil
}

// CheckOSProxyPodNotReady 检查指定节点上的 os-proxy Pod 是否感知到 os-agent 不可达
// 判断条件：Pod 非 Running，或日志中包含连接失败/socket 相关关键字
func (k *KubeOSChecker) CheckOSProxyPodNotReady(nodeName string) (bool, string, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return false, "", err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get pods -n %s -l %s --field-selector spec.nodeName=%s -o jsonpath='{.items[0].status.phase}'", KubeOSAddonNamespace, OSProxyPodLabel, nodeName)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, "", err
	}
	phase := strings.TrimSpace(strings.Trim(result.Stdout, "'"))
	if phase != "Running" {
		return true, fmt.Sprintf("os-proxy Pod 状态为 %s（非 Running）", phase), nil
	}

	// Pod 仍 Running，检查日志中是否有连接失败关键字
	// os-proxy 感知 os-agent 不可达后会在日志中输出错误信息，但 Pod 可能不会重启
	logs, _ := k.GetOSProxyPodLogs(nodeName)
	if logs == "" {
		return false, "", nil
	}
	// 匹配连接失败、socket 错误、os-agent 不可达等诊断关键字
	failureKeywords := []string{
		"connect", "refused", "failed", "error", "unreachable",
		"socket", "os-agent", "dial", "connection", "timeout",
		"rpc error", "no such file", "broken pipe",
	}
	lowerLogs := strings.ToLower(logs)
	for _, kw := range failureKeywords {
		if strings.Contains(lowerLogs, kw) {
			return true, fmt.Sprintf("os-proxy 日志中包含诊断关键字: %s", kw), nil
		}
	}
	return false, "", nil
}

// CheckNodeReady 检查指定节点是否 Ready
func (k *KubeOSChecker) CheckNodeReady(nodeName string) (bool, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	args := fmt.Sprintf("get node %s -o jsonpath='{.status.conditions[?(@.type==\"Ready\")].status}'", nodeName)
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, err
	}
	return strings.TrimSpace(strings.Trim(result.Stdout, "'")) == "True", nil
}

// GetNodeNameByIP 通过节点 IP 获取 Kubernetes Node 名称
func (k *KubeOSChecker) GetNodeNameByIP(ip string) (string, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return "", err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	// 通过地址匹配节点
	args := fmt.Sprintf("get nodes -o jsonpath='{range .items[*]}{.metadata.name}{\":\"}{.status.addresses[?(@.type==\"InternalIP\")].address}{\"\\n\"}{end}'")
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return "", err
	}

	lines := strings.Split(strings.TrimSpace(result.Stdout), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(strings.Trim(line, "'"))
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 && parts[1] == ip {
			return parts[0], nil
		}
	}
	return "", fmt.Errorf("未找到 IP 为 %s 的节点", ip)
}

// CheckKubeOSAddonInstalled 检查 KubeOS addon 是否已安装（CRD、os-operator、os-proxy 均存在）
func (k *KubeOSChecker) CheckKubeOSAddonInstalled() (bool, error) {
	kubeconfigPath, err := k.clusterChecker.SaveKubeconfigToFile()
	if err != nil {
		return false, err
	}
	defer k.clusterChecker.cleanupKubeconfig(kubeconfigPath)

	// 检查 CRD 是否存在
	args := "get crd os.upgrade.openeuler.org -o name 2>/dev/null || echo notfound"
	result, err := k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, err
	}
	if strings.TrimSpace(result.Stdout) == "notfound" {
		return false, nil
	}

	// 检查 os-operator Deployment 是否存在
	args = fmt.Sprintf("get deployment %s -n %s -o name 2>/dev/null || echo notfound", OSOperatorDeploymentName, KubeOSAddonNamespace)
	result, err = k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, err
	}
	if strings.TrimSpace(result.Stdout) == "notfound" {
		return false, nil
	}

	// 检查 os-proxy DaemonSet 是否存在
	args = fmt.Sprintf("get daemonset %s -n %s -o name 2>/dev/null || echo notfound", OSProxyDaemonSetName, KubeOSAddonNamespace)
	result, err = k.clusterChecker.execKubectl(kubeconfigPath, args)
	if err != nil {
		return false, err
	}
	if strings.TrimSpace(result.Stdout) == "notfound" {
		return false, nil
	}

	return true, nil
}

// GetBKEClusterAnnotations 获取 BKECluster 的所有 annotations
// 使用 kubectl -o json + jq 解析，避免 jsonpath 无法处理含 "." 和 "/" 的 key
func (k *KubeOSChecker) GetBKEClusterAnnotations(clusterName string) (map[string]string, error) {
	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	// 用 jq 提取 annotations 为 key\tvalue 格式，避免 jsonpath 的 key 解析问题
	cmd := fmt.Sprintf("kubectl get bc %s -n %s -o json | jq -r '.metadata.annotations // {} | to_entries[] | \"\\(.key)\\t\\(.value)\"'", bcName, namespace)
	result, err := k.clusterManager.executor.Exec(cmd)
	if err != nil {
		return nil, fmt.Errorf("获取 BKECluster %s annotations 失败: %w", clusterName, err)
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("获取 BKECluster %s annotations 失败: %s", clusterName, result.Stderr)
	}

	annotations := make(map[string]string)
	for _, line := range strings.Split(strings.TrimSpace(result.Stdout), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line == "null" {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) == 2 {
			annotations[parts[0]] = parts[1]
		}
	}
	return annotations, nil
}

// CheckClusterCreateRejected 检查集群创建是否被 webhook 拒绝
// 通过检查创建命令的输出/日志中是否包含拒绝相关关键字
func (k *KubeOSChecker) CheckClusterCreateRejected(logPath string) (bool, string, error) {
	cmd := fmt.Sprintf("grep -iE 'denied|rejected|forbidden|validation error|conflict' %s 2>/dev/null || true", logPath)
	result, err := k.clusterManager.executor.Exec(cmd)
	if err != nil {
		return false, "", fmt.Errorf("检查创建拒绝日志失败: %w", err)
	}
	output := strings.TrimSpace(result.Stdout)
	return output != "", output, nil
}

// parseIntOrDefault 安全解析整数，失败返回默认值
func parseIntOrDefault(s string, defaultVal int) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultVal
	}
	var n int
	_, err := fmt.Sscanf(s, "%d", &n)
	if err != nil {
		return defaultVal
	}
	return n
}

// EnsureKubeOSCheckerFromUnstructured 从 unstructured BKECluster 对象获取 immutable-os annotation
// 用于在不依赖 ClusterManager 的情况下直接从对象读取
func EnsureKubeOSCheckerFromUnstructured(bc *unstructured.Unstructured) string {
	annotations := bc.GetAnnotations()
	if annotations == nil {
		return ""
	}
	return annotations[ImmutableOSAnnotationKey]
}
