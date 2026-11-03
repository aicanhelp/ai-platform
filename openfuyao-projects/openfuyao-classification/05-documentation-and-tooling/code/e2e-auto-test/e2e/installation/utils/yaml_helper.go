package utils

import (
	"fmt"
	"strconv"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

// 模板文件路径（引导节点上已存在的模板，拆分后只包含集群配置）
const (
	Template1MasterCluster      = "/bke/cluster/1master-cluster.yaml"
	Template1Master1NodeCluster = "/bke/cluster/1master1node-cluster.yaml"
	Template3MasterCluster      = "/bke/cluster/3master-cluster.yaml"

	TemplateContainerdConfig = "/bke/tmpl/containerd_default.yaml"
	TemplateKubeletConfig    = "/bke/tmpl/kubelet_default.yaml"
)

// ClusterConfigGenerator 集群配置文件生成器
type ClusterConfigGenerator struct {
	executor *executor.SSHExecutor
}

// NewClusterConfigGenerator 创建配置生成器
func NewClusterConfigGenerator(exec *executor.SSHExecutor) *ClusterConfigGenerator {
	return &ClusterConfigGenerator{executor: exec}
}

// GetTemplatePathByNodeCount 根据节点数量获取集群模板路径
// 拆分后模板只包含集群配置（不含节点信息），节点配置由 GenerateNodeConfig 单独生成
func GetTemplatePathByNodeCount(nodeCount int) string {
	switch {
	case nodeCount >= 3:
		return Template3MasterCluster
	case nodeCount == 2:
		return Template1Master1NodeCluster
	default:
		return Template1MasterCluster
	}
}

// GenerateFromTemplate 基于模板文件生成集群配置和节点配置
// 拆分后生成两个文件：
// 1. 集群配置（BKECluster）：从引导节点模板复制并修改
// 2. 节点配置（BKENode）：根据节点列表从头生成
// 返回 (clusterConfigPath, nodeConfigPath, error)
func (g *ClusterConfigGenerator) GenerateFromTemplate(cfg *config.BKEClusterConfig, templatePath string) (string, string, error) {
	clusterConfigPath := fmt.Sprintf("/tmp/bke-cluster-%s.yaml", cfg.ClusterName)
	nodeConfigPath := fmt.Sprintf("/tmp/bke-nodes-%s.yaml", cfg.ClusterName)

	// === 生成集群配置文件（BKECluster）===
	copyCmd := fmt.Sprintf("cp %s %s", templatePath, clusterConfigPath)
	if result, err := g.executor.Exec(copyCmd); err != nil || result.ExitCode != 0 {
		return "", "", fmt.Errorf("复制模板文件失败: %v, stderr: %s", err, result.Stderr)
	}

	bcName := "bke-" + cfg.ClusterName
	if err := g.yqSet(clusterConfigPath, ".metadata.name", bcName); err != nil {
		return "", "", fmt.Errorf("设置 metadata.name 失败: %w", err)
	}
	if err := g.yqSet(clusterConfigPath, ".metadata.namespace", bcName); err != nil {
		return "", "", fmt.Errorf("设置 metadata.namespace 失败: %w", err)
	}

	// 拆分后不再在集群配置中设置节点信息（节点在 BKENode 中）
	// 删除集群配置中可能残留的 nodes 字段
	delNodesCmd := fmt.Sprintf(`yq -i 'del(.spec.clusterConfig.nodes)' %s`, clusterConfigPath)
	g.executor.Exec(delNodesCmd) // 忽略错误，字段可能不存在

	// Addon 处理优先级：
	// 1. 如果 cfg.Addons 非空，完全替换模板的 addon 列表（向后兼容）
	// 2. 否则保留模板中的 addon，按需排除和追加
	if len(cfg.Addons) > 0 {
		if err := g.updateAddons(clusterConfigPath, cfg.Addons); err != nil {
			return "", "", fmt.Errorf("更新 addon 列表失败: %w", err)
		}
	} else {
		// 从模板中排除指定的 addon
		if len(cfg.ExcludeAddons) > 0 {
			if err := g.excludeAddons(clusterConfigPath, cfg.ExcludeAddons); err != nil {
				return "", "", fmt.Errorf("排除 addon 失败: %w", err)
			}
		}
	}

	// 追加额外的 addon（无论是否使用了 Addons 替换）
	if len(cfg.ExtraAddons) > 0 {
		if err := g.appendExtraAddons(clusterConfigPath, cfg.ExtraAddons); err != nil {
			return "", "", fmt.Errorf("追加额外 addon 失败: %w", err)
		}
	}

	if cfg.ImageRepo != nil {
		if err := g.updateImageRepo(clusterConfigPath, cfg.ImageRepo); err != nil {
			return "", "", fmt.Errorf("更新镜像仓库配置失败: %w", err)
		}
	}

	if cfg.ChartRepo != nil {
		if err := g.updateChartRepo(clusterConfigPath, cfg.ChartRepo); err != nil {
			return "", "", fmt.Errorf("更新chart仓库配置失败: %w", err)
		}
	}
	if cfg.OpenFuyaoVersion != "" {
		if err := g.yqSet(clusterConfigPath, ".spec.clusterConfig.cluster.openFuyaoVersion", cfg.OpenFuyaoVersion); err != nil {
			return "", "", fmt.Errorf("设置 openFuyaoVersion 失败: %w", err)
		}
	}

	if cfg.KubernetesVersion != "" {
		if err := g.yqSet(clusterConfigPath, ".spec.clusterConfig.cluster.kubernetesVersion", cfg.KubernetesVersion); err != nil {
			return "", "", fmt.Errorf("设置 kubernetesVersion 失败: %w", err)
		}
	}

	if cfg.LoadBalanceType != "" {
		if err := g.yqSet(clusterConfigPath, ".spec.clusterConfig.cluster.loadBalanceType", cfg.LoadBalanceType); err != nil {
			return "", "", fmt.Errorf("设置 loadBalanceType 失败: %w", err)
		}
	}

	for addonName, version := range cfg.AddonVersionOverrides {
		if err := g.updateAddonVersion(clusterConfigPath, addonName, version); err != nil {
			return "", "", fmt.Errorf("设置 addon %s version 失败: %w", addonName, err)
		}
	}

	for addonName, params := range cfg.AddonParamOverrides {
		for paramKey, paramValue := range params {
			if err := g.updateAddonParam(clusterConfigPath, addonName, paramKey, paramValue); err != nil {
				return "", "", fmt.Errorf("设置 addon %s param.%s 失败: %w", addonName, paramKey, err)
			}
		}
	}

	// Handle ControlPlaneEndpoint: update if provided, set to empty object if not needed
	if cfg.ControlPlaneEndpoint != nil && cfg.ControlPlaneEndpoint.Host != "" {
		if err := g.updateControlPlaneEndpoint(clusterConfigPath, cfg.ControlPlaneEndpoint); err != nil {
			return "", "", fmt.Errorf("更新 controlPlaneEndpoint 失败: %w", err)
		}
	} else {
		if err := g.setControlPlaneEndpointEmpty(clusterConfigPath); err != nil {
			return "", "", fmt.Errorf("设置 controlPlaneEndpoint 为空对象失败: %w", err)
		}
	}

	if cfg.CustomExtra != nil {
		if err := g.updateClusterCustomExtra(clusterConfigPath, cfg.CustomExtra); err != nil {
			return "", "", fmt.Errorf("更新 spec.customExtra 失败: %w", err)
		}
	}

	if len(cfg.Annotations) > 0 {
		if err := g.updateAnnotations(clusterConfigPath, cfg.Annotations); err != nil {
			return "", "", fmt.Errorf("更新 annotations 失败: %w", err)
		}
	}

	kubeletConfigPath := fmt.Sprintf("/tmp/kubelet-%s.yaml", cfg.ClusterName)
	containerdConfigPath := fmt.Sprintf("/tmp/containerd-%s.yaml", cfg.ClusterName)

	// 检查 kubelet 自定义配置文件是否存在
	if g.fileExists(kubeletConfigPath) {
		if g.fieldExists(clusterConfigPath, ".spec.KubeletConfigRef.name") {
			// 从 kubelet 配置文件中读取 metadata.name
			kubeletName, err := g.yqGet(kubeletConfigPath, ".metadata.name")
			if err != nil {
				return "", "", fmt.Errorf("读取 kubelet metadata.name 失败: %w", err)
			}
			// 修改 KubeletConfigRef.name 为从配置文件中读取的值
			if err = g.yqSet(clusterConfigPath, ".spec.KubeletConfigRef.name", kubeletName); err != nil {
				return "", "", fmt.Errorf("设置 KubeletConfigRef.name 失败: %w", err)
			}
		}
	}

	// 检查 containerd 自定义配置文件是否存在
	if g.fileExists(containerdConfigPath) {
		// 检查集群配置中是否存在 containerdConfigRef 字段
		if g.fieldExists(clusterConfigPath, ".spec.clusterConfig.cluster.containerdConfigRef.name") {
			// 从 containerd 配置文件中读取 metadata.name
			containerdName, err := g.yqGet(containerdConfigPath, ".metadata.name")
			if err != nil {
				return "", "", fmt.Errorf("读取 containerd metadata.name 失败: %w", err)
			}
			// 修改 containerdConfigRef.name 为从配置文件中读取的值
			if err = g.yqSet(clusterConfigPath, ".spec.clusterConfig.cluster.containerdConfigRef.name", containerdName); err != nil {
				return "", "", fmt.Errorf("设置 containerdConfigRef.name 失败: %w", err)
			}
		}
	}

	// === 生成节点配置文件（BKENode）===
	if err := g.generateNodeConfig(cfg, nodeConfigPath); err != nil {
		return "", "", fmt.Errorf("生成节点配置失败: %w", err)
	}

	return clusterConfigPath, nodeConfigPath, nil
}

// updateImageRepo 更新镜像仓库配置
func (g *ClusterConfigGenerator) updateChartRepo(configPath string, chartRepo *config.ChartRepoConfig) error {
	if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.chartRepo.domain", chartRepo.Domain); err != nil {
		return err
	}
	if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.chartRepo.ip", chartRepo.IP); err != nil {
		return err
	}
	if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.chartRepo.port", chartRepo.Port); err != nil {
		return err
	}
	if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.chartRepo.prefix", chartRepo.Prefix); err != nil {
		return err
	}
	return nil
}

// yqSet 使用 yq 设置 YAML 字段值
func (g *ClusterConfigGenerator) yqSet(filePath, path, value string) error {
	// 使用 yq 的 -i 选项直接修改文件
	cmd := fmt.Sprintf(`yq -i '%s = "%s"' %s`, path, value, filePath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return err
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("yq 设置失败: %s", result.Stderr)
	}
	return nil
}

// fileExists 检查文件是否存在
func (g *ClusterConfigGenerator) fileExists(filePath string) bool {
	cmd := fmt.Sprintf("test -f %s && echo 'exists' || echo 'notfound'", filePath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return false
	}
	return strings.Contains(result.Stdout, "exists")
}

// fieldExists 检查 YAML 字段是否存在
func (g *ClusterConfigGenerator) fieldExists(filePath, path string) bool {
	// 使用 yq 读取字段值，如果字段不存在会返回 null 或空
	cmd := fmt.Sprintf(`yq '%s' %s`, path, filePath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return false
	}
	if result.ExitCode != 0 {
		return false
	}
	// 如果字段不存在，yq 可能返回 null、空字符串或错误
	value := strings.TrimSpace(result.Stdout)
	return value != "" && value != "null" && !strings.Contains(value, "Error")
}

// RemoveBinaryConfigRefs 删除 BKECluster 配置中的 kubelet/containerd 自定义配置引用
// 主要用于测试场景：不配置 kubelet/containerd 的自定义配置时，确保 BC 中不包含对应的 Ref 字段
func (g *ClusterConfigGenerator) RemoveBinaryConfigRefs(configPath string) error {
	// 删除 .spec.KubeletConfigRef 整个字段
	delKubeletCmd := fmt.Sprintf(`yq -i 'del(.spec.KubeletConfigRef)' %s`, configPath)
	result, err := g.executor.Exec(delKubeletCmd)
	if err != nil {
		return fmt.Errorf("删除 KubeletConfigRef 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("删除 KubeletConfigRef 失败: %s", result.Stderr)
	}

	// 删除 .spec.clusterConfig.cluster.containerdConfigRef 整个字段
	delContainerdCmd := fmt.Sprintf(`yq -i 'del(.spec.clusterConfig.cluster.containerdConfigRef)' %s`, configPath)
	result, err = g.executor.Exec(delContainerdCmd)
	if err != nil {
		return fmt.Errorf("删除 containerdConfigRef 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("删除 containerdConfigRef 失败: %s", result.Stderr)
	}

	return nil
}

// ModBkeagentHealthPort 修改 BKECluster 配置中的 agentHealthPort 字段
func (g *ClusterConfigGenerator) ModBkeagentHealthPort(configPath, selfPort string) error {
	if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.agentHealthPort", selfPort); err != nil {
		return fmt.Errorf("设置 agentHealthPort %s 失败: %w", selfPort, err)
	}
	return nil
}

// ModBkeagentDeployerPort 修改 bkeagent-deployer.yaml 文件中的监听端口
// 需要修改的地方：
// 1. spec.template.spec.containers[0].env 中 HEALTH_PORT 的 value
// 2. spec.template.spec.containers[0].readinessProbe.httpGet.port
// 3. spec.template.spec.containers[0].livenessProbe.httpGet.port
// 注意：该文件是多文档 YAML（包含 Namespace 和 DaemonSet），且包含 Go 模板语法
// 由于 yq 无法直接解析包含模板语法的 YAML，使用 sed 直接替换端口号
func (g *ClusterConfigGenerator) ModBkeagentDeployerPort(oldPort, newPort string) error {
	deployerYamlPath := "/etc/openFuyao/addons/manifests/kubernetes/bkeagent-deployer/latest/bkeagent-deployer.yaml"

	if !g.fileExists(deployerYamlPath) {
		return fmt.Errorf("bkeagent-deployer.yaml 文件不存在: %s", deployerYamlPath)
	}

	// 使用 sed 直接替换端口号，因为文件包含 Go 模板语法，yq 无法正确解析
	// 1. 修改 HEALTH_PORT 环境变量的 value（在 name: HEALTH_PORT 和 valueFrom: 之间的 value: "8443"）
	cmd := fmt.Sprintf(`sed -i '/name: HEALTH_PORT/,/valueFrom:/ s/value: "%s"/value: "%s"/' %s`, oldPort, newPort, deployerYamlPath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("修改 HEALTH_PORT 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("修改 HEALTH_PORT 失败: %s", result.Stderr)
	}

	// 2. 修改 readinessProbe.httpGet.port（在 readinessProbe 块内匹配 port: 8443）
	cmd = fmt.Sprintf(`sed -i '/readinessProbe:/,/failureThreshold: 3/ s/port: %s/port: %s/' %s`, oldPort, newPort, deployerYamlPath)
	result, err = g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("修改 readinessProbe port 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("修改 readinessProbe port 失败: %s", result.Stderr)
	}

	// 3. 修改 livenessProbe.httpGet.port（在 livenessProbe 块内匹配 port: 8443）
	cmd = fmt.Sprintf(`sed -i '/livenessProbe:/,/failureThreshold: 3/ s/port: %s/port: %s/' %s`, oldPort, newPort, deployerYamlPath)
	result, err = g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("修改 livenessProbe port 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("修改 livenessProbe port 失败: %s", result.Stderr)
	}

	return nil
}

// ModBkeControllerManagerPort 修改 002-cluster-api-bke.yaml 中 bke-controller-manager 的健康检查监听端口。
// 需要修改的字段：
// 1. command 参数中的 --health-probe-port=<port>
// 2. livenessProbe.httpGet.port
// 3. readinessProbe.httpGet.port
// 4. ports 中 name: health-probe 对应的 containerPort
// 注意：该文件包含 Go 模板语法，yq 解析不稳定，这里使用 sed 按上下文范围替换。
func (g *ClusterConfigGenerator) ModBkeControllerManagerPort(oldPort, newPort string) error {
	bkeControllerYamlPath := "/etc/openFuyao/addons/manifests/kubernetes/cluster-api/v1.4.3/002-cluster-api-bke.yaml"

	if !g.fileExists(bkeControllerYamlPath) {
		return fmt.Errorf("002-cluster-api-bke.yaml 文件不存在: %s", bkeControllerYamlPath)
	}

	// 1) 修改 command 中 --health-probe-port
	cmd := fmt.Sprintf(`sed -i '/name: bke-controller-manager/,/readinessProbe:/ s/--health-probe-port=%s/--health-probe-port=%s/' %s`, oldPort, newPort, bkeControllerYamlPath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("修改 --health-probe-port 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("修改 --health-probe-port 失败: %s", result.Stderr)
	}

	// 2) 修改 readinessProbe.httpGet.port
	cmd = fmt.Sprintf(`sed -i '/readinessProbe:/,/periodSeconds:/ s/port: %s/port: %s/' %s`, oldPort, newPort, bkeControllerYamlPath)
	result, err = g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("修改 readinessProbe port 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("修改 readinessProbe port 失败: %s", result.Stderr)
	}

	// 3) 修改 livenessProbe.httpGet.port
	cmd = fmt.Sprintf(`sed -i '/name: bke-controller-manager/,/readinessProbe:/ s/port: %s/port: %s/' %s`, oldPort, newPort, bkeControllerYamlPath)
	result, err = g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("修改 livenessProbe port 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("修改 livenessProbe port 失败: %s", result.Stderr)
	}

	// 4) 修改 ports 中 health-probe 的 containerPort
	// webhook-server 端口是 9443，不会被 oldPort(默认 9444) 命中。
	cmd = fmt.Sprintf(`sed -i '/name: bke-controller-manager/,/readinessProbe:/ s/containerPort: %s/containerPort: %s/' %s`, oldPort, newPort, bkeControllerYamlPath)
	result, err = g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("修改 health-probe containerPort 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("修改 health-probe containerPort 失败: %s", result.Stderr)
	}

	return nil
}

// yqGet 使用 yq 读取 YAML 字段值
func (g *ClusterConfigGenerator) yqGet(filePath, path string) (string, error) {
	cmd := fmt.Sprintf(`yq '%s' %s`, path, filePath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return "", err
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("yq 读取失败: %s", result.Stderr)
	}
	// 去除引号和换行符
	value := strings.TrimSpace(result.Stdout)
	// 去除可能的引号
	value = strings.Trim(value, `"'`)
	return value, nil
}

// generateNodeConfig 生成 BKENode 资源的 YAML 文件
// 每个节点生成一个独立的 BKENode CRD 资源，多个资源用 --- 分隔
func (g *ClusterConfigGenerator) generateNodeConfig(cfg *config.BKEClusterConfig, nodeConfigPath string) error {
	bcName := "bke-" + cfg.ClusterName

	var sb strings.Builder
	for i, node := range cfg.Nodes {
		if i > 0 {
			sb.WriteString("---\n")
		}
		sb.WriteString("apiVersion: bke.bocloud.com/v1beta1\n")
		sb.WriteString("kind: BKENode\n")
		sb.WriteString("metadata:\n")
		sb.WriteString(fmt.Sprintf("  name: %s\n", node.Hostname))
		sb.WriteString(fmt.Sprintf("  namespace: %s\n", bcName))
		sb.WriteString("  labels:\n")
		sb.WriteString(fmt.Sprintf("    cluster.x-k8s.io/cluster-name: %s\n", bcName))
		sb.WriteString("spec:\n")
		sb.WriteString(fmt.Sprintf("  hostname: %s\n", node.Hostname))
		sb.WriteString(fmt.Sprintf("  ip: %s\n", node.IP))
		sb.WriteString(fmt.Sprintf("  password: '%s'\n", node.Password))
		sb.WriteString(fmt.Sprintf("  port: \"%s\"\n", node.Port))
		sb.WriteString(fmt.Sprintf("  username: %s\n", node.Username))
		sb.WriteString("  role:\n")
		for _, r := range node.Role {
			sb.WriteString(fmt.Sprintf("  - %s\n", r))
		}
		if len(node.Labels) > 0 {
			sb.WriteString("  labels:\n")
			for _, l := range node.Labels {
				sb.WriteString(fmt.Sprintf("  - key: %s\n", l.Key))
				sb.WriteString(fmt.Sprintf("    value: %s\n", l.Value))
			}
		}
		sb.WriteString("status: {}\n")
	}

	// 写入远程文件
	content := sb.String()
	writeCmd := fmt.Sprintf("cat > %s << 'BKENODE_EOF'\n%sBKENODE_EOF", nodeConfigPath, content)
	result, err := g.executor.Exec(writeCmd)
	if err != nil {
		return fmt.Errorf("写入节点配置文件失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("写入节点配置文件失败: %s", result.Stderr)
	}

	return nil
}

// GenerateSingleBKENodeYAML 生成单个 BKENode 的 YAML 内容（用于扩容）
func GenerateSingleBKENodeYAML(clusterName string, node config.NodeInfo) string {
	bcName := "bke-" + clusterName

	var sb strings.Builder
	sb.WriteString("apiVersion: bke.bocloud.com/v1beta1\n")
	sb.WriteString("kind: BKENode\n")
	sb.WriteString("metadata:\n")
	sb.WriteString(fmt.Sprintf("  name: %s\n", node.Hostname))
	sb.WriteString(fmt.Sprintf("  namespace: %s\n", bcName))
	sb.WriteString("  labels:\n")
	sb.WriteString(fmt.Sprintf("    cluster.x-k8s.io/cluster-name: %s\n", bcName))
	sb.WriteString("spec:\n")
	sb.WriteString(fmt.Sprintf("  hostname: %s\n", node.Hostname))
	sb.WriteString(fmt.Sprintf("  ip: %s\n", node.IP))
	sb.WriteString(fmt.Sprintf("  password: '%s'\n", node.Password))
	sb.WriteString(fmt.Sprintf("  port: \"%s\"\n", node.Port))
	sb.WriteString(fmt.Sprintf("  username: %s\n", node.Username))
	sb.WriteString("  role:\n")
	for _, r := range node.Role {
		sb.WriteString(fmt.Sprintf("  - %s\n", r))
	}
	if len(node.Labels) > 0 {
		sb.WriteString("  labels:\n")
		for _, l := range node.Labels {
			sb.WriteString(fmt.Sprintf("  - key: %s\n", l.Key))
			sb.WriteString(fmt.Sprintf("    value: %s\n", l.Value))
		}
	}
	sb.WriteString("status: {}\n")
	return sb.String()
}

// updateAddons 更新 addon 列表
func (g *ClusterConfigGenerator) updateAddons(configPath string, addons []config.AddonConfig) error {
	// 先清空现有 addon 列表
	clearCmd := fmt.Sprintf(`yq -i '.spec.clusterConfig.addons = []' %s`, configPath)
	if result, err := g.executor.Exec(clearCmd); err != nil || result.ExitCode != 0 {
		return fmt.Errorf("清空 addon 列表失败: %v", err)
	}

	// 逐个添加 addon
	for i, addon := range addons {
		addonYAML := g.buildAddonYAML(addon)

		addCmd := fmt.Sprintf(`yq -i '.spec.clusterConfig.addons[%d] = %s' %s`, i, addonYAML, configPath)
		if result, err := g.executor.Exec(addCmd); err != nil || result.ExitCode != 0 {
			return fmt.Errorf("添加 addon %s 失败: %v, stderr: %s", addon.Name, err, result.Stderr)
		}
	}

	return nil
}

// updateAddonVersion 按名称覆盖模板中 addon 的 version 字段。
func (g *ClusterConfigGenerator) updateAddonVersion(configPath, addonName, version string) error {
	cmd := fmt.Sprintf(`yq -i '(.spec.clusterConfig.addons[] | select(.name == "%s")).version = "%s"' %s`,
		addonName, version, configPath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return err
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("yq 执行失败: %s", result.Stderr)
	}
	return nil
}

// updateAddonParam 按名称覆盖模板中 addon 的 param 字段。
func (g *ClusterConfigGenerator) updateAddonParam(configPath, addonName, paramKey, paramValue string) error {
	cmd := fmt.Sprintf(`yq -i '(.spec.clusterConfig.addons[] | select(.name == "%s")).param.%s = "%s"' %s`,
		addonName, paramKey, paramValue, configPath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return err
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("yq 执行失败: %s", result.Stderr)
	}
	return nil
}

// excludeAddons 从模板的 addon 列表中移除指定名称的 addon
func (g *ClusterConfigGenerator) excludeAddons(configPath string, names []string) error {
	for _, name := range names {
		cmd := fmt.Sprintf(`yq -i 'del(.spec.clusterConfig.addons[] | select(.name == "%s"))' %s`, name, configPath)
		if result, err := g.executor.Exec(cmd); err != nil || result.ExitCode != 0 {
			return fmt.Errorf("排除 addon %s 失败: %v, stderr: %s", name, err, result.Stderr)
		}
	}
	return nil
}

// appendExtraAddons 向模板的 addon 列表末尾追加额外的 addon
func (g *ClusterConfigGenerator) appendExtraAddons(configPath string, addons []config.AddonConfig) error {
	for _, addon := range addons {
		addonYAML := g.buildAddonYAML(addon)
		cmd := fmt.Sprintf(`yq -i '.spec.clusterConfig.addons += [%s]' %s`, addonYAML, configPath)
		if result, err := g.executor.Exec(cmd); err != nil || result.ExitCode != 0 {
			return fmt.Errorf("追加 addon %s 失败: %v, stderr: %s", addon.Name, err, result.Stderr)
		}
	}
	return nil
}

// buildAddonYAML 构建 addon 的 YAML 表达式
func (g *ClusterConfigGenerator) buildAddonYAML(addon config.AddonConfig) string {
	// 基础配置
	parts := []string{
		fmt.Sprintf(`"name": "%s"`, addon.Name),
		fmt.Sprintf(`"version": "%s"`, addon.Version),
	}

	if addon.Block {
		parts = append(parts, `"block": true`)
	}
	if addon.Type != "" {
		parts = append(parts, fmt.Sprintf(`"type": "%s"`, addon.Type))
	}
	if addon.ReleaseName != "" {
		parts = append(parts, fmt.Sprintf(`"releaseName": "%s"`, addon.ReleaseName))
	}
	if addon.Namespace != "" {
		parts = append(parts, fmt.Sprintf(`"namespace": "%s"`, addon.Namespace))
	}
	// 合并 nodeSelector 到 param 中
	mergedParam := make(map[string]string)
	for k, v := range addon.Param {
		mergedParam[k] = v
	}
	if len(addon.NodeSelector) > 0 {
		// 构建 nodeSelector JSON 字符串: {"key":"value"}
		nsParts := make([]string, 0, len(addon.NodeSelector))
		for k, v := range addon.NodeSelector {
			nsParts = append(nsParts, fmt.Sprintf(`\"%s\":\"%s\"`, k, v))
		}
		mergedParam["nodeselector"] = "{" + strings.Join(nsParts, ",") + "}"
	} else if addon.Name == "redis" {
		mergedParam["nodeselector"] = "{}"
	}
	if addon.ValuesConfigMapRef != nil {
		valuesParts := []string{
			fmt.Sprintf(`"name": "%s"`, addon.ValuesConfigMapRef.Name),
		}
		if addon.ValuesConfigMapRef.Namespace != "" {
			valuesParts = append(valuesParts, fmt.Sprintf(`"namespace": "%s"`, addon.ValuesConfigMapRef.Namespace))
		}
		if addon.ValuesConfigMapRef.ValuesKey != "" {
			valuesParts = append(valuesParts, fmt.Sprintf(`"valuesKey": "%s"`, addon.ValuesConfigMapRef.ValuesKey))
		}
		parts = append(parts, fmt.Sprintf(`"valuesConfigMapRef": {%s}`, strings.Join(valuesParts, ", ")))
	}

	// param
	if len(mergedParam) > 0 {
		params := make([]string, 0, len(mergedParam))
		for k, v := range mergedParam {
			params = append(params, fmt.Sprintf(`"%s": "%s"`, k, v))
		}
		parts = append(parts, fmt.Sprintf(`"param": {%s}`, strings.Join(params, ", ")))
	}

	return "{" + strings.Join(parts, ", ") + "}"
}

// updateImageRepo 更新镜像仓库配置
func (g *ClusterConfigGenerator) updateImageRepo(configPath string, imageRepo *config.ImageRepoConfig) error {
	if imageRepo.Domain != "" {
		if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.imageRepo.domain", imageRepo.Domain); err != nil {
			return err
		}
	}
	if imageRepo.IP != "" {
		if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.imageRepo.ip", imageRepo.IP); err != nil {
			return err
		}
	}
	if imageRepo.Port != "" {
		if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.imageRepo.port", imageRepo.Port); err != nil {
			return err
		}
	}
	if imageRepo.Prefix != "" {
		if err := g.yqSet(configPath, ".spec.clusterConfig.cluster.imageRepo.prefix", imageRepo.Prefix); err != nil {
			return err
		}
	}
	return nil
}

// updateControlPlaneEndpoint 更新控制平面端点（高可用 VIP）
func (g *ClusterConfigGenerator) updateControlPlaneEndpoint(configPath string, endpoint *config.ControlPlaneEndpoint) error {
	if endpoint.Host != "" {
		if err := g.yqSet(configPath, ".spec.controlPlaneEndpoint.host", endpoint.Host); err != nil {
			return err
		}
	}
	if endpoint.Port != "" {
		// Port 必须是数字类型，不能带引号，所以不使用 yqSet（它会加引号）
		// 直接使用 yq 设置数字值
		cmd := fmt.Sprintf(`yq -i '.spec.controlPlaneEndpoint.port = %s' %s`, endpoint.Port, configPath)
		result, err := g.executor.Exec(cmd)
		if err != nil {
			return err
		}
		if result.ExitCode != 0 {
			return fmt.Errorf("yq 设置 port 失败: %s", result.Stderr)
		}
	}
	return nil
}

func (g *ClusterConfigGenerator) updateClusterCustomExtra(configPath string, extra *config.ClusterCustomExtra) error {
	if extra == nil {
		return nil
	}
	if extra.ExtraLoadBalanceIP != "" {
		// 确保 spec.clusterConfig.customExtra 为对象，避免模板未定义该字段时 yq 写入失败
		initCmd := fmt.Sprintf(`yq -i '.spec.clusterConfig.customExtra = (.spec.clusterConfig.customExtra // {})' %s`, configPath)
		if result, err := g.executor.Exec(initCmd); err != nil || result.ExitCode != 0 {
			return fmt.Errorf("初始化 spec.clusterConfig.customExtra 失败: %w, stderr: %s", err, result.Stderr)
		}
		if err := g.yqSet(configPath, ".spec.clusterConfig.customExtra.extraLoadBalanceIP", extra.ExtraLoadBalanceIP); err != nil {
			return err
		}
	}
	return nil
}

// setControlPlaneEndpointEmpty 将控制平面端点设置为空对象 {}（当不需要 VIP 时）
func (g *ClusterConfigGenerator) setControlPlaneEndpointEmpty(configPath string) error {
	// 使用 yq 设置 controlPlaneEndpoint 为空对象 {}
	// 先删除 host 和 port 字段（如果存在），然后设置为空对象
	cmd := fmt.Sprintf("yq -i '.spec.controlPlaneEndpoint = {}' %s", configPath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("设置 controlPlaneEndpoint 为空对象失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("设置 controlPlaneEndpoint 为空对象失败: %s", result.Stderr)
	}
	return nil
}

// GenerateAndUpload 生成配置并上传到引导节点
// 返回 (clusterConfigPath, nodeConfigPath, error)
func (g *ClusterConfigGenerator) GenerateAndUpload(cfg *config.BKEClusterConfig) (string, string, error) {
	templatePath := GetTemplatePathByNodeCount(len(cfg.Nodes))
	return g.GenerateFromTemplate(cfg, templatePath)
}

// CleanupConfig 清理配置文件
func (g *ClusterConfigGenerator) CleanupConfig(configPath string) error {
	cmd := fmt.Sprintf("rm -f %s", configPath)
	_, err := g.executor.Exec(cmd)
	return err
}

// CleanupPathRecursive 递归清理路径（用于目录或包含子文件的路径）
func (g *ClusterConfigGenerator) CleanupPathRecursive(path string) error {
	cmd := fmt.Sprintf("rm -rf %s", path)
	_, err := g.executor.Exec(cmd)
	return err
}

// ValidateConfigOnNode 在引导节点上验证配置文件
func (g *ClusterConfigGenerator) ValidateConfigOnNode(configPath string) error {
	cmd := fmt.Sprintf("test -f %s && echo 'exists' || echo 'not found'", configPath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("检查配置文件失败: %w", err)
	}
	if !strings.Contains(result.Stdout, "exists") {
		return fmt.Errorf("配置文件不存在: %s", configPath)
	}
	return nil
}

// ReadTemplateFromNode 从引导节点读取模板文件
func (g *ClusterConfigGenerator) ReadTemplateFromNode(templatePath string) (string, error) {
	cmd := fmt.Sprintf("cat %s 2>/dev/null", templatePath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return "", fmt.Errorf("读取模板文件失败: %w", err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("读取模板文件失败: %s", result.Stderr)
	}
	return result.Stdout, nil
}

// ListGeneratedConfigs 列出已生成的配置文件
func (g *ClusterConfigGenerator) ListGeneratedConfigs() ([]string, error) {
	cmd := "ls -1 /tmp/bke-cluster-*.yaml 2>/dev/null || true"
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return nil, err
	}

	configs := []string{}
	for _, line := range strings.Split(strings.TrimSpace(result.Stdout), "\n") {
		if line != "" {
			configs = append(configs, line)
		}
	}
	return configs, nil
}

// CleanupAllGeneratedConfigs 清理所有生成的配置文件
func (g *ClusterConfigGenerator) CleanupAllGeneratedConfigs() error {
	cmd := "rm -f /tmp/bke-cluster-*.yaml"
	_, err := g.executor.Exec(cmd)
	return err
}

// GetConfigPath 获取集群配置文件路径
func GetConfigPath(clusterName string) string {
	return fmt.Sprintf("/tmp/bke-cluster-%s.yaml", clusterName)
}

// GetNodeConfigPath 获取节点配置文件路径
func GetNodeConfigPath(clusterName string) string {
	return fmt.Sprintf("/tmp/bke-nodes-%s.yaml", clusterName)
}

func GetIntValue(value interface{}, defaultValue int) int {
	if val, ok := value.(string); ok {
		if intVal, err := strconv.Atoi(val); err == nil {
			return intVal
		}
	}
	return defaultValue
}

// GenerateBinaryConfig 生成二进制的自定义配置并上传到引导节点
// 目前逻辑：从引导节点上的默认模板复制到 /tmp 目录下，按集群名称区分文件名
func (g *ClusterConfigGenerator) GenerateBinaryConfig(cfg *config.BKEClusterConfig) (string, string, error) {
	// 目标路径：按集群名称区分
	kubeletConfigPath := fmt.Sprintf("/tmp/kubelet-%s.yaml", cfg.ClusterName)
	containerdConfigPath := fmt.Sprintf("/tmp/containerd-%s.yaml", cfg.ClusterName)

	// kubelet
	exportKubeletCmd := fmt.Sprintf("kubectl get kct -n bke-kubelet bke-kubelet -o yaml > %s", kubeletConfigPath)
	if result, err := g.executor.Exec(exportKubeletCmd); err != nil || result.ExitCode != 0 {
		return "", "", fmt.Errorf("从集群导出 kubelet kct 失败: %v, stderr: %s", err, result.Stderr)
	}

	// containerd
	exportContainerdCmd := fmt.Sprintf("kubectl get cct -n bke-containerd bke-containerd -o yaml > %s", containerdConfigPath)
	if result, err := g.executor.Exec(exportContainerdCmd); err != nil || result.ExitCode != 0 {
		return "", "", fmt.Errorf("从集群导出 containerd cct 失败: %v, stderr: %s", err, result.Stderr)
	}

	// 修改 containerd 配置文件的字段
	// 1. 修改 metadata.name 为 bke-containerd-2
	if err := g.yqSet(containerdConfigPath, ".metadata.name", "bke-containerd-2"); err != nil {
		return "", "", fmt.Errorf("设置 containerd metadata.name 失败: %w", err)
	}

	// 2. 修改 spec.service.startLimitBurst 为 10（数字类型，不加引号）
	cmd := fmt.Sprintf(`yq -i '.spec.service.startLimitBurst = 10' %s`, containerdConfigPath)
	if result, err := g.executor.Exec(cmd); err != nil || result.ExitCode != 0 {
		return "", "", fmt.Errorf("设置 containerd startLimitBurst 失败: %v, stderr: %s", err, result.Stderr)
	}

	// 修改 kubelet 配置文件的字段
	// 1. 修改 metadata.name 为 bke-kubelet-2
	if err := g.yqSet(kubeletConfigPath, ".metadata.name", "bke-kubelet-2"); err != nil {
		return "", "", fmt.Errorf("设置 kubelet metadata.name 失败: %w", err)
	}

	// 2. 修改 spec.service.startLimitBurst 为 10（数字类型，不加引号）
	cmd = fmt.Sprintf(`yq -i '.spec.kubeletService.service.startLimitBurst = 10' %s`, kubeletConfigPath)
	if result, err := g.executor.Exec(cmd); err != nil || result.ExitCode != 0 {
		return "", "", fmt.Errorf("设置 kubelet startLimitBurst 失败: %v, stderr: %s", err, result.Stderr)
	}

	return kubeletConfigPath, containerdConfigPath, nil
}

// MakeKubeletConfigInvalid 修改 kubelet 自定义配置为错误参数，用于验证集群部署失败场景
// 当前实现：将 spec.kubeletService.service.execStart 设置为 /bin/false，期望导致 kubelet 服务启动失败
func (g *ClusterConfigGenerator) MakeKubeletConfigInvalid(kubeletConfigPath string) error {
	cmd := fmt.Sprintf(`yq -i '.spec.kubeletService.service.execStart = "/bin/false"' %s`, kubeletConfigPath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("设置错误的 kubelet execStart 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("设置错误的 kubelet execStart 失败: %s", result.Stderr)
	}
	return nil
}

// EnableKubeletDynamicVariableSubstitution 配置 kubelet KCT 动态变量替换。
// 通过 ${EXPR|...|END} 在节点侧执行脚本并将输出渲染到 kubelet.service 环境变量中。
func (g *ClusterConfigGenerator) EnableKubeletDynamicVariableSubstitution(kubeletConfigPath string) error {
	commands := []string{
		fmt.Sprintf(`yq -i '.metadata.name = "bke-kubelet-dynamic"' %s`, kubeletConfigPath),
		fmt.Sprintf(`yq -i '.spec.kubeletService.service.environment = ["BKE_DYNAMIC_HOSTNAME=${EXPR|hostname|END}", "BKE_DYNAMIC_MARK=${EXPR|printf kubelet-dynamic|END}"]' %s`, kubeletConfigPath),
	}
	for _, cmd := range commands {
		result, err := g.executor.Exec(cmd)
		if err != nil {
			return fmt.Errorf("设置 kubelet 动态变量配置失败: %w", err)
		}
		if result.ExitCode != 0 {
			return fmt.Errorf("设置 kubelet 动态变量配置失败: %s", result.Stderr)
		}
	}
	return nil
}

// MakeContainerdConfigInvalid 修改 containerd 自定义配置为错误参数，用于验证集群部署失败场景
// 当前实现：将 spec.service.execStart 设置为 /bin/false，期望导致 containerd 服务启动失败
func (g *ClusterConfigGenerator) MakeContainerdConfigInvalid(containerdConfigPath string) error {
	cmd := fmt.Sprintf(`yq -i '.spec.service.execStart = "/bin/false"' %s`, containerdConfigPath)
	result, err := g.executor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("设置错误的 containerd execStart 失败: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("设置错误的 containerd execStart 失败: %s", result.Stderr)
	}
	return nil
}

func (g *ClusterConfigGenerator) updateAnnotations(configPath string, annotations map[string]string) error {
	for key, value := range annotations {
		cmd := fmt.Sprintf(`yq -i '.metadata.annotations["%s"] = "%s"' %s`, key, value, configPath)
		result, err := g.executor.Exec(cmd)
		if err != nil {
			return fmt.Errorf("设置 annotation %s 失败: %w", key, err)
		}
		if result.ExitCode != 0 {
			return fmt.Errorf("设置 annotation %s 失败: %s", key, result.Stderr)
		}
	}
	return nil
}
