package utils

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// NodeConfig 一个被测集群节点（IP/角色/SSH 凭据，全部来自环境变量）
type NodeConfig struct {
	IP       string
	Hostname string // 可选：主机名（TEST_NODE{i}_HOSTNAME）；kubectl 对账匹配与日志展示用
	Role     string // master / worker；master 断言 g1-g6，worker 仅 g6（判定口径见 node_functions.go）
	Port     int
	User     string
	Password string
	Slot     int // 环境变量槽位号（=TEST_NODE{i} 下标，2~5），报错指引与对账提示用
}

// Config compliance-hardening E2E 配置。节点连接信息来自环境变量，固定测试参数来自 const.go。
type Config struct {
	Entry NodeConfig
	Nodes []NodeConfig

	HardeningDir string // entry_node 上被测工具目录
	// HardeningSourceDir 是 Runner 上的被测工具源码目录，仅在 entry_node 尚未部署工具时使用。
	HardeningSourceDir string
	ClusterYAML        string // entry_node 上 cluster.yaml 路径（含口令，仅存于节点）
	BackupDir          string // 各节点本地备份目录
	ReportDir          string // entry_node 集群统一报告目录
	LogDir             string // entry_node 编排日志目录（cluster.yaml log_dir）
	WorkDir            string // entry_node 测试工作目录（指纹/复扫产物/状态文件）

	KubeletConfig string // kubelet 配置文件路径

	ScanMode       string // auto / on / off
	KubeBenchImage string
	OpenSCAPImage  string

	BizNamespace  string
	BizDeployment string
	BizImage      string

	// RequireNodeOverride 要求 cluster.yaml 至少 1 个节点写独立 password 覆盖（E2E-A 前置 3）
	RequireNodeOverride bool
}

// Masters 返回 master 节点（按 TEST_NODE 槽位顺序）
func (c *Config) Masters() []NodeConfig {
	return c.nodesByRole(RoleMaster)
}

// Workers 返回 worker 节点（按 TEST_NODE 槽位顺序）
func (c *Config) Workers() []NodeConfig {
	return c.nodesByRole(RoleWorker)
}

func (c *Config) nodesByRole(role string) []NodeConfig {
	var out []NodeConfig
	for _, n := range c.Nodes {
		if n.Role == role {
			out = append(out, n)
		}
	}
	return out
}

// ScanEnabledOn 判断复扫是否可用：off 恒关；on 恒开；auto 交由调用方探测镜像后决定。
func (c *Config) ScanEnabledOn() bool {
	return strings.ToLower(c.ScanMode) != "off"
}

// LoadConfig 从环境变量读取节点配置并校验必填项，同时装配 const.go 中的固定测试参数。
//
// 节点/entry 连接信息按三级取值（口径见 README「配置复用与节点角色判定」）：
//
//	COMPLIANCE_* 显式值 > TEST_NODE_* 复用值 > 缺配置即停
//
// 拓扑口径：业务集群节点集合 = TEST_NODE2~5（3 master + 1 worker），entry_node = TEST_NODE2
// （首个业务 master）；TEST_NODE1（=GUIDE_NODE，管理集群与引导节点共部署）不参与本套件。
// 显式覆盖键 COMPLIANCE_NODE{i}_* 与 TEST_NODE 槽位同下标（i=2..5）。填写 COMPLIANCE_* 的
// 旧用法保持兼容。节点角色不在本函数判定（见 node_functions.go），显式 COMPLIANCE_NODE{i}_ROLE
// 在此处校验取值并保留，其余留空交由 ResolveNodeRoles 填充。
func LoadConfig() (*Config, error) {
	cfg := &Config{
		HardeningDir:        ComplianceHardeningDir,
		HardeningSourceDir:  resolveHardeningSourceDir(),
		ClusterYAML:         ComplianceClusterYAML,
		BackupDir:           ComplianceBackupDir,
		ReportDir:           ComplianceReportDir,
		LogDir:              ComplianceLogDir,
		WorkDir:             ComplianceWorkDir,
		KubeletConfig:       ComplianceKubeletConfig,
		ScanMode:            ComplianceScanMode,
		KubeBenchImage:      ComplianceKubeBenchImage,
		OpenSCAPImage:       ComplianceOpenSCAPImage,
		BizNamespace:        ComplianceBizNamespace,
		BizDeployment:       ComplianceBizDeployment,
		BizImage:            ComplianceBizImage,
		RequireNodeOverride: ComplianceRequireNodeOverride,
	}

	// 节点列表：业务集群槽位 TEST_NODE2~5（下标与 TEST_NODE 对齐，TEST_NODE2~4=业务 master、
	// TEST_NODE5=业务 worker；TEST_NODE1/GUIDE_NODE 为管理引导面，不参与被测集群），
	// COMPLIANCE_NODE{i}_* 显式覆盖；角色仅识别显式 COMPLIANCE_NODE{i}_ROLE（无内置默认，
	// 杜绝静默猜测），未显式指定的角色由 ResolveNodeRoles 按 kubectl 角色标签（权威源）或主机名前缀推断。
	var entryFallback *NodeConfig // 槽位 2：首个业务 master = compliance entry_node 的复用源
	for i := FirstClusterNodeSlot; i <= LastClusterNodeSlot; i++ {
		ip := firstNonEmpty(os.Getenv(nodeEnv(i, "IP")), os.Getenv(testNodeEnv(i, "IP")))
		if ip == "" {
			continue // 空槽位：未参与本轮测试
		}
		role := os.Getenv(nodeEnv(i, "ROLE"))
		if role != "" && role != RoleMaster && role != RoleWorker {
			return nil, fmt.Errorf("%s 取值非法：%s（仅支持 master/worker；不填则自动判定）", nodeEnv(i, "ROLE"), role)
		}
		port, err := envIntChain(22, nodeEnv(i, "PORT"), testNodeEnv(i, "PORT"))
		if err != nil {
			return nil, err
		}
		node := NodeConfig{
			IP:       ip,
			Hostname: strings.TrimSpace(os.Getenv(testNodeEnv(i, "HOSTNAME"))),
			Role:     role,
			Port:     port,
			User:     firstNonEmpty(os.Getenv(nodeEnv(i, "USER")), os.Getenv(testNodeEnv(i, "USER")), "root"),
			Password: firstNonEmpty(os.Getenv(nodeEnv(i, "PASSWORD")), os.Getenv(testNodeEnv(i, "PASSWORD"))),
			Slot:     i,
		}
		if node.Password == "" {
			return nil, fmt.Errorf("TEST_NODE%d 密码未配置（%s 与 %s 均为空）：节点凭据必须通过环境变量注入，不得写入代码", i, nodeEnv(i, "PASSWORD"), testNodeEnv(i, "PASSWORD"))
		}
		if i == FirstClusterNodeSlot {
			n := node
			entryFallback = &n
		}
		cfg.Nodes = append(cfg.Nodes, node)
	}
	if len(cfg.Nodes) == 0 {
		return nil, fmt.Errorf("未发现任何被测集群节点：请通过环境变量提供 TEST_NODE2~5_IP（业务集群 3 master + 1 worker），或以 COMPLIANCE_NODE{2..5}_IP 显式覆盖")
	}

	// entry_node：COMPLIANCE_ENTRY_* 显式 > TEST_NODE2_*（槽位 2 节点实际取值，含 COMPLIANCE_NODE2_*
	// 显式覆盖）> 缺配置即停。GUIDE_NODE_*（管理/引导面）不参与 compliance 读取链。
	entryIP := ""
	entryDefPort := 22
	entryUser := "root"
	entryPassword := ""
	if entryFallback != nil {
		entryIP = entryFallback.IP
		entryDefPort = entryFallback.Port
		entryUser = entryFallback.User
		entryPassword = entryFallback.Password
	}
	entryIP = firstNonEmpty(os.Getenv(EnvEntryHost), entryIP)
	entryPort, err := envIntChain(entryDefPort, EnvEntryPort)
	if err != nil {
		return nil, err
	}
	entryUser = firstNonEmpty(os.Getenv(EnvEntryUser), entryUser)
	entryPassword = firstNonEmpty(os.Getenv(EnvEntryPassword), entryPassword)
	if entryIP == "" {
		return nil, fmt.Errorf("entry_node 未配置：%s 与 TEST_NODE2_* 均为空——compliance entry_node 默认为 TEST_NODE2（首个业务 master），请通过环境变量提供", EnvEntryHost)
	}
	if entryPassword == "" {
		return nil, fmt.Errorf("entry_node 密码未配置（%s 与 TEST_NODE2_PASSWORD 均为空）：entry_node 凭据必须通过环境变量注入，不得写入代码", EnvEntryPassword)
	}
	cfg.Entry = NodeConfig{
		IP:       entryIP,
		Port:     entryPort,
		User:     entryUser,
		Password: entryPassword,
	}
	return cfg, nil
}

func nodeEnv(i int, field string) string {
	return fmt.Sprintf(EnvNodeN, i, field)
}

func testNodeEnv(i int, field string) string {
	return fmt.Sprintf(EnvTestNodeN, i, field)
}

// firstNonEmpty 返回第一个非空（去空白后）取值；全空返回空串。
func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

// envIntChain 依次尝试各环境变量键，取第一个可解析为正整数的值；全空/不可解析时返回 def
// （存在「已设置但非法」的键时返回错误，避免静默吞掉配置错误）。
func envIntChain(def int, keys ...string) (int, error) {
	for _, key := range keys {
		v := strings.TrimSpace(os.Getenv(key))
		if v == "" {
			continue
		}
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 {
			return 0, fmt.Errorf("%s 取值非法：%q（应为正整数端口号）", key, v)
		}
		return n, nil
	}
	return def, nil
}
