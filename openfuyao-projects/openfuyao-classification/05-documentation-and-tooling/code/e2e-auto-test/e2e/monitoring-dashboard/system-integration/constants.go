package monitoring_dashboard

// ExpectedImages 为各组件期望的镜像列表
var ExpectedImages = map[string][]string{
	"monitoring-dashboard-service": {"cr.openfuyao.cn/openfuyao/monitoring-dashboard-service:"},
	"monitoring-dashboard-website": {"cr.openfuyao.cn/openfuyao/monitoring-dashboard-website:"},
}



// 安装指令常量
const (
	MONITOR_INSTALL_CMD       = "helm install monitoring-dashboard cropenfuyao/monitoring-dashboard --version 0.15.0 -n default"
	NUMA_INSTALL_CMD          = "helm install numa cropenfuyao/numa-affinity-package --version 0.15.0 -n default"
	COLOCATION_INSTALL_CMD    = "helm install colocation cropenfuyao/colocation-package --version 0.15.0 -n default"
	RAY_INSTALL_CMD           = "helm install ray cropenfuyao/ray-package --version 0.15.0 -n default"
	MULTI_CLUSTER_INSTALL_CMD = "helm install multi-cluster-service cropenfuyao/multi-cluster-service --version 0.15.0 -n default"
	LOGGING_INSTALL_CMD       = "helm install logging cropenfuyao/logging-package --version 0.15.0 -n default"
)

// 卸载指令常量
const (
	MONITOR_UNINSTALL_CMD       = "helm uninstall monitoring-dashboard -n default"
	NUMA_UNINSTALL_CMD          = "helm uninstall numa -n default"
	COLOCATION_UNINSTALL_CMD    = "helm uninstall colocation -n default"
	RAY_UNINSTALL_CMD           = "helm uninstall ray -n default"
	MULTI_CLUSTER_UNINSTALL_CMD = "helm uninstall multi-cluster-service -n default"
	LOGGING_UNINSTALL_CMD       = "helm uninstall logging -n default"
)

// ComponentTestConfig 组件测试配置结构体
type ComponentTestConfig struct {
	Name                      string // 组件名称（用作标识）
	DisplayName               string // 显示名称
	InstallCmd                string
	UninstallCmd              string
	KeepMonitorAfterTest      bool   // 测试完成后是否保留 MONITOR 组件
	Description               string // 测试描述
	WaitSecondsAfterInstall   int    // 安装后等待时间（秒）
	WaitSecondsAfterUninstall int    // 卸载后等待时间（秒）
}

// 预定义的组件测试配置列表
var AllComponentTestConfigs = []ComponentTestConfig{
	{
		Name:                      "NUMA",
		DisplayName:               "NUMA",
		InstallCmd:                NUMA_INSTALL_CMD,
		UninstallCmd:              NUMA_UNINSTALL_CMD,
		KeepMonitorAfterTest:      false,
		Description:               "测试 NUMA 组件与 Monitoring 的安装冲突",
		WaitSecondsAfterInstall:   30, //30s
		WaitSecondsAfterUninstall: 60,
	},
	{
		Name:                      "COLOCATION",
		DisplayName:               "COLOCATION",
		InstallCmd:                COLOCATION_INSTALL_CMD,
		UninstallCmd:              COLOCATION_UNINSTALL_CMD,
		KeepMonitorAfterTest:      false,
		Description:               "测试 COLOCATION 组件与 Monitoring 的安装冲突",
		WaitSecondsAfterInstall:   30,
		WaitSecondsAfterUninstall: 90,
	},
	{
		Name:                      "RAY",
		DisplayName:               "RAY",
		InstallCmd:                RAY_INSTALL_CMD,
		UninstallCmd:              RAY_UNINSTALL_CMD,
		KeepMonitorAfterTest:      false,
		Description:               "测试 RAY 组件与 Monitoring 的安装冲突",
		WaitSecondsAfterInstall:   30,
		WaitSecondsAfterUninstall: 60,
	},
	{
		Name:                      "MULTI_CLUSTER",
		DisplayName:               "MULTI_CLUSTER",
		InstallCmd:                MULTI_CLUSTER_INSTALL_CMD,
		UninstallCmd:              MULTI_CLUSTER_UNINSTALL_CMD,
		KeepMonitorAfterTest:      false,
		Description:               "测试 MULTI_CLUSTER 组件与 Monitoring 的安装冲突",
		WaitSecondsAfterInstall:   30,
		WaitSecondsAfterUninstall: 120,
	},
	{
		Name:                      "LOGGING",
		DisplayName:               "LOGGING",
		InstallCmd:                LOGGING_INSTALL_CMD,
		UninstallCmd:              LOGGING_UNINSTALL_CMD,
		KeepMonitorAfterTest:      true, // 最后一次冲突测试不卸载 MONITOR
		Description:               "测试 LOGGING 组件与 Monitoring 的安装冲突（保留 Monitoring）",
		WaitSecondsAfterInstall:   30,
		WaitSecondsAfterUninstall: 60,
	},
}
