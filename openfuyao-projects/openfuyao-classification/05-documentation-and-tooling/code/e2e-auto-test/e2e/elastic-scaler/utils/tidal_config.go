package utils

// tidal 子 chart 把控制器 fullname 写死成 tidal-controller-manager（不含 release 名，见
// charts/tidal/templates/_helpers.tpl），于是它创建的 5 个 ClusterRole + 2 个 ClusterRoleBinding
// 是集群级、跨 release 同名的：同集群已有别的 pd-orchestrator/tidal 安装时，helm 会因归属校验拒绝安装。
// 这里用 fullnameOverride 给 tidal 的资源名加上 e2e 命名空间前缀，从而与别人的安装共存（不去清理别人的资源）。
var (
	TidalFullnameOverride   = ElasticScalerE2ENamespace + "-tidal"
	TidalServiceAccountName = ElasticScalerE2ENamespace + "-tidal-manager"
)

// TidalPodPrefixes tidal-scheduler 控制器相关的 Pod 名称前缀（改名后按 TidalFullnameOverride 匹配）
var TidalPodPrefixes = []string{
	TidalFullnameOverride,
	"tidal-scheduler",
}

// TidalHelm 安装 tidal-scheduler 的 helm 配置
const (
	TidalHelmReleaseName = "scaler"
	TidalHelmChart       = "oci://cr.openfuyao.cn/charts/pd-orchestrator"
	TidalHelmVersion     = "0.0.0-latest"
)

// TidalHelmNamespace / TidalTestNamespace 统一指向 e2e 专属命名空间（不再用 chart 默认的 scaling-system）
var (
	TidalHelmNamespace = ElasticScalerE2ENamespace
	TidalTestNamespace = ElasticScalerE2ENamespace
)

// TidalTest 测试用（对应 xlsx 用例）
const (
	TidalTestCRName          = "tidal1"    // 用例2-5
	TidalTestCRCase3Name     = "tidal-c3"  // 用例3 每分钟触发
	TidalTestCRCase5Name     = "tidal-c5"  // 用例5 缩容到零
	TidalTestCRCase6Name     = "tidal-c6"  // 用例6 Deployment
	TidalTestCRCase7Name     = "tidal-c7"  // 用例7 StatefulSet
	TidalTestCRCase8Name     = "tidal-c8"  // 用例8 DaemonSet
	TidalTestCRCase9Name     = "tidal-c9"  // 用例9 CRD
	TidalTestCRCase10Name    = "tidal-c10" // 用例10 规则动态更新
	TidalTestCRCase11Name    = "tidal-c11" // 用例11 下一个触发规则准确性
	TidalTestCRCase12Name    = "tidal-c12" // 用例12 多规则最大值
	TidalTestCRCase13Name    = "tidal-c13" // 用例13 时区敏感
	TidalTestCRCase14Prefix  = "tidal-c14" // 用例14 高并发相同规则：-1,-2,-3
	TidalTestCRCase15Prefix  = "tidal-c15" // 用例15 高并发不同规则：-1,-2,-3
	TidalTestCRCase16Name    = "tidal-c16" // 用例16 Labels缺失
	TidalTestCRCase17Name    = "tidal-c17" // 用例17 无效cron
	TidalTestCRCase18Name    = "tidal-c18" // 用例18 重复规则名
	TidalTestCRCase19Name    = "tidal-c19" // 用例19 目标不存在
	TidalTestDeploymentName  = "test-deploy"
	TidalTestDeploymentC15_1 = "test-deploy-c15-1" // 用例15 每个CR独立目标
	TidalTestDeploymentC15_2 = "test-deploy-c15-2"
	TidalTestDeploymentC15_3 = "test-deploy-c15-3"
	TidalTestNotExistDeploy  = "deploy-not-exist" // 用例19 不存在的目标
	TidalTestStatefulSetName = "test-sts"
	TidalTestDaemonSetName   = "test-ds"
	TidalTestCRDResourceName = "test-scalable" // 用例9 自定义CRD资源名
)
