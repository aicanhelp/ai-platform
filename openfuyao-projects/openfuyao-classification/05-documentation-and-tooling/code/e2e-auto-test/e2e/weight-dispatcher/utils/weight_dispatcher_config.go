package utils

// weight-dispatcher的helm配置
const (
	WeightDispatcherHelmReleaseName = "weight-dispatcher"
	WeightDispatcherHelmChart       = "oci://cr.openfuyao.cn/charts/weight-dispatcher"
	WeightDispatcherHelmVersion     = "0.0.0-latest"
	WeightDispatcherHelmNamespace   = "default"
)

// WeightDispatcherRelaeseName 保留原拼写以兼容旧引用。
const WeightDispatcherRelaeseName = WeightDispatcherHelmReleaseName

const (
	WeightDispatcherTestNamespace = "weight-dispatcher-e2e"
	ModelWarmupJobCRDName         = "modelwarmupjobs.warmup.openfuyao.com"
	ModelWarmupJobAPIVersion      = "warmup.openfuyao.com/v1alpha1"
	ModelWarmupJobKind            = "ModelWarmupJob"
	WeightDispatcherPodKeyword    = "weight-dispatcher"
	WeightDispatcherControllerKey = "controller"
	WeightDispatcherNodeAgentKey  = "node-agent"
)

const (
	ModelWarmupJobFIT2Name       = "modelwarmupjob-fit-2"
	ModelWarmupJobFIT3Name       = "modelwarmupjob-fit-3"
	ModelWarmupJobFIT4Name       = "modelwarmupjob-fit-4"
	ModelWarmupJobFIT4Join       = "modelwarmupjob-fit-4-join"
	ModelWarmupJobFIT5Name       = "modelwarmupjob-fit-5"
	ModelWarmupJobFIT6Name       = "modelwarmupjob-fit-6"
	ModelWarmupJobFIT7Name       = "modelwarmupjob-fit-7"
	ModelWarmupJobFIT9Name       = "modelwarmupjob-fit-9"
	ModelWarmupJobFIT10Name      = ModelWarmupJobFIT2Name
	ModelWarmupJobFIT11Name      = "modelwarmupjob-fit-11"
	ModelWarmupJobFIT12Name      = "modelwarmupjob-fit-12"
	ModelWarmupJobFIT13Name      = "modelwarmupjob-fit-13"
	ModelWarmupJobFIT14Name      = "modelwarmupjob-fit-14"
	ModelWarmupJobFIT15Base      = "modelwarmupjob-fit-15"
	ModelWarmupJobFIT16Name      = "modelwarmupjob-fit-16"
	ModelWarmupJobFIT17Name      = "modelwarmupjob-fit-17"
	ModelWarmupJobFIT18Name      = "modelwarmupjob-fit-18"
	ModelWarmupJobFIT19Name      = "modelwarmupjob-fit-19"
	ModelWarmupJobFIT20Name      = "modelwarmupjob-fit-20"
	ModelWarmupJobFIT21Name      = "modelwarmupjob-fit-21"
	ModelWarmupJobFIT22Name      = "modelwarmupjob-fit-22"
	ModelWarmupJobFIT23Name      = "modelwarmupjob-fit-23"
	ModelWarmupJobFIT24Name      = "modelwarmupjob-fit-24"
	ModelWarmupJobFIT24RetryName = "modelwarmupjob-fit-24-retry"
	ModelWarmupJobFIT25Name      = "modelwarmupjob-fit-25"
	ModelWarmupJobFIT26Name      = "modelwarmupjob-fit-26"

	ModelWarmupJobDefaultArtifactKey          = "qwen3-8b:2026-06-05"
	ModelWarmupJobDefaultSourcePath           = "/home/llm_cache/huggingface/hub/models--Qwen--Qwen3-8B"
	ModelWarmupJobDefaultSecondSourcePath     = "/var/lib/weight-dispatcher/cache/e2e/source-alt/qwen3-8b-link"
	ModelWarmupJobDefaultSingleFileSourcePath = "/var/lib/weight-dispatcher/cache/e2e/source-single/qwen3-8b/model.safetensors"
	ModelWarmupJobDefaultMismatchSourcePath   = "/var/lib/weight-dispatcher/cache/e2e/source-mismatch/qwen3-8b"
	ModelWarmupJobDefaultTimeoutSourcePath    = "/var/lib/weight-dispatcher/cache/e2e/source-timeout/qwen3-8b"
	ModelWarmupJobDefaultTargetPath           = "/var/lib/weight-dispatcher/cache/e2e/ready"
	ModelWarmupJobFIT5LabelKey                = "weight-dispatcher-e2e.openfuyao.com/fit5"
	ModelWarmupJobFIT5LabelValue              = "true"
	ModelWarmupJobFIT14ArtifactKey            = "qwen/3:70b alpha"
	ModelWarmupJobFIT15BatchCount             = 10
	ModelWarmupJobValidationFailed            = "ValidationFailed"
	ModelWarmupJobTCPFallbackPath             = "TCP_FALLBACK"
)

const (
	WeightDispatcherDefaultControllerLogLevel = "debug"
	WeightDispatcherDefaultNodeAgentLogLevel  = "debug"
	WeightDispatcherDefaultNodeAgentCacheRoot = "/var/lib/weight-dispatcher/cache"
	WeightDispatcherDefaultHostHomeCacheRoot  = "/home/llm_cache"

	WeightDispatcherDefaultRDMAEnabled              = "true"
	WeightDispatcherDefaultTCPFallbackEnabled       = "true"
	WeightDispatcherDefaultHostNetwork              = "true"
	WeightDispatcherDefaultPrivileged               = "true"
	WeightDispatcherDefaultPostWriteCRC32CEnabled   = "false"
	WeightDispatcherDefaultInjectCRCMismatchOnce    = "false"
	WeightDispatcherDefaultForceTCPFallback         = "false"
	WeightDispatcherDefaultNodeAgentRuntimeLogLevel = "debug"
)
