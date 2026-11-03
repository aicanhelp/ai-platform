package utils

import (
	"os"
	"strings"
	"time"
)

// MetricsTestNamespace 与 ES 套件共用同一个 e2e 专属命名空间（见 ElasticScalerE2ENamespace）
var MetricsTestNamespace = ElasticScalerTestNamespace

// 通用扩缩容决策框架（oFEP-0030/0044）E2E 测试常量
const (
	// ES CR 名称前缀（按 ES-E2E-xxx 编号）
	MetricsESNamePrefix = "metrics-es-"

	MetricsESName001 = MetricsESNamePrefix + "001"
	MetricsESName002 = MetricsESNamePrefix + "002"
	MetricsESName003 = MetricsESNamePrefix + "003"
	MetricsESName004 = MetricsESNamePrefix + "004"
	MetricsESName005 = MetricsESNamePrefix + "005"
	MetricsESName006 = MetricsESNamePrefix + "006"
	MetricsESName007 = MetricsESNamePrefix + "007"
	MetricsESName008 = MetricsESNamePrefix + "008"
	MetricsESName009 = MetricsESNamePrefix + "009"
	MetricsESName010 = MetricsESNamePrefix + "010"
	MetricsESName011 = MetricsESNamePrefix + "011"
	MetricsESName012 = MetricsESNamePrefix + "012"
	MetricsESName013 = MetricsESNamePrefix + "013"
	MetricsESName014 = MetricsESNamePrefix + "014"
	MetricsESName015 = MetricsESNamePrefix + "015"
	MetricsESName016 = MetricsESNamePrefix + "016"
	MetricsESName017 = MetricsESNamePrefix + "017"
	MetricsESName018 = MetricsESNamePrefix + "018"
	MetricsESName019 = MetricsESNamePrefix + "019"
	MetricsESName020 = MetricsESNamePrefix + "020"
	MetricsESName021 = MetricsESNamePrefix + "021"

	// 工作负载
	MetricsTestDeployName    = "metrics-framework-deploy"
	MetricsTestDeployName005 = "metrics-framework-deploy-005"
	MetricsTestDeployName006 = "metrics-framework-deploy-006"
	MetricsTestDeployName007 = "metrics-framework-deploy-007"

	// ES-E2E-008 轻量推理服务（模拟 /v1/chat/completions HTTP 200）
	MetricsInferenceDeployName     = "metrics-inference-deploy"
	MetricsInferenceSvcName        = "metrics-inference-svc"
	MetricsInferencePort           = int32(8000)
	MetricsInferenceProbeJobName     = "metrics-inference-probe"
	MetricsInferenceLoadTestJobName = "metrics-inference-load-test"
	// infer_qps ExternalServer target（低于默认 PromQL 返回值以触发 apa 扩容）
	MetricsE2E008InferQPSTarget = "50"
	// ES-E2E-009 缩容前目标副本数
	MetricsE2E009ScaleUpReplicas = int32(3)

	// ES-E2E-010 Resource/cpu + 推理服务（独立 targetRef，避免与 metrics-es-008 冲突）
	MetricsInferenceDeployName010     = "metrics-inference-deploy-010"
	MetricsInferenceSvcName010        = "metrics-inference-svc-010"
	MetricsInferenceProbeJobName010   = "metrics-inference-probe-010"
	MetricsInferenceLoadTestJobName010 = "metrics-inference-load-test-010"
	// ES-E2E-010 仅需非零 CPU request 供 metrics-server 计算 utilization；用例验证 PodList 采集与 apa 对齐，不依赖扩容量级 CPU。
	// memory 需留足压测期间 nginx 连接缓冲（64Mi limit 在 load-test 下会 OOMKilled，导致 pod metrics 不可用）。
	MetricsInference010CPURequest = "10m"
	MetricsInference010MemRequest = "64Mi"
	MetricsInference010CPULimit   = "100m"
	MetricsInference010MemLimit   = "128Mi"

	// ES-E2E-011 ExternalServer.query 拼接审计（endpoint 固定参数 + query PromQL/step）
	MetricsInferenceDeployName011      = "metrics-inference-deploy-011"
	MetricsInferenceSvcName011         = "metrics-inference-svc-011"
	MetricsInferenceProbeJobName011    = "metrics-inference-probe-011"
	AuditPrometheusExternalServerDeployName = "metrics-audit-prometheus"
	AuditPrometheusExternalServerSvcName    = "metrics-audit-prometheus"
	AuditPrometheusExternalServerPort       = int32(8080)
	AuditPrometheusEndpointTenant             = "e2e011"
	// AuditPrometheusEndpointNamespaceParam 是 ES-E2E-011 审计用例自造的 endpoint 固定查询参数值
	// （用于验证 controller 会把 endpoint 固定参数与 spec.query 一起拼接透传），与部署命名空间无关，
	// 因此不跟随 ElasticScalerE2ENamespace。
	AuditPrometheusEndpointNamespaceParam     = "scaling-system"
	AuditPrometheusQueryStep                  = "15s"

	// ES-E2E-012 Resource/cpu + ExternalServer/infer_qps 多指标全量成功
	MetricsInferenceDeployName012      = "metrics-inference-deploy-012"
	MetricsInferenceSvcName012         = "metrics-inference-svc-012"
	MetricsInferenceProbeJobName012    = "metrics-inference-probe-012"
	MetricsInferenceLoadTestJobName012   = "metrics-inference-load-test-012"
	ApaDualMetricsProcessedCount  = 2

	// ES-E2E-013 双 ExternalServer 部分成功 + apa 算法仍输出 ScalingCalculated
	MetricsInferenceDeployName013    = "metrics-inference-deploy-013"
	MetricsInferenceSvcName013       = "metrics-inference-svc-013"
	MetricsInferenceProbeJobName013  = "metrics-inference-probe-013"
	MetricsInferenceLoadTestJobName013 = "metrics-inference-load-test-013"

	// ES-E2E-014 指标策略注解 windowSeconds=120 + aggregationType=Max 窗口聚合
	MetricsInferenceDeployName014      = "metrics-inference-deploy-014"
	MetricsInferenceSvcName014         = "metrics-inference-svc-014"
	MetricsInferenceProbeJobName014    = "metrics-inference-probe-014"
	MetricsInferenceLoadTestJobName014 = "metrics-inference-load-test-014"
	MetricsWindowAggregationSeconds         = int64(120)

	// ES-E2E-015 Custom 指标 + 推理服务（custom.metrics.k8s.io；无 adapter 时由 mock APIService 提供）
	MetricsInferenceDeployName015      = "metrics-inference-deploy-015"
	MetricsInferenceSvcName015         = "metrics-inference-svc-015"
	MetricsInferenceProbeJobName015    = "metrics-inference-probe-015"
	MetricsInferenceLoadTestJobName015 = "metrics-inference-load-test-015"
	DefaultCustomMetricsMetricName     = "infer_qps"
	DefaultCustomMetricsTarget         = int64(50)

	MockCustomMetricsAPIServerDeployName = "metrics-custom-metrics-mock"
	MockCustomMetricsAPIServerSvcName    = "metrics-custom-metrics-mock"
	MockCustomMetricsAPIServerConfigName = "metrics-custom-metrics-mock-cfg"
	MockCustomMetricsAPIServerTLSSecret  = "metrics-custom-metrics-mock-tls"
	MockCustomMetricsAPIServerPort       = int32(443)
	MockCustomMetricsAPIServerSvcPort    = int32(443)
	MockCustomMetricsAPIServiceName      = "v1beta2.custom.metrics.k8s.io"
	MockCustomMetricsManagedByLabel      = "e2e-metrics-framework"

	// ES-E2E-016 RSG GroupReplication + ES targetRef 基线（caseID 116 避免与 RSG 套件 rsg-e2e-16 冲突）
	MetricsFrameworkRSGName   = "metrics-rsg-016"
	MetricsFrameworkRSGCaseID = 116
	MetricsFrameworkRSGGroupName   = "pd116"

	// ES-E2E-017 RSG 组数扩容（复用 016 RSG/ES；独立推理 Service 产生压测流量）
	MetricsInferenceDeployName017      = "metrics-inference-deploy-017"
	MetricsInferenceSvcName017         = "metrics-inference-svc-017"
	MetricsInferenceProbeJobName017    = "metrics-inference-probe-017"
	MetricsInferenceLoadTestJobName017 = "metrics-inference-load-test-017"

	// ES-E2E-019 RSG targetRef + Resource/cpu PodList（caseID 119）
	MetricsFrameworkRSGResourceName              = "metrics-rsg-019"
	MetricsFrameworkRSGResourceCaseID            = 119
	MetricsFrameworkRSGResourceGroupName            = "pd119"
	MetricsFrameworkRSGLoadTestJobName   = "metrics-rsg-019-load-test"

	// ExternalServer 双端点场景（ES-E2E-006/007）：成功 Prometheus 响应 / 失败非 Prometheus 响应
	OKPrometheusExternalServerDeployName = "metrics-ok-external-server"
	OKPrometheusExternalServerSvcName    = "metrics-ok-external-server"
	OKPrometheusExternalServerPort       = int32(8080)

	MetricsExternalServerInferQPS        = "infer_qps"
	MetricsExternalServerBrokenLatency   = "broken_latency"
	BrokenExternalServerDeployName = "metrics-broken-external-server"
	BrokenExternalServerSvcName    = "metrics-broken-external-server"
	BrokenExternalServerPort       = int32(8080)

	// 集群 Prometheus（ExternalServer 成功路径；METRICS_PROMETHEUS_PROTOCOL / METRICS_PROMETHEUS_ENDPOINT_URL 可覆盖）
	DefaultPrometheusNamespace = "monitoring"
	DefaultPrometheusService   = "prometheus-k8s"
	DefaultPrometheusPort      = int32(9090)
	DefaultPrometheusQueryPath = "/api/v1/query"

	// apa 算法名（与 elastic-scaler DefaultAlgorithmName 一致）
	MetricsScalingAlgorithmApa = "apa"

	// MetricsPipeline / ScalingCalculated 条件 Reason
	MetricsReasonCollectionSucceeded = "MetricsCollectionSucceeded"
	MetricsReasonPartialSuccess    = "MetricsPartialSuccess"
	MetricsReasonScalingSucceeded  = "ScalingSucceeded"

	// 策略注解键（与 pkg/elasticscaler/context/builder.go 常量一致）
	AnnotationMetricsWindowSeconds       = "elasticscaler.io/metrics.windowSeconds"
	AnnotationMetricsAggregationType     = "elasticscaler.io/metrics.aggregationType"
	AnnotationMetricsMissingDataPolicy     = "elasticscaler.io/metrics.missingDataPolicy"
	AnnotationMetricsEnableOutlierFilter   = "elasticscaler.io/metrics.enableOutlierFilter"
	AnnotationMetricsGCFactor              = "elasticscaler.io/metrics.gcFactor"
	AnnotationMetricsOutlierThreshold      = "elasticscaler.io/metrics.outlierThreshold"
)

// MetricsPodMetricsWaitTimeout 等待 metrics-server 产出 Pod CPU 指标的超时（慢集群可通过 METRICS_POD_METRICS_WAIT_TIMEOUT 覆盖，如 12m）。
func MetricsPodMetricsWaitTimeout() time.Duration {
	return metricsEnvDuration("METRICS_POD_METRICS_WAIT_TIMEOUT", 10*time.Minute)
}

// MetricsPipelineWaitTimeout 等待 MetricsPipeline 条件收敛的超时。
func MetricsPipelineWaitTimeout() time.Duration {
	return metricsEnvDuration("METRICS_PIPELINE_WAIT_TIMEOUT", 10*time.Minute)
}

// MetricsApaScaleDownWaitTimeout 等待 apa 缩容至 minReplicas 的超时。
// controller 默认 scaleDown.stabilizationWindowSeconds=180、maxPodsPerStep=1，3→2→1 在 ES 创建后
// 首次缩容通常需 ≥180s 稳定窗口 + 分步缩容，默认 10m（可通过 METRICS_APA_SCALE_DOWN_WAIT_TIMEOUT 覆盖）。
func MetricsApaScaleDownWaitTimeout() time.Duration {
	return metricsEnvDuration("METRICS_APA_SCALE_DOWN_WAIT_TIMEOUT", 10*time.Minute)
}

// ClusterPrometheusWaitTimeout 等待集群 Prometheus Endpoints 就绪的最长时限（超时则 Skip 依赖 Prometheus 的用例）。
func ClusterPrometheusWaitTimeout() time.Duration {
	return metricsEnvDuration("METRICS_PROMETHEUS_WAIT_TIMEOUT", 5*time.Minute)
}

func metricsEnvDuration(key string, fallback time.Duration) time.Duration {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return fallback
	}
	return d
}
