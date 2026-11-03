package infernex

import "strings"

// InfernexHelmConfig holds the Helm chart configuration for Infernex deployment.
// ChartURL: supports both OCI remote URL (oci://...) and local chart directory/file path.
//   - OCI URL: e.g. "oci://cr.openfuyao.cn/charts/infernex" (default)
//   - Local path: e.g. "/path/to/infernex-chart" or "./charts/infernex"
//   - Local tgz: e.g. "/home/zsc/zhangshichang/temp/infernex-0.0.0-latest.tgz"
//
// Version: set to a specific semver (e.g. "0.22.0") to pass --version;
// set to "latest" or "" to omit --version (uses the repository's default/latest tag).
// Ignored when ChartURL is a local path.
var InfernexHelmConfig = struct {
	ReleaseName string
	ChartURL    string
	Version     string
	Namespace   string
}{
	ReleaseName: "infernex",
	ChartURL:    "oci://cr.openfuyao.cn/charts/infernex",
	Version:     "0.0.0-latest",
	Namespace:   "ai-inference",
}

// IsLocalChart returns true if ChartURL is a local file/directory path (not an OCI URL).
func IsLocalChart() bool {
	return InfernexHelmConfig.ChartURL != "" && !strings.HasPrefix(InfernexHelmConfig.ChartURL, "oci://")
}

// PDTestEnvVarName / PDTestEnvVarValue 与 config/Qwen3-8B-pd-test-env.yaml 中 prefill/decode env 一致（FIT_38/39）。
const (
	PDTestEnvVarName  = "TEST_ENV_VAR"
	PDTestEnvVarValue = "1"
)

// PDBackendTestEnvVar1Name / PDBackendTestEnvVar1Value 与 config/Qwen3-8B-pd-test-env.yaml 中 inference-backend.env 一致（FIT_40）。
const (
	PDBackendTestEnvVar1Name  = "TEST_ENV_VAR_1"
	PDBackendTestEnvVar1Value = "1"
)

// PDExtraArgMarkers 与 config/Qwen3-8B-pd-extra-args.yaml 中 prefill/decode extraArgs 一致（FIT_42/43）。
// 传入 utils 校验 vllm serve 进程命令行时作为 expected 片段列表。
var PDExtraArgMarkers = []string{
	"--enforce-eager",
}

// PDVllmConfigExtraArgMarkers 与 config/Qwen3-8B-pd-vllm-config-params.yaml 一致（FIT_23~26 / FIT_31~34，v26.09 extraArgs 变更）。
var PDVllmConfigExtraArgMarkers = []string{
	"--no-enable-prefix-caching",
	"--max-model-len 4096",
	"--max-num-batched-tokens 10000",
	"--gpu-memory-utilization 0.85",
}

const (
	fit45ExternalMooncakeNS = "ai-mooncake-ext"
	fit45SecondNamespace    = "ai-inference-2"
	fit45SecondGatewayPort  = 30188
)
