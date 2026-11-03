package infernex_eagle_eye

import "strings"

// InfernexHelmConfig 与 e2e/eagle-eye/constants.go 中 EagleEyeHelmConfig 用法一致。
// ChartURL: supports both OCI remote URL (oci://...) and local chart directory/file path.
//   - OCI URL: e.g. "oci://cr.openfuyao.cn/charts/infernex" (default)
//   - Local path: e.g. "/path/to/infernex-chart" or "./charts/infernex"
//
// Version：
//   - 写具体 semver（如 0.22.0）→ helm install 会带 --version 0.22.0；
//   - 写 "latest" 或 "" → 不传 --version（等价于「跟仓库默认/最新」），切勿写成 helm 的 --version latest（会报 improper constraint）。
//   - 使用本地 chart 路径时 Version 字段被忽略。
var InfernexHelmConfig = struct {
	ReleaseName string
	ChartURL    string
	Version     string
	Namespace   string
}{
	ReleaseName: "infernex",
	ChartURL:    "oci://cr.openfuyao.cn/charts/infernex",
	Version:     "latest",
	Namespace:   "ai-inference",
}

// IsLocalChart returns true if ChartURL is a local file/directory path (not an OCI URL).
func IsLocalChart() bool {
	return InfernexHelmConfig.ChartURL != "" && !strings.HasPrefix(InfernexHelmConfig.ChartURL, "oci://")
}
