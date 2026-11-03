package infernex_checker

// InfernexCheckerConfig stores external artifact locations used by the
// black-box infernex-checker e2e tests. Fill NodesYAMLURL with the third-party
// nodes.yaml URL before running the suite. values.yaml is generated from the
// Infernex Helm chart with `helm show values`.
var InfernexCheckerConfig = struct {
	BinaryURLPattern string
	BinaryURL        string
	NodesYAMLURL     string
	NodesYAMLPath    string
	EnvFilePath      string
	KubeconfigPath   string
}{
	BinaryURLPattern: "https://static.openfuyao.cn/openFuyao/infernex/releases/download/latest/bin/linux/%s/infernex-checker",
	BinaryURL:        "",
	NodesYAMLURL:     "",
	NodesYAMLPath:    "",
	EnvFilePath:      ".env",
	KubeconfigPath:   "",
}

var InfernexCheckerHelmConfig = struct {
	ReleaseName string
	ChartURL    string
	Version     string
	Namespace   string
}{
	ReleaseName: "infernex",
	ChartURL:    "oci://cr.openfuyao.cn/charts/infernex",
	Version:     "0.0.0-latest",
	Namespace:   "default",
}
