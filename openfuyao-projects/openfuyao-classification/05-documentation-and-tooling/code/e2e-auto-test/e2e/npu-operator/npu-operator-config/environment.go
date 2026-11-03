package npu_operator_config

// Prefixes npu_operator 相关的 pods
var Prefixes = []string{
	"ascend-runtime-containerd",
	"npu-driver",
	"npu-operator",
	"npu-feature-discovery",
	"mindio-acp",
	"mindio-tft",
	"ascend-device-plugin",
	"ascend-operator-manager",
	"clusterd",
	"noded",
	"resilience-controller",
	"npu-exporter",
	"volcano-controllers",
	"volcano-scheduler",
}

var Deployments = []string{
	"npu-operator",
	"ascend-operator-manager",
	"clusterd",
	"resilience-controller",
	"volcano-controllers",
	"volcano-scheduler",
}

var Daemonsets = []string{
	"ascend-runtime-containerd",
	"npu-driver",
	"npu-feature-discovery",
	"mindio-acp",
	"mindio-tft",
	"ascend-device-plugin",
	"noded",
	"npu-exporter",
}

var NpuTypes = []string{
	"d100",
	"d500",
	"d801",
	"d802",
	"d803",
}
