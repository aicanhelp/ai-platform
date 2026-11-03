package kae_operator

var ExpectedImages = map[string][]string{
	"kae-operator":                      {"cr.openfuyao.cn/openfuyao/kae-operator:v25.09"},
	"kae-node-feature-discovery-gc":     {"registry.k8s.io/nfd/node-feature-discovery:v0.16.4"},
	"kae-node-feature-discovery-master": {"registry.k8s.io/nfd/node-feature-discovery:v0.16.4"},
	"kae-node-feature-discovery-worker": {"registry.k8s.io/nfd/node-feature-discovery:v0.16.4"},
}

// 如果你仍希望在代码中硬编码，请自行在测试开始时设置环境变量。
// 安装指令常量
const (
	NUMA_INSTALL_CMD = "helm install numa cropenfuyao/numa-affinity-package --version 0.15.0 -n default"
	KAE_INSTALL_CMD  = "helm install kae cropenfuyao/kae-operator --version 0.15.0 -n default"
	NPU_INSTALL_CMD  = "helm install npu cropenfuyao/npu-operator --version 0.15.0 -n default"
)

// 卸载指令常量
const (
	NUMA_UNINSTALL_CMD = "helm uninstall numa -n default"
	KAE_UNINSTALL_CMD  = "helm uninstall kae -n default"
	NPU_UNINSTALL_CMD  = "helm uninstall npu -n default"
)
