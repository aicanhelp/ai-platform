package npu_operator_config

const (
	// HelmURL 拉取chart包url
	//HelmURL = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/npu-operator-1.1.1.tgz"
	//HelmURL = "../../npu-operator"
	HelmURL = "oci://cr.openfuyao.cn/charts/npu-operator"
	// HelmVersion 默认安装包含 vNPU/DRA 集成能力的 latest 开发 chart；需要固定其他版本时通过
	// NPU_OPERATOR_HELM_VERSION 覆盖。
	HelmVersion = "0.0.0-latest"
	// ReleaseName 部署名称
	ReleaseName = "npu"
	// InstallationNamespace 部署命名空间
	InstallationNamespace = "default"
	// K8SNamespace k8s 的命名空间
	K8SNamespace = "k8s.io"
	// ContainerImage 验证 Ascend Docker Runtime 用到的容器镜像
	ContainerImage = "docker.io/library/ubuntu:22.04"
	// ContainerImagePrefix 验证 Ascend Docker Runtime 用到的容器镜像的源仓库
	ContainerImagePrefix = "docker.io/library/"
	// ContainerID 验证 Ascend Docker Runtime 用到的容器ID
	ContainerID = "13"
)

var ExpectedImages = map[string][]string{
	"ascend-runtime-containerd":              {"cr.openfuyao.cn/openfuyao/ascend-docker-runtime:v7.3.0"},
	"npu-driver":                             {"cr.openfuyao.cn/openfuyao/npu-driver-installer:latest"},
	"npu-operator":                           {"cr.openfuyao.cn/openfuyao/npu-operator:latest"},
	"npu-operator-node-feature-discovery-gc": {"hub.oepkgs.net/openfuyao/nfd/node-feature-discovery:v0.16.4"},
	"mindio-acp":                             {"cr.openfuyao.cn/openfuyao/npu-node-provision:latest"},
	"mindio-tft":                             {"cr.openfuyao.cn/openfuyao/npu-node-provision:latest"},
	"ascend-device-plugin":                   {"hub.oepkgs.net/openfuyao/ascendhub/ascend-k8sdeviceplugin:v7.3.0"},
	"ascend-operator-manager":                {"hub.oepkgs.net/openfuyao/ascendhub/ascend-operator:v7.3.0"},
	"clusterd":                               {"hub.oepkgs.net/openfuyao/ascendhub/clusterd:v7.3.0"},
	"noded":                                  {"hub.oepkgs.net/openfuyao/ascendhub/noded:v7.3.0"},
	"resilience-controller":                  {"hub.oepkgs.net/openfuyao/ascendhub/resilience-controller:v7.1.RC1"},
	"npu-exporter":                           {"hub.oepkgs.net/openfuyao/ascendhub/npu-exporter:v7.3.0"},
	"volcano-controllers":                    {"hub.oepkgs.net/openfuyao/ascendhub/vc-controller-manager:v1.9.0-v7.3.0"},
	"volcano-scheduler":                      {"hub.oepkgs.net/openfuyao/ascendhub/vc-scheduler:v1.9.0-v7.3.0"},
}
