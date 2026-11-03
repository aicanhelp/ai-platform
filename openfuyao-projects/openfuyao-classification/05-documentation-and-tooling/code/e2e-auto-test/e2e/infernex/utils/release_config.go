package utils

// ReleaseConfig 描述 Helm Chart 的基本信息
type ReleaseConfig struct {
	ReleaseName         string   `yaml:"releaseName"`
	HelmURL             string   `yaml:"helmUrl"`
	Version             string   `yaml:"version"`
	Namespace           string   `yaml:"namespace"`
	PredictionModelPath string   `yaml:"predictionModelPath"` // 预测模型目录；相对路径以 hermes-router-config 为基准，加载后解析为绝对路径供 hostPath 挂载
	EppNode             string   `yaml:"-"`                   // 运行时填充：LoadHelmConfig 按本机 hostname/IP 对照集群 Node 识别，不从 YAML 读取
	SetValues           []string `yaml:"-"`                   // --set 参数列表，不序列化到 YAML
	ValuesFile          string   `yaml:"-"`                   // 临时 values 文件路径，使用 -f 参数加载
}
