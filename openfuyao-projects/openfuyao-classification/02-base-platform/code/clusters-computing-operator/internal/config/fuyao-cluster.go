package config

type FuyaoCluster struct {
	APIVersion string    `yaml:"apiVersion"`
	Kind       string    `yaml:"kind"`
	Metadata   FuyaoMeta `yaml:"metadata"`
	Spec       FuyaoSpec `yaml:"spec"`
}

type FuyaoMeta struct {
	Name string `yaml:"name"`
}

type FuyaoSpec struct {
	Nodes FuyaoNodes `yaml:"nodes"`
}

type FuyaoNodes struct {
	Masters []FuyaoNode `yaml:"masters"`
	Workers []FuyaoNode `yaml:"workers"`
}

type FuyaoNode struct {
	Hostname       string       `yaml:"hostname"`
	User           string       `yaml:"user"`
	IP             string       `yaml:"ip"`
	Port           int32        `yaml:"port"`
	PrivateKeyPath string       `yaml:"privateKeyPath"`
	RemoveTaints   bool         `yaml:"removeTaints,omitempty"`
	Labels         []FuyaoLabel `yaml:"labels,omitempty"`
}

// FuyaoLabel 定义了标签
type FuyaoLabel struct {
	Name  string `yaml:"name"`
	Value string `yaml:"value"`
}
