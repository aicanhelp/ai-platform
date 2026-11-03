package config

import (
	"fmt"
	"gopkg.in/yaml.v2"
	"os"
)

type FuyaoCreateClusterConfig struct {
	Name      string   `yaml:"name"`
	Namespace string   `yaml:"namespace"`
	Image     string   `yaml:"image"`
	Command   []string `yaml:"command"`
	Args      []string `yaml:"args"`
	Volumes   []struct {
		Name     string `yaml:"name"`
		HostPath string `yaml:"hostPath"`
	} `yaml:"volumes"`
	VolumeMounts []struct {
		Name      string `yaml:"name"`
		MountPath string `yaml:"mountPath"`
	} `yaml:"volumeMounts"`
}

func LoadConfig(path string) (*FuyaoCreateClusterConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config file error:%w", err)
	}

	var config FuyaoCreateClusterConfig
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return &config, nil
}
