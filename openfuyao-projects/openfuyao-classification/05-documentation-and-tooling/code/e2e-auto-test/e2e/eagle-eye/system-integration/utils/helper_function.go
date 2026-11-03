package utils

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gopkg.in/yaml.v3"
)

type ReleaseConfig struct {
	ReleaseName string `yaml:"releaseName"`
	HelmURl     string `yaml:"helmUrl"`
	Version     string `yaml:"version"`
	Namespace   string `yaml:"namespace"`
}

// LoadHelmConfig 加载helm配置
func LoadHelmConfig() (map[string]*ReleaseConfig, error) {
	parentPath, err := getParentPath()
	if err != nil {
		return nil, err
	}
	// 使用framework/helm下的配置
	configPath := filepath.Join(parentPath, "framework", "helm", "helm_config.yaml")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("read helm config failed, %v", err)
	}

	var config map[string]*ReleaseConfig
	err = yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf("helm config yaml parse failed, %v", err)
	}
	return config, nil
}

// getParentPath 获取父目录路径
func getParentPath() (string, error) {
	currentDir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		templatesDir := filepath.Join(currentDir, "framework")
		if _, err := os.Stat(templatesDir); err == nil {
			return currentDir, nil
		}
		parent := filepath.Dir(currentDir)
		if parent == currentDir {
			break
		}
		currentDir = parent
	}
	return "", fmt.Errorf("parent path not found")
}

// HelmInstall 执行helm install
func HelmInstall(releaseName string, helmUrl string, version string, namespace string, sshClient *executor.SSHExecutor, installArgs ...string) error {
	cmdArgs := strings.Join(installArgs, " ")
	cmd := fmt.Sprintf("helm install %s %s --version %s -n %s", releaseName, helmUrl, version, namespace)
	if cmdArgs != "" {
		cmd = fmt.Sprintf("%s %s", cmd, cmdArgs)
	}
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("helm install %s failed : %v", releaseName, err)
	}
	return nil
}

// HelmUninstall 执行helm uninstall
func HelmUninstall(releaseName string, namespace string, sshClient *executor.SSHExecutor) error {
	cmd := fmt.Sprintf("helm uninstall %s -n %s", releaseName, namespace)
	_, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("helm uninstall %s failed : %v", releaseName, err)
	}
	return nil
}

// CheckHelmRelease 检查helm release是否存在
func CheckHelmRelease(releaseName string, namespace string, sshClient *executor.SSHExecutor) (bool, error) {
	cmd := fmt.Sprintf("helm list -n %s -o json", namespace)
	res, err := sshClient.Exec(cmd)
	if err != nil {
		return false, err
	}
	output := res.Stdout
	var helmList []map[string]interface{}
	err = json.Unmarshal([]byte(output), &helmList)
	if err != nil {
		return false, fmt.Errorf("failed to parse helm list : %v", err)
	}
	for _, each := range helmList {
		if releaseName == each["name"] {
			return true, nil
		}
	}
	return false, nil
}
