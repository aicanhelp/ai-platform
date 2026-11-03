package helm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gopkg.in/yaml.v3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
)

type ReleaseConfig struct {
	ReleaseName string `yaml:"releaseName"`
	HelmURrl    string `yaml:"helmUrl"`
	Version     string `yaml:"version"`
	Namespace   string `yaml:"namespace"`
}

// load helm_config
func LoadHelmConfig() (map[string]*ReleaseConfig, error) {
	parentPath, err := getParentPath()
	if err != nil {
		return nil, err
	}
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

// get parent path
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

// release check
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

func CheckHelmReleaseLocal(releaseName string, namespace string, localexecutor *executor.LocalExecutor) (bool, error) {
	cmd := fmt.Sprintf("helm list -n %s -o json", namespace)
	res, err := localexecutor.Exec(cmd)
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

// helm install
func HelmInstall(releaseName string, helmUrl string, version string, namespace string, sshClient *executor.SSHExecutor, installArgs ...string) error {
	cmdArgs := strings.Join(installArgs, " ")
	cmd := fmt.Sprintf("helm install %s %s --version %s -n %s", releaseName, helmUrl, version, namespace)
	if cmdArgs != "" {
		cmd = fmt.Sprintf("%s %s", cmd, cmdArgs)
	}
	res, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("helm install %s failed: %w, stderr: %s, stdout: %s", releaseName, err, strings.TrimSpace(res.Stderr), strings.TrimSpace(res.Stdout))
	}
	return nil
}

func HelmInstallLocal(releaseName string, helmUrl string, version string, namespace string, localexecutor *executor.LocalExecutor, installArgs ...string) error {
	cmdArgs := strings.Join(installArgs, " ")
	cmd := fmt.Sprintf("helm install %s %s --version %s -n %s", releaseName, helmUrl, version, namespace)
	if cmdArgs != "" {
		cmd = fmt.Sprintf("%s %s", cmd, cmdArgs)
	}
	res, err := localexecutor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("helm install %s failed: %w, stderr: %s, stdout: %s", releaseName, err, strings.TrimSpace(res.Stderr), strings.TrimSpace(res.Stdout))
	}
	return nil
}

// helm uninstall
func HelmUninstall(releaseName string, namespace string, sshClient *executor.SSHExecutor) error {
	cmd := fmt.Sprintf("helm uninstall %s -n %s", releaseName, namespace)
	res, err := sshClient.Exec(cmd)
	if err != nil {
		return fmt.Errorf("helm uninstall %s failed: %w, stderr: %s, stdout: %s", releaseName, err, strings.TrimSpace(res.Stderr), strings.TrimSpace(res.Stdout))
	}
	return nil
}

func HelmUninstallLocal(releaseName string, namespace string, localexecutor *executor.LocalExecutor) error {
	cmd := fmt.Sprintf("helm uninstall %s -n %s", releaseName, namespace)
	res, err := localexecutor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("helm uninstall %s failed: %w, stderr: %s, stdout: %s", releaseName, err, strings.TrimSpace(res.Stderr), strings.TrimSpace(res.Stdout))
	}
	return nil
}

func HelmUninstallNamespaceCheck(releaseName string, k8sClient *k8s.K8SClient, ctx context.Context) (bool, error) {

	helmInstallNamespace := make(map[string][]string)

	helmInstallNamespace["logging-package"] = []string{"loki"}
	helmInstallNamespace["colocation-package"] = []string{"openfuyao-colocation"}
	helmInstallNamespace["multi-cluster-service"] = []string{"karmada-system"}
	helmInstallNamespace["monitoring-dashboard"] = []string{"monitoring-dashboard"}
	helmInstallNamespace["ray-package"] = []string{"vcjob"}
	helmInstallNamespace["npu-operator"] = []string{"mindx-dl", "npu-exporter"}
	helmInstallNamespace["infernex"] = []string{"scaling-system"}

	var err error
	flag := true

	if _, exists := helmInstallNamespace[releaseName]; exists {
		for _, namespace := range helmInstallNamespace[releaseName] {
			_, err = k8sClient.GetNamespace(ctx, namespace)
			if err == nil {
				flag = false
			} else {
				if !apierrors.IsNotFound(err) {
					return false, fmt.Errorf("get namespace failed : %v", err)
				}
			}
		}
	}
	return flag, nil
}
