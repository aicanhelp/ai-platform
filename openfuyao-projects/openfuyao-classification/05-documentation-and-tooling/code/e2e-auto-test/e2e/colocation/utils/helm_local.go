package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

func requireKubeconfigFromEnv() error {
	raw := strings.TrimSpace(os.Getenv("KUBECONFIG"))
	if raw == "" {
		return fmt.Errorf("KUBECONFIG 未设置，请设为 kubeconfig 文件路径后再跑混部套件")
	}
	first := strings.Split(raw, string(os.PathListSeparator))[0]
	first = strings.TrimSpace(first)
	fi, err := os.Stat(first)
	if err != nil {
		return fmt.Errorf("KUBECONFIG 路径 %q 不可用: %w", first, err)
	}
	if fi.IsDir() {
		return fmt.Errorf("KUBECONFIG 路径 %q 是目录，应为文件", first)
	}
	return nil
}

func helmListJSON(ctx context.Context, namespace string) ([]map[string]interface{}, error) {
	cmd := exec.CommandContext(ctx, "helm", "list", "-n", namespace, "-o", "json")
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("helm list: %w", err)
	}
	out = bytes.TrimSpace(out)
	if len(out) == 0 {
		return nil, nil
	}
	var list []map[string]interface{}
	if err := json.Unmarshal(out, &list); err != nil {
		return nil, fmt.Errorf("parse helm list json: %w", err)
	}
	return list, nil
}

// HelmReleaseExists 查询 namespace 下是否已存在指定 helm release。
func HelmReleaseExists(ctx context.Context, helmConfig map[string]string) (bool, error) {
	list, err := helmListJSON(ctx, helmConfig["namespace"])
	if err != nil {
		return false, err
	}
	for _, item := range list {
		if name, ok := item["name"].(string); ok && name == helmConfig["releaseName"] {
			return true, nil
		}
	}
	return false, nil
}

func helmReleaseStatusDeployed(ctx context.Context, releaseName, namespace string) (bool, string, error) {
	cmd := exec.CommandContext(ctx, "helm", "status", releaseName, "-n", namespace, "-o", "json")
	out, err := cmd.Output()
	if err != nil {
		return false, "", err
	}
	var parsed struct {
		Info struct {
			Status string `json:"status"`
		} `json:"info"`
	}
	if err := json.Unmarshal(bytes.TrimSpace(out), &parsed); err != nil {
		return false, "", fmt.Errorf("parse helm status json: %w", err)
	}
	st := strings.ToLower(strings.TrimSpace(parsed.Info.Status))
	return st == "deployed", st, nil
}

func WaitHelmReleaseDeployed(ctx context.Context, releaseName, namespace string, timeout, poll time.Duration) error {
	deadline := time.Now().Add(timeout)
	var last string
	for time.Now().Before(deadline) {
		ok, st, err := helmReleaseStatusDeployed(ctx, releaseName, namespace)
		if err == nil && ok {
			return nil
		}
		if err == nil {
			last = st
		}
		time.Sleep(poll)
	}
	return fmt.Errorf("helm release %q in namespace %q not deployed within %v; last status: %q", releaseName, namespace, timeout, last)
}

// HelmInstallLocal 在本机执行 helm install（依赖 KUBECONFIG）。
func HelmInstallLocal(ctx context.Context, helmConfig map[string]string) error {
	releaseName := helmConfig["releaseName"]
	chartURL := helmConfig["helmUrl"]
	version := helmConfig["version"]
	namespace := helmConfig["namespace"]
	args := []string{"install", releaseName, chartURL, "--version", version, "-n", namespace}
	cmd := exec.CommandContext(ctx, "helm", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm install %s: %w", releaseName, err)
	}
	return nil
}

// HelmUninstallLocal 在本机执行 helm uninstall。
func HelmUninstallLocal(ctx context.Context, helmConfig map[string]string) error {
	releaseName := helmConfig["releaseName"]
	namespace := helmConfig["namespace"]
	cmd := exec.CommandContext(ctx, "helm", "uninstall", releaseName, "-n", namespace)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm uninstall %s: %w", releaseName, err)
	}
	return nil
}
