package utils

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// HelmInstallChart helm install chart [--version 合法 semver] --set eagle-eye.enabled=...
// 支持 OCI URL (oci://...) 和本地 chart 路径。
// version 为空或为 "latest" 时不传 --version（Helm 不接受 --version latest，会报 improper constraint）。
// 使用本地 chart 路径时 version 字段被忽略。
func HelmInstallChart(releaseName, chartURL, version, namespace string, eagleEyeEnabled bool) error {
	args := []string{"install", releaseName, chartURL}
	isLocal := chartURL != "" && !strings.HasPrefix(chartURL, "oci://")
	if !isLocal {
		v := strings.TrimSpace(version)
		if v != "" && !strings.EqualFold(v, "latest") {
			args = append(args, "--version", v)
		}
	}
	args = append(args,
		"-n", namespace,
		"--set", "eagle-eye.enabled="+strconv.FormatBool(eagleEyeEnabled),
	)
	cmd := exec.Command("helm", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm install failed: %w\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
	}
	return nil
}

// HelmUninstall 卸载 release
func HelmUninstall(releaseName, namespace string) error {
	cmd := exec.Command("helm", "uninstall", releaseName, "-n", namespace)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm uninstall %s/%s: %w stderr: %s", namespace, releaseName, err, stderr.String())
	}
	return nil
}

// HelmUninstallIfExists 若 release 存在则卸载
func HelmUninstallIfExists(releaseName, namespace string) error {
	cmd := exec.Command("helm", "list", "-n", namespace, "-o", "json")
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	var list []map[string]any
	if json.Unmarshal(out, &list) != nil {
		return nil
	}
	for _, item := range list {
		if name, ok := item["name"].(string); ok && name == releaseName {
			return HelmUninstall(releaseName, namespace)
		}
	}
	return nil
}
