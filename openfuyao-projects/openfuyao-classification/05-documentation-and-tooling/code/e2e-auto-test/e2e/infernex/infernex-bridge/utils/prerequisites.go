package utils

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	bridgeReleaseName         = "infernex-bridge"
	defaultBridgeChartOCI     = "oci://cr.openfuyao.cn/charts/infernex-bridge"
	defaultBridgeChartVersion = "0.0.0-latest"
)

// EagleEyeDepsDir returns directory with values-nats.yaml / values-prometheus.yaml (optional).
func EagleEyeDepsDir() string {
	if v := os.Getenv("EAGLE_EYE_DEPS_DIR"); v != "" {
		return v
	}
	dir, err := packageDir()
	if err != nil {
		return ""
	}
	candidate := filepath.Clean(filepath.Join(dir, "..", "..", "..", "..", "eagle-eye-deps"))
	if st, err := os.Stat(candidate); err == nil && st.IsDir() {
		return candidate
	}
	return ""
}

func helmValuesFlag(envKey, defaultRelPath string) string {
	if p := os.Getenv(envKey); p != "" {
		return fmt.Sprintf(" -f %q", p)
	}
	if root := EagleEyeDepsDir(); root != "" && defaultRelPath != "" {
		p := filepath.Join(root, defaultRelPath)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return fmt.Sprintf(" -f %q", p)
		}
	}
	return ""
}

// InstallNATS installs NATS (release/namespace nats → nats.nats:4222 for Eagle-Eye).
func InstallNATS(fitID string) error {
	ns := NATSNamespace()
	release := NATSRelease()
	values := helmValuesFlag("NATS_HELM_VALUES", "values-nats.yaml")
	args := fmt.Sprintf(
		"upgrade --install %s nats/nats --namespace %s --create-namespace%s --wait --timeout 10m",
		release, ns, values,
	)
	return runHelmUpgradeInstall(fitID, "install NATS", args)
}

// InstallKubePrometheusStack installs kube-prometheus-stack for Eagle-Eye metrics.
func InstallKubePrometheusStack(fitID string) error {
	ns := PrometheusNamespace()
	release := PrometheusRelease()
	timeout := envOr("PROMETHEUS_HELM_TIMEOUT", "20m")
	values := helmValuesFlag("PROMETHEUS_HELM_VALUES", "values-prometheus.yaml")
	wait := ""
	if PrometheusHelmWait() {
		wait = " --wait"
	}
	args := fmt.Sprintf(
		"upgrade --install %s prometheus-community/kube-prometheus-stack --namespace %s --create-namespace%s --timeout %s%s",
		release, ns, values, timeout, wait,
	)
	return runHelmUpgradeInstall(fitID, "install kube-prometheus-stack", args)
}

// prepareBridgeClusterForInstall removes leftovers from interrupted helm install/uninstall.
func prepareBridgeClusterForInstall(fitID string) {
	ns := BridgeSystemNS()
	for _, name := range []string{
		"infernex-bridge-generate-webhook-cert",
	} {
		cmd := fmt.Sprintf(
			"KUBECONFIG=%q kubectl delete job %s -n %s --ignore-not-found",
			localKubeconfig(), name, ns,
		)
		stdout, stderr, _ := ExecLocal(cmd)
		if strings.TrimSpace(stdout+stderr) != "" {
			LogStep(fitID, "cleanup orphan bridge job %s: %s", name, trimForLog(stdout+stderr))
		}
	}
}

// InstallInferNexBridge installs InferNex-Bridge via OCI chart (or local path override).
func InstallInferNexBridge(fitID string) error {
	prepareBridgeClusterForInstall(fitID)
	ns := BridgeSystemNS()
	if chartPath := BridgeChartPath(); chartPath != "" {
		args := fmt.Sprintf(
			"upgrade --install %s %q --namespace %s --create-namespace --wait --timeout 10m",
			bridgeReleaseName, chartPath, ns,
		)
		return runHelmUpgradeInstall(fitID, "install infernex-bridge (local chart)", args)
	}
	oci := envOr("INFERNEX_BRIDGE_CHART_OCI", defaultBridgeChartOCI)
	version := envOr("INFERNEX_BRIDGE_CHART_VERSION", defaultBridgeChartVersion)
	args := fmt.Sprintf(
		"upgrade --install %s %s --version %s --namespace %s --create-namespace --wait --timeout 10m",
		bridgeReleaseName, oci, version, ns,
	)
	return runHelmUpgradeInstall(fitID, "install infernex-bridge (OCI)", args)
}

// InstallBridge is an alias for InstallInferNexBridge.
func InstallBridge(fitID string) error {
	return InstallInferNexBridge(fitID)
}

func uninstallInferNexBridge(fitID string) error {
	return runHelmUninstall(fitID, "uninstall infernex-bridge", bridgeReleaseName, BridgeSystemNS())
}

func uninstallKubePrometheusStack(fitID string) error {
	return runHelmUninstall(fitID, "uninstall kube-prometheus-stack", PrometheusRelease(), PrometheusNamespace())
}

func uninstallNATS(fitID string) error {
	return runHelmUninstall(fitID, "uninstall NATS", NATSRelease(), NATSNamespace())
}

// TeardownBeforeSuitePrerequisites uninstalls helm releases installed by BeforeSuite (unless skipped).
func TeardownBeforeSuitePrerequisites(fitID string) {
	if SkipPrerequisitesUninstall() {
		LogStep(fitID, "SKIP_PREREQUISITES_UNINSTALL=true — keep NATS/prometheus/bridge helm releases")
		return
	}
	var agg []string
	if !SkipBridgeInstall() {
		if err := uninstallInferNexBridge(fitID); err != nil {
			agg = append(agg, fmt.Sprintf("infernex-bridge: %v", err))
		}
		prepareBridgeClusterForInstall(fitID)
	}
	if !SkipKubePrometheusInstall() {
		if err := uninstallKubePrometheusStack(fitID); err != nil {
			agg = append(agg, fmt.Sprintf("kube-prometheus: %v", err))
		}
	}
	if !SkipNATSInstall() {
		if err := uninstallNATS(fitID); err != nil {
			agg = append(agg, fmt.Sprintf("nats: %v", err))
		}
	}
	if len(agg) > 0 {
		LogStep(fitID, "prerequisites uninstall warnings: %s", strings.Join(agg, "; "))
	}
}

// InstallBeforeSuitePrerequisites installs NATS, Prometheus, Bridge, then optional KServe/Gateway hooks.
func InstallBeforeSuitePrerequisites(fitID string) error {
	if err := EnsureHelmRepos(fitID); err != nil {
		return fmt.Errorf("helm repos: %w", err)
	}
	if !SkipNATSInstall() {
		if err := InstallNATS(fitID); err != nil {
			return fmt.Errorf("nats: %w", err)
		}
	} else {
		LogStep(fitID, "SKIP_NATS_INSTALL=true — skip NATS helm install")
	}
	if !SkipKubePrometheusInstall() {
		if err := InstallKubePrometheusStack(fitID); err != nil {
			return fmt.Errorf("kube-prometheus-stack: %w", err)
		}
	} else {
		LogStep(fitID, "SKIP_KUBE_PROMETHEUS_INSTALL=true — skip kube-prometheus-stack helm install")
	}
	if !SkipBridgeInstall() {
		if err := InstallInferNexBridge(fitID); err != nil {
			return fmt.Errorf("infernex-bridge: %w", err)
		}
	} else {
		LogStep(fitID, "SKIP_BRIDGE_INSTALL=true — skip infernex-bridge helm install")
	}
	if err := InstallOptionalStackPrerequisites(fitID); err != nil {
		return fmt.Errorf("optional stack: %w", err)
	}
	return nil
}
