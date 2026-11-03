package utils

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	defaultBridgeSystemNS = "infernex-bridge-system"
	defaultE2ENS          = "infernex-bridge-e2e"
	defaultTemplateNS     = "infernex-bridge-system"
)

func envOr(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func envBool(key string) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	return v == "1" || v == "true" || v == "yes"
}

func BridgeSystemNS() string {
	return envOr("INFERNEX_BRIDGE_SYSTEM_NS", defaultBridgeSystemNS)
}

func E2ENS() string {
	return envOr("INFERNEX_E2E_NS", defaultE2ENS)
}

func TemplateNS() string {
	return envOr("INFERNEX_TEMPLATE_NS", defaultTemplateNS)
}

func KServeNS() string {
	return envOr("KSERVE_NS", "kserve")
}

// KServeGatewayNS is the namespace of the Envoy-provisioned Service for Gateway CR kserve-ingress-gateway.
func KServeGatewayNS() string {
	return envOr("KSERVE_GATEWAY_NS", "envoy-gateway-system")
}

// KServeGatewaySvcPrefix matches envoy-kserve-kserve-ingress-gateway-<hash> (name suffix varies).
func KServeGatewaySvcPrefix() string {
	return envOr("KSERVE_GATEWAY_SVC_PREFIX", "envoy-kserve-kserve-ingress-gateway")
}

// EnvoyGatewaySystemNS is where Envoy Gateway controller publishes NodePort/LB Services.
func EnvoyGatewaySystemNS() string {
	return KServeGatewayNS()
}

// AIGatewaySystemNS is the namespace of the envoy-ai-gateway controller (extension server +
// InferencePool status writer).
func AIGatewaySystemNS() string {
	return envOr("AI_GATEWAY_NS", "envoy-ai-gateway-system")
}

func BridgeChartPath() string {
	return envOr("INFERNEX_BRIDGE_CHART_PATH", "")
}

func SkipBridgeInstall() bool { return envBool("SKIP_BRIDGE_INSTALL") }

func SkipNATSInstall() bool { return envBool("SKIP_NATS_INSTALL") }

func SkipKubePrometheusInstall() bool {
	if envBool("SKIP_KUBE_PROMETHEUS_INSTALL") {
		return true
	}
	return envBool("SKIP_PROMETHEUS") // alias used by eagle-eye-deps/install.sh
}

// SkipKServeInstall defaults false: clean E2E clones KServe and runs llmisvc-full-install-helm.sh.
func SkipKServeInstall() bool { return envBoolDefault("SKIP_KSERVE_INSTALL", false) }

// SkipGatewayStackInstall defaults true (gateway deps ship with KServe bootstrap). Set false only with SKIP_KSERVE_INSTALL=true.
func SkipGatewayStackInstall() bool { return envBoolDefault("SKIP_GATEWAY_STACK_INSTALL", true) }

func KServeGitURL() string { return envOr("KSERVE_GIT_URL", defaultKServeGitURL) }

// KServeVersion resolves the git ref / Helm chart version (tag or branch).
// Priority: KSERVE_VERSION > SET_KSERVE_VERSION > KSERVE_GIT_REF > default.
func KServeVersion() string {
	for _, key := range []string{"KSERVE_VERSION", "SET_KSERVE_VERSION", "KSERVE_GIT_REF"} {
		if v := strings.TrimSpace(os.Getenv(key)); v != "" {
			return normalizeKServeGitRef(v)
		}
	}
	return defaultKServeGitRef
}

// KServeGitRef is an alias of KServeVersion for checkout.
func KServeGitRef() string { return KServeVersion() }

// KServeHelmVersion is passed to manage.*.sh as SET_KSERVE_VERSION.
func KServeHelmVersion() string { return KServeVersion() }

func SkipKServeExternalLB() bool { return envBool("KSERVE_SKIP_EXTERNAL_LB") }

// SkipKServeUninstall keeps cluster releases after the suite (debug only); temp clone is still removed.
func SkipKServeUninstall() bool { return envBool("SKIP_KSERVE_UNINSTALL") }

// SkipPrerequisitesUninstall keeps BeforeSuite helm releases (debug only).
func SkipPrerequisitesUninstall() bool { return envBool("SKIP_PREREQUISITES_UNINSTALL") }

// KServeHelmReinstall runs manage.kserve-helm.sh --reinstall to avoid orphan RBAC from partial uninstalls.
func KServeHelmReinstall() bool { return envBoolDefault("KSERVE_HELM_REINSTALL", true) }

// SkipKServeUninstallShared keeps cert-manager/envoy/gateway CRDs on cluster after suite (debug on shared clusters).
func SkipKServeUninstallShared() bool { return envBool("SKIP_KSERVE_UNINSTALL_SHARED") }

func envBoolDefault(key string, def bool) bool {
	if strings.TrimSpace(os.Getenv(key)) == "" {
		return def
	}
	return envBool(key)
}

func NATSNamespace() string { return envOr("NATS_NAMESPACE", "nats") }

func NATSRelease() string { return envOr("NATS_RELEASE", "nats") }

func PrometheusNamespace() string { return envOr("PROM_NAMESPACE", "monitoring") }

func PrometheusRelease() string { return envOr("PROM_RELEASE", "kube-prometheus") }

func PrometheusHelmWait() bool { return envBool("PROMETHEUS_HELM_WAIT") }

func SkipLLMISVCE2E() bool { return envBool("SKIP_LLMISVC_E2E") }

// SkipInferenceE2E is opt-out only: unset env ⇒ run chat/completions + Mooncake KV checks (default on).
// Set SKIP_INFERENCE_E2E=true only when gateway/NPU inference is unavailable (local static checks).
func SkipInferenceE2E() bool   { return envBool("SKIP_INFERENCE_E2E") }
func SkipNPUWorkloadE2E() bool { return envBool("SKIP_NPU_WORKLOAD_E2E") }

// SkipLWSE2E skips FIT 56–59 when the cluster cannot run LeaderWorkerSet workloads.
func SkipLWSE2E() bool { return envBool("SKIP_LWS_E2E") }

// SuiteContextTimeout is the shared ctx deadline for all FITs (default 10h).
func SuiteContextTimeout() time.Duration {
	if v := strings.TrimSpace(os.Getenv("INFERNEX_BRIDGE_SUITE_TIMEOUT")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			return d
		}
	}
	return 10 * time.Hour
}

func SkipNoKubeconfigReason() string {
	if _, err := os.Stat(localKubeconfig()); err != nil {
		return "kubeconfig not available: " + err.Error()
	}
	return ""
}

func localKubeconfig() string {
	if v := strings.TrimSpace(os.Getenv("KUBECONFIG")); v != "" {
		return v
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".kube", "config")
}

func ControllerLogTailLines() int {
	if v := strings.TrimSpace(os.Getenv("INFERNEX_BRIDGE_LOG_TAIL")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 80
}
