package utils

import (
	"os"
	"path/filepath"
	"strings"
)

var (
	MCOHelmConfig = map[string]string{
		"releaseName": "many-core-orchestrator",
		"namespace":   "mco-system",
	}
)

func GetChartPath() string {
	v := strings.TrimSpace(os.Getenv("MCO_CHART_PATH"))
	if v != "" {
		return v
	}
	return MCOChartOCIURL
}

func GetChartVersion() string {
	v := strings.TrimSpace(os.Getenv("MCO_CHART_VERSION"))
	if v != "" {
		return v
	}
	return MCOChartVersion
}

func IsOCIChart(chartPath string) bool {
	return strings.HasPrefix(chartPath, "oci://")
}

func FindRepoRoot() string {
	v := strings.TrimSpace(os.Getenv("MCO_REPO_ROOT"))
	if v != "" {
		return v
	}
	dir, err := os.Getwd()
	if err != nil {
		return dir
	}
	for i := 0; i < 10; i++ {
		if _, err := os.Stat(filepath.Join(dir, "charts", "many-core-orchestrator", "Chart.yaml")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	wd, _ := os.Getwd()
	return wd
}

const (
	MCODeployNamespace         = "mco-system"
	MCOCollectorDaemonSetName  = "mco-collector"
	MCOCollectorAppLabel       = "app=mco-collector"
	MCOCollectorMetricsPort    = 9877
	MCOAnalyzerServiceName     = "mco-analyzer"
	MCOAnalyzerDeploymentName  = "mco-analyzer"
	MCOAnalyzerAppLabel        = "app=mco-analyzer"
	MCOAnalyzerGRPCPort        = 50051
	MCOAnalyzerMetricsPort     = 9878
	MCONodeInterferenceReport  = "cluster"
	MCODetectionRuleConfig     = "cluster"
	KataDeployDaemonSetName    = "kata-deploy"
	KataReadyLabelKey          = "mco.many-core.io/kata-ready"
	KataRuntimeClassQemu       = "kata-qemu"
	KataRuntimeClassClh        = "kata-clh"
	KataTestPodName            = "kata-test-busybox"
	KataTestNamespace          = "default"
	KataPreflightContainerName = "kata-preflight"
	KataDeployContainerName    = "kata-deploy"

	MCOVolcanoNamespace        = "volcano-system"
	MCOSchedulerDeploymentName = "mco-volcano-scheduler"
	MCOSchedulerConfigMapName  = "volcano-scheduler-configmap"
	MCOSchedulerServiceAccount = "mco-scheduler"
	MCOSchedulerAppLabel       = "app=mco-volcano-scheduler"
	MCOKataReadyLabelKey       = "mco.many-core.io/kata-ready"
	MCOKataRuntimeLabelKey     = "katacontainers.io/kata-runtime"

	MCOChartOCIURL  = "oci://cr.openfuyao.cn/charts/many-core-orchestrator"
	MCOChartVersion = "0.0.0-latest"
)

var KataPhaseHelmSets = []string{
	"--set", "collector.enabled=false",
	"--set", "analyzer.enabled=false",
	"--set", "mcoPlugin.enabled=false",
}

var CollectorPhaseHelmSets = []string{
	"--set", "kata-deploy.enabled=false",
	"--set", "mcoPlugin.enabled=false",
}

var SchedulerPhaseHelmSets = []string{
	"--set", "volcano.enabled=true",
	"--set", "mcoPlugin.enabled=true",
	"--set", "kata-deploy.enabled=false",
	"--set", "collector.enabled=false",
	"--set", "analyzer.enabled=false",
	"--set", "collector.serviceMonitor.enabled=false",
	"--set", "analyzer.serviceMonitor.enabled=false",
}
