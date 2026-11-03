package custom_installation

import (
	"fmt"
	"time"

	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	. "github.com/onsi/ginkgo/v2"
)

const (
	InstallationItTimeout = 45 * time.Minute
	installTimeout        = 20 * time.Minute
	mgmtInstallTimeout    = 25 * time.Minute
	uninstallTimeout      = 15 * time.Minute
	pollInterval          = 30 * time.Second
)

func failOnClusterFailure(state, clusterStatus string) {
	if state == "DeployFailed" || clusterStatus == "InitializationFailed" {
		Fail(fmt.Sprintf("集群进入失败态: state=%s, clusterStatus=%s，不再等待 Healthy", state, clusterStatus))
	}
}

func loadMgmtNodesFromEnv(count int) []config.NodeInfo {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < count {
		return nil
	}
	return append([]config.NodeInfo(nil), nodes[:count]...)
}
