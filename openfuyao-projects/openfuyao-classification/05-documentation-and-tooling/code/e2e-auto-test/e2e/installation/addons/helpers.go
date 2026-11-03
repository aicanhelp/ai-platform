package addons

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
)

const (
	InstallationItTimeout = 45 * time.Minute
	installTimeout        = 20 * time.Minute
	uninstallTimeout      = 15 * time.Minute
	pollInterval          = 30 * time.Second
)

func failOnClusterFailure(state, clusterStatus string) {
	if state == "DeployFailed" || clusterStatus == "InitializationFailed" {
		Fail(fmt.Sprintf("集群进入失败态: state=%s, clusterStatus=%s，不再等待 Healthy", state, clusterStatus))
	}
}
