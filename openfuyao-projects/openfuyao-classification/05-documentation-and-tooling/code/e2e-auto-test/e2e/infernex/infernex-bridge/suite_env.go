package infernexbridge

import (
	"context"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
	. "github.com/onsi/ginkgo/v2"
)

// Suite-scoped clients shared by tests and workload cleanup (non-_test.go so cleanup compiles with go test).
var (
	suiteCtx    context.Context
	suiteCancel context.CancelFunc
	k8sClient   *k8s.K8SClient
)

func requireCluster(fitID string) {
	bridgeutils.LogStep(fitID, "require cluster (kubeconfig + k8s client)")
	if bridgeutils.SkipNoKubeconfigReason() != "" {
		Skip("no kubeconfig: " + bridgeutils.SkipNoKubeconfigReason())
	}
	if k8sClient == nil {
		Skip("k8s client not initialized")
	}
	bridgeutils.LogStep(fitID, "cluster ok ns=%s", bridgeutils.E2ENS())
}

// requireBridgeDeployed：FIT_4+ 前置校验 controller Ready（安装由 BeforeSuite 负责，或 SKIP_BRIDGE_INSTALL 时假定已安装）。
func requireBridgeDeployed(fitID string) {
	requireCluster(fitID)
	bridgeutils.LogStep(fitID, "require bridge controller Ready")
	bridgeutils.WaitBridgeControllerReady(suiteCtx, k8sClient.Clientset, fitID, 3*time.Minute)
}

func requireBridgeInstalled(fitID string) {
	requireBridgeDeployed(fitID)
}
