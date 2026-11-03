package nginx_proxy

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

const (
	installTimeout      = 20 * time.Minute
	mgmtInstallTimeout  = 25 * time.Minute
	uninstallTimeout    = 15 * time.Minute
	pollInterval        = 30 * time.Second
	failedStatesMsgTmpl = "集群进入失败态: state=%s, clusterStatus=%s，不再等待 Healthy"
)

// InstallationItTimeout is the default Spec timeout for tests under e2e/installation/nginx-proxy.
const InstallationItTimeout = 45 * time.Minute

func isClusterFailedState(state, clusterStatus string) bool {
	return state == "DeployFailed" || clusterStatus == "InitializationFailed"
}

func failOnClusterFailure(state, clusterStatus string) {
	if isClusterFailedState(state, clusterStatus) {
		Fail(fmt.Sprintf(failedStatesMsgTmpl, state, clusterStatus))
	}
}

// eventuallyShouldBeTrue 等价于 Eventually(condition, timeout, interval).Should(BeTrue(), msg)，
// 但在超时或失败时返回 error 而非 panic，便于调用方在失败后仍能返回已创建的资源信息以供 AfterAll 清理。
// 注意：condition 闭包内的任意 panic（含 Ginkgo Fail 与真实 bug）均会被转为 error；
// 详细失败信息与堆栈由 Ginkgo Fail 在 panic 前记录到测试输出。
func eventuallyShouldBeTrue(condition func() bool, timeout, interval time.Duration, msg string) error {
	err := fmt.Errorf("%s", msg)
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("%s: %v", msg, r)
			}
		}()
		Eventually(condition, timeout, interval).Should(BeTrue(), msg)
		err = nil
	}()
	return err
}
