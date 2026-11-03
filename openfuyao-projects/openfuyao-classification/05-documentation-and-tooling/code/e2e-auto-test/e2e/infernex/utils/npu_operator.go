package utils

import (
	"fmt"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	. "github.com/onsi/ginkgo/v2"
)

const (
	npuOperatorReleaseName = "npu-operator"
	npuOperatorNamespace   = "default"
)

// EnsureNPUOperatorInstalled 检查 default 命名空间是否已有 release npu-operator。
// 已存在则跳过；不存在则安装到 default，release 名同样为 npu-operator。
// chart 地址和版本仍取自 framework/helm/helm_config.yaml。
// 安装后的 release 留给环境继续使用，调用方不要在 AfterSuite 里卸载。
func EnsureNPUOperatorInstalled() error {
	installed, err := CheckHelmRelease(&ReleaseConfig{
		ReleaseName: npuOperatorReleaseName,
		Namespace:   npuOperatorNamespace,
	})
	if err != nil {
		return err
	}
	if installed {
		GinkgoWriter.Printf("npu-operator already installed (%s/%s), skip\n", npuOperatorNamespace, npuOperatorReleaseName)
		return nil
	}

	cfgMap, err := helm.LoadHelmConfig()
	if err != nil {
		return fmt.Errorf("load helm config: %w", err)
	}
	cfg := cfgMap["npu-operator"]
	if cfg == nil {
		return fmt.Errorf("helm config 应包含 npu-operator")
	}

	GinkgoWriter.Printf("npu-operator not found in %s, start install...\n", npuOperatorNamespace)
	local := executor.NewLocalExecutor(15 * time.Minute)
	err = helm.HelmInstallLocal(npuOperatorReleaseName, cfg.HelmURrl, cfg.Version, npuOperatorNamespace, local, "--create-namespace")
	if err != nil {
		return err
	}
	time.Sleep(90 * time.Second)
	GinkgoWriter.Println("npu-operator install success")
	return nil
}
