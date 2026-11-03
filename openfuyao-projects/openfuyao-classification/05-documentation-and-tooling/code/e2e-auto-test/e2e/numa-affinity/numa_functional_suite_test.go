package numa_affinity_functional

import (
	"context"
	"log"
	"os"
	"testing"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/numa-affinity/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var (
	sshClient *executor.SSHExecutor
	cancel    context.CancelFunc
	cfg       *utils.FunctionalConfig
)

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		log.Fatalf("Failed to load .env file: %v", err)
	}
}

func setFunctionalEnvDefaultsFromNode2() {
	if os.Getenv("FUNCTIONAL_TEST_IP") == "" {
		_ = os.Setenv("FUNCTIONAL_TEST_IP", os.Getenv("TEST_NODE2_IP"))
	}
	if os.Getenv("FUNCTIONAL_TEST_PORT") == "" {
		if port := os.Getenv("TEST_NODE2_PORT"); port != "" {
			_ = os.Setenv("FUNCTIONAL_TEST_PORT", port)
		}
	}
	if os.Getenv("FUNCTIONAL_TEST_USER") == "" {
		if user := os.Getenv("TEST_NODE2_USER"); user != "" {
			_ = os.Setenv("FUNCTIONAL_TEST_USER", user)
		}
	}
	if os.Getenv("FUNCTIONAL_TEST_PASSWORD") == "" {
		_ = os.Setenv("FUNCTIONAL_TEST_PASSWORD", os.Getenv("TEST_NODE2_PASSWORD"))
	}
}

var _ = BeforeSuite(func() {
	setFunctionalEnvDefaultsFromNode2()

	var err error
	cfg, err = utils.LoadFunctionalConfig()
	Expect(err).NotTo(HaveOccurred(), "functional test env config should be valid")

	if cfg.HelmReleaseName == "" || cfg.HelmNamespace == "" || cfg.HelmURL == "" || cfg.HelmVersion == "" {
		helmConfig, loadErr := helm.LoadHelmConfig()
		Expect(loadErr).NotTo(HaveOccurred(), "helm config should load when helm env vars are empty")

		numaConfig, exists := helmConfig["numa-affinity-package"]
		Expect(exists).To(BeTrue(), "numa-affinity-package should exist in helm config")

		if cfg.HelmReleaseName == "" {
			cfg.HelmReleaseName = numaConfig.ReleaseName
		}
		if cfg.HelmNamespace == "" {
			cfg.HelmNamespace = numaConfig.Namespace
		}
		if cfg.HelmURL == "" {
			cfg.HelmURL = numaConfig.HelmURrl
		}
		if cfg.HelmVersion == "" {
			cfg.HelmVersion = numaConfig.Version
		}
	}

	// 功能测试统一使用 latest 版本验证，避免依赖固定历史版本。
	cfg.HelmVersion = "0.0.0-latest"

	_, cancel = context.WithTimeout(context.Background(), 30*time.Minute)

	sshClient, err = executor.NewSSHExecutor(cfg.SSHHost, cfg.SSHPort, cfg.SSHUser, cfg.SSHPassword)
	Expect(err).NotTo(HaveOccurred(), "ssh client should be created")

	By("准备安装 numa-affinity 组件")
	exists, checkErr := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
	Expect(checkErr).NotTo(HaveOccurred(), "helm list should be available before install")
	if exists {
		uninstallErr := helm.HelmUninstall(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
		Expect(uninstallErr).NotTo(HaveOccurred(), "old release should be uninstalled before fresh install")
	}

	installErr := helm.HelmInstall(cfg.HelmReleaseName, cfg.HelmURL, cfg.HelmVersion, cfg.HelmNamespace, sshClient)
	Expect(installErr).NotTo(HaveOccurred(), "numa-affinity should be installed before running functional tests")

	By("等待 volcano 核心组件就绪")
	Eventually(func() error {
		return utils.WaitForVolcanoComponentsReady(sshClient, cfg.VolcanoNamespace, 10*time.Second)
	}, 5*time.Minute, 5*time.Second).Should(Succeed(), "volcano 核心组件应在 5 分钟内就绪")

	GinkgoWriter.Printf("NUMA functional target: %s:%d, release=%s/%s, autoRestore=%v\n", cfg.SSHHost, cfg.SSHPort, cfg.HelmNamespace, cfg.HelmReleaseName, cfg.AutoRestore)
})

var _ = AfterSuite(func() {
	if sshClient != nil && cfg != nil {
		_ = helm.HelmUninstall(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
	}

	if cancel != nil {
		cancel()
	}
	if sshClient != nil {
		if err := sshClient.Close(); err != nil {
			GinkgoWriter.Printf("failed to close ssh client: %v\n", err)
		}
	}
})

func TestNumaAffinityFunctional(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "NumaAffinity Functional Suite")
}
