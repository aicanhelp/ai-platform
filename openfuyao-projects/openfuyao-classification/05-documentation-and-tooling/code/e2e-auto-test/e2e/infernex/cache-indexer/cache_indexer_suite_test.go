package cacheindexer

import (
	"context"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	frameworkk8s "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var (
	k8sClient  *frameworkk8s.K8SClient
	helmConfig map[string]*utils.ReleaseConfig
)

func TestCacheIndexer(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "CacheIndexer Suite")
}

var _ = BeforeSuite(func() {
	_ = utils.LoadEnvFromFile("..", ".env")

	By("创建 K8s 客户端（使用本机 kubeconfig）")
	var err error
	k8sClient, err = frameworkk8s.NewK8SClientFromLocalKubeconfig()
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "K8s 客户端创建失败")

	By("确保 npu-operator 已安装")
	err = utils.EnsureNPUOperatorInstalled()
	Expect(err).NotTo(HaveOccurred(), "npu-operator 应当已存在或安装成功")

	By("加载 cache-indexer Helm 配置")
	helmConfig, err = utils.LoadCacheIndexerHelmConfig()
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "cache-indexer Helm 配置加载失败")

	ns := cacheIndexerTestNamespace()
	Expect(ns).NotTo(BeEmpty(), "Helm 配置中应包含可用命名空间")
	err = k8sClient.CreateNamespaceIfNotExists(context.Background(), ns)
	Expect(err).NotTo(HaveOccurred(), "命名空间创建失败")

	By("确保 LeaderWorkerSet 控制器已安装")
	lwsCtx, lwsCancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer lwsCancel()
	err = utils.EnsureLeaderWorkerSetController(k8sClient.Clientset, lwsCtx)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "LeaderWorkerSet 控制器安装失败")
})

var _ = AfterSuite(func() {
	if k8sClient == nil {
		return
	}
	By("卸载本 suite 安装的 LeaderWorkerSet 控制器")
	lwsCtx, lwsCancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer lwsCancel()
	err := utils.CleanupLeaderWorkerSetController(k8sClient.Clientset, lwsCtx)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "LeaderWorkerSet 控制器卸载失败")
})

func cacheIndexerTestNamespace() string {
	if cfg := helmConfig["infernex"]; cfg != nil && cfg.Namespace != "" {
		return cfg.Namespace
	}
	if cfg := helmConfig["cache-indexer"]; cfg != nil && cfg.Namespace != "" {
		return cfg.Namespace
	}
	return ""
}

func mustHelmRelease(name string) *utils.ReleaseConfig {
	cfg := helmConfig[name]
	Expect(cfg).NotTo(BeNil(), "Helm 配置中应当包含 %s", name)
	return cfg
}

func cleanupLeftoverInfernexAndCacheIndexer(ctx context.Context) {
	By("安装前清理遗留 infernex 与 standalone cache-indexer release")
	if standaloneCfg := helmConfig["cache-indexer"]; standaloneCfg != nil {
		err := utils.HelmUninstallIfExists(standaloneCfg.ReleaseName, standaloneCfg.Namespace)
		Expect(err).NotTo(HaveOccurred(), "安装前清理 standalone cache-indexer release 失败")
	}
	infernexCfg := mustHelmRelease("infernex")
	err := utils.CleanupCacheIndexerInfernexConflicts(ctx, k8sClient, infernexCfg.ReleaseName, infernexCfg.Namespace)
	Expect(err).NotTo(HaveOccurred(), "安装前 infernex 冲突清理失败")
}

func uninstallReleaseIfPresent(cfg *utils.ReleaseConfig, successMsg string) {
	if cfg == nil {
		return
	}
	needUninstall, err := utils.CheckHelmRelease(cfg)
	if err != nil {
		GinkgoWriter.Println(err)
		return
	}
	if !needUninstall {
		return
	}
	err = utils.HelmUninstall(cfg)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "Helm uninstall %s 执行失败", cfg.ReleaseName)
	time.Sleep(15 * time.Second)
	GinkgoWriter.Println(successMsg)
}
