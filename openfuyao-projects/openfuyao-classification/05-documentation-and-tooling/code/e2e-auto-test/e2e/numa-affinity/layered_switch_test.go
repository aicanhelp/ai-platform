package numa_affinity_functional

import (
	"fmt"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func getPodNamesOutput() (string, error) {
	cmd := fmt.Sprintf("kubectl get pods -n %s -o jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}{end}'", cfg.HelmNamespace)
	out, err := sshClient.Exec(cmd)
	if err != nil {
		return "", err
	}
	return out.Stdout, nil
}

func getResourceExporterWorkloadsOutput() (string, error) {
	cmd := "kubectl get daemonset,deployment,pod -A -o jsonpath='{range .items[*]}{.kind}{\":\"}{.metadata.namespace}{\"/\"}{.metadata.name}{\"\\n\"}{end}'"
	out, err := sshClient.Exec(cmd)
	if err != nil {
		return "", err
	}
	return out.Stdout, nil
}

func hasAnySubstring(text string, candidates ...string) bool {
	for _, candidate := range candidates {
		if strings.Contains(text, candidate) {
			return true
		}
	}
	return false
}

var _ = Describe("numa affinity 分层开关功能测试", Ordered, Label("numa-affinity-layered-switch", "with-workload-cluster"), func() {

	BeforeAll(func() {
		By("卸载 BeforeSuite 中默认安装的组件以准备分层安装测试")
		if cfg != nil && sshClient != nil {
			_ = helm.HelmUninstall(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
			time.Sleep(15 * time.Second) // 等待资源清理
		}
	})

	AfterAll(func() {
		By("恢复默认安装以供其他测试使用")
		if cfg != nil && sshClient != nil {
			_ = helm.HelmUninstall(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
			time.Sleep(15 * time.Second)
			err := helm.HelmInstall(cfg.HelmReleaseName, cfg.HelmURL, cfg.HelmVersion, cfg.HelmNamespace, sshClient)
			Expect(err).NotTo(HaveOccurred(), "应成功恢复默认配置的 release")
			time.Sleep(30 * time.Second) // 等待组件 Ready
		}
	})

	AfterEach(func() {
		By("清理当前测试用例中安装的用例")
		if cfg != nil && sshClient != nil {
			err := helm.HelmUninstall(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
			Expect(err).NotTo(HaveOccurred(), "当前测试用例结束，helm uninstall 应执行成功")
			time.Sleep(15 * time.Second)
		}
	})

	It("【VCS-INT-001】分层开关-总开关关闭", Label("VCS-INT-001", "P0"), func() {
		By("开始安装 components.enabled=false 的 numa affinity")
		args := "--set volcano-config-package.volcano.components.enabled=false"
		err := helm.HelmInstall(cfg.HelmReleaseName, cfg.HelmURL, cfg.HelmVersion, cfg.HelmNamespace, sshClient, args)
		Expect(err).NotTo(HaveOccurred(), "helm install 应执行成功")

		time.Sleep(45 * time.Second)

		By("检查 release 是否成功安装")
		exists, err := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeTrue(), "numa-affinity-package release 应存在")

		By("检查 volcano 与 resource-exporter 相关 pod 是否未创建")
		podNames, err := getPodNamesOutput()
		Expect(err).NotTo(HaveOccurred())
		Expect(hasAnySubstring(podNames, "volcano-scheduler-", "volcano-controllers-", "volcano-admission-", "resource-exporter-daemonset-")).To(BeFalse(), "volcano/resource-exporter pod 不应该被创建")
	})

	It("【VCS-INT-002】分层开关-volcano 主组件关闭", Label("VCS-INT-002", "P0"), func() {
		By("开始安装 volcano main disabled and resourceExporter enabled")
		args := "--set volcano-config-package.volcano.components.enabled=true --set volcano-config-package.volcano.components.volcano.enabled=false --set volcano-config-package.volcano.components.resourceExporter.enabled=true"
		err := helm.HelmInstall(cfg.HelmReleaseName, cfg.HelmURL, cfg.HelmVersion, cfg.HelmNamespace, sshClient, args)
		Expect(err).NotTo(HaveOccurred(), "helm install 应执行成功")

		time.Sleep(45 * time.Second)

		exists, err := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeTrue(), "numa-affinity-package release 应存在")

		By("volcano 主组件是否没有安装")
		podNames, err := getPodNamesOutput()
		Expect(err).NotTo(HaveOccurred())
		Expect(hasAnySubstring(podNames, "volcano-scheduler-", "volcano-controllers-", "volcano-admission-")).To(BeFalse(), "volcano 主组件 pod 不应该被创建")

		By("resource-exporter 是否正常安装")
		Eventually(func(g Gomega) {
			workloads, workloadErr := getResourceExporterWorkloadsOutput()
			g.Expect(workloadErr).NotTo(HaveOccurred())
			g.Expect(hasAnySubstring(workloads, "resource-exporter")).To(BeTrue(), fmt.Sprintf("当前 daemonset/deployment/pod 列表:\n%s", workloads))
		}, 5*time.Minute, 10*time.Second).Should(Succeed(), "resource-exporter 应该被创建")
	})

	It("【VCS-INT-003】分层开关-resourceExporter 关闭", Label("VCS-INT-003", "P1"), func() {
		By("开始安装 volcano main enabled and resourceExporter disabled")
		args := "--set volcano-config-package.volcano.components.enabled=true --set volcano-config-package.volcano.components.volcano.enabled=true --set volcano-config-package.volcano.components.resourceExporter.enabled=false"
		err := helm.HelmInstall(cfg.HelmReleaseName, cfg.HelmURL, cfg.HelmVersion, cfg.HelmNamespace, sshClient, args)
		Expect(err).NotTo(HaveOccurred(), "helm install 应该成功")

		time.Sleep(45 * time.Second)

		exists, err := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
		Expect(err).NotTo(HaveOccurred())
		Expect(exists).To(BeTrue(), "numa-affinity-package release 应存在")

		By("volcano 主组件是否正常安装")
		Eventually(func(g Gomega) {
			workloads, workloadErr := getResourceExporterWorkloadsOutput()
			g.Expect(workloadErr).NotTo(HaveOccurred())
			g.Expect(
				hasAnySubstring(workloads, "volcano-scheduler", "volcano-controllers", "volcano-admission"),
			).To(BeTrue(), fmt.Sprintf("当前 daemonset/deployment/pod 列表:\n%s", workloads))
		}, 5*time.Minute, 10*time.Second).Should(Succeed(), "volcano 主组件 pod 应该被创建")

		By("resource-exporter 是否没有安装")
		podNames, err := getPodNamesOutput()
		Expect(err).NotTo(HaveOccurred())
		Expect(hasAnySubstring(podNames, "resource-exporter-daemonset-")).To(BeFalse(), "resource-exporter 不应该被创建")
	})
})
