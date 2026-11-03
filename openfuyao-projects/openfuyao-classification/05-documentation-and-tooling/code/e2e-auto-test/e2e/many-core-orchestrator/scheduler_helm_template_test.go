package mco

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/many-core-orchestrator/utils"
)

var _ = Describe("众核调度插件 Helm Template 渲染测试", Label("many-core-orchestrator", "scheduler", "helm-template", "virtual-machine"), func() {
	var chartPath string

	BeforeEach(func() {
		chartPath = GetChartPath()
	})

	It("众核-SCHED-IT-001 默认全量渲染", func() {
		args := append([]string{}, SchedulerPhaseHelmSets...)
		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template volcano.enabled=true + mcoPlugin.enabled=true 应正常返回")

		Expect(output).To(ContainSubstring("mco-volcano-scheduler"), "应包含 mco-volcano-scheduler Deployment")
		Expect(output).To(ContainSubstring("--plugins-dir=/plugins"), "mco-volcano-scheduler Deployment 应包含 --plugins-dir=/plugins 参数")
		Expect(output).To(ContainSubstring("volcano-scheduler-configmap"), "应包含 volcano-scheduler-configmap ConfigMap")
		Expect(output).To(ContainSubstring("mco-scheduler"), "应包含 mco-scheduler ServiceAccount/ClusterRole/ClusterRoleBinding")

		Expect(output).To(ContainSubstring("kind: Deployment"), "应包含 Deployment 资源")
		Expect(output).To(ContainSubstring("kind: ConfigMap"), "应包含 ConfigMap 资源")
		Expect(output).To(ContainSubstring("kind: ServiceAccount"), "应包含 ServiceAccount 资源")
		Expect(output).To(ContainSubstring("kind: ClusterRole"), "应包含 ClusterRole 资源")
		Expect(output).To(ContainSubstring("kind: ClusterRoleBinding"), "应包含 ClusterRoleBinding 资源")
	})

	It("众核-SCHED-IT-002 关闭 mcoPlugin", func() {
		args := append([]string{}, SchedulerPhaseHelmSets...)
		args = append(args, "--set", "mcoPlugin.enabled=false")

		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template mcoPlugin.enabled=false 应正常返回")

		Expect(output).NotTo(ContainSubstring("mco-volcano-scheduler"), "不应包含 mco-volcano-scheduler Deployment")
		Expect(output).NotTo(ContainSubstring("volcano-scheduler-configmap"), "不应包含 volcano-scheduler-configmap")
	})

	It("众核-SCHED-IT-003 一站式安装模式", func() {
		args := append([]string{}, SchedulerPhaseHelmSets...)
		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template volcano.enabled=true 一站式模式应正常返回")

		Expect(output).To(ContainSubstring("mco-volcano-scheduler"), "应包含 mco-volcano-scheduler Deployment")
		Expect(output).To(ContainSubstring("volcano-scheduler-configmap"), "应包含 volcano-scheduler-configmap")

		volcanoStandardResources := []string{
			"volcano-admission",
			"vc-controller-manager",
			"vc-webhook-manager",
		}
		for _, resource := range volcanoStandardResources {
			Expect(output).To(ContainSubstring(resource), "一站式模式应包含 Volcano 标准资源: %s", resource)
		}

		Expect(output).To(ContainSubstring("kind: CustomResourceDefinition"), "一站式模式应包含 Volcano CRD")

		Expect(output).NotTo(ContainSubstring("name: volcano-scheduler\n"), "一站式模式不应包含默认 volcano-scheduler Deployment（应为 mco-volcano-scheduler 替代）")
	})

	It("众核-SCHED-IT-004 接入已有 Volcano", func() {
		args := []string{
			"--set", "volcano.enabled=false",
			"--set", "mcoPlugin.enabled=true",
			"--set", "kata-deploy.enabled=false",
			"--set", "collector.enabled=false",
			"--set", "analyzer.enabled=false",
		}
		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template volcano.enabled=false 接入已有模式应正常返回")

		Expect(output).To(ContainSubstring("mco-volcano-scheduler"), "应包含 mco-volcano-scheduler Deployment")
		Expect(output).To(ContainSubstring("volcano-scheduler-configmap"), "应包含 volcano-scheduler-configmap（MCO 插件配置，非 Volcano 子资源）")

		volcanoSubResources := []string{
			"volcano-admission",
			"volcano-controllers",
		}
		for _, resource := range volcanoSubResources {
			Expect(output).NotTo(ContainSubstring(resource), "接入已有模式不应包含 Volcano 子资源: %s", resource)
		}

		Expect(output).NotTo(ContainSubstring("kind: CustomResourceDefinition"), "接入已有模式不应包含 Volcano CRD")
	})

	It("众核-SCHED-IT-005 ConfigMap 自动生成", func() {
		args := append([]string{}, SchedulerPhaseHelmSets...)
		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template 应正常返回")

		Expect(output).To(ContainSubstring("volcano-scheduler-configmap"), "应包含 volcano-scheduler-configmap")
		Expect(output).To(ContainSubstring("volcano-scheduler.conf"), "ConfigMap 应包含 volcano-scheduler.conf 配置键")

		Expect(output).To(ContainSubstring("mco-plugin"), "tiers 中应注册 mco-plugin")
		Expect(output).To(ContainSubstring("crName"), "mco-plugin 配置应包含 crName")
		Expect(output).To(ContainSubstring("crStatusTTLSeconds"), "mco-plugin 配置应包含 crStatusTTLSeconds")
		Expect(output).To(ContainSubstring("hardConstraints"), "mco-plugin 配置应包含 hardConstraints")
		Expect(output).To(ContainSubstring("fallbackWhenCROutdated"), "mco-plugin 配置应包含 fallbackWhenCROutdated")

		tier0Plugins := []string{"priority", "gang", "conformance"}
		for _, plugin := range tier0Plugins {
			Expect(output).To(ContainSubstring(plugin), "tiers[0] 中应包含标准插件: %s", plugin)
		}

		tier1Plugins := []string{"overcommit", "drf", "predicates"}
		for _, plugin := range tier1Plugins {
			Expect(output).To(ContainSubstring(plugin), "tiers[1] 中应包含标准调度插件: %s", plugin)
		}
	})
})
