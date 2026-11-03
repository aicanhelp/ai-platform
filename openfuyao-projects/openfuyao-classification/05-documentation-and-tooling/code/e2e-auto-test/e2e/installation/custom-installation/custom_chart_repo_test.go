package custom_installation

import (
	"fmt"
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

var _ = Describe("自定义 HTTP Chart 仓库安装", Label("custom-installation", "post-init", "online"), func() {
	var (
		sshExecutor     *executor.SSHExecutor
		clusterManager  *utils.ClusterManager
		resourceManager *utils.ResourceManager
		clusterName     string
		clusterConfig   *config.BKEClusterConfig
		configPath      string
		nodeConfigPath  string
	)

	BeforeEach(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty())
		Expect(guideConfig.Password).NotTo(BeEmpty())
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideConfig.Host, guideConfig.Port, guideConfig.Username, guideConfig.Password)
		Expect(err).NotTo(HaveOccurred())
		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred())
		dynamicClient, err := dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred())
		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
		clientSet, err := kubernetes.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred())
		resourceManager = utils.NewResourceManager(sshExecutor, dynamicClient, clientSet)
		clusterName = fmt.Sprintf("test-http-chart-%d", time.Now().Unix())
		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要一个节点")
		masterNode := nodes[0]
		masterNode.Role = []string{"master/node", "etcd"}
		clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})
	})

	AfterEach(func() {
		if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
			if err := clusterManager.DeleteClusterWithKubeconfig(clusterName, ""); err != nil {
				GinkgoWriter.Printf("Failed to delete cluster: %v\n", err)
			}
			Eventually(func() bool {
				return !clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
			}, uninstallTimeout, 10*time.Second).Should(BeTrue())
		}
		if configPath != "" {
			_ = clusterManager.CleanupConfig(configPath)
		}
		if nodeConfigPath != "" {
			_ = clusterManager.CleanupConfig(nodeConfigPath)
		}
	})

	// 用例名称：应该支持从http格式仓库安装chart
	// 用例步骤：1) 参考 bkeadm pkg/initialize/bkeconsole（在线模式 FUYAO_REPO=oci://cr.openfuyao.cn/charts），从 openFuyao OCI 仓下载 logging-package chart 并上传到引导节点 http chart 仓库（存储目录 /bke/mount/charts）；2) 配置 ChartRepo 为 http 源（必要时先安装 plugin-management-service）；3) 通过引导节点生成并上传集群配置；4) 执行 `bke cluster create` 创建集群并等待 Healthy；5) 校验 logging 相关 chart 组件 Pod 是否就绪。
	// 预期结果：http 仓库安装流程成功，logging/loki/logging-operator 等关键组件均能正常部署并运行。
	// Author: shizilong1
	It("应该支持从http格式仓库安装chart", Label("required_nodes:1", "install-chart-from-http-repo"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("下载logging chart并放到引导节点/bke/mount/charts下")
		// 引导节点 /bke/mount/charts 仅有核心 chart（参考 bkeadm/assets/offline-artifacts.yaml），
		// logging-package 属于扩展组件，在线模式从 openFuyao OCI chart 仓拉取
		// （参考 bkeadm pkg/initialize/bkeconsole/consts.sh: FUYAO_REPO="oci://cr.openfuyao.cn/charts"），
		// 需先下载并通过 chart 仓库 API 上传到 /bke/mount/charts 后方可安装
		publishChartCmds := []string{
			"chmod 777 /bke/mount/charts",
			"rm -f /tmp/logging-package-0.0.0-latest.tgz",
			"helm pull oci://cr.openfuyao.cn/charts/logging-package --version 0.0.0-latest -d /tmp",
			"if curl -fsS http://127.0.0.1:38080/api/charts/logging-package/0.0.0-latest >/dev/null 2>&1; then echo 'logging-package already exists, skip upload'; else curl --data-binary @/tmp/logging-package-0.0.0-latest.tgz http://127.0.0.1:38080/api/charts; fi",
		}
		for _, cmd := range publishChartCmds {
			result, execErr := sshExecutor.Exec(cmd)
			if result != nil {
				GinkgoWriter.Printf("[guide-node] %s\nexit=%d\nstdout: %s\nstderr: %s\n", cmd, result.ExitCode, result.Stdout, result.Stderr)
			}
			Expect(result).NotTo(BeNil(), "命令应该返回结果: %s, err: %v", cmd, execErr)
			Expect(result.ExitCode).To(Equal(0), "命令应该执行成功: %s\nstdout: %s\nstderr: %s", cmd, result.Stdout, result.Stderr)
		}

		// 验证 logging chart 已在引导节点 chart 仓库中生效
		Eventually(func() bool {
			result, err := sshExecutor.Exec("curl -fsS http://127.0.0.1:38080/index.yaml | grep -q logging-package")
			if err != nil || result == nil {
				return false
			}
			return result.ExitCode == 0
		}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "logging chart 应该出现在引导节点 chart 仓库索引中")

		hasPluginManagementService := false
		for _, addon := range clusterConfig.Addons {
			if addon.Name == "plugin-management-service" || addon.Name == "openfuyao-system-controller" {
				hasPluginManagementService = true
			}
		}

		// 不安装plugin-management-service直接安装logging-package会失败，所以安装plugin-management-service
		if !hasPluginManagementService {
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:        "plugin-management-service",
				Version:     "0.0.0-latest",
				ReleaseName: "plugin-management",
				Namespace:   "openfuyao-system",
				Type:        "chart",
			})
		}

		clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
			Name:        "logging-package",
			Version:     "0.0.0-latest",
			ReleaseName: "logging",
			Type:        "chart",
			Namespace:   "logging",
		})

		clusterConfig.ChartRepo = config.GuideNodeChartRepoConfig()

		By("生成集群配置文件")
		var err error
		configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

		By("执行集群创建命令")
		err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

		By("等待集群状态变为 Ready")
		Eventually(func() string {
			phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			if err != nil {
				GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
				return ""
			}
			GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus
		}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

		By("验证日志组件")
		checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

		// 验证 logging-website 组件已经安装
		for _, component := range []string{"proxy-logging"} {
			Eventually(func() bool {
				for _, labelKey := range []string{"app"} {
					nodeName, _ := checker.GetPodNodeName("logging", fmt.Sprintf("%s=%s", labelKey, component))
					if nodeName != "" {
						return true
					}
				}
				return false
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), fmt.Sprintf("应该能找到 %s 组件", component))
		}

		// 验证 loki 组件已经安装
		for _, component := range []string{"loki"} {
			Eventually(func() bool {
				for _, labelKey := range []string{"app", "name"} {
					nodeName, _ := checker.GetPodNodeName("loki", fmt.Sprintf("%s=%s", labelKey, component))
					if nodeName != "" {
						return true
					}
				}
				return false
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), fmt.Sprintf("应该能找到 %s 组件", component))
		}

		// 验证 logging-operator 组件已经安装
		for _, component := range []string{"logging-operator"} {
			Eventually(func() bool {
				for _, labelKey := range []string{"app"} {
					nodeName, _ := checker.GetPodNodeName("openfuyao-system", fmt.Sprintf("%s=%s", labelKey, component))
					if nodeName != "" {
						return true
					}
				}
				return false
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), fmt.Sprintf("应该能找到 %s 组件", component))
		}
	})

	// 用例名称：配置chart仓库不存在的chart
	// 用例步骤：1) 将一个不存在的 chart（如 `elon-musk`）加入 addon 配置；2) 配置 ChartRepo 并通过引导节点生成/上传集群配置；3) 执行 `bke cluster create`；4) 校验该 chart 对应命名空间下无 Pod。
	// 预期结果：集群整体仍可达到 Healthy，但不存在的 chart 不会成功部署，命名空间下无 Pod。
	// Author: shizilong1
	It("配置chart仓库不存在的chart", Label("required_nodes:1", "set-chart-repo-not-exist-chart"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
			Name:        "elon-musk",
			Version:     "0.0.0-elon-musk",
			ReleaseName: "elon-musk",
			Namespace:   "elon-musk-1",
			Type:        "chart",
		})
		clusterConfig.ChartRepo = config.OpenFuyaoChartRepoConfig()

		By("生成集群配置文件")
		var err error
		configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

		By("执行集群创建命令")
		err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

		By("等待集群状态变为 Ready")
		Eventually(func() string {
			phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			if err != nil {
				GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
				return ""
			}
			GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus
		}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

		By("验证组件是否安装")
		checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
		// 验证 elon-musk 空间下是否有Pod
		_, notReady, _ := checker.CheckNamespacePodsReadyWithKubeconfig("elon-musk-1", "")
		Expect(notReady).Should(Equal("no pods found"), "elon-musk-1 命名空间下应该无Pod")
	})

	// 用例名称：配置异常chart仓库应该安装集群失败
	// 用例步骤：1) 配置一个异常的 ChartRepo（domain/IP/port 等不合法或不可达）；2) 通过引导节点生成并上传集群配置；3) 执行 `bke cluster create`；4) 验证集群 BC/组件未能正确创建。
	// 预期结果：由于 chart 仓库异常，相关 BC 不存在或获取失败，集群创建流程无法完成预期组件部署。
	// Author: shizilong1
	It("配置异常chart仓库应该安装集群失败", Label("required_nodes:1", "set-err-chart-repo-install-cluster-failed"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
			Name:        "plugin-management-service",
			Version:     "0.0.0-latest",
			ReleaseName: "plugin-management",
			Namespace:   "openfuyao-system",
			Type:        "chart",
		})
		clusterConfig.ChartRepo = &config.ChartRepoConfig{
			Domain: "my.chart.repo.elon.musk",
			IP:     "192.168.1",
			Port:   "443z",
			Prefix: "",
		}

		By("生成集群配置文件")
		var err error
		configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

		By("执行集群创建命令")
		err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "创建集群命令应该未报错，但是未创建集群")

		By("获取bc")
		_, err = clusterManager.GetBkeCluster(clusterConfig.ClusterName, clusterConfig.Namespace)
		Expect(err).To(HaveOccurred(), "应该不能获取到bc")
	})

	// 用例名称：应该支持安装chart包时配置values.yaml
	// 用例步骤：1) 在 chart addon 配置中提供 `values.yaml`（例如 logging-operator/loki-stack 的镜像仓库、卷挂载、规则等）；2) 通过引导节点生成并上传集群配置；3) 执行 `bke cluster create` 创建集群并等待 Healthy；4) 校验 values.yaml 相关组件是否部署成功并能反映预期配置。
	// 预期结果：使用 values.yaml 的 chart addon 能成功安装，相关组件 Pod 可查询到且运行符合配置期望。
	// Author: shizilong1
	It("应该支持安装chart包时配置values.yaml", Label("required_nodes:1", "set-values-yaml-and-install-chart"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		valuesYaml := `
logging-operator:
  namespace:
    backend: logging
  images:
    core:
      repository: "cr.openfuyao.cn/openfuyao/logging-operator"
      pullPolicy: Always
loki-stack:
  namespace: loki
  promtail:
    defaultVolumes:
      - name: containers
        hostPath:
          path: /var/lib/docker/containers
      - name: pods
        hostPath:
          path: /var/log/pods
      - name: random
        hostPath:
          path: /var/log/random
    defaultVolumeMounts:
      - name: containers
        mountPath: /var/lib/docker/containers
        readOnly: true
      - name: pods
        mountPath: /var/log/pods
        readOnly: true
      - name: random
        mountPath: /var/log/random
        readOnly: true
  loki:
    config:
      limits_config:
        reject_old_samples_max_age: 168h
      table_manager:
        retention_deletes_enabled: true
        retention_period: 240h
    alerting_groups:
      - name: GenericHighErrorRate
        rules:
          - alert: GenericHighErrorRate
            expr: |
              sum by (job, instance) (rate({job=~".+"} |= "error" [5m])) > 0
            for: 5m
            labels:
              severity: critical
              loki: logging/k8s.io
            annotations:
              summary: "High error rate detected in {{ $labels.job }} instance {{ $labels.instance }}"
              description: "Job {{ $labels.job }} on instance {{ $labels.instance }} has a high rate of error logs."
      - name: GenericExceptionDetected
        rules:
          - alert: GenericExceptionDetected
            expr: |
              sum by (job, instance) (rate({job=~".+"} |= "Exception" [5m])) > 0
            for: 5m
            labels:
              severity: warning
              loki: logging/k8s.io
            annotations:
              summary: "Exception detected in {{ $labels.job }} instance {{ $labels.instance }}"
              description: "Job {{ $labels.job }} on instance {{ $labels.instance }} has logs containing 'Exception'."
      - name: GenericLogVolumeSpike
        rules:
          - alert: GenericLogVolumeSpike
            expr: |
              sum by (job, instance) (rate({job=~".+"}[5m])) > 1
            for: 5m
            labels:
              severity: warning
              loki: logging/k8s.io
            annotations:
              summary: "Log volume spike detected in {{ $labels.job }} instance {{ $labels.instance }}"
              description: "Job {{ $labels.job }} on instance {{ $labels.instance }} has a spike in log volume."
      - name: GenericErrorRateThresholdExceeded
        rules:
          - alert: GenericErrorRateThresholdExceeded
            expr: |
              (sum by (job, instance) (rate({job=~".+"} |= "error" [5m])) / sum by (job, instance) (rate({job=~".+"}[5m]))) > 0.05
            for: 5m
            labels:
              severity: critical
              loki: logging/k8s.io
            annotations:
              summary: "High error rate threshold exceeded in {{ $labels.job }} instance {{ $labels.instance }}"
              description: "Job {{ $labels.job }} on instance {{ $labels.instance }} has an error rate exceeding the threshold."
      - name: GenericServiceUnavailable
        rules:
          - alert: GenericServiceUnavailable
            expr: |
              sum by (job, instance) (rate({job=~".+"} |= "service unavailable" [5m])) > 0
            for: 5m
            labels:
              severity: critical
              loki: logging/k8s.io
            annotations:
              summary: "Service unavailable detected in {{ $labels.job }} instance {{ $labels.instance }}"
              description: "Job {{ $labels.job }} on instance {{ $labels.instance }} has logs indicating 'service unavailable'."
      - name: GenericApplicationStartupFailure
        rules:
          - alert: GenericApplicationStartupFailure
            expr: |
              sum by (job, instance) (rate({job=~".+"} |= "startup failure" [5m])) > 0
            for: 5m
            labels:
              severity: critical
              loki: logging/k8s.io
            annotations:
              summary: "Application startup failure detected in {{ $labels.job }} instance {{ $labels.instance }}"
              description: "Job {{ $labels.job }} on instance {{ $labels.instance }} has logs indicating 'startup failure'."
proxy-logging:
  namespace: logging
  enableOAuth: true
  images:
    core:
      repository: "cr.openfuyao.cn/openfuyao/logging-website"
      pullPolicy: Always
    oauth:
      repository: "cr.openfuyao.cn/openfuyao/oauth-proxy"
      pullPolicy: Always
`

		cmNamespace := fmt.Sprintf("elon-%s", clusterConfig.Namespace)
		cmName := fmt.Sprintf("elon-%s", clusterName)
		err := resourceManager.CreateConfigMap(cmName, cmNamespace, "values.yaml", valuesYaml)
		Expect(err).NotTo(HaveOccurred(), "创建values.yaml configMap失败")

		hasPluginManagementService := false
		for _, addon := range clusterConfig.Addons {
			if addon.Name == "plugin-management-service" || addon.Name == "openfuyao-system-controller" {
				hasPluginManagementService = true
			}
		}

		// 不安装plugin-management-service直接安装logging-package会失败，所以安装plugin-management-service
		if !hasPluginManagementService {
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:        "plugin-management-service",
				Version:     "0.0.0-latest",
				ReleaseName: "plugin-management",
				Namespace:   "openfuyao-system",
				Type:        "chart",
				Block:       true,
			})
		}

		clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
			Name:        "logging-package",
			Version:     "1.0.3",
			Namespace:   "logging",
			ReleaseName: "logging",
			Type:        "chart",
			ValuesConfigMapRef: &config.ValuesConfigMapRef{
				Name:      cmName,
				Namespace: cmNamespace,
			},
		})

		clusterConfig.ChartRepo = config.OpenFuyaoChartRepoConfig()

		By("生成集群配置文件")
		configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

		By("执行集群创建命令")
		err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

		By("等待集群状态变为 Ready")
		Eventually(func() string {
			phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			if err != nil {
				GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
				return ""
			}
			GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus
		}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

		By("验证日志组件")
		checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

		// 检查 loki 命名空间下的 Pod 状态
		Eventually(func() bool {
			allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("loki", "")
			if err != nil {
				return false
			}
			if !allReady {
				GinkgoWriter.Printf("等待 loki Pod 就绪: %s\n", notReady)
				return false
			}
			return true
		}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "loki 命名空间下的 Pod 应该全部就绪")

		// 进一步验证具体的 logging-package 组件
		for _, component := range []string{"proxy-logging", "logging-operator"} {
			Eventually(func() bool {
				for _, labelKey := range []string{"app"} {
					nodeName, _ := checker.GetPodNodeName("logging", fmt.Sprintf("%s=%s", labelKey, component))
					if nodeName != "" {
						return true
					}
				}
				return false
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), fmt.Sprintf("应该能找到 %s 组件", component))
		}
	})
})
