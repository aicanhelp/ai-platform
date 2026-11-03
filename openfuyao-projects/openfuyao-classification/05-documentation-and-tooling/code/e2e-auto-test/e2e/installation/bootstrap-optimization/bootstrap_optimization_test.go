package bootstrap_optimization

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

/*
  特性：引导节点安装优化
  用例：
      1）安装参数校验
      2）使用oci格式制备引导节点安装包，并进行引导节点初始化
*/

var _ = Describe("引导节点安装优化", Label("bootstrap-deploy-opt", "pre-init", "online"), func() {
	var (
		sshExecutor *executor.SSHExecutor
		guideConfig *config.GuideNodeConfig
	)

	BeforeEach(func() {
		// 加载引导节点配置
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")

		// 创建 SSH 执行器连接引导节点
		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")
	})

	/*
	 用例名称：1）安装参数校验
	 用例步骤：1）bke init携带参数--installConsole（为true）和--imageFilePath，预期结果1
	         2）bke init携带参数--otherRepo（为空）和--imageFilePath，预期结果1
	         3）bke init携带参数--imageFilePath，模拟路径不对或者非tar.gz/tgz文件，预期结果1
	 预期结果：1）参数校验失败，无法继续执行命令
	*/
	Describe("安装参数校验", Label("init-param-check"), func() {
		// Author: xuyouhong
		It("安装参数校验，使用错误的参数就无法下发引导节点的初始化命令", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("执行 bke init 命令")

			// bke init携带参数--installConsole（为true）和--imageFilePath，参数校验失败
			initCmd := "bke init --installConsole=true --imageFilePath=/tmp/a.tar.gz"
			_, err := sshExecutor.Exec(initCmd)
			GinkgoWriter.Printf("%s 使用引导节点安装优化，无法安装带openFuyao管理面的引导节点\n", initCmd)
			Expect(err).To(HaveOccurred(), "使用引导节点安装优化，无法安装带openFuyao管理面的引导节点")

			// bke init携带参数--otherRepo（为空）和--imageFilePath
			initCmd = "bke init --imageFilePath=/tmp/a.tar.gz"
			_, err = sshExecutor.Exec(initCmd)
			GinkgoWriter.Printf("%s 使用引导节点安装优化，离线场景下不能使用\n", initCmd)
			Expect(err).To(HaveOccurred(), "使用引导节点安装优化，离线场景下不能使用")

			// bke init携带参数--imageFilePath，指定的文件有误，参数校验失败
			initCmd = "bke init --otherRepo=test.image.cn --imageFilePath=/tmp/a.ta"
			_, err = sshExecutor.Exec(initCmd)
			GinkgoWriter.Printf("%s imageFilePath指定的文件格式有误\n", initCmd)
			Expect(err).To(HaveOccurred(), "imageFilePath指定的文件格式有误")

			initCmd = "bke init --otherRepo=test.image.cn --imageFilePath=/tmp/a.tar.gz"
			_, err = sshExecutor.Exec(initCmd)
			GinkgoWriter.Printf("%s imageFilePath指定的文件不存在\n", initCmd)
			Expect(err).To(HaveOccurred(), "imageFilePath指定的文件不存在")
		})
	})
})

var _ = Describe("使用oci格式制备引导节点安装包，并进行引导节点初始化", Label("bootstrap-deploy-opt", "bootstrap-init", "pre-init", "online"), func() {
	var (
		sshExecutor *executor.SSHExecutor
		guideNode   *config.GuideNodeConfig
	)

	BeforeEach(func() {
		// 加载引导节点配置
		guideNode = config.LoadGuideNodeFromEnv()
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideNode.Host, guideNode.Port, guideNode.Username, guideNode.Password)
		Expect(err).NotTo(HaveOccurred(), "连接引导节点失败")

		By("重置引导节点环境 (bke reset)")

		resetCmd := "echo Y | bke reset --all --mount"
		result, err := sshExecutor.Exec(resetCmd)
		if err != nil {
			GinkgoWriter.Printf("Reset 命令执行返回错误 (可能环境已清理): %v\nStderr: %s\n", err, result.Stderr)
		} else {
			GinkgoWriter.Printf("Reset 命令执行成功: %s\n", result.Stdout)
		}

		time.Sleep(10 * time.Second)

		// 构造引导节点初始化安装包
		By("在引导节点生成 sample.yaml（默认 amd64）")
		yamlContent := `
repos:
  - architecture:
      - amd64
    needDownload: true
    subImages:
      - sourceRepo: hub.oepkgs.net/openfuyao
        targetRepo: kubernetes
        imageTrack: ""
        images:
          - name: registry
            tag:
              - 2.8.1
          - name: nginx
            tag:
              - 1.23.0-alpine
          - name: helm/chartmuseum
            tag:
              - v0.16.2
          - name: rancher/k3s
            tag:
              - v1.25.16-k3s4
          - name: rancher/mirrored-pause
            tag:
              - 3.6
  - architecture:
      - amd64
    needDownload: true
    subImages:
      - sourceRepo: hub.oepkgs.net/openfuyao/cluster-api
        targetRepo: kubernetes/cluster-api
        images:
          - name: cluster-api-controller
            usedPodInfo:
              - podPrefix: capi-controller-manager
                namespace: cluster-system
            tag:
              - v1.4.3
          - name: kubeadm-bootstrap-controller
            usedPodInfo:
              - podPrefix: capi-controller-manager
                namespace: cluster-system
            tag:
              - v1.4.3
          - name: kubeadm-control-plane-controller
            usedPodInfo:
              - podPrefix: capi-controller-manager
                namespace: cluster-system
            tag:
              - v1.4.3
  - architecture:
      - amd64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: cluster-api-provider-bke
            tag:
              - latest
          - name: bke-manifests
            tag:
              - latest
patches:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/
    files:
      - fileName: Core-VersionConfig-latest.yaml
files:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/registry/
    files:
      - fileName: registry.image-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containerd/containerd/releases/download/v2.1.1-origin/
    files:
      - fileName: containerd-v2.1.1-linux-amd64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containernetworking/plugins/releases/download/v1.4.1/
    files:
      - fileName: cni-plugins-linux-amd64-v1.4.1.tgz
`
		writeYamlCmd := fmt.Sprintf("cat > /tmp/sample.yaml << 'EOF_SAMPLE'\n%s\nEOF_SAMPLE", yamlContent)
		_, err = sshExecutor.Exec(writeYamlCmd)
		Expect(err).NotTo(HaveOccurred(), "写入 /tmp/sample.yaml 应该成功")

		By("根据引导节点架构调整 YAML（arm64 时替换 amd64 为 arm64）")
		archAdjustCmd := `
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
  sed -i 's/amd64/arm64/g' /tmp/sample.yaml
fi
`
		_, err = sshExecutor.Exec(archAdjustCmd)
		Expect(err).NotTo(HaveOccurred(), "根据架构调整 sample.yaml 应该成功")

		By("使用 OCI 策略制备安装包")
		buildCmd := "cd /tmp && bke build patch -f sample.yaml -t image.tar.gz --strategy=oci"
		buildRes, err := sshExecutor.Exec(buildCmd)
		if err != nil {
			GinkgoWriter.Printf("制备安装包失败: %v\nstdout:\n%s\nstderr:\n%s\n", err, buildRes.Stdout, buildRes.Stderr)
		}
		Expect(err).NotTo(HaveOccurred(), "bke build patch 应成功")
		Expect(buildRes.ExitCode).To(Equal(0), "bke build patch 退出码应为 0")
	})

	AfterEach(func() {
		By("测试后清理：重置引导节点环境")
		resetCmd := "echo Y | bke reset --all --mount"
		result, err := sshExecutor.Exec(resetCmd)
		if err != nil {
			GinkgoWriter.Printf("AfterEach Reset 失败: %v\nStderr: %s\n", err, result.Stderr)
		} else {
			GinkgoWriter.Printf("AfterEach Reset 成功: %s\n", result.Stdout)
		}
		By("测试后清理：删除制备的安装包与 YAML")
		cleanupPkgCmd := "rm -f /tmp/image.tar.gz /tmp/sample.yaml"
		_, _ = sshExecutor.Exec(cleanupPkgCmd)
		time.Sleep(5 * time.Second)
	})

	// bke init --otherRepo cr.openfuyao.cn/openfuyao --installConsole=false --imageFilePath=/tmp/image.tar.gz
	runBootstrapInit := func(args string) (*executor.ExecResult, error) {
		// 使用 echo Y 自动确认初始化操作
		cmd := fmt.Sprintf("echo Y | bke init %s", args)
		GinkgoWriter.Printf("执行引导节点初始化命令: %s\n", cmd)

		res, err := sshExecutor.Exec(cmd)

		if res != nil {
			lines := strings.Split(res.Stdout, "\n")
			start := 0
			if len(lines) > 20 {
				start = len(lines) - 20
			}
			GinkgoWriter.Printf("====== Last 20 lines of Stdout ======\n%s\n", strings.Join(lines[start:], "\n"))

			if res.Stderr != "" {
				GinkgoWriter.Printf("====== Stderr ======\n%s\n", res.Stderr)
			}
			GinkgoWriter.Printf("=====================================\n")
		}
		return res, err
	}

	checkPodsReady := func() {
		By("检查引导集群 Pod 状态")
		Eventually(func() bool {
			// 获取所有 Pod
			cmd := "kubectl get pod -A --no-headers"
			res, err := sshExecutor.Exec(cmd)
			if err != nil || res.ExitCode != 0 {
				return false
			}

			lines := strings.Split(strings.TrimSpace(res.Stdout), "\n")
			if len(lines) == 0 {
				return false
			}

			allReady := true
			for _, line := range lines {
				if strings.TrimSpace(line) == "" {
					continue
				}
				fields := strings.Fields(line)
				if len(fields) < 4 {
					continue
				}
				status := fields[3]
				// 允许的状态: Running 或 Completed
				if status != "Running" && status != "Completed" {
					GinkgoWriter.Printf("Pod 未就绪: %s (Status: %s)\n", fields[1], status)
					allReady = false
				}
			}
			return allReady
		}, 20*time.Minute, 10*time.Second).Should(BeTrue(), "引导集群所有 Pod 应该处于 Running 或 Completed 状态")
	}

	// Author: xuyouhong
	It("OCI 安装包制备并初始化应成功", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("使用制备的安装包进行引导节点初始化")
		// 选择与样本 YAML 相匹配的仓库前缀
		initArgs := "--installConsole=false --otherRepo cr.openfuyao.cn/openfuyao --imageFilePath=/tmp/image.tar.gz"
		res, err := runBootstrapInit(initArgs)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.ExitCode).To(Equal(0), "安装命令应该成功退出")
		Expect(res.Stdout).To(ContainSubstring("BKE initialization is complete"), "日志应包含初始化完成信息")

		checkPodsReady()

		By("校验所用镜像来自本地仓库（127.0.0.1）")
		checkImageCmd := `nerdctl images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | grep -E '(^|/)127\.0\.0\.1([/:]|$)' || true`
		imgRes, _ := sshExecutor.Exec(checkImageCmd)
		GinkgoWriter.Printf("本地镜像检查输出:\n%s\n", imgRes.Stdout)
		Expect(strings.TrimSpace(imgRes.Stdout)).NotTo(BeEmpty(), "应当存在来自 127.0.0.1 的镜像条目")
	})
})
