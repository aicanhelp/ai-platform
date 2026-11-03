package installation

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
)

var _ = Describe("envCheck 前置校验工具测试", Label("installation", "env-check", "25.12", "post-init", "online"), Ordered, func() {
	var (
		sshExecutor      *executor.SSHExecutor
		guideConfig      *config.GuideNodeConfig
		envCheckBinPath  string
		downloaderTmpDir string
		createdPaths     []string
	)

	run := func(cmd string) *executor.ExecResult {
		res, err := sshExecutor.Exec(cmd)
		if err != nil {
			GinkgoWriter.Printf("cmd: %s\nexit=%d\nstdout:\n%s\nstderr:\n%s\n", cmd, res.ExitCode, res.Stdout, res.Stderr)
		}
		Expect(err).NotTo(HaveOccurred(), "命令执行应成功: %s", cmd)
		Expect(res).NotTo(BeNil())
		Expect(res.ExitCode).To(Equal(0), "命令退出码应为0: %s", cmd)
		return res
	}

	trackPath := func(path string) {
		if strings.TrimSpace(path) == "" {
			return
		}
		createdPaths = append(createdPaths, path)
	}

	createTempDir := func() string {
		res := run("mktemp -d /tmp/envcheck-e2e-XXXXXX")
		dir := strings.TrimSpace(res.Stdout)
		trackPath(dir)
		return dir
	}

	writeFile := func(path, content string) {
		cmd := fmt.Sprintf("cat > %s <<'EOF'\n%s\nEOF", path, content)
		run(cmd)
		trackPath(path)
	}

	BeforeAll(func() {
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")

		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")

		By("按系统架构下载 envCheck 并添加可执行权限")
		archRes := run("uname -m")
		rawArch := strings.TrimSpace(archRes.Stdout)
		arch := rawArch
		switch rawArch {
		case "x86_64", "amd64":
			arch = "amd64"
		case "aarch64", "arm64":
			arch = "arm64"
		}
		Expect(arch == "amd64" || arch == "arm64").To(BeTrue(), "不支持的架构: %s", rawArch)

		url := "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/envCheck"
		if arch == "arm64" {
			url = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/envCheck"
		}

		downloaderTmpDir = createTempDir()
		envCheckBinPath = fmt.Sprintf("%s/envCheck", downloaderTmpDir)
		run(fmt.Sprintf("curl -fL %s -o %s", url, envCheckBinPath))
		run(fmt.Sprintf("chmod +x %s", envCheckBinPath))
		GinkgoWriter.Printf("download to : %s\n", downloaderTmpDir)
		run(fmt.Sprintf("%s -h", envCheckBinPath))
	})

	AfterAll(func() {
		if sshExecutor != nil {
			for i := len(createdPaths) - 1; i >= 0; i-- {
				p := strings.TrimSpace(createdPaths[i])
				if p == "" {
					continue
				}
				_, _ = sshExecutor.Exec(fmt.Sprintf("if [ -e %s ]; then rm -rf %s; fi", p, p))
			}
			_ = sshExecutor.Close()
		}
	})

	// 用例名称：显示帮助文档
	// 用例步骤：1) 在引导节点环境下执行 `envCheck -h`；2) 读取命令行输出内容。
	// 预期结果：输出包含 `Usage:` 与 `Available Commands:` 等帮助信息。
	It("显示帮助文档", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		// 步骤: 执行 envCheck -h / --help
		// 预期: 输出帮助文档
		By("执行 envCheck -h")
		res := run(fmt.Sprintf("%s -h", envCheckBinPath))
		Expect(res.Stdout).To(ContainSubstring("Usage:"))
		Expect(res.Stdout).To(ContainSubstring("Available Commands:"))
	})

	// 用例名称：时钟不同步时检查应返回 fail（执行前后恢复同步）
	// 用例步骤：1) 记录引导节点当前时间；2) 在 master 上将时间偏移 30 秒制造不同步；3) 执行 `envCheck clock`；4) 恢复 master 时间与引导节点同步。
	// 预期结果：`envCheck clock` 输出包含 `fail`，且恢复后 master 时间设置命令退出码为 0。
	It("时钟不同步时检查应返回 fail（执行前后恢复同步）", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		tmpDir := createTempDir()
		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要一个节点用于检测时钟是否同步")

		masterNode := nodes[0]

		// 连接到 master 节点以调整时间
		masterPort, err := strconv.Atoi(masterNode.Port)
		Expect(err).NotTo(HaveOccurred(), "应能将 master 节点端口转换为整数")
		masterSSH, err := executor.NewSSHExecutor(
			masterNode.IP,
			masterPort,
			masterNode.Username,
			masterNode.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "应能连接到 master 节点")
		defer func() { _ = masterSSH.Close() }()

		// 记录引导节点当前时间（秒）以便恢复
		guideTSRes := run("date +%s")
		guideTS := strings.TrimSpace(guideTSRes.Stdout)
		Expect(guideTS).NotTo(BeEmpty())

		// 在 master 上将时间设置为基于引导节点时间向前偏移 30 秒
		By("在 master 上设置时间偏移")
		guideSec, err := strconv.ParseInt(guideTS, 10, 64)
		Expect(err).NotTo(HaveOccurred(), "应能解析引导节点时间戳")
		target := guideSec + 30
		setOffsetCmd := fmt.Sprintf("sudo date -s \"@%d\"", target)
		offRes, err := masterSSH.Exec(setOffsetCmd)
		if err != nil {
			GinkgoWriter.Printf("set offset exit=%d stdout=%s stderr=%s\n", offRes.ExitCode, offRes.Stdout, offRes.Stderr)
		}
		Expect(err).NotTo(HaveOccurred())
		Expect(offRes).NotTo(BeNil())
		Expect(offRes.ExitCode).To(Equal(0), "应能在 master 上设置偏移时间")

		// 写入检查用配置并执行检查
		clockCfgPath := fmt.Sprintf("%s/clockConfig.json", tmpDir)
		clockCfg := fmt.Sprintf(`{
  "output_format": "text",
  "hosts": [
    {
      "ip": %q,
      "username": %q,
      "password": %q,
      "port": %q,
      "role": ["bootstrap"]
    },
    {
      "ip": %q,
      "username": %q,
      "password": %q,
      "port": %q,
      "role": ["master"]
    }
  ],
  "clock_threshold": 10
}`,
			guideConfig.Host, guideConfig.Username, guideConfig.Password, strconv.Itoa(guideConfig.Port),
			masterNode.IP, masterNode.Username, masterNode.Password, masterNode.Port,
		)
		writeFile(clockCfgPath, clockCfg)
		By("执行时钟同步检查")
		resCheck := run(fmt.Sprintf("%s clock --config=%s", envCheckBinPath, clockCfgPath))
		// 期望检查结果包含 fail（不区分大小写）
		Expect(strings.Contains(strings.ToLower(resCheck.Stdout), "fail")).To(BeTrue(), "期望时钟不同步时检查返回 fail")

		// 在恢复前重新读取引导节点当前时间
		By("恢复 master 时间与引导节点同步")
		guideTSAfterRes := run("date +%s")
		guideTSAfter := strings.TrimSpace(guideTSAfterRes.Stdout)
		Expect(guideTSAfter).NotTo(BeEmpty())

		restoreCmd := fmt.Sprintf("sudo date -s \"@%s\"", guideTSAfter)
		restRes, err := masterSSH.Exec(restoreCmd)
		Expect(err).NotTo(HaveOccurred())
		Expect(restRes).NotTo(BeNil())
		Expect(restRes.ExitCode).To(Equal(0), "应能在 master 上恢复时间")
	})

	// 用例名称：时钟同步检查使用指定配置文件
	// 用例步骤：1) 生成 `clockConfig.json` 指定 hosts（包含引导节点 bootstrap 与 master）；2) 执行 `envCheck clock --config=...`。
	// 预期结果：输出包含 `Clock Check Result: pass`，并包含各 Host 的检查信息。
	It("时钟同步检查使用指定配置文件", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		// 步骤: envCheck clock --config=./clockConfig.json
		// 预期: 使用指定配置中的 hosts 做时钟同步检查
		tmpDir := createTempDir()
		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要一个节点用于检测时钟是否同步")

		masterNode := nodes[0]
		masterNode.Role = []string{"master/node", "etcd"}

		clockCfgPath := fmt.Sprintf("%s/clockConfig.json", tmpDir)
		clockCfg := fmt.Sprintf(`{
  "output_format": "text",
  "hosts": [
    {
      "ip": %q,
      "username": %q,
      "password": %q,
      "port": %q,
      "role": ["bootstrap"]
    },
    {
      "ip": %q,
      "username": %q,
      "password": %q,
      "port": %q,
      "role": ["master"]
    }
  ],
  "clock_threshold": 10
}`,
			guideConfig.Host, guideConfig.Username, guideConfig.Password, strconv.Itoa(guideConfig.Port),
			masterNode.IP, masterNode.Username, masterNode.Password, masterNode.Port,
		)
		writeFile(clockCfgPath, clockCfg)

		By("执行时钟同步检查")
		res := run(fmt.Sprintf("%s clock --config=%s", envCheckBinPath, clockCfgPath))
		Expect(res.Stdout).To(ContainSubstring("Clock Check Result: pass"))
		Expect(res.Stdout).To(ContainSubstring("Host"))
	})

	// 用例名称：命令行缺失配置时输出帮助文档
	// 用例步骤：1) 直接执行 `envCheck`（不带子命令/配置参数）；2) 读取命令行输出。
	// 预期结果：输出包含 `Usage:` 与 `Available Commands:`，提示用户补全必填参数。
	It("命令行缺失配置时输出帮助文档", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		// 步骤: 执行 envCheck（不带子命令）
		// 预期: 输出帮助文档
		By("执行 envCheck（不带子命令）")
		res := run(fmt.Sprintf("%s", envCheckBinPath))
		Expect(res.Stdout).To(ContainSubstring("Usage:"))
		Expect(res.Stdout).To(ContainSubstring("Available Commands:"))
	})

	// 用例名称：强制删除字段冲突时应以 --force 为准
	// 用例步骤：1) 准备目标文件并生成 `config.json`（clean_force=false）；2) 执行 `envCheck clean --force`；3) 检查目标文件最终是否被删除。
	// 预期结果：开启 `--force` 后目标文件被删除，且命令输出包含 `Total Deleted`。
	It("强制删除字段冲突时应以 --force 为准", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		// 步骤: 配置 clean_force=false，执行 envCheck clean --force
		// 预期: 直接删除目标文件
		tmpDir := createTempDir()

		targetPath := fmt.Sprintf("%s/to-clean.txt", tmpDir)
		cfgPath := fmt.Sprintf("%s/config.json", tmpDir)

		run(fmt.Sprintf("echo test-data > %s", targetPath))
		cfg := fmt.Sprintf(`{
  "output_format": "text",
  "paths": [%q],
  "clean_force": false
}`,
			targetPath,
		)
		writeFile(cfgPath, cfg)

		By("执行 clean 命令")
		res := run(fmt.Sprintf("%s clean --force --config=%s", envCheckBinPath, cfgPath))
		Expect(res.Stdout).To(ContainSubstring("Total Deleted"))

		check := run(fmt.Sprintf("[ ! -e %s ] && echo DELETED || echo EXISTS", targetPath))
		Expect(strings.TrimSpace(check.Stdout)).To(Equal("DELETED"), "开启 --force 后目标文件应被删除")
	})

	// 用例名称：envCheck query 查询引导节点 kubeconfig 与默认 kubeconfig
	// 用例步骤：1) 生成 `query-config.json`（包含需要查询的 kubeconfig 路径，用于引导节点场景）；2) 执行 `envCheck query --config=...`；3) 校验输出统计字段。
	// 预期结果：stdout 包含指定 kubeconfig 路径，并且 `Total Checked/Total Exists` 与实际文件状态一致。
	It("envCheck query 查询引导节点 kubeconfig 与默认 kubeconfig", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		// 步骤: 执行 envCheck query
		// 预期: 能查询到引导节点 kubeconfig 文件和默认 kubeconfig 文件
		tmpDir := createTempDir()
		queryCfgPath := fmt.Sprintf("%s/query-config.json", tmpDir)
		queryCfg := `{
  "output_format": "text",
  "paths": [
    "/etc/rancher/k3s/k3s.yaml"
  ]
}`
		writeFile(queryCfgPath, queryCfg)

		res := run(fmt.Sprintf("%s query --config=%s", envCheckBinPath, queryCfgPath))
		Expect(res.Stdout).To(ContainSubstring("/etc/rancher/k3s/k3s.yaml"))
		Expect(res.Stdout).To(ContainSubstring("Total Checked"))
		Expect(res.Stdout).To(ContainSubstring("Total Exists"))
	})

	// 用例名称：创建1master1worker后缩容worker并在worker执行envCheck query检测残留
	// 用例步骤：1) 通过引导节点创建 1master1worker 集群；2) 缩容 worker 并等待集群节点数与状态更新；3) 在 worker 上下载 envCheck 并执行 `query` 检测 kubeconfig 残留。
	// 预期结果：`envCheck query` 输出中包含 kubeconfig 路径，且查询结果与 worker 节点实际文件存在性一致。
	It("创建1master1worker后缩容worker并在worker执行envCheck query检测残留", Label("skip-temporarily", "required_nodes:2"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 2), "1Master1Worker 集群需要至少 2 个节点")

		clusterName := fmt.Sprintf("test-envcheck-scalein-%d", time.Now().Unix())
		masterNode := nodes[0]
		masterNode.Role = []string{"master/node", "etcd"}
		workerNode := nodes[1]
		workerNode.Role = []string{"node"}

		var dynamicClient dynamic.Interface
		var clusterManager *utils.ClusterManager

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
		clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode, workerNode})

		var configPath string
		var nodeConfigPath string
		DeferCleanup(func() {
			By("清理测试集群")
			if clusterManager != nil && clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
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

		By("创建 1master1worker 集群")
		configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
		err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

		Eventually(func() string {
			phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus
		}, installTimeout, pollInterval).Should(Equal("Ready"))
		checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
		oldCount, _ := checker.GetNodeCountWithKubeconfig("")
		GinkgoWriter.Printf("当前节点数量: nodeCount=%d\n", oldCount)

		By("缩容 worker 节点")
		err = clusterManager.ScaleInNodeWithKubeconfig(clusterName, workerNode.IP, "")
		Expect(err).NotTo(HaveOccurred(), "应能成功缩容 worker 节点")

		kubeconfigPath, err := checker.SaveKubeconfigToFile()
		Expect(err).NotTo(HaveOccurred())
		trackPath(kubeconfigPath)

		Eventually(func() bool {
			phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			count, _ := checker.GetNodeCountWithKubeconfig("")
			GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s, nodeCount=%d\n", phase, state, clusterStatus, count)
			return count == oldCount-1 && clusterStatus == "Ready"
		}, installTimeout, pollInterval).Should(BeTrue())

		By("在 worker 节点下载 envCheck 并执行 query 检测 kubeconfig 残留")
		workerPort, err := strconv.Atoi(workerNode.Port)
		Expect(err).NotTo(HaveOccurred())
		workerSSH, err := executor.NewSSHExecutor(workerNode.IP, workerPort, workerNode.Username, workerNode.Password)
		Expect(err).NotTo(HaveOccurred(), "应能连接到 worker 节点")
		defer func() { _ = workerSSH.Close() }()

		runOnWorker := func(cmd string) *executor.ExecResult {
			res, execErr := workerSSH.Exec(cmd)
			if execErr != nil {
				if res != nil {
					GinkgoWriter.Printf("worker cmd: %s\nexit=%d\nstdout:\n%s\nstderr:\n%s\n", cmd, res.ExitCode, res.Stdout, res.Stderr)
				} else {
					GinkgoWriter.Printf("worker cmd: %s\nerror: %v\n", cmd, execErr)
				}
			}
			Expect(execErr).NotTo(HaveOccurred(), "worker 命令执行应成功: %s", cmd)
			Expect(res).NotTo(BeNil())
			Expect(res.ExitCode).To(Equal(0), "worker 命令退出码应为0: %s", cmd)
			return res
		}

		workerTmp := strings.TrimSpace(runOnWorker("mktemp -d /tmp/envcheck-worker-XXXXXX").Stdout)
		defer func() {
			_, _ = workerSSH.Exec(fmt.Sprintf("rm -rf %s", workerTmp))
		}()

		workerArchRaw := strings.TrimSpace(runOnWorker("uname -m").Stdout)
		workerArch := workerArchRaw
		switch workerArchRaw {
		case "x86_64", "amd64":
			workerArch = "amd64"
		case "aarch64", "arm64":
			workerArch = "arm64"
		}
		Expect(workerArch == "amd64" || workerArch == "arm64").To(BeTrue(), "不支持的 worker 架构: %s", workerArchRaw)

		workerURL := "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/amd64/envCheck"
		if workerArch == "arm64" {
			workerURL = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/env-check/releases/download/latest/bin/linux/arm64/envCheck"
		}

		workerEnvCheck := fmt.Sprintf("%s/envCheck", workerTmp)
		runOnWorker(fmt.Sprintf("curl -fL %s -o %s", workerURL, workerEnvCheck))
		runOnWorker(fmt.Sprintf("chmod +x %s", workerEnvCheck))

		workerQueryCfg := fmt.Sprintf("%s/query-config.json", workerTmp)
		runOnWorker(fmt.Sprintf("cat > %s <<'EOF'\n{\n  \"output_format\": \"text\",\n  \"paths\": [\"/root/.kube/config\"]\n}\nEOF", workerQueryCfg))

		expectStatus := strings.TrimSpace(runOnWorker("[ -e /root/.kube/config ] && echo Exists || echo Missing").Stdout)
		queryRes := runOnWorker(fmt.Sprintf("%s query --config=%s", workerEnvCheck, workerQueryCfg))
		Expect(queryRes.Stdout).To(ContainSubstring("/root/.kube/config"))
		Expect(queryRes.Stdout).To(ContainSubstring(expectStatus), "query 结果应与 worker 节点实际文件状态一致")

	})

	// 用例名称：清理步骤后不遗留测试文件
	// 用例步骤：1) 校验 envCheck 二进制在下载目录存在；2) 创建一次性 marker 文件并加入统一清理列表。
	// 预期结果：后续统一清理逻辑应移除临时文件，避免残留影响其他用例。
	It("清理步骤后不遗留测试文件", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("确认下载目录在用例执行阶段存在")
		res := run(fmt.Sprintf("[ -x %s ] && echo EXISTS || echo MISSING", envCheckBinPath))
		Expect(strings.TrimSpace(res.Stdout)).To(Equal("EXISTS"))

		By("创建一个一次性文件并记录到统一清理列表")
		marker := fmt.Sprintf("%s/marker-%d.txt", downloaderTmpDir, time.Now().UnixNano())
		run(fmt.Sprintf("echo marker > %s", marker))
		trackPath(marker)
	})
})
