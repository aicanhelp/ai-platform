package performance_test

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// 性能测试结果输出文件
const performanceResultFile = "performance_test_results.txt"

// writePerformanceResult 将性能测试结果写入文件（覆盖模式）
func writePerformanceResult(content string) {
	// 使用覆盖模式打开文件
	file, err := os.OpenFile(performanceResultFile, os.O_WRONLY|os.O_TRUNC|os.O_CREATE, 0644)
	if err != nil {
		GinkgoWriter.Printf("写入性能测试结果文件失败: %v\n", err)
		return
	}
	defer file.Close()

	// 写入内容
	_, err = file.WriteString(content)
	if err != nil {
		GinkgoWriter.Printf("写入性能测试结果内容失败: %v\n", err)
	}
}

// appendPerformanceResult 将性能测试结果追加到文件
func appendPerformanceResult(content string) {
	// 使用追加模式打开文件
	file, err := os.OpenFile(performanceResultFile, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0644)
	if err != nil {
		GinkgoWriter.Printf("追加性能测试结果文件失败: %v\n", err)
		return
	}
	defer file.Close()

	// 写入内容
	_, err = file.WriteString(content)
	if err != nil {
		GinkgoWriter.Printf("追加性能测试结果内容失败: %v\n", err)
	}
}

var _ = Describe("BKE Online Installation Performance Test", Label("online-installation-performance", "skip-temporarily", "online"), func() {
	var (
		guideNodeExecutor   *executor.SSHExecutor
		backendNodeExecutor *executor.SSHExecutor
	)

	BeforeEach(func() {
		// 创建引导节点的 SSH 执行器
		var err error
		guideHost := os.Getenv("GUIDE_NODE_HOST")
		guidePortStr := os.Getenv("GUIDE_NODE_PORT")
		guidePort, _ := strconv.Atoi(guidePortStr)
		guideUser := os.Getenv("GUIDE_NODE_USER")
		guidePassword := os.Getenv("GUIDE_NODE_PASSWORD")

		guideNodeExecutor, err = executor.NewSSHExecutor(guideHost, guidePort, guideUser, guidePassword)
		Expect(err).NotTo(HaveOccurred(), "应该能够创建到引导节点的SSH连接")

		// 创建后端节点的 SSH 执行器
		backendHost := os.Getenv("BACKEND_NODE_HOST")
		backendPortStr := os.Getenv("BACKEND_NODE_PORT")
		backendPort, _ := strconv.Atoi(backendPortStr)
		backendUser := os.Getenv("BACKEND_NODE_USER")
		backendPassword := os.Getenv("BACKEND_NODE_PASSWORD")

		backendNodeExecutor, err = executor.NewSSHExecutor(backendHost, backendPort, backendUser, backendPassword)
		Expect(err).NotTo(HaveOccurred(), "应该能够创建到后端节点的SSH连接")
	})

	AfterEach(func() {
		// 关闭引导节点的SSH连接
		if guideNodeExecutor != nil {
			guideNodeExecutor.Close()
		}

		// 关闭后端节点的SSH连接
		if backendNodeExecutor != nil {
			backendNodeExecutor.Close()
		}
	})

	It("should complete bke init with performance metrics", func() {
		var err error

		// 记录总开始时间
		startTime := time.Now()

		// 步骤1: 在引导节点下载 BKE 二进制文件
		GinkgoWriter.Println("=====步骤1: 在引导节点下载 BKE 二进制文件=====")
		downloadStartTime := time.Now()

		GinkgoWriter.Printf("执行下载指令: %s\n", BKE_DOWNLOAD_COMMAND)
		downloadResult, err := guideNodeExecutor.Exec(BKE_DOWNLOAD_COMMAND)
		Expect(err).NotTo(HaveOccurred(), "BKE 二进制文件下载应该成功")
		GinkgoWriter.Printf("下载输出: %s\n", downloadResult.Stdout)
		if downloadResult.Stderr != "" {
			GinkgoWriter.Printf("下载错误信息: %s\n", downloadResult.Stderr)
		}

		downloadDuration := time.Since(downloadStartTime)
		GinkgoWriter.Printf("下载完成，耗时: %v\n", downloadDuration)

		// 步骤2: 在引导节点执行初始化指令
		GinkgoWriter.Println("=====步骤2: 在引导节点执行初始化指令=====")
		initStartTime := time.Now()

		GinkgoWriter.Printf("执行初始化指令: %s\n", BKE_INIT_COMMAND)

		// 为初始化指令添加10分钟超时控制
		var initResult *executor.ExecResult
		var initErr error

		done := make(chan struct{}, 1)
		go func() {
			initResult, initErr = guideNodeExecutor.Exec(BKE_INIT_COMMAND)
			done <- struct{}{}
		}()

		// 设置10分钟超时
		select {
		case <-done:
			// 命令执行完成
		case <-time.After(10 * time.Minute):
			// 超时
			GinkgoWriter.Println("初始化指令执行超时 (10分钟)")
			Fail("初始化指令执行超时")
		}

		Expect(initErr).NotTo(HaveOccurred(), "BKE 初始化指令应该成功启动")
		GinkgoWriter.Printf("初始化启动输出: %s\n", initResult.Stdout)
		if initResult.Stderr != "" {
			GinkgoWriter.Printf("初始化启动错误信息: %s\n", initResult.Stderr)
		}

		// 步骤3: 等待初始化完成
		GinkgoWriter.Println("=====步骤3: 等待初始化完成=====")

		// 设置检查超时时间为10分钟
		timeoutTime := 10 * time.Minute
		checkInterval := 15 * time.Second

		var initComplete bool
		var initDuration time.Duration

		Eventually(func() bool {
			// 获取所有 Pod 状态
			cmd := "kubectl get pod -A --no-headers"
			logResult, err := guideNodeExecutor.Exec(cmd)
			if err != nil || logResult.ExitCode != 0 {
				GinkgoWriter.Printf("执行kubectl命令失败: %v\n", err)
				return false
			}

			lines := strings.Split(strings.TrimSpace(logResult.Stdout), "\n")
			if len(lines) == 0 {
				GinkgoWriter.Println("没有找到Pod，初始化可能还在进行中...")
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
					GinkgoWriter.Printf("Pod 未就绪: %s (状态: %s)\n", fields[1], status)
					allReady = false
				}
			}

			if allReady {
				initComplete = true
				initDuration = time.Since(initStartTime)
				return true
			}

			GinkgoWriter.Println("初始化中... 等待下一次检查")
			return false
		}, timeoutTime, checkInterval).Should(BeTrue(), fmt.Sprintf("初始化应该在 %v 内完成", timeoutTime))

		// 计算总耗时
		totalDuration := time.Since(startTime)

		// 构建性能结果内容
		resultContent := "测试执行时间: " + time.Now().Format("2006-01-02 15:04:05") + "\n"
		resultContent += "\n" + strings.Repeat("=", 60) + "\n"
		resultContent += "=====        性能测试结果        =====\n"
		resultContent += strings.Repeat("=", 60) + "\n"
		resultContent += "总耗时: " + totalDuration.String() + "\n"
		resultContent += "下载耗时: " + downloadDuration.String() + "\n"
		resultContent += "初始化耗时: " + initDuration.String() + "\n"
		resultContent += strings.Repeat("=", 60) + "\n\n"

		// 输出到控制台
		GinkgoWriter.Println(resultContent)

		// 写入文件（覆盖模式）
		writePerformanceResult(resultContent)

		// 验证初始化成功
		Expect(initComplete).To(BeTrue(), "BKE 初始化应该成功完成")
		GinkgoWriter.Println("引导节点初始化完成！")
	})

	It("should create cluster from guide node and verify on backend node", func() {
		var err error

		// 记录集群创建开始时间
		clusterCreateStartTime := time.Now()

		// 步骤1: 读取并修改节点配置文件
		GinkgoWriter.Println("=====步骤1: 读取并修改节点配置文件=====")
		nodeConfigPath := "/bke/cluster/1master-node.yaml"
		// 备份文件路径
		backupNodeConfigPath := "/bke/cluster/1master-node-backup.yaml"

		// 备份原节点配置文件
		GinkgoWriter.Printf("备份原节点配置文件到: %s\n", backupNodeConfigPath)
		backupCmd := fmt.Sprintf("cp %s %s", nodeConfigPath, backupNodeConfigPath)
		_, err = guideNodeExecutor.Exec(backupCmd)
		Expect(err).NotTo(HaveOccurred(), "应该能够备份节点配置文件")

		// 从环境变量获取后端节点信息
		backendIP := os.Getenv("BACKEND_NODE_HOST")
		backendPassword := os.Getenv("BACKEND_NODE_PASSWORD")

		// 修改备份文件的IP字段 (使用yq命令，修改spec下的ip字段)
		ipCmd := fmt.Sprintf("yq -i '.spec.ip = \"%s\"' %s", backendIP, backupNodeConfigPath)
		_, err = guideNodeExecutor.Exec(ipCmd)
		Expect(err).NotTo(HaveOccurred(), "应该能够修改备份节点配置文件的IP字段")

		// 修改备份文件的password字段 (使用yq命令，修改spec下的password字段)
		passwordCmd := fmt.Sprintf("yq -i '.spec.password = \"%s\"' %s", backendPassword, backupNodeConfigPath)
		_, err = guideNodeExecutor.Exec(passwordCmd)
		Expect(err).NotTo(HaveOccurred(), "应该能够修改备份节点配置文件的password字段")

		GinkgoWriter.Println("节点配置文件备份并修改完成")

		// 步骤2: 执行集群创建命令
		GinkgoWriter.Println("=====步骤2: 执行集群创建命令=====")
		// 使用备份文件路径创建集群
		clusterCreateCmd := fmt.Sprintf("bke cluster create -f /bke/cluster/1master-cluster.yaml -n %s", backupNodeConfigPath)

		// 为集群创建命令添加10分钟超时控制
		var createResult *executor.ExecResult
		var createErr error

		done := make(chan struct{}, 1)
		go func() {
			createResult, createErr = guideNodeExecutor.Exec(clusterCreateCmd)
			done <- struct{}{}
		}()

		// 设置10分钟超时
		select {
		case <-done:
			// 命令执行完成
		case <-time.After(10 * time.Minute):
			// 超时
			GinkgoWriter.Println("集群创建命令执行超时 (10分钟)")
			Fail("集群创建命令执行超时")
		}

		Expect(createErr).NotTo(HaveOccurred(), "集群创建命令应该成功执行")
		GinkgoWriter.Printf("集群创建命令输出: %s\n", createResult.Stdout)
		if createResult.Stderr != "" {
			GinkgoWriter.Printf("集群创建命令错误信息: %s\n", createResult.Stderr)
		}

		// 步骤3:
		GinkgoWriter.Println("=====步骤3: 等待集群创建进行中...=====")

		// 步骤4: 在后端节点检查所有Pod状态
		GinkgoWriter.Println("=====步骤4: 在后端节点检查集群状态=====")
		timeoutTime := 10 * time.Minute
		checkInterval := 15 * time.Second

		var clusterReady bool
		var clusterCreateDuration time.Duration

		Eventually(func() bool {
			// 获取所有Pod状态
			cmd := "kubectl get pod -A --no-headers"
			logResult, err := backendNodeExecutor.Exec(cmd)
			if err != nil || logResult.ExitCode != 0 {
				GinkgoWriter.Printf("执行kubectl命令失败: %v\n", err)
				return false
			}

			lines := strings.Split(strings.TrimSpace(logResult.Stdout), "\n")
			if len(lines) == 0 {
				GinkgoWriter.Println("没有找到Pod，集群创建可能还在进行中...")
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
					GinkgoWriter.Printf("Pod 未就绪: %s (状态: %s)\n", fields[1], status)
					allReady = false
				}
			}

			if allReady {
				clusterReady = true
				clusterCreateDuration = time.Since(clusterCreateStartTime)
				return true
			}

			GinkgoWriter.Println("集群创建中... 等待下一次检查")
			return false
		}, timeoutTime, checkInterval).Should(BeTrue(), fmt.Sprintf("集群创建应该在 %v 内完成", timeoutTime))

		// 构建集群创建性能结果内容
		clusterResultContent := "\n" + strings.Repeat("=", 60) + "\n"
		clusterResultContent += "=====      集群创建性能结果      =====\n"
		clusterResultContent += strings.Repeat("=", 60) + "\n"
		clusterResultContent += "集群创建总耗时: " + clusterCreateDuration.String() + "\n"
		clusterResultContent += strings.Repeat("=", 60) + "\n\n"

		// 输出到控制台
		GinkgoWriter.Println(clusterResultContent)

		// 追加到文件
		appendPerformanceResult(clusterResultContent)

		// 验证集群创建成功
		Expect(clusterReady).To(BeTrue(), "集群创建应该成功完成")
		GinkgoWriter.Println("集群创建完成！")

		// 步骤5: 统计集群中Pod的资源占用情况
		resourceResultContent := "\n" + strings.Repeat("=", 60) + "\n"
		resourceResultContent += "=====    集群资源占用统计结果    =====\n"
		resourceResultContent += strings.Repeat("=", 60) + "\n"

		// 使用kubectl top pod -A命令获取资源使用情况
		topCmd := "kubectl top pod -A --no-headers"
		topResult, err := backendNodeExecutor.Exec(topCmd)
		Expect(err).NotTo(HaveOccurred(), "应该能够执行kubectl top命令")

		// 打印原始输出内容，用于调试
		GinkgoWriter.Printf("原始kubectl top输出:\n%s\n", topResult.Stdout)

		if topResult.Stderr != "" {
			GinkgoWriter.Printf("kubectl top命令的错误信息:\n%s\n", topResult.Stderr)
		}

		// 解析输出并统计资源占用
		type NamespaceResource struct {
			CPU         string
			Memory      string
			CPUValue    int
			MemoryValue int
		}

		namespaceStats := make(map[string]NamespaceResource)
		totalCPUValue := 0
		totalMemoryValue := 0

		lines := strings.Split(strings.TrimSpace(topResult.Stdout), "\n")
		GinkgoWriter.Printf("解析到 %d 行pod信息\n", len(lines))

		for i, line := range lines {
			if strings.TrimSpace(line) == "" {
				GinkgoWriter.Printf("第 %d 行为空，跳过\n", i+1)
				continue
			}

			fields := strings.Fields(line)
			if len(fields) < 4 {
				GinkgoWriter.Printf("第 %d 行字段不足4个，跳过: %s\n", i+1, line)
				continue
			}

			GinkgoWriter.Printf("处理第 %d 行: %s\n", i+1, line)

			namespace := fields[0]
			podName := fields[1]
			cpuStr := fields[2]
			memoryStr := fields[3]

			GinkgoWriter.Printf("  命名空间: %s, Pod名称: %s, CPU: %s, 内存: %s\n", namespace, podName, cpuStr, memoryStr)

			// 解析CPU值 (只支持m单位)
			var cpuValue int
			if strings.HasSuffix(cpuStr, "m") {
				cpuStr = strings.TrimSuffix(cpuStr, "m")
				cpuValue, err = strconv.Atoi(cpuStr)
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("解析CPU值失败: %s", cpuStr))
				GinkgoWriter.Printf("  CPU解析结果: %d m\n", cpuValue)
			} else {
				GinkgoWriter.Printf("CPU值格式错误，只支持m单位: %s\n", cpuStr)
				Fail("CPU值格式错误")
			}

			// 解析内存值 (只支持Mi单位)
			var memoryValue int
			if strings.HasSuffix(memoryStr, "Mi") {
				memoryStr = strings.TrimSuffix(memoryStr, "Mi")
				memoryValue, err = strconv.Atoi(memoryStr)
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("解析内存值失败: %s", memoryStr))
				GinkgoWriter.Printf("  内存解析结果: %d Mi\n", memoryValue)
			} else {
				GinkgoWriter.Printf("内存值格式错误，只支持Mi单位: %s\n", memoryStr)
				Fail("内存值格式错误")
			}

			// 按命名空间汇总
			nsResource := namespaceStats[namespace]
			nsResource.CPU = cpuStr + "m"
			nsResource.Memory = memoryStr + "Mi"
			nsResource.CPUValue += cpuValue
			nsResource.MemoryValue += memoryValue
			namespaceStats[namespace] = nsResource
			GinkgoWriter.Printf("  命名空间 %s 当前累计: CPU=%d m, Memory=%d Mi\n", namespace, nsResource.CPUValue, nsResource.MemoryValue)

			// 总汇总
			totalCPUValue += cpuValue
			totalMemoryValue += memoryValue
			GinkgoWriter.Printf("  总累计: CPU=%d m, Memory=%d Mi\n", totalCPUValue, totalMemoryValue)
		}

		// 输出各命名空间资源占用统计
		resourceResultContent += "\n各命名空间资源占用统计:\n"
		resourceResultContent += "命名空间\tCPU(m)\t内存(Mi)\n"
		for ns, resource := range namespaceStats {
			resourceResultContent += ns + "\t" + strconv.Itoa(resource.CPUValue) + "\t" + strconv.Itoa(resource.MemoryValue) + "\n"
		}

		// 输出总资源占用统计
		resourceResultContent += "\n总资源占用统计:\n"
		resourceResultContent += "总CPU占用: " + strconv.Itoa(totalCPUValue) + " m\n"
		resourceResultContent += "总内存占用: " + strconv.Itoa(totalMemoryValue) + " Mi\n"
		resourceResultContent += strings.Repeat("=", 60) + "\n\n"
		resourceResultContent += "集群资源占用统计完成！\n"

		// 输出到控制台
		GinkgoWriter.Println(resourceResultContent)

		// 追加到文件
		appendPerformanceResult(resourceResultContent)
	})
})

// 业务集群资源占用测试
// 多节点集群创建测试
var _ = Describe("BKE Multi-node Cluster Creation Test", Label("multi-node", "cluster-creation", "skip-temporarily", "online"), func() {
	var (
		guideExecutors []*executor.SSHExecutor
		sceneExecutors map[int][]*executor.SSHExecutor // 场景索引 -> 节点执行器列表
		envVars        map[string]string
	)

	BeforeEach(func() {
		// 从环境变量加载配置
		envVars = map[string]string{
			"GUIDE1_NODE_HOST":     os.Getenv("GUIDE1_NODE_HOST"),
			"GUIDE1_NODE_PORT":     os.Getenv("GUIDE1_NODE_PORT"),
			"GUIDE1_NODE_USER":     os.Getenv("GUIDE1_NODE_USER"),
			"GUIDE1_NODE_PASSWORD": os.Getenv("GUIDE1_NODE_PASSWORD"),
			"GUIDE2_NODE_HOST":     os.Getenv("GUIDE2_NODE_HOST"),
			"GUIDE2_NODE_PORT":     os.Getenv("GUIDE2_NODE_PORT"),
			"GUIDE2_NODE_USER":     os.Getenv("GUIDE2_NODE_USER"),
			"GUIDE2_NODE_PASSWORD": os.Getenv("GUIDE2_NODE_PASSWORD"),
			"GUIDE3_NODE_HOST":     os.Getenv("GUIDE3_NODE_HOST"),
			"GUIDE3_NODE_PORT":     os.Getenv("GUIDE3_NODE_PORT"),
			"GUIDE3_NODE_USER":     os.Getenv("GUIDE3_NODE_USER"),
			"GUIDE3_NODE_PASSWORD": os.Getenv("GUIDE3_NODE_PASSWORD"),
			"GUIDE4_NODE_HOST":     os.Getenv("GUIDE4_NODE_HOST"),
			"GUIDE4_NODE_PORT":     os.Getenv("GUIDE4_NODE_PORT"),
			"GUIDE4_NODE_USER":     os.Getenv("GUIDE4_NODE_USER"),
			"GUIDE4_NODE_PASSWORD": os.Getenv("GUIDE4_NODE_PASSWORD"),
			"GUIDE5_NODE_HOST":     os.Getenv("GUIDE5_NODE_HOST"),
			"GUIDE5_NODE_PORT":     os.Getenv("GUIDE5_NODE_PORT"),
			"GUIDE5_NODE_USER":     os.Getenv("GUIDE5_NODE_USER"),
			"GUIDE5_NODE_PASSWORD": os.Getenv("GUIDE5_NODE_PASSWORD"),
			// 场景1: 1master1worker
			"SCENE1_MASTER1_HOST":     os.Getenv("SCENE1_MASTER1_HOST"),
			"SCENE1_MASTER1_PASSWORD": os.Getenv("SCENE1_MASTER1_PASSWORD"),
			"SCENE1_WORKER1_HOST":     os.Getenv("SCENE1_WORKER1_HOST"),
			"SCENE1_WORKER1_PASSWORD": os.Getenv("SCENE1_WORKER1_PASSWORD"),
			// 场景2: 1master2worker
			"SCENE2_MASTER1_HOST":     os.Getenv("SCENE2_MASTER1_HOST"),
			"SCENE2_MASTER1_PASSWORD": os.Getenv("SCENE2_MASTER1_PASSWORD"),
			"SCENE2_WORKER1_HOST":     os.Getenv("SCENE2_WORKER1_HOST"),
			"SCENE2_WORKER1_PASSWORD": os.Getenv("SCENE2_WORKER1_PASSWORD"),
			"SCENE2_WORKER2_HOST":     os.Getenv("SCENE2_WORKER2_HOST"),
			"SCENE2_WORKER2_PASSWORD": os.Getenv("SCENE2_WORKER2_PASSWORD"),
			// 场景3: 1master3worker
			"SCENE3_MASTER1_HOST":     os.Getenv("SCENE3_MASTER1_HOST"),
			"SCENE3_MASTER1_PASSWORD": os.Getenv("SCENE3_MASTER1_PASSWORD"),
			"SCENE3_WORKER1_HOST":     os.Getenv("SCENE3_WORKER1_HOST"),
			"SCENE3_WORKER1_PASSWORD": os.Getenv("SCENE3_WORKER1_PASSWORD"),
			"SCENE3_WORKER2_HOST":     os.Getenv("SCENE3_WORKER2_HOST"),
			"SCENE3_WORKER2_PASSWORD": os.Getenv("SCENE3_WORKER2_PASSWORD"),
			"SCENE3_WORKER3_HOST":     os.Getenv("SCENE3_WORKER3_HOST"),
			"SCENE3_WORKER3_PASSWORD": os.Getenv("SCENE3_WORKER3_PASSWORD"),
			// 场景4: 1master4worker
			"SCENE4_MASTER1_HOST":     os.Getenv("SCENE4_MASTER1_HOST"),
			"SCENE4_MASTER1_PASSWORD": os.Getenv("SCENE4_MASTER1_PASSWORD"),
			"SCENE4_WORKER1_HOST":     os.Getenv("SCENE4_WORKER1_HOST"),
			"SCENE4_WORKER1_PASSWORD": os.Getenv("SCENE4_WORKER1_PASSWORD"),
			"SCENE4_WORKER2_HOST":     os.Getenv("SCENE4_WORKER2_HOST"),
			"SCENE4_WORKER2_PASSWORD": os.Getenv("SCENE4_WORKER2_PASSWORD"),
			"SCENE4_WORKER3_HOST":     os.Getenv("SCENE4_WORKER3_HOST"),
			"SCENE4_WORKER3_PASSWORD": os.Getenv("SCENE4_WORKER3_PASSWORD"),
			"SCENE4_WORKER4_HOST":     os.Getenv("SCENE4_WORKER4_HOST"),
			"SCENE4_WORKER4_PASSWORD": os.Getenv("SCENE4_WORKER4_PASSWORD"),
			// 场景5: 3master2worker
			"SCENE5_MASTER1_HOST":     os.Getenv("SCENE5_MASTER1_HOST"),
			"SCENE5_MASTER1_PASSWORD": os.Getenv("SCENE5_MASTER1_PASSWORD"),
			"SCENE5_MASTER2_HOST":     os.Getenv("SCENE5_MASTER2_HOST"),
			"SCENE5_MASTER2_PASSWORD": os.Getenv("SCENE5_MASTER2_PASSWORD"),
			"SCENE5_MASTER3_HOST":     os.Getenv("SCENE5_MASTER3_HOST"),
			"SCENE5_MASTER3_PASSWORD": os.Getenv("SCENE5_MASTER3_PASSWORD"),
			"SCENE5_WORKER1_HOST":     os.Getenv("SCENE5_WORKER1_HOST"),
			"SCENE5_WORKER1_PASSWORD": os.Getenv("SCENE5_WORKER1_PASSWORD"),
			"SCENE5_WORKER2_HOST":     os.Getenv("SCENE5_WORKER2_HOST"),
			"SCENE5_WORKER2_PASSWORD": os.Getenv("SCENE5_WORKER2_PASSWORD"),
		}

		// 验证必要的环境变量
		for key, value := range envVars {
			if (strings.Contains(key, "GUIDE") && strings.Contains(key, "NODE_HOST")) ||
				(strings.Contains(key, "SCENE") && (strings.Contains(key, "MASTER") || strings.Contains(key, "WORKER"))) {
				if value == "" {
					Fail(fmt.Sprintf("缺少必要的环境变量: %s", key))
				}
			}
		}

		// 创建5个引导节点的SSH执行器
		guideExecutors = make([]*executor.SSHExecutor, 5)
		for i := 1; i <= 5; i++ {
			port, err := strconv.Atoi(envVars[fmt.Sprintf("GUIDE%d_NODE_PORT", i)])
			if err != nil {
				port = 22 // 默认端口
			}

			executor, err := executor.NewSSHExecutor(
				envVars[fmt.Sprintf("GUIDE%d_NODE_HOST", i)],
				port,
				envVars[fmt.Sprintf("GUIDE%d_NODE_USER", i)],
				envVars[fmt.Sprintf("GUIDE%d_NODE_PASSWORD", i)],
			)
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("创建引导节点%d的SSH执行器失败", i))
			guideExecutors[i-1] = executor
		}

		// 初始化场景节点执行器映射
		sceneExecutors = make(map[int][]*executor.SSHExecutor)

		// 定义每个场景的节点配置: [场景索引] -> [master数量, worker数量]
		sceneNodeConfigs := map[int][2]int{
			1: {1, 1}, // 场景1: 1master1worker
			2: {1, 2}, // 场景2: 1master2worker
			3: {1, 3}, // 场景3: 1master3worker
			4: {1, 4}, // 场景4: 1master4worker
			5: {3, 2}, // 场景5: 3master2worker
		}

		// 循环创建每个场景的节点执行器
		for sceneIndex, config := range sceneNodeConfigs {
			masterCount, workerCount := config[0], config[1]
			totalNodes := masterCount + workerCount
			sceneExecutors[sceneIndex] = make([]*executor.SSHExecutor, totalNodes)

			// 创建master节点执行器
			for i := 1; i <= masterCount; i++ {
				var hostKey, passwordKey string
				if sceneIndex == 5 { // 场景5的master节点命名不同
					hostKey = fmt.Sprintf("SCENE%d_MASTER%d_HOST", sceneIndex, i)
					passwordKey = fmt.Sprintf("SCENE%d_MASTER%d_PASSWORD", sceneIndex, i)
				} else {
					hostKey = fmt.Sprintf("SCENE%d_MASTER1_HOST", sceneIndex) // 其他场景只有1个master
					passwordKey = fmt.Sprintf("SCENE%d_MASTER1_PASSWORD", sceneIndex)
					break // 其他场景只有1个master，创建后跳出循环
				}

				executor, err := executor.NewSSHExecutor(envVars[hostKey], 22, "root", envVars[passwordKey])
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("创建场景%d master%d节点的SSH执行器失败", sceneIndex, i))
				sceneExecutors[sceneIndex][i-1] = executor
			}

			// 创建worker节点执行器
			for i := 1; i <= workerCount; i++ {
				var hostKey, passwordKey string
				hostKey = fmt.Sprintf("SCENE%d_WORKER%d_HOST", sceneIndex, i)
				passwordKey = fmt.Sprintf("SCENE%d_WORKER%d_PASSWORD", sceneIndex, i)

				// 计算worker节点在执行器列表中的索引
				index := masterCount + i - 1
				executor, err := executor.NewSSHExecutor(envVars[hostKey], 22, "root", envVars[passwordKey])
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("创建场景%d worker%d节点的SSH执行器失败", sceneIndex, i))
				sceneExecutors[sceneIndex][index] = executor
			}
		}
	})

	AfterEach(func() {
		// 关闭所有SSH连接
		for _, executor := range guideExecutors {
			if executor != nil {
				executor.Close()
			}
		}

		// 关闭所有集群节点的SSH连接
		for _, nodeExecutors := range sceneExecutors {
			for _, executor := range nodeExecutors {
				if executor != nil {
					executor.Close()
				}
			}
		}
	})

	It("should create multi-node clusters with different configurations", func() {
		var wg sync.WaitGroup
		resultChan := make(chan error, len(guideExecutors))

		// 步骤1: 并发向所有引导节点发送下载BKE二进制文件的指令
		GinkgoWriter.Println("步骤1: 向5个引导节点并发发送下载BKE二进制文件的指令")
		wg.Add(len(guideExecutors))
		for i, executor := range guideExecutors {
			i := i + 1
			executor := executor
			go func() {
				defer wg.Done()
				GinkgoWriter.Printf("  开始向引导节点%d发送下载BKE二进制文件的指令...\n", i)
				// 发送下载指令
				_, err := executor.Exec(BKE_DOWNLOAD_COMMAND)
				if err != nil {
					GinkgoWriter.Printf("  引导节点%d下载BKE二进制文件失败: %v\n", i, err)
					resultChan <- err
					return
				}
				GinkgoWriter.Printf("  引导节点%d下载BKE二进制文件成功\n", i)
				resultChan <- nil
			}()
		}

		// 等待所有下载指令发送完成
		wg.Wait()
		close(resultChan)

		// 检查下载结果
		var downloadFailedCount int
		for err := range resultChan {
			if err != nil {
				downloadFailedCount++
			}
		}
		if downloadFailedCount > 0 {
			GinkgoWriter.Printf("  警告: %d个引导节点下载BKE二进制文件失败\n", downloadFailedCount)
		} else {
			GinkgoWriter.Println("  所有引导节点下载BKE二进制文件成功")
		}

		// 步骤2: 并发向所有引导节点发送初始化指令
		GinkgoWriter.Println("\n步骤2: 向5个引导节点并发发送初始化指令")
		wg.Add(len(guideExecutors))
		resultChan = make(chan error, len(guideExecutors))
		for i, executor := range guideExecutors {
			i := i + 1
			executor := executor
			go func() {
				defer wg.Done()
				GinkgoWriter.Printf("  开始向引导节点%d发送初始化指令...\n", i)
				// 直接发送指令，不等待控制台返回
				_, err := executor.Exec(BKE_INIT_COMMAND)
				if err != nil {
					GinkgoWriter.Printf("  引导节点%d发送初始化指令失败: %v\n", i, err)
					resultChan <- err
					return
				}
				GinkgoWriter.Printf("  引导节点%d初始化指令发送成功\n", i)
				resultChan <- nil
			}()
		}

		// 等待所有初始化指令发送完成
		wg.Wait()
		close(resultChan)

		// 检查初始化指令发送结果
		var initFailedCount int
		for err := range resultChan {
			if err != nil {
				initFailedCount++
			}
		}
		if initFailedCount > 0 {
			GinkgoWriter.Printf("  警告: %d个引导节点初始化指令发送失败\n", initFailedCount)
		} else {
			GinkgoWriter.Println("  所有引导节点初始化指令发送成功")
		}

		// 检查每个引导节点上的集群是否就绪
		readyNodeCount := 0
		for i, executor := range guideExecutors {
			i := i + 1
			GinkgoWriter.Printf("  检查引导节点%d上的集群就绪状态...\n", i)

			// 使用kubectl get pod -A检查所有Pod状态
			cmd := "kubectl get pod -A"
			output, err := executor.Exec(cmd)
			if err != nil {
				GinkgoWriter.Printf("  引导节点%d执行kubectl命令失败: %v\n", i, err)
				continue
			}

			// 解析输出，检查所有Pod是否处于Running或Completed状态
			lines := strings.Split(output.Stdout, "\n")
			allPodsReady := true
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "NAMESPACE") {
					continue // 跳过空行和表头
				}

				// 分割行内容，获取Pod状态
				parts := strings.Fields(line)
				if len(parts) < 4 {
					continue // 格式不正确，跳过
				}

				status := parts[3]
				if status != "Running" && status != "Completed" {
					GinkgoWriter.Printf("  引导节点%d上的Pod未就绪: %s\n", i, line)
					allPodsReady = false
					break
				}
			}

			if allPodsReady {
				GinkgoWriter.Printf("  引导节点%d上的集群已就绪\n", i)
				readyNodeCount++
			} else {
				GinkgoWriter.Printf("  引导节点%d上的集群未完全就绪\n", i)
			}
		}

		// 输出就绪状态统计
		GinkgoWriter.Printf("\n集群就绪状态统计: %d/5个集群已就绪\n", readyNodeCount)

		// 步骤4: 为就绪的引导节点创建集群
		GinkgoWriter.Println("\n步骤4: 为就绪的引导节点创建集群")

		// 定义集群创建结果结构体
		type ClusterCreateResult struct {
			GuideNodeIndex int
			StartTime      time.Time
			EndTime        time.Time
			Duration       time.Duration
			Success        bool
			Error          string
		}

		// 存储集群创建结果
		clusterResults := make([]ClusterCreateResult, 0)

		// 为每个就绪的引导节点创建集群
		for i, executor := range guideExecutors {
			i := i + 1 // 转换为1-based索引

			// 跳过未就绪的引导节点
			cmd := "kubectl get pod -A"
			output, err := executor.Exec(cmd)
			if err != nil {
				GinkgoWriter.Printf("  跳过引导节点%d: 无法执行kubectl命令\n", i)
				continue
			}

			// 检查是否所有Pod都已就绪
			allPodsReady := true
			lines := strings.Split(output.Stdout, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "NAMESPACE") {
					continue
				}
				parts := strings.Fields(line)
				if len(parts) >= 4 {
					status := parts[3]
					if status != "Running" && status != "Completed" {
						allPodsReady = false
						break
					}
				}
			}

			if !allPodsReady {
				GinkgoWriter.Printf("  跳过引导节点%d: 集群未完全就绪\n", i)
				continue
			}

			GinkgoWriter.Printf("  开始在引导节点%d上创建集群...\n", i)
			result := ClusterCreateResult{
				GuideNodeIndex: i,
				StartTime:      time.Now(),
			}

			// 根据不同的场景创建不同的集群配置
			var createErr error
			switch i {
			case 1: // 场景1: 1master1worker
				GinkgoWriter.Printf("    创建1master1worker集群...\n")
				// 备份原始节点配置文件
				backupCmd := "mkdir -p /bke/cluster/backup && cp /bke/cluster/1master1node-nodes.yaml /bke/cluster/backup/1master1node-nodes.yaml.bak"
				_, err = executor.Exec(backupCmd)
				if err != nil {
					createErr = fmt.Errorf("备份节点配置文件失败: %v", err)
					break
				}

				// 修改节点配置文件
				// 修改master节点
				modifyMasterCmd := fmt.Sprintf(`yq -i '.spec.ip = "%s" | .spec.password = "%s" | .spec.username = "root"' /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE1_MASTER1_HOST"], envVars["SCENE1_MASTER1_PASSWORD"])
				_, err = executor.Exec(modifyMasterCmd)
				if err != nil {
					createErr = fmt.Errorf("修改master节点配置失败: %v", err)
					break
				}

				// 修改worker节点
				modifyWorkerCmd := fmt.Sprintf(`yq -i 'select(.metadata.name == "bke-cluster-n1").spec.ip = "%s" | select(.metadata.name == "bke-cluster-n1").spec.password = "%s" | select(.metadata.name == "bke-cluster-n1").spec.username = "root"' /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE1_WORKER1_HOST"], envVars["SCENE1_WORKER1_PASSWORD"])
				_, err = executor.Exec(modifyWorkerCmd)
				if err != nil {
					createErr = fmt.Errorf("修改worker节点配置失败: %v", err)
					break
				}

				// 创建集群
				clusterCreateCmd := "bke cluster create -f /bke/cluster/1master1node-cluster.yaml -n /bke/cluster/backup/1master1node-nodes.yaml.bak"
				_, err = executor.Exec(clusterCreateCmd)
				if err != nil {
					createErr = fmt.Errorf("创建集群失败: %v", err)
				}

			case 2: // 场景2: 1master2worker
				GinkgoWriter.Printf("    创建1master2worker集群...\n")
				// 备份原始节点配置文件
				backupCmd := "mkdir -p /bke/cluster/backup && cp /bke/cluster/1master1node-nodes.yaml /bke/cluster/backup/1master1node-nodes.yaml.bak"
				_, err = executor.Exec(backupCmd)
				if err != nil {
					createErr = fmt.Errorf("备份节点配置文件失败: %v", err)
					break
				}

				// 修改节点配置文件
				// 修改master节点
				modifyMasterCmd := fmt.Sprintf(`yq -i '.spec.ip = "%s" | .spec.password = "%s" | .spec.username = "root"' /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE2_MASTER1_HOST"], envVars["SCENE2_MASTER1_PASSWORD"])
				_, err = executor.Exec(modifyMasterCmd)
				if err != nil {
					createErr = fmt.Errorf("修改master节点配置失败: %v", err)
					break
				}

				// 修改worker1节点
				modifyWorker1Cmd := fmt.Sprintf(`yq -i 'select(.metadata.name == "bke-cluster-n1").spec.ip = "%s" | select(.metadata.name == "bke-cluster-n1").spec.password = "%s" | select(.metadata.name == "bke-cluster-n1").spec.username = "root"' /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE2_WORKER1_HOST"], envVars["SCENE2_WORKER1_PASSWORD"])
				_, err = executor.Exec(modifyWorker1Cmd)
				if err != nil {
					createErr = fmt.Errorf("修改worker1节点配置失败: %v", err)
					break
				}

				// 添加worker2节点
				addWorker2Cmd := fmt.Sprintf(`echo -e "---\napiVersion: bke.bocloud.com/v1beta1\nkind: BKENode\nmetadata:\n  creationTimestamp: null\n  labels:\n    cluster.x-k8s.io/cluster-name: bke-cluster\n  name: bke-cluster-n2\n  namespace: bke-cluster\nspec:\n  hostname: n2\n  ip: %s\n  password: '%s'\n  port: \"22\"\n  role:\n  - node\n  username: root\nstatus: {}" >> /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE2_WORKER2_HOST"], envVars["SCENE2_WORKER2_PASSWORD"])
				_, err = executor.Exec(addWorker2Cmd)
				if err != nil {
					createErr = fmt.Errorf("添加worker2节点配置失败: %v", err)
					break
				}

				// 创建集群
				clusterCreateCmd := "bke cluster create -f /bke/cluster/1master1node-cluster.yaml -n /bke/cluster/backup/1master1node-nodes.yaml.bak"
				_, err = executor.Exec(clusterCreateCmd)
				if err != nil {
					createErr = fmt.Errorf("创建集群失败: %v", err)
				}
			case 3: // 场景3: 1master3worker
				GinkgoWriter.Printf("    创建1master3worker集群...\n")
				// 备份原始节点配置文件
				backupCmd := "mkdir -p /bke/cluster/backup && cp /bke/cluster/1master1node-nodes.yaml /bke/cluster/backup/1master1node-nodes.yaml.bak"
				_, err = executor.Exec(backupCmd)
				if err != nil {
					createErr = fmt.Errorf("备份节点配置文件失败: %v", err)
					break
				}

				// 修改节点配置文件
				// 修改master节点
				modifyMasterCmd := fmt.Sprintf(`yq -i '.spec.ip = "%s" | .spec.password = "%s" | .spec.username = "root"' /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE3_MASTER1_HOST"], envVars["SCENE3_MASTER1_PASSWORD"])
				_, err = executor.Exec(modifyMasterCmd)
				if err != nil {
					createErr = fmt.Errorf("修改master节点配置失败: %v", err)
					break
				}

				// 修改worker1节点
				modifyWorker1Cmd := fmt.Sprintf(`yq -i 'select(.metadata.name == "bke-cluster-n1").spec.ip = "%s" | select(.metadata.name == "bke-cluster-n1").spec.password = "%s" | select(.metadata.name == "bke-cluster-n1").spec.username = "root"' /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE3_WORKER1_HOST"], envVars["SCENE3_WORKER1_PASSWORD"])
				_, err = executor.Exec(modifyWorker1Cmd)
				if err != nil {
					createErr = fmt.Errorf("修改worker1节点配置失败: %v", err)
					break
				}

				// 添加worker2节点
				addWorker2Cmd := fmt.Sprintf(`echo -e "---\napiVersion: bke.bocloud.com/v1beta1\nkind: BKENode\nmetadata:\n  creationTimestamp: null\n  labels:\n    cluster.x-k8s.io/cluster-name: bke-cluster\n  name: bke-cluster-n2\n  namespace: bke-cluster\nspec:\n  hostname: n2\n  ip: %s\n  password: '%s'\n  port: \"22\"\n  role:\n  - node\n  username: root\nstatus: {}" >> /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE3_WORKER2_HOST"], envVars["SCENE3_WORKER2_PASSWORD"])
				_, err = executor.Exec(addWorker2Cmd)
				if err != nil {
					createErr = fmt.Errorf("添加worker2节点配置失败: %v", err)
					break
				}

				// 添加worker3节点
				addWorker3Cmd := fmt.Sprintf(`echo -e "---\napiVersion: bke.bocloud.com/v1beta1\nkind: BKENode\nmetadata:\n  creationTimestamp: null\n  labels:\n    cluster.x-k8s.io/cluster-name: bke-cluster\n  name: bke-cluster-n3\n  namespace: bke-cluster\nspec:\n  hostname: n3\n  ip: %s\n  password: '%s'\n  port: \"22\"\n  role:\n  - node\n  username: root\nstatus: {}" >> /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE3_WORKER3_HOST"], envVars["SCENE3_WORKER3_PASSWORD"])
				_, err = executor.Exec(addWorker3Cmd)
				if err != nil {
					createErr = fmt.Errorf("添加worker3节点配置失败: %v", err)
					break
				}

				// 创建集群
				clusterCreateCmd := "bke cluster create -f /bke/cluster/1master1node-cluster.yaml -n /bke/cluster/backup/1master1node-nodes.yaml.bak"
				_, err = executor.Exec(clusterCreateCmd)
				if err != nil {
					createErr = fmt.Errorf("创建集群失败: %v", err)
				}
			case 4: // 场景4: 1master4worker
				GinkgoWriter.Printf("    创建1master4worker集群...\n")
				// 备份原始节点配置文件
				backupCmd := "mkdir -p /bke/cluster/backup && cp /bke/cluster/1master1node-nodes.yaml /bke/cluster/backup/1master1node-nodes.yaml.bak"
				_, err = executor.Exec(backupCmd)
				if err != nil {
					createErr = fmt.Errorf("备份节点配置文件失败: %v", err)
					break
				}

				// 修改节点配置文件
				// 修改master节点
				modifyMasterCmd := fmt.Sprintf(`yq -i '.spec.ip = "%s" | .spec.password = "%s" | .spec.username = "root"' /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE4_MASTER1_HOST"], envVars["SCENE4_MASTER1_PASSWORD"])
				_, err = executor.Exec(modifyMasterCmd)
				if err != nil {
					createErr = fmt.Errorf("修改master节点配置失败: %v", err)
					break
				}

				// 修改worker1节点
				modifyWorker1Cmd := fmt.Sprintf(`yq -i 'select(.metadata.name == "bke-cluster-n1").spec.ip = "%s" | select(.metadata.name == "bke-cluster-n1").spec.password = "%s" | select(.metadata.name == "bke-cluster-n1").spec.username = "root"' /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE4_WORKER1_HOST"], envVars["SCENE4_WORKER1_PASSWORD"])
				_, err = executor.Exec(modifyWorker1Cmd)
				if err != nil {
					createErr = fmt.Errorf("修改worker1节点配置失败: %v", err)
					break
				}

				// 添加worker2节点
				addWorker2Cmd := fmt.Sprintf(`echo -e "---\napiVersion: bke.bocloud.com/v1beta1\nkind: BKENode\nmetadata:\n  creationTimestamp: null\n  labels:\n    cluster.x-k8s.io/cluster-name: bke-cluster\n  name: bke-cluster-n2\n  namespace: bke-cluster\nspec:\n  hostname: n2\n  ip: %s\n  password: '%s'\n  port: \"22\"\n  role:\n  - node\n  username: root\nstatus: {}" >> /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE4_WORKER2_HOST"], envVars["SCENE4_WORKER2_PASSWORD"])
				_, err = executor.Exec(addWorker2Cmd)
				if err != nil {
					createErr = fmt.Errorf("添加worker2节点配置失败: %v", err)
					break
				}

				// 添加worker3节点
				addWorker3Cmd := fmt.Sprintf(`echo -e "---\napiVersion: bke.bocloud.com/v1beta1\nkind: BKENode\nmetadata:\n  creationTimestamp: null\n  labels:\n    cluster.x-k8s.io/cluster-name: bke-cluster\n  name: bke-cluster-n3\n  namespace: bke-cluster\nspec:\n  hostname: n3\n  ip: %s\n  password: '%s'\n  port: \"22\"\n  role:\n  - node\n  username: root\nstatus: {}" >> /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE4_WORKER3_HOST"], envVars["SCENE4_WORKER3_PASSWORD"])
				_, err = executor.Exec(addWorker3Cmd)
				if err != nil {
					createErr = fmt.Errorf("添加worker3节点配置失败: %v", err)
					break
				}

				// 添加worker4节点
				addWorker4Cmd := fmt.Sprintf(`echo -e "---\napiVersion: bke.bocloud.com/v1beta1\nkind: BKENode\nmetadata:\n  creationTimestamp: null\n  labels:\n    cluster.x-k8s.io/cluster-name: bke-cluster\n  name: bke-cluster-n4\n  namespace: bke-cluster\nspec:\n  hostname: n4\n  ip: %s\n  password: '%s'\n  port: \"22\"\n  role:\n  - node\n  username: root\nstatus: {}" >> /bke/cluster/backup/1master1node-nodes.yaml.bak`,
					envVars["SCENE4_WORKER4_HOST"], envVars["SCENE4_WORKER4_PASSWORD"])
				_, err = executor.Exec(addWorker4Cmd)
				if err != nil {
					createErr = fmt.Errorf("添加worker4节点配置失败: %v", err)
					break
				}

				// 创建集群
				clusterCreateCmd := "bke cluster create -f /bke/cluster/1master1node-cluster.yaml -n /bke/cluster/backup/1master1node-nodes.yaml.bak"
				_, err = executor.Exec(clusterCreateCmd)
				if err != nil {
					createErr = fmt.Errorf("创建集群失败: %v", err)
				}

			case 5: // 场景5: 3master2worker
				GinkgoWriter.Printf("    创建3master2worker集群...\n")
				// 备份原始节点配置文件
				backupCmd := "mkdir -p /bke/cluster/backup && cp /bke/cluster/3master-nodes.yaml /bke/cluster/backup/3master-nodes.yaml.bak"
				_, err = executor.Exec(backupCmd)
				if err != nil {
					createErr = fmt.Errorf("备份节点配置文件失败: %v", err)
					break
				}

				// 修改节点配置文件
				// 修改master1节点
				modifyMaster1Cmd := fmt.Sprintf(`yq -i 'select(.metadata.name == "bke-cluster-master-1").spec.ip = "%s" | select(.metadata.name == "bke-cluster-master-1").spec.password = "%s" | select(.metadata.name == "bke-cluster-master-1").spec.username = "root"' /bke/cluster/backup/3master-nodes.yaml.bak`,
					envVars["SCENE5_MASTER1_HOST"], envVars["SCENE5_MASTER1_PASSWORD"])
				_, err = executor.Exec(modifyMaster1Cmd)
				if err != nil {
					createErr = fmt.Errorf("修改master1节点配置失败: %v", err)
					break
				}

				// 修改master2节点
				modifyMaster2Cmd := fmt.Sprintf(`yq -i 'select(.metadata.name == "bke-cluster-master-2").spec.ip = "%s" | select(.metadata.name == "bke-cluster-master-2").spec.password = "%s" | select(.metadata.name == "bke-cluster-master-2").spec.username = "root"' /bke/cluster/backup/3master-nodes.yaml.bak`,
					envVars["SCENE5_MASTER2_HOST"], envVars["SCENE5_MASTER2_PASSWORD"])
				_, err = executor.Exec(modifyMaster2Cmd)
				if err != nil {
					createErr = fmt.Errorf("修改master2节点配置失败: %v", err)
					break
				}

				// 修改master3节点
				modifyMaster3Cmd := fmt.Sprintf(`yq -i 'select(.metadata.name == "bke-cluster-master-3").spec.ip = "%s" | select(.metadata.name == "bke-cluster-master-3").spec.password = "%s" | select(.metadata.name == "bke-cluster-master-3").spec.username = "root"' /bke/cluster/backup/3master-nodes.yaml.bak`,
					envVars["SCENE5_MASTER3_HOST"], envVars["SCENE5_MASTER3_PASSWORD"])
				_, err = executor.Exec(modifyMaster3Cmd)
				if err != nil {
					createErr = fmt.Errorf("修改master3节点配置失败: %v", err)
					break
				}

				// 添加worker1节点
				addWorker1Cmd := fmt.Sprintf(`echo -e "---\napiVersion: bke.bocloud.com/v1beta1\nkind: BKENode\nmetadata:\n  creationTimestamp: null\n  labels:\n    cluster.x-k8s.io/cluster-name: bke-cluster\n  name: bke-cluster-worker-1\n  namespace: bke-cluster\nspec:\n  hostname: worker-1\n  ip: %s\n  password: '%s'\n  port: \"22\"\n  role:\n  - node\n  username: root\nstatus: {}" >> /bke/cluster/backup/3master-nodes.yaml.bak`,
					envVars["SCENE5_WORKER1_HOST"], envVars["SCENE5_WORKER1_PASSWORD"])
				_, err = executor.Exec(addWorker1Cmd)
				if err != nil {
					createErr = fmt.Errorf("添加worker1节点配置失败: %v", err)
					break
				}

				// 添加worker2节点
				addWorker2Cmd := fmt.Sprintf(`echo -e "---\napiVersion: bke.bocloud.com/v1beta1\nkind: BKENode\nmetadata:\n  creationTimestamp: null\n  labels:\n    cluster.x-k8s.io/cluster-name: bke-cluster\n  name: bke-cluster-worker-2\n  namespace: bke-cluster\nspec:\n  hostname: worker-2\n  ip: %s\n  password: '%s'\n  port: \"22\"\n  role:\n  - node\n  username: root\nstatus: {}" >> /bke/cluster/backup/3master-nodes.yaml.bak`,
					envVars["SCENE5_WORKER2_HOST"], envVars["SCENE5_WORKER2_PASSWORD"])
				_, err = executor.Exec(addWorker2Cmd)
				if err != nil {
					createErr = fmt.Errorf("添加worker2节点配置失败: %v", err)
					break
				}

				// 创建集群
				clusterCreateCmd := "bke cluster create -f /bke/cluster/1master1node-cluster.yaml -n /bke/cluster/backup/3master-nodes.yaml.bak"
				_, err = executor.Exec(clusterCreateCmd)
				if err != nil {
					createErr = fmt.Errorf("创建集群失败: %v", err)
				}
			}

			// 记录创建结果
			result.EndTime = time.Now()
			result.Duration = result.EndTime.Sub(result.StartTime)

			if createErr != nil {
				GinkgoWriter.Printf("  引导节点%d创建集群失败: %v\n", i, createErr)
				result.Success = false
				result.Error = createErr.Error()
			} else {
				GinkgoWriter.Printf("  引导节点%d创建集群成功\n", i)
				result.Success = true
			}

			clusterResults = append(clusterResults, result)
		}

		// 步骤5: 等待2分钟后检查集群状态
		GinkgoWriter.Println("\n步骤5: 等待2分钟后检查集群状态")
		time.Sleep(2 * time.Minute)

		// 步骤6: 检查每个集群的Pod状态
		GinkgoWriter.Println("\n步骤6: 检查每个集群的Pod状态")

		// 定义集群状态结果结构体
		type ClusterStatusResult struct {
			GuideNodeIndex int
			AllPodsReady   bool
			PodCount       int
			ReadyPodCount  int
			Error          string
		}

		// 存储集群状态结果
		statusResults := make([]ClusterStatusResult, 0)

		for _, result := range clusterResults {
			if !result.Success {
				continue // 跳过创建失败的集群
			}

			statusResult := ClusterStatusResult{
				GuideNodeIndex: result.GuideNodeIndex,
			}

			// 获取当前场景的节点执行器
			nodeExecutors := sceneExecutors[result.GuideNodeIndex]
			allNodesPodsReady := true
			totalPodCount := 0
			totalReadyPodCount := 0

			// 检查每个节点的Pod状态
			for i, nodeExecutor := range nodeExecutors {
				GinkgoWriter.Printf("  检查引导节点%d的集群节点%d的Pod状态...\n", result.GuideNodeIndex, i+1)

				// 执行kubectl命令检查Pod状态
				checkPodsCmd := "kubectl get pods -A --no-headers"
				output, err := nodeExecutor.Exec(checkPodsCmd)
				if err != nil {
					GinkgoWriter.Printf("    节点%d执行kubectl命令失败: %v\n", i+1, err)
					statusResult.Error += fmt.Sprintf("节点%d执行kubectl失败: %v; ", i+1, err)
					allNodesPodsReady = false
					continue
				}

				// 解析Pod状态
				lines := strings.Split(output.Stdout, "\n")
				nodePodsReady := true
				nodePodCount := 0
				nodeReadyPodCount := 0

				for _, line := range lines {
					line = strings.TrimSpace(line)
					if line == "" {
						continue
					}

					parts := strings.Fields(line)
					if len(parts) >= 4 {
						nodePodCount++
						status := parts[3]
						if status == "Running" || status == "Completed" {
							nodeReadyPodCount++
						} else {
							nodePodsReady = false
						}
					}
				}

				// 更新统计信息
				totalPodCount += nodePodCount
				totalReadyPodCount += nodeReadyPodCount

				if !nodePodsReady {
					allNodesPodsReady = false
					GinkgoWriter.Printf("    节点%d上的部分Pod未就绪 (%d/%d)\n", i+1, nodeReadyPodCount, nodePodCount)
				} else {
					GinkgoWriter.Printf("    节点%d上的所有Pod都已就绪 (%d/%d)\n", i+1, nodeReadyPodCount, nodePodCount)
				}
			}

			statusResult.AllPodsReady = allNodesPodsReady
			statusResult.PodCount = totalPodCount
			statusResult.ReadyPodCount = totalReadyPodCount
			statusResults = append(statusResults, statusResult)

			if allNodesPodsReady {
				GinkgoWriter.Printf("  引导节点%d上的集群所有节点的Pod都已就绪 (总Pod数: %d/%d)\n", result.GuideNodeIndex, totalReadyPodCount, totalPodCount)
			} else {
				GinkgoWriter.Printf("  引导节点%d上的集群部分节点的Pod未就绪 (总Pod数: %d/%d)\n", result.GuideNodeIndex, totalReadyPodCount, totalPodCount)
			}
		}

		// 步骤7: 输出测试结果
		GinkgoWriter.Println("\n步骤7: 输出测试结果")

		// 构建测试结果内容
		resultContent := "\n" + strings.Repeat("=", 60) + "\n"
		resultContent += "===== 多节点集群创建测试结果 =====\n"
		resultContent += strings.Repeat("=", 60) + "\n"
		resultContent += "测试执行时间: " + time.Now().Format("2006-01-02 15:04:05") + "\n"
		resultContent += "\n"

		// 输出集群创建结果
		resultContent += "集群创建结果:\n"
		resultContent += "引导节点\t集群配置\t创建时长\t状态\t\t错误信息\n"
		resultContent += "--------\t--------\t--------\t------\t\t--------\n"

		for _, result := range clusterResults {
			clusterConfig := ""
			switch result.GuideNodeIndex {
			case 1:
				clusterConfig = "1master1worker"
			case 2:
				clusterConfig = "1master2worker"
			case 3:
				clusterConfig = "1master3worker"
			case 4:
				clusterConfig = "1master4worker"
			case 5:
				clusterConfig = "3master2worker"
			}

			status := "成功"
			if !result.Success {
				status = "失败"
			}

			errorMsg := ""
			if result.Error != "" {
				errorMsg = result.Error
			}

			resultContent += fmt.Sprintf("%d\t\t%s\t%s\t%s\t\t%s\n", result.GuideNodeIndex, clusterConfig, result.Duration, status, errorMsg)
		}

		resultContent += "\n"

		// 输出集群状态结果
		resultContent += "集群状态结果:\n"
		resultContent += "引导节点\t集群配置\t所有Pod就绪\tPod总数\t就绪Pod数\t错误信息\n"
		resultContent += "--------\t--------\t----------\t------\t--------\t--------\n"

		for _, statusResult := range statusResults {
			clusterConfig := ""
			switch statusResult.GuideNodeIndex {
			case 1:
				clusterConfig = "1master1worker"
			case 2:
				clusterConfig = "1master2worker"
			case 3:
				clusterConfig = "1master3worker"
			case 4:
				clusterConfig = "1master4worker"
			case 5:
				clusterConfig = "3master2worker"
			}

			allReady := "是"
			if !statusResult.AllPodsReady {
				allReady = "否"
			}

			errorMsg := ""
			if statusResult.Error != "" {
				errorMsg = statusResult.Error
			}

			resultContent += fmt.Sprintf("%d\t\t%s\t%s\t\t%d\t	%d\t	%s\n", statusResult.GuideNodeIndex, clusterConfig, allReady, statusResult.PodCount, statusResult.ReadyPodCount, errorMsg)
		}

		resultContent += strings.Repeat("=", 60) + "\n\n"
		resultContent += "多节点集群创建测试完成！\n"

		// 输出到控制台
		GinkgoWriter.Println(resultContent)

		// 追加到文件
		appendPerformanceResult(resultContent)
	})
})

var _ = Describe("业务集群资源占用测试", Label("business-cluster-resource", "skip-temporarily", "online"), func() {
	var (
		businessNodeExecutor *executor.SSHExecutor
	)

	BeforeEach(func() {
		// 创建业务节点的 SSH 执行器
		var err error
		businessHost := os.Getenv("BUSINESS_NODE_HOST")
		businessPortStr := os.Getenv("BUSINESS_NODE_PORT")
		businessPort, _ := strconv.Atoi(businessPortStr)
		businessUser := os.Getenv("BUSINESS_NODE_USER")
		businessPassword := os.Getenv("BUSINESS_NODE_PASSWORD")

		businessNodeExecutor, err = executor.NewSSHExecutor(businessHost, businessPort, businessUser, businessPassword)
		Expect(err).NotTo(HaveOccurred(), "应该能够创建到业务节点的SSH连接")
	})

	AfterEach(func() {
		// 关闭业务节点的SSH连接
		if businessNodeExecutor != nil {
			businessNodeExecutor.Close()
		}
	})

	It("should check resource usage on business node", func() {
		var err error

		GinkgoWriter.Println("=====业务集群资源占用测试=====")

		// 统计业务节点的资源占用情况
		resourceResultContent := "\n" + strings.Repeat("=", 60) + "\n"
		resourceResultContent += "===== 集群资源占用统计结果 =====\n"
		resourceResultContent += strings.Repeat("=", 60) + "\n"

		// 使用kubectl top pod -A命令获取资源使用情况
		topCmd := "kubectl top pod -A --no-headers"
		topResult, err := businessNodeExecutor.Exec(topCmd)
		Expect(err).NotTo(HaveOccurred(), "应该能够执行kubectl top命令")

		// 打印原始输出内容，用于调试
		GinkgoWriter.Printf("原始kubectl top输出:\n%s\n", topResult.Stdout)

		if topResult.Stderr != "" {
			GinkgoWriter.Printf("kubectl top命令的错误信息:\n%s\n", topResult.Stderr)
		}

		// 解析输出并统计资源占用
		type NamespaceResource struct {
			CPU         string
			Memory      string
			CPUValue    int
			MemoryValue int
		}

		namespaceStats := make(map[string]NamespaceResource)
		totalCPUValue := 0
		totalMemoryValue := 0

		lines := strings.Split(strings.TrimSpace(topResult.Stdout), "\n")
		GinkgoWriter.Printf("解析到 %d 行pod信息\n", len(lines))

		for i, line := range lines {
			if strings.TrimSpace(line) == "" {
				GinkgoWriter.Printf("第 %d 行为空，跳过\n", i+1)
				continue
			}

			fields := strings.Fields(line)
			if len(fields) < 4 {
				GinkgoWriter.Printf("第 %d 行字段不足4个，跳过: %s\n", i+1, line)
				continue
			}

			GinkgoWriter.Printf("处理第 %d 行: %s\n", i+1, line)

			namespace := fields[0]
			podName := fields[1]
			cpuStr := fields[2]
			memoryStr := fields[3]

			GinkgoWriter.Printf("  命名空间: %s, Pod名称: %s, CPU: %s, 内存: %s\n", namespace, podName, cpuStr, memoryStr)

			// 解析CPU值 (只支持m单位)
			var cpuValue int
			if strings.HasSuffix(cpuStr, "m") {
				cpuStr = strings.TrimSuffix(cpuStr, "m")
				cpuValue, err = strconv.Atoi(cpuStr)
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("解析CPU值失败: %s", cpuStr))
				GinkgoWriter.Printf("  CPU解析结果: %d m\n", cpuValue)
			} else {
				GinkgoWriter.Printf("CPU值格式错误，只支持m单位: %s\n", cpuStr)
				Fail("CPU值格式错误")
			}

			// 解析内存值 (只支持Mi单位)
			var memoryValue int
			if strings.HasSuffix(memoryStr, "Mi") {
				memoryStr = strings.TrimSuffix(memoryStr, "Mi")
				memoryValue, err = strconv.Atoi(memoryStr)
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("解析内存值失败: %s", memoryStr))
				GinkgoWriter.Printf("  内存解析结果: %d Mi\n", memoryValue)
			} else {
				GinkgoWriter.Printf("内存值格式错误，只支持Mi单位: %s\n", memoryStr)
				Fail("内存值格式错误")
			}

			// 按命名空间汇总
			nsResource := namespaceStats[namespace]
			nsResource.CPU = cpuStr + "m"
			nsResource.Memory = memoryStr + "Mi"
			nsResource.CPUValue += cpuValue
			nsResource.MemoryValue += memoryValue
			namespaceStats[namespace] = nsResource
			GinkgoWriter.Printf("  命名空间 %s 当前累计: CPU=%d m, Memory=%d Mi\n", namespace, nsResource.CPUValue, nsResource.MemoryValue)

			// 总汇总
			totalCPUValue += cpuValue
			totalMemoryValue += memoryValue
		}

		// 输出各命名空间资源占用统计
		resourceResultContent += "\n各命名空间资源占用统计:\n"
		resourceResultContent += "命名空间\tCPU(m)\t内存(Mi)\n"
		for ns, resource := range namespaceStats {
			resourceResultContent += ns + "\t" + strconv.Itoa(resource.CPUValue) + "\t" + strconv.Itoa(resource.MemoryValue) + "\n"
		}

		// 输出总资源占用统计
		resourceResultContent += "\n总资源占用统计:\n"
		resourceResultContent += "总CPU占用: " + strconv.Itoa(totalCPUValue) + " m\n"
		resourceResultContent += "总内存占用: " + strconv.Itoa(totalMemoryValue) + " Mi\n"
		resourceResultContent += strings.Repeat("=", 60) + "\n\n"
		resourceResultContent += "集群资源占用统计完成！\n"

		// 输出到控制台
		GinkgoWriter.Println(resourceResultContent)

		// 追加到文件
		appendPerformanceResult(resourceResultContent)
	})
})
