package system_integration

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

var _ = Describe("Workload Cluster Load Testing", Label("system_integration_etcd_apiserver_load_test", "with-management-cluster", "system-integration", "online", "skip-temporarily"), func() {
	var (
		sshClient     *executor.SSHExecutor
		localexecutor *executor.LocalExecutor
		k8sClient     *k8s.K8SClient
		testCtx       context.Context
		testCancel    context.CancelFunc
	)

	BeforeEach(func() {
		// 初始化测试上下文
		testCtx, testCancel = context.WithTimeout(context.Background(), 15*time.Minute)

		var err error

		// ssh client
		if isDaily {
			localexecutor = executor.NewLocalExecutor(10 * time.Minute)
		} else {
			port := 22
			if portStr := os.Getenv("TEST_NODE5_PORT"); portStr != "" {
				if portInt, err := strconv.Atoi(portStr); err == nil {
					port = portInt
				}
			}
			sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE5_IP"), port, os.Getenv("TEST_NODE5_USER"), os.Getenv("TEST_NODE5_PASSWORD"))
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "ssh Client 应当成功创建")
		}

		// k8s client
		if isDaily {
			k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
		} else {
			k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

		installEtcdctlIfNeeded(sshClient, localexecutor)

		GinkgoWriter.Printf("业务集群负载测试初始化完成\n")
	})

	AfterEach(func() {
		// 清理资源
		if testCancel != nil {
			testCancel()
		}
	})

	It("应该能够测试etcd的读写性能和稳定性", Label("required_nodes:3"), func() {
		By("收集etcd集群信息")
		etcdInfo := collectEtcdInfo(testCtx, k8sClient, sshClient, localexecutor)
		GinkgoWriter.Printf("etcd集群信息:\n")
		GinkgoWriter.Printf("  版本: %s\n", etcdInfo.Version)
		GinkgoWriter.Printf("  成员数量: %d\n", etcdInfo.MemberCount)
		GinkgoWriter.Printf("  存储大小: %s\n", etcdInfo.StorageSize)
		GinkgoWriter.Printf("  存储配额: %s\n", etcdInfo.StorageQuota)
		GinkgoWriter.Printf("  数据库大小: %s\n", etcdInfo.DbSize)
		GinkgoWriter.Printf("  修订版本: %d\n", etcdInfo.Revision)

		By("测试etcd集群健康状态")
		etcdHealth := testEtcdHealth(testCtx, k8sClient, sshClient, localexecutor)
		Expect(etcdHealth.IsHealthy).To(BeTrue(), "etcd集群应该健康")
		GinkgoWriter.Printf("etcd集群健康状态: 健康\n")
		GinkgoWriter.Printf("  领导者: %s\n", etcdHealth.Leader)
		GinkgoWriter.Printf("  每个成员的RaftTerm: %v\n", etcdHealth.RaftTerms)

		By("执行etcd读写性能基准测试")
		etcdBenchmark := benchmarkEtcdPerformance(testCtx, k8sClient, sshClient, localexecutor)

		GinkgoWriter.Printf("etcd性能基准测试结果:\n")
		GinkgoWriter.Printf("  写入性能: %.2f 次操作/秒\n", etcdBenchmark.WriteOps)
		GinkgoWriter.Printf("  读取性能: %.2f 次操作/秒\n", etcdBenchmark.ReadOps)
		GinkgoWriter.Printf("  延迟统计:\n")
		GinkgoWriter.Printf("    平均写入延迟: %.2f 毫秒\n", etcdBenchmark.WriteLatencyAvg)
		GinkgoWriter.Printf("    写入延迟P95: %.2f 毫秒\n", etcdBenchmark.WriteLatencyP95)
		GinkgoWriter.Printf("    平均读取延迟: %.2f 毫秒\n", etcdBenchmark.ReadLatencyAvg)
		GinkgoWriter.Printf("    读取延迟P95: %.2f 毫秒\n", etcdBenchmark.ReadLatencyP95)

		// 性能断言
		// Expect(etcdBenchmark.WriteOps).To(BeNumerically(">=", 100),
		// 	"etcd写入性能应该至少100 ops/秒")
		// Expect(etcdBenchmark.ReadOps).To(BeNumerically(">=", 1000),
		// 	"etcd读取性能应该至少1000 ops/秒")
		// Expect(etcdBenchmark.WriteLatencyP95).To(BeNumerically("<", 100),
		// 	"etcd写入延迟P95应该小于100毫秒")

		By("模拟正常使用负载")
		etcdLoadTest := simulateEtcdNormalLoad(testCtx, k8sClient, sshClient, localexecutor)

		GinkgoWriter.Printf("etcd正常负载测试结果:\n")
		GinkgoWriter.Printf("  总操作数: %d\n", etcdLoadTest.TotalOps)
		GinkgoWriter.Printf("  成功操作: %d (%.1f%%)\n",
			etcdLoadTest.SuccessfulOps,
			float64(etcdLoadTest.SuccessfulOps)/float64(etcdLoadTest.TotalOps)*100)
		GinkgoWriter.Printf("  操作速率: %.1f ops/秒\n", etcdLoadTest.OpsPerSecond)
		GinkgoWriter.Printf("  最大内存使用: %d MB\n", etcdLoadTest.MaxMemoryMB)
		GinkgoWriter.Printf("  平均CPU使用: %.1f%%\n", etcdLoadTest.AvgCPUPercent)

		Expect(etcdLoadTest.SuccessRate).To(BeNumerically(">=", 99.0),
			"etcd操作成功率应该至少80%%")
		// Expect(etcdLoadTest.OpsPerSecond).To(BeNumerically(">=", 50),
		// 	"etcd操作速率应该至少50 ops/秒")
	})

	It("应该能够测试API Server的请求处理能力和稳定性", Label("required_nodes:3"), func() {
		By("收集API Server信息")
		apiServerInfo := collectAPIServerInfo(testCtx, k8sClient.Clientset, sshClient, localexecutor, isDaily)

		GinkgoWriter.Printf("API Server信息:\n")
		GinkgoWriter.Printf("  版本: %s\n", apiServerInfo.Version)
		GinkgoWriter.Printf("  实例数量: %d\n", apiServerInfo.InstanceCount)
		GinkgoWriter.Printf("  运行模式: %s\n", apiServerInfo.Mode)
		GinkgoWriter.Printf("  启用的API版本: %v\n", apiServerInfo.APIVersions)

		By("测试API Server健康状态")
		apiHealth := testAPIServerHealth(testCtx, k8sClient.Clientset, sshClient, localexecutor, isDaily)
		Expect(apiHealth.IsHealthy).To(BeTrue(), "API Server应该健康")
		GinkgoWriter.Printf("API Server健康状态: 健康\n")
		GinkgoWriter.Printf("  响应时间: %.2f 毫秒\n", apiHealth.ResponseTimeMs)
		GinkgoWriter.Printf("  请求成功率: %.1f%%\n", apiHealth.SuccessRate)

		By("执行API Server并发请求测试")
		concurrentTest := testAPIServerConcurrency(testCtx, k8sClient.Clientset, sshClient, localexecutor, isDaily)

		GinkgoWriter.Printf("API Server并发测试结果:\n")
		GinkgoWriter.Printf("  并发级别: %d\n", concurrentTest.ConcurrencyLevel)
		GinkgoWriter.Printf("  总请求数: %d\n", concurrentTest.TotalRequests)
		GinkgoWriter.Printf("  成功请求: %d (%.1f%%)\n",
			concurrentTest.SuccessfulRequests,
			float64(concurrentTest.SuccessfulRequests)/float64(concurrentTest.TotalRequests)*100)
		GinkgoWriter.Printf("  吞吐量: %.1f 请求/秒\n", concurrentTest.Throughput)
		GinkgoWriter.Printf("  延迟统计:\n")
		GinkgoWriter.Printf("    平均延迟: %.2f 毫秒\n", concurrentTest.AvgLatencyMs)
		GinkgoWriter.Printf("    P95延迟: %.2f 毫秒\n", concurrentTest.P95LatencyMs)
		GinkgoWriter.Printf("    P99延迟: %.2f 毫秒\n", concurrentTest.P99LatencyMs)

		Expect(concurrentTest.SuccessRate).To(BeNumerically(">=", 80.0),
			"API Server请求成功率应该至少80%%")
		Expect(concurrentTest.AvgLatencyMs).To(BeNumerically("<", 1000),
			"API Server平均延迟应该小于1000毫秒")

		By("模拟正常集群操作负载")
		normalLoadTest := simulateNormalClusterOperations(testCtx, k8sClient.Clientset, sshClient, localexecutor, isDaily)

		GinkgoWriter.Printf("正常集群操作负载测试结果:\n")
		GinkgoWriter.Printf("  创建操作: %d\n", normalLoadTest.CreateOperations)
		GinkgoWriter.Printf("  读取操作: %d\n", normalLoadTest.ReadOperations)
		GinkgoWriter.Printf("  更新操作: %d\n", normalLoadTest.UpdateOperations)
		GinkgoWriter.Printf("  删除操作: %d\n", normalLoadTest.DeleteOperations)
		GinkgoWriter.Printf("  总操作数: %d\n", normalLoadTest.TotalOperations)
		GinkgoWriter.Printf("  操作成功率: %.1f%%\n", normalLoadTest.SuccessRate)
		GinkgoWriter.Printf("  平均吞吐量: %.1f ops/秒\n", normalLoadTest.AvgThroughput)

		Expect(normalLoadTest.SuccessRate).To(BeNumerically(">=", 80.0),
			"集群操作成功率应该至少80%%")
	})
})

// 辅助函数 - etcd相关
//
//	检查并安装etcdctl
func installEtcdctlIfNeeded(sshClient *executor.SSHExecutor, localexecutor *executor.LocalExecutor) {
	By("检查etcdctl是否已安装")

	// 检查etcdctl是否可用
	var result *executor.ExecResult
	var err error
	var whichCmd = "which etcdctl"
	if isDaily {
		result, err = localexecutor.Exec(whichCmd)
	} else {
		result, err = sshClient.Exec(whichCmd)
	}
	if err == nil && result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != "" {
		// 检查版本
		var versionResult *executor.ExecResult
		var verr error
		if isDaily {
			versionResult, verr = localexecutor.Exec("etcdctl version")
		} else {
			versionResult, verr = sshClient.Exec("etcdctl version")
		}
		if verr == nil && versionResult.ExitCode == 0 {
			GinkgoWriter.Printf("etcdctl已安装: %s\n", strings.TrimSpace(versionResult.Stdout))
			return
		}
	}

	GinkgoWriter.Printf("etcdctl未安装或不可用，开始安装...\n")

	// 创建临时目录
	if isDaily {
		result, err = localexecutor.Exec("mkdir -p /tmp/etcd-install")
	} else {
		result, err = sshClient.Exec("mkdir -p /tmp/etcd-install")
	}
	Expect(err).NotTo(HaveOccurred(), "应该能够创建临时目录")
	Expect(result.ExitCode).To(Equal(0), "创建临时目录命令失败")

	// 检测系统架构
	if isDaily {
		result, err = localexecutor.Exec("uname -m")
	} else {
		result, err = sshClient.Exec("uname -m")
	}
	Expect(err).NotTo(HaveOccurred(), "应该能够检测系统架构")
	Expect(result.ExitCode).To(Equal(0), "检测系统架构命令失败")

	arch := strings.TrimSpace(result.Stdout)
	GinkgoWriter.Printf("当前系统架构: %s\n", arch)

	// 设置etcd版本和下载URL
	etcdVer := "v3.5.21"
	var etcdURL string
	var archName string

	if arch == "aarch64" || arch == "arm64" {
		etcdURL = fmt.Sprintf("https://github.com/etcd-io/etcd/releases/download/%s/etcd-%s-linux-arm64.tar.gz", etcdVer, etcdVer)
		archName = "arm64"
	} else if arch == "x86_64" || arch == "amd64" {
		etcdURL = fmt.Sprintf("https://github.com/etcd-io/etcd/releases/download/%s/etcd-%s-linux-amd64.tar.gz", etcdVer, etcdVer)
		archName = "amd64"
	} else {
		Fail(fmt.Sprintf("不支持的架构: %s", arch))
	}

	GinkgoWriter.Printf("下载etcdctl: %s\n", etcdURL)

	// 下载etcd
	downloadCmd := fmt.Sprintf("cd /tmp/etcd-install && wget -q %s -O etcd.tar.gz", etcdURL)
	if isDaily {
		result, err = localexecutor.Exec(downloadCmd)
	} else {
		result, err = sshClient.Exec(downloadCmd)
	}
	if err != nil || result.ExitCode != 0 {
		// 尝试使用curl
		downloadCmd = fmt.Sprintf("cd /tmp/etcd-install && curl -L %s -o etcd.tar.gz", etcdURL)
		if isDaily {
			result, err = localexecutor.Exec(downloadCmd)
		} else {
			result, err = sshClient.Exec(downloadCmd)
		}
		Expect(err).NotTo(HaveOccurred(), "应该能够下载etcd")
		Expect(result.ExitCode).To(Equal(0), "下载etcd失败")
	}

	// 解压
	extractCmd := "cd /tmp/etcd-install && tar xzf etcd.tar.gz"
	if isDaily {
		result, err = localexecutor.Exec(extractCmd)
	} else {
		result, err = sshClient.Exec(extractCmd)
	}
	Expect(err).NotTo(HaveOccurred(), "应该能够解压etcd")
	Expect(result.ExitCode).To(Equal(0), "解压etcd失败")

	// 安装到/usr/local/bin
	installCmd := fmt.Sprintf("cp /tmp/etcd-install/etcd-%s-linux-%s/etcdctl /usr/local/bin/ && chmod +x /usr/local/bin/etcdctl", etcdVer, archName)
	if isDaily {
		result, err = localexecutor.Exec(installCmd)
	} else {
		result, err = sshClient.Exec(installCmd)
	}
	Expect(err).NotTo(HaveOccurred(), "应该能够安装etcdctl")
	Expect(result.ExitCode).To(Equal(0), "安装etcdctl失败")

	// 验证安装
	verifyCmd := "etcdctl version"
	if isDaily {
		result, err = localexecutor.Exec(verifyCmd)
	} else {
		result, err = sshClient.Exec(verifyCmd)
	}
	Expect(err).NotTo(HaveOccurred(), "应该能够验证etcdctl安装")
	Expect(result.ExitCode).To(Equal(0), "etcdctl安装验证失败")

	GinkgoWriter.Printf("etcdctl安装成功: %s\n", strings.TrimSpace(result.Stdout))

	// 清理临时文件
	cleanupCmd := "rm -rf /tmp/etcd-install"
	if isDaily {
		localexecutor.Exec(cleanupCmd)
	} else {
		sshClient.Exec(cleanupCmd)
	}
}

// 结构
type EtcdInfo struct {
	Version      string
	MemberCount  int
	StorageSize  string
	StorageQuota string
	DbSize       string
	Revision     int64
	Endpoint     string
}

// 设置etcdctl命令基础参数
var etcdctlBaseCmd = "ETCDCTL_API=3 etcdctl " +
	"--endpoints=https://127.0.0.1:2379 " +
	"--cacert=/etc/kubernetes/pki/etcd/ca.crt " +
	"--cert=/etc/kubernetes/pki/etcd/server.crt " +
	"--key=/etc/kubernetes/pki/etcd/server.key "

func collectEtcdInfo(ctx context.Context, k8sClient *k8s.K8SClient, sshClient *executor.SSHExecutor, localexecutor *executor.LocalExecutor) EtcdInfo {
	var info EtcdInfo
	info.Endpoint = "https://127.0.0.1:2379"

	// 获取etcd版本
	versionCmd := etcdctlBaseCmd + "version"
	var result *executor.ExecResult
	var err error
	if isDaily {
		result, err = localexecutor.Exec(versionCmd)
	} else {
		result, err = sshClient.Exec(versionCmd)
	}
	if err == nil && result.ExitCode == 0 {
		lines := strings.Split(result.Stdout, "\n")
		for _, line := range lines {
			if strings.Contains(line, "etcd Version:") {
				parts := strings.Split(line, ":")
				if len(parts) > 1 {
					info.Version = strings.TrimSpace(parts[1])
				}
			}
		}
	}

	// 获取etcd成员信息
	memberCmd := etcdctlBaseCmd + "member list --write-out=table"
	if isDaily {
		result, err = localexecutor.Exec(memberCmd)
	} else {
		result, err = sshClient.Exec(memberCmd)
	}
	if err == nil && result.ExitCode == 0 {
		lines := strings.Split(result.Stdout, "\n")
		// 跳过表头
		for i := 1; i < len(lines); i++ {
			if strings.TrimSpace(lines[i]) != "" {
				info.MemberCount++
			}
		}
	}

	// 获取etcd endpoint信息
	statusCmd := etcdctlBaseCmd + "endpoint status --write-out=json"
	if isDaily {
		result, err = localexecutor.Exec(statusCmd)
	} else {
		result, err = sshClient.Exec(statusCmd)
	}
	if err == nil && result.ExitCode == 0 {
		var status []map[string]interface{}
		if err := json.Unmarshal([]byte(result.Stdout), &status); err == nil && len(status) > 0 {
			if dbSize, ok := status[0]["DbSize"].(float64); ok {
				info.DbSize = fmt.Sprintf("%.2f MB", dbSize/1024/1024)
			}
			if revision, ok := status[0]["Status"].(map[string]interface{})["header"].(map[string]interface{})["revision"].(float64); ok {
				info.Revision = int64(revision)
			}
		}
	}

	// 获取etcd Pod的资源限制
	pods, err := k8sClient.Clientset.CoreV1().Pods("kube-system").List(ctx, metav1.ListOptions{
		LabelSelector: "component=etcd",
	})
	if err == nil && len(pods.Items) > 0 {
		etcdPod := pods.Items[0]
		for _, container := range etcdPod.Spec.Containers {
			if container.Name == "etcd" {
				if container.Resources.Limits != nil {
					if memory, ok := container.Resources.Limits["memory"]; ok {
						info.StorageSize = memory.String()
					}
				}
			}
		}
	}

	return info
}

type EtcdHealth struct {
	IsHealthy bool
	Leader    string
	RaftTerms map[string]int64
}

func testEtcdHealth(ctx context.Context, k8sClient *k8s.K8SClient, sshClient *executor.SSHExecutor, localexecutor *executor.LocalExecutor) EtcdHealth {
	var health EtcdHealth
	health.RaftTerms = make(map[string]int64)
	health.IsHealthy = false

	// 检查etcd集群健康（加长 command-timeout，避免默认 5s 在慢 IO 流水线下误判）
	healthCmd := etcdctlBaseCmd + "endpoint health --command-timeout=15s"

	// 轮询重试：总超时 3 分钟、间隔 15s，覆盖流水线瞬时抖动（compaction/leader 切换/慢 IO 等）
	Eventually(func() bool {
		var result *executor.ExecResult
		var err error
		if isDaily {
			result, err = localexecutor.Exec(healthCmd)
		} else {
			result, err = sshClient.Exec(healthCmd)
		}
		if result == nil {
			// 执行器层面失败（SSH 会话/本地进程无法启动），拿不到执行结果
			GinkgoWriter.Printf("etcd health 执行失败(无结果): %v\n", err)
			return false
		}
		// 注意：executor 在非零退出码时同时返回 (result, ExitError)，以 ExitCode 为准
		if result.ExitCode != 0 {
			GinkgoWriter.Printf("etcd health 失败 err=%v exit=%d\nstdout: %s\nstderr: %s\n",
				err, result.ExitCode, result.Stdout, result.Stderr)
			return false
		}
		// exit code=0 即健康；etcd 3.5 的健康输出在 stderr，stdout/stderr 解析仅作日志
		output := result.Stdout + "\n" + result.Stderr
		for _, line := range strings.Split(output, "\n") {
			if strings.Contains(line, "is healthy") {
				parts := strings.Split(line, " ")
				if len(parts) > 0 {
					GinkgoWriter.Printf("etcd端点 %s 健康\n", parts[0])
				}
			}
		}
		health.IsHealthy = true
		return true
	}, 3*time.Minute, 15*time.Second).Should(BeTrue(), "etcd集群应在3分钟内恢复健康")

	// 获取领导者信息
	statusCmd := etcdctlBaseCmd + "endpoint status --write-out=json"
	var result *executor.ExecResult
	var err error
	if isDaily {
		result, err = localexecutor.Exec(statusCmd)
	} else {
		result, err = sshClient.Exec(statusCmd)
	}
	if err == nil && result.ExitCode == 0 {
		var status []map[string]interface{}
		if err := json.Unmarshal([]byte(result.Stdout), &status); err == nil {
			for _, endpoint := range status {
				if member, ok := endpoint["Status"].(map[string]interface{})["leader"].(map[string]interface{}); ok {
					if leaderName, ok := member["name"].(string); ok {
						health.Leader = leaderName
					}
				}
			}
		}
	}

	return health
}

type EtcdBenchmark struct {
	WriteOps        float64
	ReadOps         float64
	WriteLatencyAvg float64
	WriteLatencyP95 float64
	ReadLatencyAvg  float64
	ReadLatencyP95  float64
}

func benchmarkEtcdPerformance(ctx context.Context, k8sClient *k8s.K8SClient, sshClient *executor.SSHExecutor, localexecutor *executor.LocalExecutor) EtcdBenchmark {
	var benchmark EtcdBenchmark

	// 使用etcd内置的benchmark工具进行测试
	GinkgoWriter.Printf("执行etcd基准测试...\n")
	// 简单的写入测试（100个键值对）
	startTime := time.Now()
	var execResult *executor.ExecResult
	var err error
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("/benchmark/key-%d", i)
		value := fmt.Sprintf("value-%d", i)
		cmd := fmt.Sprintf("%s put %s %s", etcdctlBaseCmd, key, value)

		if isDaily && localexecutor != nil {
			execResult, err = localexecutor.Exec(cmd)
		} else if sshClient != nil {
			execResult, err = sshClient.Exec(cmd)
		}
		if err != nil || execResult.ExitCode != 0 {
			GinkgoWriter.Printf("写入失败: %s\n", key)
		}
	}
	writeDuration := time.Since(startTime).Seconds()
	benchmark.WriteOps = 100.0 / writeDuration

	// 简单的读取测试
	startTime = time.Now()
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("/benchmark/key-%d", i%100)
		cmd := fmt.Sprintf("%s get %s", etcdctlBaseCmd, key)

		if isDaily && localexecutor != nil {
			execResult, err = localexecutor.Exec(cmd)
		} else if sshClient != nil {
			execResult, err = sshClient.Exec(cmd)
		} // 忽略结果，只测量时间
	}
	readDuration := time.Since(startTime).Seconds()
	benchmark.ReadOps = 1000.0 / readDuration

	// 清理测试数据
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("/benchmark/key-%d", i)
		cmd := fmt.Sprintf("%s del %s", etcdctlBaseCmd, key)

		if isDaily && localexecutor != nil {
			execResult, err = localexecutor.Exec(cmd)
		} else if sshClient != nil {
			execResult, err = sshClient.Exec(cmd)
		}
	}

	return benchmark
}

type EtcdLoadTestResult struct {
	TotalOps      int
	SuccessfulOps int
	SuccessRate   float64
	OpsPerSecond  float64
	MaxMemoryMB   int
	AvgCPUPercent float64
}

// 模拟正常的etcd操作负载 - 支持根据isDaily选择执行器
func simulateEtcdNormalLoad(ctx context.Context, k8sClient *k8s.K8SClient, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor) EtcdLoadTestResult {
	var result EtcdLoadTestResult

	// 模拟正常的etcd操作负载
	startTime := time.Now()
	operations := 100
	var execResult *executor.ExecResult
	var err error
	for i := 0; i < operations; i++ {
		// 模拟不同类型的操作
		operation := i % 4
		switch operation {
		case 0: // 写入配置
			key := fmt.Sprintf("/config/app-%d", i)
			value := fmt.Sprintf(`{"version": "1.0.%d", "enabled": true}`, i)
			cmd := fmt.Sprintf("%s put %s '%s'", etcdctlBaseCmd, key, value)
			if isDaily && localExecutor != nil {
				execResult, err = localExecutor.Exec(cmd)
			} else if sshExecutor != nil {
				execResult, err = sshExecutor.Exec(cmd)
			}
			if err == nil && execResult.ExitCode == 0 {
				result.SuccessfulOps++
			}

		case 1: // 读取配置
			key := fmt.Sprintf("/config/app-%d", i/2)
			cmd := fmt.Sprintf("%s get %s", etcdctlBaseCmd, key)
			if isDaily && localExecutor != nil {
				execResult, err = localExecutor.Exec(cmd)
			} else if sshExecutor != nil {
				execResult, err = sshExecutor.Exec(cmd)
			}
			if err == nil && execResult.ExitCode == 0 {
				result.SuccessfulOps++
			}

		case 2: // 写入租约
			cmd := fmt.Sprintf("%s lease grant 60", etcdctlBaseCmd)
			if isDaily && localExecutor != nil {
				execResult, err = localExecutor.Exec(cmd)
			} else if sshExecutor != nil {
				execResult, err = sshExecutor.Exec(cmd)
			}
			if err == nil && execResult.ExitCode == 0 {
				result.SuccessfulOps++
			}

		case 3: // 范围查询
			cmd := fmt.Sprintf("%s get --prefix /config/ --limit=5", etcdctlBaseCmd)
			if isDaily && localExecutor != nil {
				execResult, err = localExecutor.Exec(cmd)
			} else if sshExecutor != nil {
				execResult, err = sshExecutor.Exec(cmd)
			}
			if err == nil && execResult.ExitCode == 0 {
				result.SuccessfulOps++
			}
		}

		result.TotalOps++

		// 稍微延迟，模拟正常负载
		if i%10 == 0 {
			time.Sleep(50 * time.Millisecond)
		}
	}

	duration := time.Since(startTime).Seconds()
	result.OpsPerSecond = float64(result.TotalOps) / duration
	result.SuccessRate = float64(result.SuccessfulOps) / float64(result.TotalOps) * 100

	// 清理测试数据
	for i := 0; i < operations/4; i++ {
		key := fmt.Sprintf("/config/app-%d", i)
		cmd := fmt.Sprintf("%s del %s", etcdctlBaseCmd, key)
		if isDaily && localExecutor != nil {
			localExecutor.Exec(cmd)
		} else if sshExecutor != nil {
			sshExecutor.Exec(cmd)
		}

		key = fmt.Sprintf("/lease/key-%d", i*4+2)
		cmd = fmt.Sprintf("%s del %s", etcdctlBaseCmd, key)
		if isDaily && localExecutor != nil {
			localExecutor.Exec(cmd)
		} else if sshExecutor != nil {
			sshExecutor.Exec(cmd)
		}
	}

	return result
}

// 辅助函数 - API Server相关
type APIServerInfo struct {
	Version       string
	InstanceCount int
	Mode          string
	APIVersions   []string
}

func collectAPIServerInfo(ctx context.Context, clientset *kubernetes.Clientset, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool) APIServerInfo {
	var info APIServerInfo

	// 获取API Server版本
	serverVersion, err := clientset.Discovery().ServerVersion()
	if err == nil {
		info.Version = serverVersion.GitVersion
	}

	// 获取API Server实例数量
	apiPods, err := clientset.CoreV1().Pods("kube-system").List(ctx, metav1.ListOptions{
		LabelSelector: "component=kube-apiserver",
	})
	if err == nil {
		info.InstanceCount = len(apiPods.Items)

		// 检查运行模式
		if info.InstanceCount > 1 {
			info.Mode = "高可用模式"
		} else {
			info.Mode = "单实例模式"
		}
	}

	// 获取启用的API版本
	apiGroups, err := clientset.Discovery().ServerGroups()
	if err == nil {
		for _, group := range apiGroups.Groups {
			for _, version := range group.Versions {
				info.APIVersions = append(info.APIVersions, version.Version)
			}
		}
	}

	return info
}

type APIServerHealth struct {
	IsHealthy      bool
	ResponseTimeMs float64
	SuccessRate    float64
}

func testAPIServerHealth(ctx context.Context, clientset *kubernetes.Clientset, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool) APIServerHealth {
	var health APIServerHealth
	health.IsHealthy = true

	// 测试API Server响应时间和可用性
	tests := []string{
		"/api/v1/namespaces",
		"/apis/apps/v1/namespaces/default/deployments",
		"/healthz",
		"/readyz",
		"/livez",
	}

	successCount := 0
	var totalResponseTime time.Duration
	var result *executor.ExecResult
	var err error
	for _, endpoint := range tests {
		startTime := time.Now()

		// 使用kubectl get --raw测试端点
		cmd := fmt.Sprintf("kubectl get --raw '%s' 2>/dev/null || echo 'FAILED'", endpoint)

		if isDaily && localExecutor != nil {
			result, err = localExecutor.Exec(cmd)
		} else if sshExecutor != nil {
			result, err = sshExecutor.Exec(cmd)
		}

		responseTime := time.Since(startTime)
		totalResponseTime += responseTime

		if err == nil && result.ExitCode == 0 && !strings.Contains(result.Stdout, "FAILED") {
			successCount++
			GinkgoWriter.Printf("端点 %s: 成功 (响应时间: %v)\n", endpoint, responseTime)
		} else {
			GinkgoWriter.Printf("端点 %s: 失败\n", endpoint)
			health.IsHealthy = false
		}
	}

	health.ResponseTimeMs = float64(totalResponseTime.Milliseconds()) / float64(len(tests))
	health.SuccessRate = float64(successCount) / float64(len(tests)) * 100

	return health
}

type ConcurrencyTestResult struct {
	ConcurrencyLevel   int
	TotalRequests      int
	SuccessfulRequests int
	SuccessRate        float64
	Throughput         float64
	AvgLatencyMs       float64
	P95LatencyMs       float64
	P99LatencyMs       float64
}

func testAPIServerConcurrency(ctx context.Context, clientset *kubernetes.Clientset, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool) ConcurrencyTestResult {
	var result ConcurrencyTestResult
	result.ConcurrencyLevel = 10
	result.TotalRequests = 100

	// 使用简单的并发测试
	var wg sync.WaitGroup
	var mu sync.Mutex
	var latencies []time.Duration

	startTime := time.Now()

	for i := 0; i < result.ConcurrencyLevel; i++ {
		wg.Add(1)
		go func(workerID int, sshExe *executor.SSHExecutor, localExe *executor.LocalExecutor, daily bool) {
			defer wg.Done()

			for j := 0; j < result.TotalRequests/result.ConcurrencyLevel; j++ {
				requestStart := time.Now()
				var execResult *executor.ExecResult
				var err error
				// 执行API请求
				cmd := "kubectl get pods --all-namespaces --request-timeout=5s 2>&1 | head -5"

				if daily && localExe != nil {
					execResult, err = localExe.Exec(cmd)
				} else if sshExe != nil {
					execResult, err = sshExe.Exec(cmd)
				}

				latency := time.Since(requestStart)

				mu.Lock()
				latencies = append(latencies, latency)
				if err == nil && execResult.ExitCode == 0 {
					result.SuccessfulRequests++
				}
				mu.Unlock()

				// 稍微延迟避免过载
				time.Sleep(10 * time.Millisecond)
			}
		}(i, sshExecutor, localExecutor, isDaily)
	}

	wg.Wait()

	duration := time.Since(startTime).Seconds()
	result.Throughput = float64(result.TotalRequests) / duration
	result.SuccessRate = float64(result.SuccessfulRequests) / float64(result.TotalRequests) * 100

	// 计算延迟统计
	if len(latencies) > 0 {
		var totalLatency time.Duration
		for _, latency := range latencies {
			totalLatency += latency
		}
		result.AvgLatencyMs = float64(totalLatency.Milliseconds()) / float64(len(latencies))

		// 排序计算百分位数
		sort.Slice(latencies, func(i, j int) bool {
			return latencies[i] < latencies[j]
		})

		p95Index := int(float64(len(latencies)) * 0.95)
		p99Index := int(float64(len(latencies)) * 0.99)

		if p95Index < len(latencies) {
			result.P95LatencyMs = float64(latencies[p95Index].Milliseconds())
		}
		if p99Index < len(latencies) {
			result.P99LatencyMs = float64(latencies[p99Index].Milliseconds())
		}
	}

	return result
}

type NormalLoadTestResult struct {
	CreateOperations int
	ReadOperations   int
	UpdateOperations int
	DeleteOperations int
	TotalOperations  int
	SuccessRate      float64
	AvgThroughput    float64
}

func simulateNormalClusterOperations(ctx context.Context, clientset *kubernetes.Clientset, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool) NormalLoadTestResult {
	var result NormalLoadTestResult

	// 创建测试命名空间
	testNamespace := fmt.Sprintf("load-test-%d", time.Now().Unix())
	createCmd := fmt.Sprintf("kubectl create namespace %s", testNamespace)
	if isDaily && localExecutor != nil {
		localExecutor.Exec(createCmd)
	} else if sshExecutor != nil {
		sshExecutor.Exec(createCmd)
	}

	defer func() {
		// 清理测试命名空间
		cleanupCmd := fmt.Sprintf("kubectl delete namespace %s --wait=false", testNamespace)
		if isDaily && localExecutor != nil {
			localExecutor.Exec(cleanupCmd)
		} else if sshExecutor != nil {
			sshExecutor.Exec(cleanupCmd)
		}
	}()

	startTime := time.Now()
	operations := 50
	var execResult *executor.ExecResult
	var err error
	for i := 0; i < operations; i++ {
		operation := i % 10

		switch {
		case operation < 3: // 30% 创建操作
			configMapName := fmt.Sprintf("test-cm-%d", i)
			cmd := fmt.Sprintf("kubectl create configmap %s -n %s --from-literal=key=value",
				configMapName, testNamespace)

			if isDaily && localExecutor != nil {
				execResult, err = localExecutor.Exec(cmd)
			} else if sshExecutor != nil {
				execResult, err = sshExecutor.Exec(cmd)
			}
			if err == nil && execResult != nil && execResult.ExitCode == 0 {
				result.CreateOperations++
				result.TotalOperations++
			}

		case operation < 7: // 40% 读取操作
			cmd := fmt.Sprintf("kubectl get configmaps -n %s", testNamespace)

			if isDaily && localExecutor != nil {
				execResult, err = localExecutor.Exec(cmd)
			} else if sshExecutor != nil {
				execResult, err = sshExecutor.Exec(cmd)
			}
			if err == nil && execResult != nil && execResult.ExitCode == 0 {
				result.ReadOperations++
				result.TotalOperations++
			}

		case operation < 9: // 20% 更新操作
			configMapName := fmt.Sprintf("test-cm-%d", i/3)
			cmd := fmt.Sprintf("kubectl patch configmap %s -n %s -p '{\"data\":{\"updated\":\"true\"}}'",
				configMapName, testNamespace)

			if isDaily && localExecutor != nil {
				execResult, err = localExecutor.Exec(cmd)
			} else if sshExecutor != nil {
				execResult, err = sshExecutor.Exec(cmd)
			}
			if err == nil && execResult != nil && execResult.ExitCode == 0 {
				result.UpdateOperations++
				result.TotalOperations++
			}

		case operation == 9: // 10% 删除操作
			configMapName := fmt.Sprintf("test-cm-%d", i/10)
			cmd := fmt.Sprintf("kubectl delete configmap %s -n %s",
				configMapName, testNamespace)

			if isDaily && localExecutor != nil {
				execResult, err = localExecutor.Exec(cmd)
			} else if sshExecutor != nil {
				execResult, err = sshExecutor.Exec(cmd)
			}
			if err == nil && execResult != nil && execResult.ExitCode == 0 {
				result.DeleteOperations++
				result.TotalOperations++
			}
		}

		// 模拟正常操作间隔
		time.Sleep(100 * time.Millisecond)
	}

	duration := time.Since(startTime).Seconds()
	result.SuccessRate = float64(result.TotalOperations) / float64(operations) * 100
	result.AvgThroughput = float64(result.TotalOperations) / duration

	return result
}
