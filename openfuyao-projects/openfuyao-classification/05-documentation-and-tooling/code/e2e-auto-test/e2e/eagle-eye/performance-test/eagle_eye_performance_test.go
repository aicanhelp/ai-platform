package eagle_eye_performance_test

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/eagle-eye/system-integration/utils"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

var _ = ginkgo.Describe("Eagle Eye Performance Test", ginkgo.Label("eagle-eye", "performance", "npu", "skip-temporarily"), func() {

	// 定义资源使用情况结构体
	type ResourceUsage struct {
		CPU    string
		Memory string
	}

	// 定义性能测试结果结构体
	type PerformanceResult struct {
		StartTime        time.Time
		PodReadyTimes    map[string]time.Duration
		PodResourceUsage map[string]ResourceUsage
	}

	// 定义测试机器类型
	const (
		MachineType310 = "310"
		MachineType910 = "910"
	)

	// 定义机器配置
	type MachineConfig struct {
		Host       string
		Port       int
		User       string
		Password   string
		Kubeconfig string
	}

	// 获取机器配置
	getMachineConfig := func(machineType string) MachineConfig {
		prefix := fmt.Sprintf("EAGLE_EYE_%s_", machineType)

		config := MachineConfig{
			Host:       os.Getenv(prefix + "TEST_IP"),
			User:       os.Getenv(prefix + "TEST_USER"),
			Password:   os.Getenv(prefix + "TEST_PASSWORD"),
			Kubeconfig: os.Getenv(prefix + "KUBECONFIG_PATH"),
		}

		portStr := os.Getenv(prefix + "TEST_PORT")
		if portStr != "" {
			if port, err := strconv.Atoi(portStr); err == nil {
				config.Port = port
			} else {
				config.Port = 22 // 默认端口
			}
		} else {
			config.Port = 22 // 默认端口
		}

		return config
	}

	// 运行性能测试的函数
	runPerformanceTest := func(machineType string, config MachineConfig) {
		ginkgo.GinkgoWriter.Printf("=== 开始测试 %s 机器 ===\n", machineType)

		var (
			serverExec    *executor.SSHExecutor
			k8sClient     *k8s.K8SClient
			releaseConfig *utils.ReleaseConfig
			resultFile    *os.File
		)

		// 初始化性能测试结果
		performanceResult := PerformanceResult{
			StartTime:        time.Now(),
			PodReadyTimes:    make(map[string]time.Duration),
			PodResourceUsage: make(map[string]ResourceUsage),
		}

		// 创建服务器SSH执行器
		var err error
		serverExec, err = executor.NewSSHExecutor(
			config.Host,
			config.Port,
			config.User,
			config.Password,
		)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), fmt.Sprintf("创建 %s 机器SSH执行器失败", machineType))
		defer serverExec.Close()

		// 创建Kubernetes客户端
		kubeConfig, err := clientcmd.BuildConfigFromFlags("", config.Kubeconfig)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), fmt.Sprintf("加载 %s 机器kubeconfig失败", machineType))

		clientset, err := kubernetes.NewForConfig(kubeConfig)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), fmt.Sprintf("创建 %s 机器Kubernetes客户端失败", machineType))

		dynamicClient, err := dynamic.NewForConfig(kubeConfig)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), fmt.Sprintf("创建 %s 机器Dynamic客户端失败", machineType))

		k8sClient = &k8s.K8SClient{RestConfig: kubeConfig, Clientset: clientset, DynamicClient: dynamicClient}

		// 加载helm配置
		helmConfig, err := utils.LoadHelmConfig()
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "加载helm配置失败")

		// 获取eagle-eye的helm配置（注意：这里可能需要根据实际情况调整key）
		releaseConfig = helmConfig["eagle-eye"]
		if releaseConfig == nil {
			// 如果helm_config.yaml中没有eagle-eye，使用默认配置
			releaseConfig = &utils.ReleaseConfig{
				ReleaseName: "eagle-eye",
				HelmURl:     "oci://cr.openfuyao.cn/charts/eagle-eye",
				Version:     "0.21.1",
				Namespace:   "default",
			}
		}

		// 创建结果文件
		resultFileName := fmt.Sprintf("eagle-eye-performance-result-%s-%s.json", machineType, time.Now().Format("20060102150405"))
		resultFilePath := filepath.Join(".", resultFileName)
		resultFile, err = os.Create(resultFilePath)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), fmt.Sprintf("创建 %s 机器结果文件失败", machineType))
		defer resultFile.Close()

		// 步骤1: 安装eagle-eye
		ginkgo.GinkgoWriter.Printf("步骤1: 安装eagle-eye\n")
		installStartTime := time.Now()

		// 确保命名空间存在
		_, err = serverExec.Exec(fmt.Sprintf("kubectl create namespace %s --dry-run=client -o yaml | kubectl apply -f -", releaseConfig.Namespace))
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "创建命名空间失败")

		// 执行helm install
		err = utils.HelmInstall(releaseConfig.ReleaseName, releaseConfig.HelmURl, releaseConfig.Version, releaseConfig.Namespace, serverExec)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "安装eagle-eye失败")

		// 步骤2: 监控pod状态，统计从安装到ready的时长
		ginkgo.GinkgoWriter.Println("步骤2: 监控pod状态，统计从安装到ready的时长")

		// 需要监控的pod列表
		podNames := []string{
			"hardware-diagnosis",
			"hardware-monitor",
			"nats",
		}

		// 等待所有pod就绪（最多5分钟）
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()

		for _, podName := range podNames {
			ginkgo.GinkgoWriter.Printf("  等待 %s pod就绪...\n", podName)
			podReady := false

			for i := 0; i < 20; i++ { // 15秒轮询，20次共5分钟
				select {
				case <-ctx.Done():
					ginkgo.Fail(fmt.Sprintf("等待 %s pod就绪超时", podName))
				default:
					// 获取pod
					pods, err := k8sClient.Clientset.CoreV1().Pods(releaseConfig.Namespace).List(ctx, metav1.ListOptions{})
					if err != nil {
						ginkgo.GinkgoWriter.Printf("    获取pod列表失败: %v\n", err)
						time.Sleep(15 * time.Second)
						continue
					}

					// 查找匹配的pod
					for _, pod := range pods.Items {
						if strings.Contains(pod.Name, podName) && pod.Status.Phase == corev1.PodRunning {
							// 检查所有容器是否就绪
							allContainersReady := true
							for _, containerStatus := range pod.Status.ContainerStatuses {
								if !containerStatus.Ready {
									allContainersReady = false
									break
								}
							}

							if allContainersReady {
								podReadyTime := time.Since(installStartTime)
								performanceResult.PodReadyTimes[pod.Name] = podReadyTime
								ginkgo.GinkgoWriter.Printf("    %s pod就绪，耗时: %v\n", pod.Name, podReadyTime)
								podReady = true
								break
							}
						}
					}

					if podReady {
						break
					}

					time.Sleep(15 * time.Second)
				}
			}

			if !podReady {
				ginkgo.Fail(fmt.Sprintf("%s pod在5分钟内未就绪", podName))
			}
		}

		// 步骤3: 统计eagle的pod的cpu和memory
		ginkgo.GinkgoWriter.Println("步骤3: 统计eagle的pod的cpu和memory")

		// 获取所有eagle-eye相关的pod
		pods, err := k8sClient.Clientset.CoreV1().Pods(releaseConfig.Namespace).List(ctx, metav1.ListOptions{})
		gomega.Expect(err).NotTo(gomega.HaveOccurred(), "获取pod列表失败")

		for _, pod := range pods.Items {
			// 检查是否是eagle-eye相关的pod
			isEagleEyePod := false
			for _, podName := range podNames {
				if strings.Contains(pod.Name, podName) {
					isEagleEyePod = true
					break
				}
			}

			if !isEagleEyePod {
				continue
			}

			// 获取pod的资源使用情况
			cmd := fmt.Sprintf("kubectl top pod %s -n %s --no-headers", pod.Name, releaseConfig.Namespace)
			res, err := serverExec.Exec(cmd)
			if err != nil {
				ginkgo.GinkgoWriter.Printf("    获取 %s pod资源使用情况失败: %v\n", pod.Name, err)
				continue
			}

			// 解析输出
			lines := strings.Split(res.Stdout, "\n")
			for _, line := range lines {
				line = strings.TrimSpace(line)
				if line == "" {
					continue
				}

				parts := strings.Fields(line)
				if len(parts) >= 3 {
					performanceResult.PodResourceUsage[pod.Name] = ResourceUsage{
						CPU:    parts[1],
						Memory: parts[2],
					}
					ginkgo.GinkgoWriter.Printf("    %s pod资源使用情况: CPU=%s, Memory=%s\n", pod.Name, parts[1], parts[2])
					break
				}
			}
		}
	}

	// 测试310机器
	ginkgo.It("should measure eagle-eye performance on 310 machine", func() {
		config := getMachineConfig(MachineType310)
		runPerformanceTest(MachineType310, config)
	})

	// 测试910机器
	ginkgo.It("should measure eagle-eye performance on 910 machine", func() {
		config := getMachineConfig(MachineType910)
		runPerformanceTest(MachineType910, config)
	})
})
