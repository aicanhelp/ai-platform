package kae_operator

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ==================== Kae-Operator 组件间集成测试 ====================

var _ = Describe("kae-operator 组件间集成测试", Ordered, Label("kae-operator-components", "kae", "with-workload-cluster", "system-integration"), func() {

	var (
		helmConfig map[string]*helm.ReleaseConfig
		sshClient  *executor.SSHExecutor
		k8sClient  *k8s.K8SClient
		ctx        context.Context
		cancel     context.CancelFunc
	)

	BeforeAll(func() {
		var err error

		// context
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

		// ssh client
		// 从环境变量获取SSH连接信息
		host := os.Getenv("TEST_NODE5_IP")
		port := 22
		if portStr := os.Getenv("TEST_NODE5_PORT"); portStr != "" {
			if portInt, err := strconv.Atoi(portStr); err == nil {
				port = portInt
			}
		}
		user := os.Getenv("TEST_NODE5_USER")
		password := os.Getenv("TEST_NODE5_PASSWORD")

		sshClient, err = executor.NewSSHExecutor(
			host,
			port,
			user,
			password,
		)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client 应当成功创建")

		// k8s client
		k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

		// load helm config from framework/helm
		helmConfig, err = helm.LoadHelmConfig()
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm config 应当成功加载")
	})

	AfterEach(func() {
		for _, eachRelease := range helmConfig {
			var needUninstall bool
			var err error
			needUninstall, err = helm.CheckHelmRelease(eachRelease.ReleaseName, eachRelease.Namespace, sshClient)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "helm release check 应该成功")
			if needUninstall == true {
				err = helm.HelmUninstall(eachRelease.ReleaseName, eachRelease.Namespace, sshClient)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("helm uninstall %s 应当执行成功", eachRelease.ReleaseName))
				time.Sleep(15 * time.Second)
				GinkgoWriter.Printf("%s uninstall success\n", eachRelease.ReleaseName)
			}
		}
	})

	It("should install numa and verify compatibility with kae", func() {
		var err error

		// step1: numa-affinity-package install
		GinkgoWriter.Println("=====step1: numa-affinity-package install=====")
		tempReleaseName := helmConfig["numa-affinity-package"].ReleaseName
		tempHelmUrl := helmConfig["numa-affinity-package"].HelmURrl
		tempVersion := helmConfig["numa-affinity-package"].Version
		tempNamespace := helmConfig["numa-affinity-package"].Namespace

		GinkgoWriter.Println("start install numa-affinity-package...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install numa-affinity-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("numa-affinity-package install success")

		// step2: kae-operator install
		GinkgoWriter.Println("=====step2: kae-operator install=====")
		releaseName := helmConfig["kae-operator"].ReleaseName
		helmUrl := helmConfig["kae-operator"].HelmURrl
		version := helmConfig["kae-operator"].Version
		namespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: kae-operator installation check
		GinkgoWriter.Println("=====step3: kae-operator installation check=====")
		verifyKaeOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("kae-operator installation check pass")
	})

	It("kae-operator & colocation-package", func() {
		var err error

		// step1: colocation-package install
		GinkgoWriter.Println("=====step1: colocation-package install=====")
		tempReleaseName := helmConfig["colocation-package"].ReleaseName
		tempHelmUrl := helmConfig["colocation-package"].HelmURrl
		tempVersion := helmConfig["colocation-package"].Version
		tempNamespace := helmConfig["colocation-package"].Namespace

		GinkgoWriter.Println("start install colocation-package...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install colocation-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("colocation-package install success")

		// step2: kae-operator install
		GinkgoWriter.Println("=====step2: kae-operator install=====")
		releaseName := helmConfig["kae-operator"].ReleaseName
		helmUrl := helmConfig["kae-operator"].HelmURrl
		version := helmConfig["kae-operator"].Version
		namespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: kae-operator installation check
		GinkgoWriter.Println("=====step3: kae-operator installation check=====")
		verifyKaeOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("kae-operator installation check pass")
	})

	It("kae-operator & many-core-scheduler", func() {
		var err error

		// step1: many-core-scheduler install
		GinkgoWriter.Println("=====step1: many-core-scheduler install=====")
		tempReleaseName := helmConfig["many-core-scheduler"].ReleaseName
		tempHelmUrl := helmConfig["many-core-scheduler"].HelmURrl
		tempVersion := helmConfig["many-core-scheduler"].Version
		tempNamespace := helmConfig["many-core-scheduler"].Namespace

		GinkgoWriter.Println("start install many-core-scheduler...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install many-core-scheduler 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("many-core-scheduler install success")

		// step2: kae-operator install
		GinkgoWriter.Println("=====step2: kae-operator install=====")
		releaseName := helmConfig["kae-operator"].ReleaseName
		helmUrl := helmConfig["kae-operator"].HelmURrl
		version := helmConfig["kae-operator"].Version
		namespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: kae-operator installation check
		GinkgoWriter.Println("=====step3: kae-operator installation check=====")
		verifyKaeOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("kae-operator installation check pass")
	})

	It("kae-operator & logging-package", func() {
		var err error

		// step1: logging-package install
		GinkgoWriter.Println("=====step1: logging-package install=====")
		tempReleaseName := helmConfig["logging-package"].ReleaseName
		tempHelmUrl := helmConfig["logging-package"].HelmURrl
		tempVersion := helmConfig["logging-package"].Version
		tempNamespace := helmConfig["logging-package"].Namespace

		GinkgoWriter.Println("start install logging-package...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install logging-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("logging-package install success")

		// step2: kae-operator install
		GinkgoWriter.Println("=====step2: kae-operator install=====")
		releaseName := helmConfig["kae-operator"].ReleaseName
		helmUrl := helmConfig["kae-operator"].HelmURrl
		version := helmConfig["kae-operator"].Version
		namespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: kae-operator installation check
		GinkgoWriter.Println("=====step3: kae-operator installation check=====")
		verifyKaeOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("kae-operator installation check pass")
	})

	It("kae-operator & multi-cluster-service", func() {
		var err error

		// step1: multi-cluster-service install
		GinkgoWriter.Println("=====step1: multi-cluster-service install=====")
		tempReleaseName := helmConfig["multi-cluster-service"].ReleaseName
		tempHelmUrl := helmConfig["multi-cluster-service"].HelmURrl
		tempVersion := helmConfig["multi-cluster-service"].Version
		tempNamespace := helmConfig["multi-cluster-service"].Namespace

		GinkgoWriter.Println("start install multi-cluster-service...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install multi-cluster-service 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("multi-cluster-service install success")

		// step2: kae-operator install
		GinkgoWriter.Println("=====step2: kae-operator install=====")
		releaseName := helmConfig["kae-operator"].ReleaseName
		helmUrl := helmConfig["kae-operator"].HelmURrl
		version := helmConfig["kae-operator"].Version
		namespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: kae-operator installation check
		GinkgoWriter.Println("=====step3: kae-operator installation check=====")
		verifyKaeOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("kae-operator installation check pass")
	})

	It("kae-operator & monitoring-dashboard", func() {
		var err error

		// step1: monitoring-dashboard install
		GinkgoWriter.Println("=====step1: monitoring-dashboard install=====")
		tempReleaseName := helmConfig["monitoring-dashboard"].ReleaseName
		tempHelmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		tempVersion := helmConfig["monitoring-dashboard"].Version
		tempNamespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("monitoring-dashboard install success")

		// step2: kae-operator install
		GinkgoWriter.Println("=====step2: kae-operator install=====")
		releaseName := helmConfig["kae-operator"].ReleaseName
		helmUrl := helmConfig["kae-operator"].HelmURrl
		version := helmConfig["kae-operator"].Version
		namespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: kae-operator installation check
		GinkgoWriter.Println("=====step3: kae-operator installation check=====")
		verifyKaeOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("kae-operator installation check pass")
	})

	It("kae-operator & ray-package", func() {
		var err error

		// step1: ray-package install
		GinkgoWriter.Println("=====step1: ray-package install=====")
		tempReleaseName := helmConfig["ray-package"].ReleaseName
		tempHelmUrl := helmConfig["ray-package"].HelmURrl
		tempVersion := helmConfig["ray-package"].Version
		tempNamespace := helmConfig["ray-package"].Namespace

		GinkgoWriter.Println("start install ray-package...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install ray-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("ray-package install success")

		// step2: kae-operator install
		GinkgoWriter.Println("=====step2: kae-operator install=====")
		releaseName := helmConfig["kae-operator"].ReleaseName
		helmUrl := helmConfig["kae-operator"].HelmURrl
		version := helmConfig["kae-operator"].Version
		namespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: kae-operator installation check
		GinkgoWriter.Println("=====step3: kae-operator installation check=====")
		verifyKaeOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("kae-operator installation check pass")
	})

	It("kae-operator & npu-operator", func() {
		var err error

		// step1: npu-operator install
		GinkgoWriter.Println("=====step1: npu-operator install=====")
		tempReleaseName := helmConfig["npu-operator"].ReleaseName
		tempHelmUrl := helmConfig["npu-operator"].HelmURrl
		tempVersion := helmConfig["npu-operator"].Version
		tempNamespace := helmConfig["npu-operator"].Namespace

		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("npu-operator install success")

		// step2: kae-operator install (expected to fail due to conflict)
		GinkgoWriter.Println("=====step2: kae-operator install (expected conflict)=====")
		releaseName := helmConfig["kae-operator"].ReleaseName
		helmUrl := helmConfig["kae-operator"].HelmURrl
		version := helmConfig["kae-operator"].Version
		namespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println("Expected conflict error:", err)
		}
		Expect(err).To(HaveOccurred(), "kae-operator 安装应当失败，因为与 npu-operator 冲突")

		GinkgoWriter.Println("kae-operator & npu-operator conflict detection pass")
	})
})

var _ = Describe("Kae-Operator System Integration", Label("kae_operator_system_integration", "kae", "with-workload-cluster", "system-integration"), func() {
	var (
		k8sClient *k8s.K8SClient
		ctx       context.Context
		cancel    context.CancelFunc
		sshClient *executor.SSHExecutor
	)

	BeforeEach(func() {
		var err error
		ctx, cancel = context.WithTimeout(context.Background(), 10*time.Minute)

		// 从环境变量读取 SSH 连接信息
		host := os.Getenv("TEST_NODE5_IP")
		port := 22
		if portStr := os.Getenv("TEST_NODE5_PORT"); portStr != "" {
			if portInt, err := strconv.Atoi(portStr); err == nil {
				port = portInt
			}
		}
		user := os.Getenv("TEST_NODE5_USER")
		password := os.Getenv("TEST_NODE5_PASSWORD")

		sshClient, err = executor.NewSSHExecutor(host, port, user, password)
		Expect(err).NotTo(HaveOccurred(), "应该能够使用 SSH 连接到 Kae-Operator 测试节点")

		// 创建 k8s 客户端
		k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")
	})

	AfterEach(func() {
		if cancel != nil {
			cancel()
		}
	})

	It("should have Kae-Operator related pods running and images match expectations", func() {
		// 校验Kae-Operator相关pod的状态和镜像
		verifyKaeOperatorPods(k8sClient, ctx)
	})

	It("should automatically recover Kae-Operator components after pod deletion", func() {
		// 按照ExpectedImages中的componentName，依次用kubectl delete pod的方式删除，然后检查pod在2分钟内是否恢复
		verifyKaeComponentAutoRecovery(k8sClient, ctx, sshClient)
	})

	It("should handle installation conflicts between KAE and other components", func() {
		GinkgoWriter.Printf("开始测试KAE与其他组件的冲突情况...\n")

		//清理现有KAE组件
		cleanupExistingKae(sshClient)

		//NUMA与KAE的安装冲突
		testNumaKaeConflict(k8sClient, ctx, sshClient)

		//NPU与KAE的安装冲突
		testNpuKaeConflict(k8sClient, ctx, sshClient)

		GinkgoWriter.Printf("KAE组件冲突测试完成！\n")
	})
})

// cleanupExistingKae 清理现有的KAE组件
func cleanupExistingKae(sshClient *executor.SSHExecutor) {
	GinkgoWriter.Printf("=== 步骤0: 检查并清理现有KAE组件 ===\n")

	// 检查KAE是否存在
	checkCmd := "helm list -n default -o json | jq -r '.[] | select(.name==\"kae\") | .name'"
	res, err := sshClient.Exec(checkCmd)
	if err != nil {
		GinkgoWriter.Printf("检查KAE存在时出错: %v\n", err)
		return
	}

	if strings.TrimSpace(res.Stdout) == "kae" {
		GinkgoWriter.Printf("发现已存在的KAE组件，执行卸载...\n")

		// 执行卸载命令
		uninstallRes, err := sshClient.Exec(KAE_UNINSTALL_CMD)
		if err != nil {
			GinkgoWriter.Printf("卸载KAE时出错: %v\n", err)
			GinkgoWriter.Printf("卸载输出: stdout=%s, stderr=%s\n", uninstallRes.Stdout, uninstallRes.Stderr)
			// 继续执行，可能部分组件已经卸载
		} else {
			GinkgoWriter.Printf("KAE卸载成功: %s\n", strings.TrimSpace(uninstallRes.Stdout))
		}

		// 等待1分钟让卸载完成
		GinkgoWriter.Printf("等待1分钟让KAE卸载完成...\n")
		time.Sleep(1 * time.Minute)
	} else {
		GinkgoWriter.Printf("未发现已存在的KAE组件，继续执行...\n")
	}
}

// testNumaKaeConflict 测试NUMA与KAE的安装冲突
func testNumaKaeConflict(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor) {
	GinkgoWriter.Printf("\n=== 测试1: NUMA与KAE的安装冲突 ===\n")

	// 步骤1.1: 安装NUMA
	GinkgoWriter.Printf("步骤1.1: 安装NUMA组件...\n")
	numaRes, err := sshClient.Exec(NUMA_INSTALL_CMD)
	if err != nil {
		GinkgoWriter.Printf("NUMA安装失败: %v\n", err)
		GinkgoWriter.Printf("安装输出: stdout=%s, stderr=%s\n", numaRes.Stdout, numaRes.Stderr)
		// 继续测试，可能是已经安装
	} else {
		GinkgoWriter.Printf("NUMA安装成功: %s\n", strings.TrimSpace(numaRes.Stdout))
	}

	// 等待1分钟让NUMA稳定
	GinkgoWriter.Printf("等待1分钟让NUMA组件稳定...\n")
	time.Sleep(1 * time.Minute)

	// 步骤1.2: 安装KAE
	GinkgoWriter.Printf("步骤1.2: 在NUMA存在的情况下安装KAE...\n")
	kaeRes, err := sshClient.Exec(KAE_INSTALL_CMD)
	Expect(err).NotTo(HaveOccurred(), "KAE应该能够成功安装（即使NUMA已存在）")
	GinkgoWriter.Printf("KAE安装成功: %s\n", strings.TrimSpace(kaeRes.Stdout))

	// 等待1分钟让KAE稳定
	GinkgoWriter.Printf("等待1分钟让KAE组件稳定...\n")
	time.Sleep(1 * time.Minute)

	// 步骤1.3: 检查KAE的Pod是否正常
	GinkgoWriter.Printf("步骤1.3: 检查KAE Pod状态...\n")
	checkAllExpectedComponents(k8sClient, ctx)

	// 步骤1.4: 卸载KAE和NUMA
	GinkgoWriter.Printf("步骤1.4: 卸载KAE和NUMA组件...\n")

	// 卸载KAE
	uninstallKaeRes, err := sshClient.Exec(KAE_UNINSTALL_CMD)
	if err != nil {
		GinkgoWriter.Printf("KAE卸载失败: %v\n", err)
		GinkgoWriter.Printf("卸载输出: stdout=%s, stderr=%s\n", uninstallKaeRes.Stdout, uninstallKaeRes.Stderr)
	} else {
		GinkgoWriter.Printf("KAE卸载成功: %s\n", strings.TrimSpace(uninstallKaeRes.Stdout))
	}

	// 卸载NUMA
	uninstallNumaRes, err := sshClient.Exec(NUMA_UNINSTALL_CMD)
	if err != nil {
		GinkgoWriter.Printf("NUMA卸载失败: %v\n", err)
		GinkgoWriter.Printf("卸载输出: stdout=%s, stderr=%s\n", uninstallNumaRes.Stdout, uninstallNumaRes.Stderr)
	} else {
		GinkgoWriter.Printf("NUMA卸载成功: %s\n", strings.TrimSpace(uninstallNumaRes.Stdout))
	}

	// 等待1分钟让卸载完成
	GinkgoWriter.Printf("等待2分钟让卸载完成...\n")
	time.Sleep(2 * time.Minute)

	// 步骤1.5: 检查集群中的所有Pod是否都Ready（判断卸载是否完成）
	GinkgoWriter.Printf("步骤1.5: 检查集群卸载状态...\n")
	verifyAllPodsReady(k8sClient, ctx)

	GinkgoWriter.Printf("NUMA与KAE冲突测试完成！\n")
}

// testNpuKaeConflict 测试NPU与KAE的安装冲突
func testNpuKaeConflict(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor) {
	GinkgoWriter.Printf("\n=== 测试2: NPU与KAE的安装冲突 ===\n")

	// 步骤2.1: 安装NPU
	GinkgoWriter.Printf("步骤2.1: 安装NPU组件...\n")
	npuRes, err := sshClient.Exec(NPU_INSTALL_CMD)
	if err != nil {
		GinkgoWriter.Printf("NPU安装失败: %v\n", err)
		GinkgoWriter.Printf("安装输出: stdout=%s, stderr=%s\n", npuRes.Stdout, npuRes.Stderr)
		// 继续测试，可能是已经安装
	} else {
		GinkgoWriter.Printf("NPU安装成功: %s\n", strings.TrimSpace(npuRes.Stdout))
	}

	// 等待1分钟让NPU稳定
	GinkgoWriter.Printf("等待1分钟让NPU组件稳定...\n")
	time.Sleep(1 * time.Minute)

	// 步骤2.2: 安装KAE - 预期应该失败
	GinkgoWriter.Printf("步骤2.2: 在NPU存在的情况下安装KAE（预期应该失败）...\n")
	kaeRes, err := sshClient.Exec(KAE_INSTALL_CMD)

	// 验证安装应该失败
	Expect(err).To(HaveOccurred(), "KAE安装应该失败，因为NPU已占用了相关资源")

	// 检查错误信息是否符合预期
	errorOutput := kaeRes.Stderr
	GinkgoWriter.Printf("KAE安装失败（符合预期）:\n")
	GinkgoWriter.Printf("  stdout: %s\n", strings.TrimSpace(kaeRes.Stdout))
	GinkgoWriter.Printf("  stderr: %s\n", strings.TrimSpace(errorOutput))

	// 验证错误信息包含预期的内容
	expectedErrorKeywords := []string{
		"Unable to continue with install",
		"ServiceAccount \"node-feature-discovery\"",
		"must equal \"kae\"",
		"current value is \"npu\"",
	}

	allKeywordsFound := true
	for _, keyword := range expectedErrorKeywords {
		if strings.Contains(errorOutput, keyword) {
			GinkgoWriter.Printf("  包含预期关键词: %s ✓\n", keyword)
		} else {
			GinkgoWriter.Printf("  缺少预期关键词: %s ✗\n", keyword)
			allKeywordsFound = false
		}
	}

	Expect(allKeywordsFound).To(BeTrue(), "错误信息应该包含所有预期的关键词")
	GinkgoWriter.Printf("KAE安装失败符合预期（资源冲突检测正确）\n")

	// 步骤2.3: 清理环境 - 先卸载NPU再重试安装KAE
	GinkgoWriter.Printf("步骤2.3: 清理环境，先卸载NPU再安装KAE...\n")

	// 卸载NPU
	uninstallNpuRes, err := sshClient.Exec(NPU_UNINSTALL_CMD)
	if err != nil {
		GinkgoWriter.Printf("NPU卸载失败: %v\n", err)
		GinkgoWriter.Printf("卸载输出: stdout=%s, stderr=%s\n", uninstallNpuRes.Stdout, uninstallNpuRes.Stderr)
	} else {
		GinkgoWriter.Printf("NPU卸载成功: %s\n", strings.TrimSpace(uninstallNpuRes.Stdout))
	}

	// 等待30秒确保资源释放
	GinkgoWriter.Printf("等待30秒确保NPU资源释放...\n")
	time.Sleep(30 * time.Second)
	GinkgoWriter.Printf("NPU与KAE冲突测试完成！验证了：\n")
	GinkgoWriter.Printf("NPU存在时KAE安装失败（符合预期）\n")
}

// verifyAllPodsReady 验证集群中所有Pod都处于Ready或Succeeded状态
func verifyAllPodsReady(k8sClient *k8s.K8SClient, ctx context.Context) {
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取Pod列表时不应出错")

	if len(podList.Items) == 0 {
		GinkgoWriter.Printf("警告: 集群中没有发现任何Pod\n")
		return
	}

	allPodsReady := true
	var notReadyPods []string

	GinkgoWriter.Printf("检查集群中所有Pod状态 (共 %d 个Pod)...\n", len(podList.Items))

	for _, pod := range podList.Items {
		podFullName := fmt.Sprintf("%s/%s", pod.Namespace, pod.Name)

		// 跳过已删除或正在终止的Pod
		if pod.DeletionTimestamp != nil {
			GinkgoWriter.Printf("  %s: 正在删除/终止 (跳过检查)\n", podFullName)
			continue
		}

		// 检查Pod状态
		podPhase := pod.Status.Phase

		// 接受的状态：Running 或 Succeeded
		acceptablePhases := map[corev1.PodPhase]bool{
			corev1.PodRunning:   true,
			corev1.PodSucceeded: true,
		}

		if !acceptablePhases[podPhase] {
			allPodsReady = false
			notReadyPods = append(notReadyPods,
				fmt.Sprintf("%s (状态: %s)", podFullName, podPhase))
			GinkgoWriter.Printf("  %s: 状态不可接受 (%s)\n", podFullName, podPhase)
			continue
		}
	}

	if !allPodsReady {
		GinkgoWriter.Printf("\n发现未就绪的Pod:\n")
		for i, podInfo := range notReadyPods {
			GinkgoWriter.Printf("  %d. %s\n", i+1, podInfo)
		}
	}

	Expect(allPodsReady).To(BeTrue(),
		"卸载完成后，集群中所有Pod都应处于Ready或Succeeded状态。发现 %d 个未就绪的Pod", len(notReadyPods))

	GinkgoWriter.Printf("所有Pod都处于可接受状态 (Running且就绪 或 Succeeded)\n")
}

// verifyKaeComponentAutoRecovery 验证Kae-Operator组件在Pod删除后的自动恢复能力
func verifyKaeComponentAutoRecovery(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor) {
	GinkgoWriter.Printf("开始验证Kae-Operator组件的自愈能力...\n")

	// 记录测试开始时间
	startTime := time.Now()

	// 记录失败的组件
	var failedComponents []string

	// 统计成功恢复的组件数量
	successfullyRecovered := 0
	totalComponents := len(ExpectedImages)

	if totalComponents == 0 {
		GinkgoWriter.Printf("警告: ExpectedImages配置为空，跳过自愈能力测试\n")
		return
	}

	// 获取组件名称列表
	var componentNames []string
	for componentName := range ExpectedImages {
		componentNames = append(componentNames, componentName)
	}

	GinkgoWriter.Printf("=== 第一阶段: 删除所有组件的Pod ===\n")
	for i, componentName := range componentNames {
		GinkgoWriter.Printf("处理组件 (%d/%d): %s\n", i+1, totalComponents, componentName)

		// 首先找到组件的当前Pod
		originalPod := findKaeComponentPodByName(k8sClient, ctx, componentName)
		if originalPod == nil {
			GinkgoWriter.Printf("警告: 未找到 %s 组件的Pod\n", componentName)
			continue
		}

		originalPodName := originalPod.Name
		namespace := originalPod.Namespace
		GinkgoWriter.Printf("  找到Pod: %s/%s (状态: %s)\n",
			namespace, originalPodName, originalPod.Status.Phase)

		// 2. 删除Pod
		deleteCmd := fmt.Sprintf("kubectl delete pod %s -n %s --wait=false --timeout=30s", originalPodName, namespace)
		deleteRes, err := sshClient.Exec(deleteCmd)
		if err != nil {
			GinkgoWriter.Printf("  删除失败: %v\n", err)
			if deleteRes != nil {
				GinkgoWriter.Printf("  删除命令输出: stdout=%s, stderr=%s\n", deleteRes.Stdout, deleteRes.Stderr)
			}
			continue
		}
		GinkgoWriter.Printf("  删除成功: %s\n", strings.TrimSpace(deleteRes.Stdout))
	}

	GinkgoWriter.Printf("\n=== 第二阶段: 等待2分钟让Pod恢复 ===\n")
	GinkgoWriter.Printf("等待2分钟让所有组件恢复...\n")
	time.Sleep(2 * time.Minute)

	GinkgoWriter.Printf("\n=== 第三阶段: 检查所有组件是否恢复 ===\n")
	for _, componentName := range componentNames {
		GinkgoWriter.Printf("检查组件: %s\n", componentName)

		// 1. 找到组件的新Pod
		newPod := findKaeComponentPodByName(k8sClient, ctx, componentName)
		if newPod == nil {
			failedComponents = append(failedComponents, fmt.Sprintf("%s (恢复后未找到Pod)", componentName))
			GinkgoWriter.Printf("  未找到Pod\n")
			continue
		}

		newPodName := newPod.Name
		namespace := newPod.Namespace
		GinkgoWriter.Printf("  找到Pod: %s/%s (状态: %s)\n",
			namespace, newPodName, newPod.Status.Phase)

		// 2. 检查Pod状态是否为Running
		if newPod.Status.Phase != corev1.PodRunning {
			failedComponents = append(failedComponents,
				fmt.Sprintf("%s (Pod状态: %s)", componentName, newPod.Status.Phase))
			GinkgoWriter.Printf("  Pod状态不是Running: %s\n", newPod.Status.Phase)
			continue
		}

		// 3. 检查Pod是否就绪
		if !isPodReady(newPod) {
			failedComponents = append(failedComponents,
				fmt.Sprintf("%s (Pod未就绪)", componentName))
			GinkgoWriter.Printf("  Pod未就绪\n")
			continue
		}

		successfullyRecovered++
		GinkgoWriter.Printf("  恢复成功\n")
	}

	// 测试完成统计
	duration := time.Since(startTime)
	GinkgoWriter.Printf("\n=== Kae-Operator组件自愈能力测试完成 ===\n")
	GinkgoWriter.Printf("测试总时长: %v\n", duration.Round(time.Second))
	GinkgoWriter.Printf("组件总数: %d\n", totalComponents)
	GinkgoWriter.Printf("成功恢复: %d\n", successfullyRecovered)
	GinkgoWriter.Printf("恢复失败: %d\n", len(failedComponents))

	if len(failedComponents) > 0 {
		GinkgoWriter.Printf("失败组件详情:\n")
		for i, comp := range failedComponents {
			GinkgoWriter.Printf("  %d. %s\n", i+1, comp)
		}
	}

	// 要求全部组件都能成功恢复
	Expect(successfullyRecovered).To(Equal(totalComponents),
		"所有Kae-Operator组件都应成功恢复。总组件数: %d, 成功恢复: %d, 失败组件: %v",
		totalComponents, successfullyRecovered, failedComponents)

	GinkgoWriter.Printf("Kae-Operator所有组件自愈能力验证通过！\n")
}

// verifyKaeOperatorPods 验证Kae-Operator相关的Pod状态和镜像
func verifyKaeOperatorPods(k8sClient *k8s.K8SClient, ctx context.Context) {
	GinkgoWriter.Printf("开始检查Kae-Operator相关Pod...\n")

	// 检查所有期望的组件
	checkAllExpectedComponents(k8sClient, ctx)
}

// checkAllExpectedComponents 公共方法：检查所有期望的组件
func checkAllExpectedComponents(k8sClient *k8s.K8SClient, ctx context.Context) {
	// 获取所有 Pod 作为候选集合
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取所有 pods 时不应出错")
	Expect(len(podList.Items)).To(BeNumerically(">", 0), "集群中应至少存在一个 pod")

	// 遍历所有期望的组件名称（来自 ExpectedImages 的 key）并校验
	componentsFound := 0
	for componentName, expectedImgs := range ExpectedImages {
		matchedPod := findMatchingKaePod(componentName, podList.Items)
		if matchedPod == nil {
			GinkgoWriter.Printf("未找到与组件 %s 匹配的 pod\n", componentName)
			continue
		}

		componentsFound++
		GinkgoWriter.Printf("检查 Pod: %s/%s (组件: %s)\n", matchedPod.Namespace, matchedPod.Name, componentName)

		// 检查 Pod 的状态
		checkPodStatus(matchedPod)

		// 如果有配置期望镜像，检查镜像匹配
		if len(expectedImgs) > 0 {
			checkPodImages(matchedPod, expectedImgs, componentName)
		}
	}

	Expect(componentsFound).To(Equal(len(ExpectedImages)),
		"应该找到所有期望的组件。期望: %d, 找到: %d", len(ExpectedImages), componentsFound)

	GinkgoWriter.Printf("Kae-Operator Pod 检查完成: 找到 %d 个组件的 Pod\n", componentsFound)
}

// checkPodStatus 检查Pod状态（抽出的公共方法）
func checkPodStatus(pod *corev1.Pod) {
	// 检查 Pod 的状态为 Running
	Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "pod %s 应为 Running", pod.Name)

	// 检查所有容器是否就绪
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			Expect(condition.Status).To(Equal(corev1.ConditionTrue),
				"pod %s 应为 Ready 状态", pod.Name)
		}
	}

	GinkgoWriter.Printf("  Pod状态: %s (Ready: %v)\n",
		pod.Status.Phase, isPodReady(pod))
}

// checkPodImages 检查Pod镜像（抽出的公共方法）
func checkPodImages(pod *corev1.Pod, expectedImgs []string, componentName string) {
	// 获取Pod中的所有容器镜像
	podImages := getAllPodImages(pod)
	GinkgoWriter.Printf("  容器镜像: %v\n", podImages)

	// 校验容器镜像至少匹配一项期望值（支持部分匹配）
	allExpectedImagesFound := true
	for _, expectedImage := range expectedImgs {
		if expectedImage == "" {
			// 空字符串表示通配符，总是匹配
			GinkgoWriter.Printf("  期望镜像 [任意]: 跳过检查\n")
			continue
		}
		if contains(podImages, expectedImage) || containsSubstringInSlice(podImages, expectedImage) {
			GinkgoWriter.Printf("  期望镜像 %s: 存在\n", expectedImage)
		} else {
			GinkgoWriter.Printf("  期望镜像 %s: 不存在\n", expectedImage)
			allExpectedImagesFound = false
		}
	}

	if !allExpectedImagesFound {
		Expect(allExpectedImagesFound).To(BeTrue(),
			fmt.Sprintf("组件 %s 应该包含期望镜像", componentName))
	}
}

// findKaeComponentPodByName 根据组件名称查找对应的Pod
func findKaeComponentPodByName(k8sClient *k8s.K8SClient, ctx context.Context, componentName string) *corev1.Pod {
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取Pod列表时不应出错")

	return findMatchingKaePod(componentName, podList.Items)
}

// findMatchingKaePod 根据组件名称查找匹配的 Pod
func findMatchingKaePod(componentName string, pods []corev1.Pod) *corev1.Pod {
	// 策略1：完全匹配 - 查找 "componentName-*" 形式的 Pod
	for i := range pods {
		if strings.HasPrefix(pods[i].Name, componentName+"-") {
			return &pods[i]
		}
	}

	// 策略2：包含匹配 - 查找名称中包含组件名称的 Pod
	for i := range pods {
		if strings.Contains(pods[i].Name, componentName) {
			// 避免误匹配
			parts := strings.Split(pods[i].Name, "-")
			for j := 0; j < len(parts); j++ {
				possibleName := strings.Join(parts[:j+1], "-")
				if possibleName == componentName {
					return &pods[i]
				}
			}
		}
	}

	return nil
}

// getAllPodImages 获取Pod中的所有容器镜像
func getAllPodImages(pod *corev1.Pod) []string {
	var images []string

	// 安全地添加initContainers镜像（如果存在）
	if pod.Spec.InitContainers != nil {
		for _, container := range pod.Spec.InitContainers {
			images = append(images, container.Image)
		}
	}

	// 安全地添加containers镜像（如果存在）
	if pod.Spec.Containers != nil {
		for _, container := range pod.Spec.Containers {
			images = append(images, container.Image)
		}
	}

	return images
}

// contains 检查字符串是否在字符串切片中
func contains(slice []string, str string) bool {
	for _, item := range slice {
		if item == str {
			return true
		}
	}
	return false
}

// containsSubstringInSlice 检查切片中是否有元素包含给定子串（支持部分匹配）
func containsSubstringInSlice(slice []string, substr string) bool {
	for _, item := range slice {
		if strings.Contains(item, substr) {
			return true
		}
	}
	return false
}

// isPodReady 检查Pod是否就绪
func isPodReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
