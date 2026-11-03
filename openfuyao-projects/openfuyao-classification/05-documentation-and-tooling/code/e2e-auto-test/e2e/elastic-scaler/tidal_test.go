package elastic_scaler

import (
	"time"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// tidal controller 不修改 targetRef 的 replicas，仅校验 CR 的 status.desiredReplicas；各用例在创建 CR 前需保证 targetRef 指向的资源已存在。
var _ = Describe("tidal-scheduler 单组件测试", Ordered, Label("tidal", "with-workload-cluster"), func() {
	BeforeAll(func() {
		// 避免上次任务残留：先删除本套 e2e 专属命名空间并等待彻底删除，再重新创建（不会动别人的命名空间）。
		_ = DeleteNamespaceIfExists(clientset, ctx, TidalHelmNamespace)
		Eventually(func() bool {
			_, err := clientset.CoreV1().Namespaces().Get(ctx, TidalHelmNamespace, metav1.GetOptions{})
			return apierrors.IsNotFound(err)
		}, 3*time.Minute, 5*time.Second).Should(BeTrue(), "namespace %s 应先被清理干净", TidalHelmNamespace)
		Expect(EnsureNamespace(clientset, ctx, TidalTestNamespace)).NotTo(HaveOccurred())
	})

	// 用例1：安装部署 - 控制器部署与健康验证（后续用例均依赖本用例）
	It("1-安装部署-控制器部署与健康验证", func() {
		Expect(InstallTidal()).NotTo(HaveOccurred())
		VerifyHelmReleaseStatus(TidalHelmReleaseName, TidalHelmNamespace)
		VerifyTidalPodsRunning(clientset, ctx, 5*time.Minute, 10*time.Second)
	})

	// 用例2：功能验证 - 基础调度规则创建（前置：已有Deployment）
	It("2-功能验证-基础调度规则创建", func() {
		Expect(CreateTestDeployment(clientset, ctx, TidalTestDeploymentName, TidalTestNamespace, 1)).NotTo(HaveOccurred())
		time.Sleep(10 * time.Second) // 等待 Deployment 就绪
		Expect(CreateTidalCR(dynamicClient, ctx, TidalTestCRName, TidalTestNamespace)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRName, TidalTestNamespace)
		VerifyTidalStatusNextTrigger(dynamicClient, ctx, TidalTestCRName, TidalTestNamespace)
		VerifyTidalStatusReady(dynamicClient, ctx, TidalTestCRName, TidalTestNamespace)
	})

	// 用例3：功能验证 - 定时触发与副本变更（每分钟触发，等待1分钟）
	It("3-功能验证-定时触发与副本变更", func() {
		Expect(CreateTidalCRWithRules(dynamicClient, ctx, TidalTestCRCase3Name, TidalTestNamespace, Case3TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase3Name, TidalTestNamespace)
		time.Sleep(70 * time.Second)                                                                // 等待1分钟让定时规则至少触发一次
		VerifyTidalDesiredReplicas(dynamicClient, ctx, TidalTestCRCase3Name, TidalTestNamespace, 2) // rule replicas=2
	})

	// 用例4：功能验证 - 控制器重启与状态恢复
	It("4-功能验证-控制器重启与状态恢复", func() {
		restartAt := time.Now()
		Expect(RestartTidalController(TidalTestNamespace)).NotTo(HaveOccurred())
		time.Sleep(30 * time.Second) // 等待控制器重启
		VerifyTidalPodsRunning(clientset, ctx, 2*time.Minute, 10*time.Second)
		// 控制器恢复就绪
		VerifyTidalStatusReady(dynamicClient, ctx, TidalTestCRName, TidalTestNamespace)
		// 使用每分钟触发的用例3 CR，验证重启后规则仍然会触发，lastTriggerTime 在重启时间之后更新，且 desiredReplicas 仍为 2
		VerifyTidalLastTriggerAfter(dynamicClient, ctx, TidalTestCRCase3Name, TidalTestNamespace, restartAt)
		VerifyTidalDesiredReplicas(dynamicClient, ctx, TidalTestCRCase3Name, TidalTestNamespace, 2)
	})

	// 用例5：功能验证 - 缩容到零场景验证
	It("5-功能验证-缩容到零场景验证", func() {
		Expect(CreateTidalCRWithRules(dynamicClient, ctx, TidalTestCRCase5Name, TidalTestNamespace, Case5TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase5Name, TidalTestNamespace)
		time.Sleep(130 * time.Second) // 等待每2分钟规则触发，缩容到 0
		VerifyTidalDesiredReplicas(dynamicClient, ctx, TidalTestCRCase5Name, TidalTestNamespace, 0)
	})

	// 用例6：兼容性 - Deployment资源类型支持
	It("6-兼容性-Deployment资源类型支持", func() {
		target := TidalTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: TidalTestDeploymentName, Namespace: TidalTestNamespace}
		Expect(CreateTidalCRWithTargetAndRules(dynamicClient, ctx, TidalTestCRCase6Name, TidalTestNamespace, target, Case2TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase6Name, TidalTestNamespace)
		VerifyTidalStatusReady(dynamicClient, ctx, TidalTestCRCase6Name, TidalTestNamespace)
	})

	// 用例7：兼容性 - StatefulSet资源类型支持
	It("7-兼容性-StatefulSet资源类型支持", func() {
		Expect(CreateTestStatefulSet(clientset, ctx, TidalTestStatefulSetName, TidalTestNamespace, 1)).NotTo(HaveOccurred())
		time.Sleep(15 * time.Second)
		target := TidalTargetRef{APIVersion: "apps/v1", Kind: "StatefulSet", Name: TidalTestStatefulSetName, Namespace: TidalTestNamespace}
		Expect(CreateTidalCRWithTargetAndRules(dynamicClient, ctx, TidalTestCRCase7Name, TidalTestNamespace, target, Case2TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase7Name, TidalTestNamespace)
		VerifyTidalStatusReady(dynamicClient, ctx, TidalTestCRCase7Name, TidalTestNamespace)
	})

	// 用例8：兼容性 - DaemonSet资源类型支持
	It("8-兼容性-DaemonSet资源类型支持", func() {
		Expect(CreateTestDaemonSet(clientset, ctx, TidalTestDaemonSetName, TidalTestNamespace)).NotTo(HaveOccurred())
		time.Sleep(15 * time.Second)
		target := TidalTargetRef{APIVersion: "apps/v1", Kind: "DaemonSet", Name: TidalTestDaemonSetName, Namespace: TidalTestNamespace}
		Expect(CreateTidalCRWithTargetAndRules(dynamicClient, ctx, TidalTestCRCase8Name, TidalTestNamespace, target, Case2TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase8Name, TidalTestNamespace)
		VerifyTidalStatusReady(dynamicClient, ctx, TidalTestCRCase8Name, TidalTestNamespace)
	})

	// 用例9：兼容性 - CRD资源类型支持（使用 e2e 自建测试 CRD）
	It("9-兼容性-CRD资源类型支持", func() {
		if err := CreateTestTidalTestResourceCRD(dynamicClient, ctx, TidalTestCRDResourceName, TidalTestNamespace); err != nil {
			GinkgoWriter.Printf("跳过用例9：测试 CRD 创建失败: %v\n", err)
			return
		}
		time.Sleep(10 * time.Second)
		target := TidalTargetRef{APIVersion: "e2e.tidal.io/v1alpha1", Kind: "TidalTestResource", Name: TidalTestCRDResourceName, Namespace: TidalTestNamespace}
		Expect(CreateTidalCRWithTargetAndRules(dynamicClient, ctx, TidalTestCRCase9Name, TidalTestNamespace, target, Case2TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase9Name, TidalTestNamespace)
		VerifyTidalStatusReady(dynamicClient, ctx, TidalTestCRCase9Name, TidalTestNamespace)
	})

	// 用例10：功能验证 - 规则动态更新与重新调度（修改 CR 规则，验证 nextTriggerTime 和副本变更）
	It("10-功能验证-规则动态更新与重新调度", func() {
		Expect(CreateTidalCRWithRules(dynamicClient, ctx, TidalTestCRCase10Name, TidalTestNamespace, Case2TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase10Name, TidalTestNamespace)
		Expect(UpdateTidalCRRules(dynamicClient, ctx, TidalTestCRCase10Name, TidalTestNamespace, Case10TidalRulesNew)).NotTo(HaveOccurred())
		VerifyTidalStatusNextTriggerUpdated(dynamicClient, ctx, TidalTestCRCase10Name, TidalTestNamespace)
		VerifyTidalStatusReady(dynamicClient, ctx, TidalTestCRCase10Name, TidalTestNamespace)
		// 等待新规则（每2分钟扩容到4）生效后，校验 desiredReplicas 及目标副本数
		time.Sleep(130 * time.Second)
		VerifyTidalDesiredReplicas(dynamicClient, ctx, TidalTestCRCase10Name, TidalTestNamespace, 4)
	})

	// 用例11：功能验证 - 下一个触发规则准确性验证（上午9点、下午6点）
	It("11-功能验证-下一个触发规则准确性验证", func() {
		// 这里只关注 nextTriggerRule 的正确性，因此仅使用一个规则，便于断言
		singleRule := []TidalRule{Case11TidalRules[0]} // 只用 morning-9 规则
		Expect(CreateTidalCRWithRules(dynamicClient, ctx, TidalTestCRCase11Name, TidalTestNamespace, singleRule)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase11Name, TidalTestNamespace)
		VerifyTidalStatusNextTriggerRule(dynamicClient, ctx, TidalTestCRCase11Name, TidalTestNamespace, "morning-9")
	})

	// 用例12：功能验证 - 多规则最大值计算验证（同时触发取最大）
	It("12-功能验证-多规则最大值计算验证", func() {
		Expect(CreateTidalCRWithRules(dynamicClient, ctx, TidalTestCRCase12Name, TidalTestNamespace, Case12TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase12Name, TidalTestNamespace)
		time.Sleep(70 * time.Second) // 等待到某分钟0秒触发
		VerifyTidalDesiredReplicas(dynamicClient, ctx, TidalTestCRCase12Name, TidalTestNamespace, 7)
		VerifyTidalStatusActiveRule(dynamicClient, ctx, TidalTestCRCase12Name, TidalTestNamespace, "rule-7")
	})

	// 用例13：功能验证 - 时区敏感调度验证（Beijing），并验证副本变更
	It("13-功能验证-时区敏感调度验证", func() {
		Expect(CreateTidalCRWithRules(dynamicClient, ctx, TidalTestCRCase13Name, TidalTestNamespace, Case13TidalRules)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase13Name, TidalTestNamespace)
		VerifyTidalStatusNextTrigger(dynamicClient, ctx, TidalTestCRCase13Name, TidalTestNamespace)
		VerifyTidalStatusReady(dynamicClient, ctx, TidalTestCRCase13Name, TidalTestNamespace)
		// 使用同一时区下的高频规则（每2分钟）在测试时间窗口内验证 replicas=2
		time.Sleep(130 * time.Second)
		VerifyTidalDesiredReplicas(dynamicClient, ctx, TidalTestCRCase13Name, TidalTestNamespace, 2)
	})

	// 用例14：功能验证 - 高并发处理多CR的相同时间规则
	It("14-功能验证-高并发处理多CR的相同时间规则", func() {
		for _, name := range []string{TidalTestCRCase14Prefix + "-1", TidalTestCRCase14Prefix + "-2", TidalTestCRCase14Prefix + "-3"} {
			Expect(CreateTidalCRWithRules(dynamicClient, ctx, name, TidalTestNamespace, Case14TidalRules)).NotTo(HaveOccurred())
			VerifyTidalCRExists(dynamicClient, ctx, name, TidalTestNamespace)
		}
		time.Sleep(35 * time.Second) // 使用每10秒高频规则，缩短等待时间
		VerifyTidalDesiredReplicas(dynamicClient, ctx, TidalTestCRCase14Prefix+"-1", TidalTestNamespace, 4)
	})

	// 用例15：功能验证 - 高并发处理多CR的不同时间规则（每个 CR 时间规则不同，各用独立目标以验证 replicas）
	It("15-功能验证-高并发处理多CR的不同时间规则", func() {
		// 每个 CR 指向独立 Deployment，便于验证各自规则触发的 desiredReplicas
		Expect(CreateTestDeployment(clientset, ctx, TidalTestDeploymentC15_1, TidalTestNamespace, 1)).NotTo(HaveOccurred())
		Expect(CreateTestDeployment(clientset, ctx, TidalTestDeploymentC15_2, TidalTestNamespace, 1)).NotTo(HaveOccurred())
		Expect(CreateTestDeployment(clientset, ctx, TidalTestDeploymentC15_3, TidalTestNamespace, 1)).NotTo(HaveOccurred())
		time.Sleep(15 * time.Second)
		targets := []TidalTargetRef{
			{APIVersion: "apps/v1", Kind: "Deployment", Name: TidalTestDeploymentC15_1, Namespace: TidalTestNamespace},
			{APIVersion: "apps/v1", Kind: "Deployment", Name: TidalTestDeploymentC15_2, Namespace: TidalTestNamespace},
			{APIVersion: "apps/v1", Kind: "Deployment", Name: TidalTestDeploymentC15_3, Namespace: TidalTestNamespace},
		}
		rules := [][]TidalRule{Case15TidalRules1, Case15TidalRules2, Case15TidalRules3}
		names := []string{TidalTestCRCase15Prefix + "-1", TidalTestCRCase15Prefix + "-2", TidalTestCRCase15Prefix + "-3"}
		for i := range names {
			Expect(CreateTidalCRWithTargetAndRules(dynamicClient, ctx, names[i], TidalTestNamespace, targets[i], rules[i])).NotTo(HaveOccurred())
			VerifyTidalCRExists(dynamicClient, ctx, names[i], TidalTestNamespace)
		}
		time.Sleep(40 * time.Second) // 使用高频规则（10/15/20秒）缩短等待时间
		// 分别验证三个 CR 的 desiredReplicas 是否等于各自规则中配置的副本数
		VerifyTidalDesiredReplicas(dynamicClient, ctx, names[0], TidalTestNamespace, 2)
		VerifyTidalDesiredReplicas(dynamicClient, ctx, names[1], TidalTestNamespace, 3)
		VerifyTidalDesiredReplicas(dynamicClient, ctx, names[2], TidalTestNamespace, 4)
	})

	// 用例16：输入验证 - Labels缺失处理（缺少 app.kubernetes.io/name）
	It("16-输入验证-Labels缺失处理", func() {
		err := CreateTidalCRWithoutRequiredLabels(dynamicClient, ctx, TidalTestCRCase16Name, TidalTestNamespace)
		if err != nil {
			GinkgoWriter.Printf("创建失败（可能被 webhook 拒绝）: %v\n", err)
			return
		}
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase16Name, TidalTestNamespace)
		VerifyTidalConditionsDegradedWithReason(dynamicClient, ctx, TidalTestCRCase16Name, TidalTestNamespace, "MissingRequiredLabels")
	})

	// 用例17：输入验证 - 非法输入与格式处理（无效 cron "* * *"）
	// 对于非法 cron，可能有两种行为：
	// 1）被 CRD/webhook 拒绝，创建直接失败；
	// 2）创建成功，但在 controller 计算 nextTriggerTime 时失败，设置 Degraded，reason=UpdateNextTriggerTimeFailed。
	It("17-输入验证-非法输入与格式处理", func() {
		err := CreateTidalCRWithInvalidCron(dynamicClient, ctx, TidalTestCRCase17Name, TidalTestNamespace)
		if err != nil {
			GinkgoWriter.Printf("非法 cron 在创建阶段被拒绝: %v\n", err)
			return
		}
		// 若创建成功，则应进入 Reconcile，并在 status.conditions 中标记 Degraded, reason=UpdateNextTriggerTimeFailed
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase17Name, TidalTestNamespace)
		VerifyTidalConditionsDegradedWithReason(dynamicClient, ctx, TidalTestCRCase17Name, TidalTestNamespace, "UpdateNextTriggerTimeFailed")
	})

	// 用例18：输入验证 - 重复规则名验证
	It("18-输入验证-重复规则名验证", func() {
		Expect(CreateTidalCRWithDuplicateRuleNames(dynamicClient, ctx, TidalTestCRCase18Name, TidalTestNamespace)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase18Name, TidalTestNamespace)
		VerifyTidalConditionsDegradedWithReason(dynamicClient, ctx, TidalTestCRCase18Name, TidalTestNamespace, "DuplicateRuleNames")
	})

	// 用例19：输入验证 - 目标资源不存在处理
	It("19-输入验证-目标资源不存在处理", func() {
		Expect(CreateTidalCRWithNonExistentTarget(dynamicClient, ctx, TidalTestCRCase19Name, TidalTestNamespace)).NotTo(HaveOccurred())
		VerifyTidalCRExists(dynamicClient, ctx, TidalTestCRCase19Name, TidalTestNamespace)
		VerifyTidalConditionsDegradedWithReason(dynamicClient, ctx, TidalTestCRCase19Name, TidalTestNamespace, "TargetRefNotFound")
	})

	AfterAll(func() {
		CleanupTidalTest(dynamicClient, clientset, ctx)
	})
})
