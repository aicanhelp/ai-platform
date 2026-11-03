package installation

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

var _ = SIGDescribe("Installation Case Matrix", Label("installation-matrix"), func() {
	var (
		sshExecutor    *executor.SSHExecutor
		localExecutor  *executor.LocalExecutor
		guideConfig    *config.GuideNodeConfig
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager
	)

	BeforeEach(func() {
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
		Expect(err).NotTo(HaveOccurred())

		localExecutor = executor.NewLocalExecutor(30 * time.Second)

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred())

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred())

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
	})

	Describe("异常处理完整测试", Label("full-flow", "post-init", "online"), func() {
		var (
			caseCtx              *CaseContext
			mgmtClusterName      string
			mgmtStatusKubeconfig string
			workloadClusterName  string
			faultNode            config.NodeInfo
		)

		AfterEach(func() {
			By("AfterEach: 清理集群")
			if caseCtx != nil && strings.TrimSpace(mgmtClusterName) != "" {
				abnormalDeleteMgmtCluster(caseCtx, mgmtClusterName, mgmtStatusKubeconfig)
			}
			if caseCtx != nil && strings.TrimSpace(mgmtClusterName) == "" && strings.TrimSpace(workloadClusterName) != "" {
				err := caseCtx.ClusterManager.DeleteClusterWithKubeconfig(workloadClusterName, "")
				if err != nil {
					GinkgoWriter.Printf("AfterEach 删除业务集群失败: cluster=%s err=%v\n", workloadClusterName, err)
				} else {
					Eventually(func() bool {
						return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(workloadClusterName, "")
					}, uninstallTimeout, pollInterval).Should(BeTrue(), "业务集群应被删除")
				}
			}
			if caseCtx != nil {
				caseCtx.CleanupGeneratedConfigs()
			}
			if strings.TrimSpace(mgmtClusterName) != "" {
				cleanupRemoteGeneratedYAMLs(caseCtx, mgmtClusterName)
			}
			if strings.TrimSpace(workloadClusterName) != "" {
				cleanupRemoteGeneratedYAMLs(caseCtx, workloadClusterName)
			}
		})

		It("引导节点安装1m1n管理集群-扩容-缩容-注入异常-创建1m1n业务集群-删除集群-异常节点恢复-再次注入异常-异常下扩容-缩容-删除异常节点", Label("skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = fmt.Sprintf("fullflow-1m1n-mgmt-%d", time.Now().Unix())
			mgmtStatusKubeconfig = ""
			workloadClusterName = ""
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{
				ClusterName:    mgmtClusterName,
				SSHExecutor:    sshExecutor,
				LocalExecutor:  localExecutor,
				ClusterManager: clusterManager,
			}

			By("1. 创建 1m1n 管理集群")
			prepareClusterWithParentKubeconfig(caseCtx, mgmtClusterName, ClusterKindMgmt, ClusterSpec1M1N, HANone, "", 0)
			waitClusterReady(caseCtx, mgmtClusterName, mgmtStatusKubeconfig)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "mgmt-installed")

			bootstrapKubeconfig := mgmtStatusKubeconfig
			mgmtSelfClusterName := "cluster"
			checker := utils.NewClusterCheckerWithParentKubeconfig(caseCtx.LocalExecutor, mgmtClusterName, bootstrapKubeconfig)
			mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "获取管理集群 kubeconfig 应成功")
			mgmtKubeconfigPath = ensureKubeconfigReachableOnReadyMaster(caseCtx, mgmtClusterName, bootstrapKubeconfig, mgmtKubeconfigPath)
			caseCtx.MgmtKubeconfigPath = mgmtKubeconfigPath
			caseCtx.ParentClusterName = mgmtClusterName
			caseCtx.AddGeneratedConfigPaths(mgmtKubeconfigPath)

			By("2. 管理集群扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec1M1N, OperationScaleOut, bootstrapKubeconfig)
			AssertClusterStatusEventually(caseCtx, mgmtSelfClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageSucceeded, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtSelfClusterName, mgmtKubeconfigPath, "mgmt-scaleout-1")

			By("3. 管理集群缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec1M1N, OperationScaleIn, bootstrapKubeconfig)
			AssertClusterStatusEventually(caseCtx, mgmtSelfClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageSucceeded, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtSelfClusterName, mgmtKubeconfigPath, "mgmt-scalein-1")

			faultNode = pickFaultTargetNode(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, InstallRoleGuide, ClusterKindMgmt, ClusterSpec1M1N, FaultWorkerDown)
			By("4. 注入异常：停 kubelet")
			injectStopKubeletFault(faultNode)
			mgmtStatusKubeconfig = ensureKubeconfigReachableOnReadyMaster(caseCtx, mgmtClusterName, "", mgmtStatusKubeconfig)
			waitBKEControllerManagerReady(caseCtx, mgmtStatusKubeconfig)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, faultNode.IP, "NotReady")
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "mgmt-stop-kubelet")

			By("5. 创建 1m1n 业务集群")
			workloadClusterName = fmt.Sprintf("fullflow-workload-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec1M1N, HANone, mgmtKubeconfigPath, 3)
			waitBKEClusterCreated(caseCtx, workloadClusterName, mgmtKubeconfigPath)
			waitClusterReady(caseCtx, workloadClusterName, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "workload-installed")

			By("6. 删除业务集群")
			err = caseCtx.ClusterManager.DeleteClusterWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "业务集群应被删除")
			collectClusterRuntimeStatus(caseCtx, mgmtSelfClusterName, mgmtKubeconfigPath, "workload-deleted")

			By("7. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "abnormal-node-recovered")

			By("8. 再次注入异常")
			injectStopKubeletFault(faultNode)
			mgmtStatusKubeconfig = ensureKubeconfigReachableOnReadyMaster(caseCtx, mgmtClusterName, "", mgmtStatusKubeconfig)
			waitBKEControllerManagerReady(caseCtx, mgmtStatusKubeconfig)
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering", "Unhealthy"}}, installTimeout)

			By("9. 异常下再次扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec1M1N, OperationScaleOut, mgmtStatusKubeconfig)
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty(), "异常扩容后应记录扩容节点 IP")
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "mgmt-scaleout-2")

			By("10. 异常下再次缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec1M1N, OperationScaleIn, mgmtStatusKubeconfig)
			waitBKENodeDeleted(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "mgmt-scalein-2")

			By("11. 在管理集群上删除异常节点")
			targetClusterName := "cluster"
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(targetClusterName, faultNode.IP, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "管理集群上删除异常节点应成功")
			waitBKENodeDeleted(caseCtx, targetClusterName, mgmtKubeconfigPath, faultNode.IP)
			AssertClusterStatusEventually(caseCtx, targetClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, targetClusterName, mgmtKubeconfigPath, "1m1n-abnormal-node-deleted")

			By("12. 用例结束")
		})

		It("引导节点安装3m1n管理集群-扩容-缩容-注入异常-创建1m1n业务集群-删除集群-异常节点恢复-再次注入异常-异常下扩容-缩容-删除异常节点", Label("skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = fmt.Sprintf("fullflow-3m1n-mgmt-%d", time.Now().Unix())
			mgmtStatusKubeconfig = ""
			workloadClusterName = ""
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{
				ClusterName:    mgmtClusterName,
				SSHExecutor:    sshExecutor,
				LocalExecutor:  localExecutor,
				ClusterManager: clusterManager,
			}

			By("1. 创建 3m1n 高可用管理集群")
			prepareClusterWithParentKubeconfig(caseCtx, mgmtClusterName, ClusterKindMgmt, ClusterSpec3M1N, HAEnabled, "", 0)
			waitClusterReady(caseCtx, mgmtClusterName, mgmtStatusKubeconfig)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-installed")

			By("2. 管理集群扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, OperationScaleOut, mgmtStatusKubeconfig)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageSucceeded, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-scaleout-1")

			By("3. 管理集群缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, OperationScaleIn, mgmtStatusKubeconfig)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageSucceeded, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-scalein-1")

			faultNode = pickFaultTargetNode(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, FaultWorkerDown)
			By("4. 注入异常：停 kubelet")
			injectStopKubeletFault(faultNode)
			waitBKEControllerManagerReady(caseCtx, mgmtStatusKubeconfig)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering", "Unhealthy"}}, installTimeout)
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, faultNode.IP, "NotReady")
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-stop-kubelet")

			By("5. 创建 1m1n 业务集群")
			checker := utils.NewClusterCheckerWithParentKubeconfig(caseCtx.LocalExecutor, mgmtClusterName, mgmtStatusKubeconfig)
			mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "获取管理集群 kubeconfig 应成功")
			caseCtx.MgmtKubeconfigPath = mgmtKubeconfigPath
			caseCtx.ParentClusterName = mgmtClusterName
			caseCtx.AddGeneratedConfigPaths(mgmtKubeconfigPath)

			workloadClusterName = fmt.Sprintf("fullflow-3m1n-workload-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec1M1N, HANone, mgmtKubeconfigPath, 4)
			waitClusterReady(caseCtx, workloadClusterName, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "3m1n-workload-installed")

			By("6. 删除业务集群")
			err = caseCtx.ClusterManager.DeleteClusterWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "业务集群应被删除")
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-workload-deleted")

			By("7. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-abnormal-node-recovered")

			By("8. 再次注入异常")
			injectStopKubeletFault(faultNode)
			waitBKEControllerManagerReady(caseCtx, mgmtStatusKubeconfig)
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering", "Unhealthy"}}, installTimeout)

			By("9. 异常下再次扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, OperationScaleOut, mgmtStatusKubeconfig)
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty(), "异常扩容后应记录扩容节点 IP")
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-scaleout-2")

			By("10. 异常下再次缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, OperationScaleIn, mgmtStatusKubeconfig)
			waitBKENodeDeleted(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{Stage: StatusStageSucceeded, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-scalein-2")

			By("11. 在管理集群上删除异常节点")
			targetClusterName := "cluster"
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(targetClusterName, faultNode.IP, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "管理集群上删除异常节点应成功")
			waitBKENodeDeleted(caseCtx, targetClusterName, mgmtKubeconfigPath, faultNode.IP)
			AssertClusterStatusEventually(caseCtx, targetClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, targetClusterName, mgmtKubeconfigPath, "3m1n-abnormal-node-deleted")

			By("12. 用例结束")
		})

		It("管理节点安装1m1n业务集群-注入异常1worker-创建1m业务-删除1m业务-异常节点恢复-再次注入异常1worker-异常下扩容-缩容-删除异常节点", Label("skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = fmt.Sprintf("fullflow-workload-parent-%d", time.Now().Unix())
			mgmtStatusKubeconfig = ""
			workloadClusterName = ""
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{ClusterName: mgmtClusterName, SSHExecutor: sshExecutor, LocalExecutor: localExecutor, ClusterManager: clusterManager}

			By("1. 引导节点安装 1m 管理集群")
			prepareClusterWithParentKubeconfig(caseCtx, mgmtClusterName, ClusterKindMgmt, ClusterSpec1M, HANone, "", 0)
			waitClusterReady(caseCtx, mgmtClusterName, mgmtStatusKubeconfig)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "mgmt-1m-installed")

			checker := utils.NewClusterCheckerWithParentKubeconfig(caseCtx.LocalExecutor, mgmtClusterName, mgmtStatusKubeconfig)
			mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "获取管理集群 kubeconfig 应成功")
			caseCtx.MgmtKubeconfigPath = mgmtKubeconfigPath
			caseCtx.ParentClusterName = mgmtClusterName
			caseCtx.AddGeneratedConfigPaths(mgmtKubeconfigPath)

			By("2. 在管理集群安装 1m1n 业务集群")
			workloadClusterName = fmt.Sprintf("fullflow-workload-1m1n-%d", time.Now().Unix())
			caseCtx.ClusterName = workloadClusterName
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec1M1N, HANone, mgmtKubeconfigPath, 1)
			waitClusterReady(caseCtx, workloadClusterName, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "workload-1m1n-installed")

			faultNode = pickFaultTargetNode(caseCtx, workloadClusterName, mgmtKubeconfigPath, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec1M1N, FaultWorkerDown)
			By("3. 注入异常：业务集群 worker 停 kubelet")
			injectStopKubeletFault(faultNode)
			waitBKEControllerManagerReady(caseCtx, mgmtKubeconfigPath)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP, "NotReady")

			By("4. 创建 1m 业务集群（仅 master）")
			subWorkload := fmt.Sprintf("fullflow-workload-1m-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, subWorkload, ClusterKindWorkload, ClusterSpec1M, HANone, mgmtKubeconfigPath, 3)
			waitClusterReady(caseCtx, subWorkload, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, subWorkload, mgmtKubeconfigPath, "workload-1m-created")

			By("5. 删除 1m 业务集群")
			err = caseCtx.ClusterManager.DeleteClusterWithKubeconfig(subWorkload, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除 1m 业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(subWorkload, mgmtKubeconfigPath)
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "1m 业务集群应被删除")

			By("6. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)

			By("7. 再次注入异常")
			injectStopKubeletFault(faultNode)
			waitBKEControllerManagerReady(caseCtx, mgmtKubeconfigPath)
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)

			By("8. 异常下扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec1M1N, OperationScaleOut, mgmtKubeconfigPath)
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty())
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"}}, installTimeout)

			By("9. 异常下缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec1M1N, OperationScaleIn, mgmtKubeconfigPath)
			waitBKENodeDeleted(caseCtx, workloadClusterName, mgmtKubeconfigPath, scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy"}}, installTimeout)

			By("10. 删除异常节点")
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(workloadClusterName, faultNode.IP, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除异常节点应成功")
			waitBKENodeDeleted(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
		})

		It("引导节点安装1m1n业务集群-注入异常1worker-创建1m业务-删除1m业务-异常节点恢复-再次注入异常1worker-异常下扩容-缩容-删除异常节点", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = ""
			mgmtStatusKubeconfig = ""
			workloadClusterName = ""
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{ClusterName: "", SSHExecutor: sshExecutor, LocalExecutor: localExecutor, ClusterManager: clusterManager}

			By("1. 引导节点安装 1m1n 业务集群")
			workloadClusterName = fmt.Sprintf("fullflow-workload-1m1n-%d", time.Now().Unix())
			caseCtx.ClusterName = workloadClusterName
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec1M1N, HANone, "", 0)
			waitClusterReady(caseCtx, workloadClusterName, "")
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "workload-1m1n-installed")

			faultNode = pickFaultTargetNode(caseCtx, workloadClusterName, "", InstallRoleGuide, ClusterKindWorkload, ClusterSpec1M1N, FaultWorkerDown)
			By("2. 注入异常：业务集群 worker 停 kubelet")
			injectStopKubeletFault(faultNode)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)
			waitBKENodeState(caseCtx, workloadClusterName, "", faultNode.IP, "NotReady")

			By("3. 创建 1m 业务集群（仅 master）")
			subWorkload := fmt.Sprintf("fullflow-workload-1m-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, subWorkload, ClusterKindWorkload, ClusterSpec1M, HANone, "", 2)
			waitClusterReady(caseCtx, subWorkload, "")
			collectClusterRuntimeStatus(caseCtx, subWorkload, "", "workload-1m-created")

			By("4. 删除 1m 业务集群")
			err := caseCtx.ClusterManager.DeleteClusterWithKubeconfig(subWorkload, "")
			Expect(err).NotTo(HaveOccurred(), "删除 1m 业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(subWorkload, "")
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "1m 业务集群应被删除")

			By("5. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)

			By("6. 再次注入异常")
			injectStopKubeletFault(faultNode)
			waitBKENodeState(caseCtx, workloadClusterName, "", faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)

			By("7. 异常下扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindWorkload, ClusterSpec1M1N, OperationScaleOut, "")
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty())
			waitBKENodeState(caseCtx, workloadClusterName, "", scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"}}, installTimeout)

			By("8. 异常下缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindWorkload, ClusterSpec1M1N, OperationScaleIn, "")
			waitBKENodeDeleted(caseCtx, workloadClusterName, "", scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy"}}, installTimeout)

			By("9. 删除异常节点")
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(workloadClusterName, faultNode.IP, "")
			Expect(err).NotTo(HaveOccurred(), "删除异常节点应成功")
			waitBKENodeDeleted(caseCtx, workloadClusterName, "", faultNode.IP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
		})

		It("管理节点安装3m1n业务集群-注入异常1worker-创建1m业务-删除1m业务-异常节点恢复-再次注入异常1worker-异常下扩容-缩容-删除异常节点", Label("skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = fmt.Sprintf("fullflow-workload3m1n-parent-%d", time.Now().Unix())
			mgmtStatusKubeconfig = ""
			workloadClusterName = ""
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{ClusterName: mgmtClusterName, SSHExecutor: sshExecutor, LocalExecutor: localExecutor, ClusterManager: clusterManager}

			By("1. 引导节点安装 1m 管理集群")
			prepareClusterWithParentKubeconfig(caseCtx, mgmtClusterName, ClusterKindMgmt, ClusterSpec1M, HANone, "", 0)
			waitClusterReady(caseCtx, mgmtClusterName, mgmtStatusKubeconfig)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "mgmt-1m-installed")

			checker := utils.NewClusterCheckerWithParentKubeconfig(caseCtx.LocalExecutor, mgmtClusterName, mgmtStatusKubeconfig)
			mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "获取管理集群 kubeconfig 应成功")
			caseCtx.MgmtKubeconfigPath = mgmtKubeconfigPath
			caseCtx.ParentClusterName = mgmtClusterName
			caseCtx.AddGeneratedConfigPaths(mgmtKubeconfigPath)

			By("2. 在管理集群安装 3m1n 高可用业务集群")
			workloadClusterName = fmt.Sprintf("fullflow-workload-3m1n-ha-%d", time.Now().Unix())
			caseCtx.ClusterName = workloadClusterName
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec3M1N, HAEnabled, mgmtKubeconfigPath, 1)
			waitClusterReady(caseCtx, workloadClusterName, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "workload-3m1n-ha-installed")

			faultNode = pickFaultTargetNode(caseCtx, workloadClusterName, mgmtKubeconfigPath, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec3M1N, FaultWorkerDown)
			By("3. 注入异常：业务集群 worker 停 kubelet")
			injectStopKubeletFault(faultNode)
			waitBKEControllerManagerReady(caseCtx, mgmtKubeconfigPath)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP, "NotReady")

			By("4. 创建 1m 业务集群（仅 master）")
			subWorkload := fmt.Sprintf("fullflow-ha-subworkload-1m-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, subWorkload, ClusterKindWorkload, ClusterSpec1M, HANone, mgmtKubeconfigPath, 5)
			waitClusterReady(caseCtx, subWorkload, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, subWorkload, mgmtKubeconfigPath, "workload-1m-created")

			By("5. 删除 1m 业务集群")
			err = caseCtx.ClusterManager.DeleteClusterWithKubeconfig(subWorkload, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除 1m 业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(subWorkload, mgmtKubeconfigPath)
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "1m 业务集群应被删除")

			By("6. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)

			By("7. 再次注入异常")
			injectStopKubeletFault(faultNode)
			waitBKEControllerManagerReady(caseCtx, mgmtKubeconfigPath)
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)

			By("8. 异常下扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleOut, mgmtKubeconfigPath)
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty())
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"}}, installTimeout)

			By("9. 异常下缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleIn, mgmtKubeconfigPath)
			waitBKENodeDeleted(caseCtx, workloadClusterName, mgmtKubeconfigPath, scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy"}}, installTimeout)

			By("10. 删除异常节点")
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(workloadClusterName, faultNode.IP, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除异常节点应成功")
			waitBKENodeDeleted(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
		})

		It("引导节点安装3m1n业务集群-注入异常1worker-创建1m业务-删除1m业务-异常节点恢复-再次注入异常1worker-异常下扩容-缩容-删除异常节点", Label("skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = ""
			mgmtStatusKubeconfig = ""
			workloadClusterName = fmt.Sprintf("fullflow-workload-3m1n-ha-%d", time.Now().Unix())
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{ClusterName: "", SSHExecutor: sshExecutor, LocalExecutor: localExecutor, ClusterManager: clusterManager}

			By("1. 引导节点安装 3m1n 高可用业务集群")
			workloadClusterName = fmt.Sprintf("fullflow-workload-3m1n-ha-%d", time.Now().Unix())
			caseCtx.ClusterName = workloadClusterName
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec3M1N, HAEnabled, "", 0)
			waitClusterReady(caseCtx, workloadClusterName, "")
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "workload-3m1n-ha-installed")

			faultNode = pickFaultTargetNode(caseCtx, workloadClusterName, "", InstallRoleGuide, ClusterKindWorkload, ClusterSpec3M1N, FaultWorkerDown)
			By("2. 注入异常：业务集群 worker 停 kubelet")
			injectStopKubeletFault(faultNode)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)
			waitBKENodeState(caseCtx, workloadClusterName, "", faultNode.IP, "NotReady")

			By("3. 创建 1m 业务集群（仅 master）")
			subWorkload := fmt.Sprintf("fullflow-ha-subworkload-1m-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, subWorkload, ClusterKindWorkload, ClusterSpec1M, HANone, "", 4)
			waitClusterReady(caseCtx, subWorkload, "")
			collectClusterRuntimeStatus(caseCtx, subWorkload, "", "workload-1m-created")

			By("4. 删除 1m 业务集群")
			err := caseCtx.ClusterManager.DeleteClusterWithKubeconfig(subWorkload, "")
			Expect(err).NotTo(HaveOccurred(), "删除 1m 业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(subWorkload, "")
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "1m 业务集群应被删除")

			By("5. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)

			By("6. 再次注入异常 1 worker")
			injectStopKubeletFault(faultNode)
			waitBKENodeState(caseCtx, workloadClusterName, "", faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"}}, installTimeout)

			By("7. 异常下扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleOut, "")
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty())
			waitBKENodeState(caseCtx, workloadClusterName, "", scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"}}, installTimeout)

			By("8. 异常下缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleIn, "")
			waitBKENodeDeleted(caseCtx, workloadClusterName, "", scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageInProgress, AllowedStates: []string{"Unhealthy"}, AllowedStatus: []string{"Unhealthy"}}, installTimeout)

			By("9. 删除异常节点")
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(workloadClusterName, faultNode.IP, "")
			Expect(err).NotTo(HaveOccurred(), "删除异常节点应成功")
			waitBKENodeDeleted(caseCtx, workloadClusterName, "", faultNode.IP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{Stage: StatusStageRecovered, AllowedStates: []string{"Healthy"}, AllowedStatus: []string{"Ready"}}, installTimeout)
		})

		It("引导节点安装3m1n管理集群-1master异常-扩容-缩容-创建1m业务-删除1m业务-恢复异常节点-再次注入异常-异常下扩容-缩容-删除异常节点", Label("skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = fmt.Sprintf("fullflow-3m1n-mgmt-master-%d", time.Now().Unix())
			mgmtStatusKubeconfig = ""
			workloadClusterName = ""
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{
				ClusterName:    mgmtClusterName,
				SSHExecutor:    sshExecutor,
				LocalExecutor:  localExecutor,
				ClusterManager: clusterManager,
			}

			By("1. 创建 3m1n 高可用管理集群")
			prepareClusterWithParentKubeconfig(caseCtx, mgmtClusterName, ClusterKindMgmt, ClusterSpec3M1N, HAEnabled, "", 0)
			waitClusterReady(caseCtx, mgmtClusterName, mgmtStatusKubeconfig)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-installed")

			checker := utils.NewClusterCheckerWithParentKubeconfig(caseCtx.LocalExecutor, mgmtClusterName, mgmtStatusKubeconfig)
			mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "获取管理集群 kubeconfig 应成功")
			mgmtKubeconfigPath = ensureKubeconfigReachableOnReadyMaster(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, mgmtKubeconfigPath)
			caseCtx.MgmtKubeconfigPath = mgmtKubeconfigPath
			caseCtx.ParentClusterName = mgmtClusterName
			caseCtx.AddGeneratedConfigPaths(mgmtKubeconfigPath)

			By("2. 管理集群扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, OperationScaleOut, mgmtStatusKubeconfig)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-scaleout-1")

			By("3. 管理集群缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, OperationScaleIn, mgmtStatusKubeconfig)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-scalein-1")

			faultNode = pickFaultTargetNode(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, FaultMasterDown)
			By("4. 注入异常：停 kubelet")
			injectStopKubeletFault(faultNode)
			mgmtStatusKubeconfig = ensureKubeconfigReachableOnReadyMaster(caseCtx, mgmtClusterName, "", mgmtStatusKubeconfig)
			waitBKEControllerManagerReady(caseCtx, mgmtStatusKubeconfig)
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-master-down")

			By("5. 创建 1m 业务集群")
			workloadClusterName = fmt.Sprintf("fullflow-3m1n-mgmt-workload-1m-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec1M, HANone, mgmtKubeconfigPath, 4)
			waitClusterReady(caseCtx, workloadClusterName, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "3m1n-mgmt-workload-1m-installed")

			By("6. 删除 1m 业务集群")
			err = caseCtx.ClusterManager.DeleteClusterWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除 1m 业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "1m 业务集群应被删除")
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-workload-1m-deleted")

			By("7. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{
				Stage:         StatusStageRecovered,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-master-recovered")

			By("8. 再次注入异常")
			injectStopKubeletFault(faultNode)
			mgmtStatusKubeconfig = ensureKubeconfigReachableOnReadyMaster(caseCtx, mgmtClusterName, "", mgmtStatusKubeconfig)
			waitBKEControllerManagerReady(caseCtx, mgmtStatusKubeconfig)
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering", "Unhealthy"},
			}, installTimeout)

			By("9. 异常下再次扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, OperationScaleOut, mgmtStatusKubeconfig)
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty(), "异常扩容后应记录扩容节点 IP")
			waitBKENodeState(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-master-scaleout-2")

			By("10. 异常下再次缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindMgmt, ClusterSpec3M1N, OperationScaleIn, mgmtStatusKubeconfig)
			waitBKENodeDeleted(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "3m1n-mgmt-master-scalein-2")

			By("11. 在管理集群上删除异常节点")
			targetClusterName := "cluster"
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(targetClusterName, faultNode.IP, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "管理集群上删除异常节点应成功")
			waitBKENodeDeleted(caseCtx, targetClusterName, mgmtKubeconfigPath, faultNode.IP)
			AssertClusterStatusEventually(caseCtx, targetClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageRecovered,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, targetClusterName, mgmtKubeconfigPath, "3m1n-mgmt-master-deleted")
		})

		It("引导节点安装3m1n业务集群-1master异常-扩容-缩容-创建1m业务-删除1m业务-恢复异常节点-再次注入异常-异常下扩容-缩容-删除异常节点", Label("skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = ""
			mgmtStatusKubeconfig = ""
			workloadClusterName = fmt.Sprintf("fullflow-3m1n-workload-master-%d", time.Now().Unix())
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{
				ClusterName:    workloadClusterName,
				SSHExecutor:    sshExecutor,
				LocalExecutor:  localExecutor,
				ClusterManager: clusterManager,
			}

			By("1. 创建 3m1n 高可用业务集群")
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec3M1N, HAEnabled, "", 0)
			waitClusterReady(caseCtx, workloadClusterName, "")
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-installed")

			By("2. 业务集群扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleOut, "")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-scaleout-1")

			By("3. 业务集群缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleIn, "")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-scalein-1")

			faultNode = pickFaultTargetNode(caseCtx, workloadClusterName, "", InstallRoleGuide, ClusterKindWorkload, ClusterSpec3M1N, FaultMasterDown)
			By("4. 注入异常：停 kubelet")
			injectStopKubeletFault(faultNode)
			waitBKEControllerManagerReady(caseCtx, "")
			waitBKENodeState(caseCtx, workloadClusterName, "", faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-master-down")

			By("5. 创建 1m 业务集群")
			subWorkload := fmt.Sprintf("fullflow-3m1n-workload-1m-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, subWorkload, ClusterKindWorkload, ClusterSpec1M, HANone, "", 4)
			waitClusterReady(caseCtx, subWorkload, "")
			collectClusterRuntimeStatus(caseCtx, subWorkload, "", "3m1n-workload-1m-installed")

			By("6. 删除 1m 业务集群")
			err := caseCtx.ClusterManager.DeleteClusterWithKubeconfig(subWorkload, "")
			Expect(err).NotTo(HaveOccurred(), "删除 1m 业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(subWorkload, "")
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "1m 业务集群应被删除")
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-1m-deleted")

			By("7. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{
				Stage:         StatusStageRecovered,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-master-recovered")

			By("8. 再次注入异常")
			injectStopKubeletFault(faultNode)
			waitBKENodeState(caseCtx, workloadClusterName, "", faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"},
			}, installTimeout)

			By("9. 异常下再次扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleOut, "")
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty(), "异常扩容后应记录扩容节点 IP")
			waitBKENodeState(caseCtx, workloadClusterName, "", scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-master-scaleout-2")

			By("10. 异常下再次缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleGuide, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleIn, "")
			waitBKENodeDeleted(caseCtx, workloadClusterName, "", scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-master-scalein-2")

			By("11. 删除异常节点")
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(workloadClusterName, faultNode.IP, "")
			Expect(err).NotTo(HaveOccurred(), "删除异常节点应成功")
			waitBKENodeDeleted(caseCtx, workloadClusterName, "", faultNode.IP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, "", ClusterStatusExpectation{
				Stage:         StatusStageRecovered,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, "", "3m1n-workload-master-deleted")
		})

		It("管理节点安装3m1n业务集群-1master异常-扩容-缩容-创建1m业务-删除1m业务-恢复异常节点-再次注入异常-异常下扩容-缩容-删除异常节点", Label("skip-temporarily"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			mgmtClusterName = fmt.Sprintf("fullflow-3m1n-workload-parent-%d", time.Now().Unix())
			mgmtStatusKubeconfig = ""
			workloadClusterName = fmt.Sprintf("fullflow-3m1n-workload-master-%d", time.Now().Unix())
			faultNode = config.NodeInfo{}
			caseCtx = &CaseContext{
				ClusterName:    mgmtClusterName,
				SSHExecutor:    sshExecutor,
				LocalExecutor:  localExecutor,
				ClusterManager: clusterManager,
			}

			By("1. 创建 1m 管理集群")
			prepareClusterWithParentKubeconfig(caseCtx, mgmtClusterName, ClusterKindMgmt, ClusterSpec1M, HANone, "", 0)
			waitClusterReady(caseCtx, mgmtClusterName, mgmtStatusKubeconfig)
			collectClusterRuntimeStatus(caseCtx, mgmtClusterName, mgmtStatusKubeconfig, "mgmt-1m-installed")

			checker := utils.NewClusterCheckerWithParentKubeconfig(caseCtx.LocalExecutor, mgmtClusterName, mgmtStatusKubeconfig)
			mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "获取管理集群 kubeconfig 应成功")
			caseCtx.MgmtKubeconfigPath = mgmtKubeconfigPath
			caseCtx.ParentClusterName = mgmtClusterName
			caseCtx.AddGeneratedConfigPaths(mgmtKubeconfigPath)

			By("2. 在管理集群安装 3m1n 高可用业务集群")
			prepareClusterWithParentKubeconfig(caseCtx, workloadClusterName, ClusterKindWorkload, ClusterSpec3M1N, HAEnabled, mgmtKubeconfigPath, 1)
			waitClusterReady(caseCtx, workloadClusterName, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "3m1n-workload-installed")
			caseCtx.ClusterName = workloadClusterName

			By("3. 业务集群扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleOut, mgmtKubeconfigPath)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "3m1n-workload-scaleout-1")

			By("4. 业务集群缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleIn, mgmtKubeconfigPath)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "3m1n-workload-scalein-1")

			faultNode = pickFaultTargetNode(caseCtx, workloadClusterName, mgmtKubeconfigPath, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec3M1N, FaultMasterDown)
			By("5. 注入异常：停 kubelet")
			injectStopKubeletFault(faultNode)
			waitBKEControllerManagerReady(caseCtx, mgmtKubeconfigPath)
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"},
			}, installTimeout)
			collectClusterRuntimeStatus(caseCtx, workloadClusterName, mgmtKubeconfigPath, "3m1n-workload-master-down")

			By("6. 创建 1m 业务集群")
			subWorkload := fmt.Sprintf("fullflow-3m1n-workload-1m-%d", time.Now().Unix())
			prepareClusterWithParentKubeconfig(caseCtx, subWorkload, ClusterKindWorkload, ClusterSpec1M, HANone, mgmtKubeconfigPath, 5)
			waitClusterReady(caseCtx, subWorkload, mgmtKubeconfigPath)
			collectClusterRuntimeStatus(caseCtx, subWorkload, mgmtKubeconfigPath, "3m1n-workload-1m-installed")

			By("7. 删除 1m 业务集群")
			err = caseCtx.ClusterManager.DeleteClusterWithKubeconfig(subWorkload, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除 1m 业务集群触发应成功")
			Eventually(func() bool {
				return !caseCtx.ClusterManager.ClusterExistsWithKubeconfig(subWorkload, mgmtKubeconfigPath)
			}, uninstallTimeout, pollInterval).Should(BeTrue(), "1m 业务集群应被删除")

			By("8. 恢复异常节点")
			recoverKubeletOnNode(faultNode)
			waitKubeletActive(faultNode)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageRecovered,
				AllowedStates: []string{"Healthy"},
				AllowedStatus: []string{"Ready"},
			}, installTimeout)

			By("9. 再次注入异常")
			injectStopKubeletFault(faultNode)
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP, "NotReady")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Initializing", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "Recovering"},
			}, installTimeout)

			By("10. 异常下再次扩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleOut, mgmtKubeconfigPath)
			scaleOutNodeIP := strings.TrimSpace(caseCtx.LastScaleOutNodeIP)
			Expect(scaleOutNodeIP).NotTo(BeEmpty())
			waitBKENodeState(caseCtx, workloadClusterName, mgmtKubeconfigPath, scaleOutNodeIP, "Ready")
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageInProgress,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"},
			}, installTimeout)

			By("11. 异常下再次缩容")
			executeScaleOperationWithKubeconfig(caseCtx, InstallRoleMgmt, ClusterKindWorkload, ClusterSpec3M1N, OperationScaleIn, mgmtKubeconfigPath)
			waitBKENodeDeleted(caseCtx, workloadClusterName, mgmtKubeconfigPath, scaleOutNodeIP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageSucceeded,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy"},
			}, installTimeout)

			By("12. 删除异常节点")
			err = caseCtx.ClusterManager.ScaleInNodeWithKubeconfig(workloadClusterName, faultNode.IP, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "删除异常节点失败")
			waitBKENodeDeleted(caseCtx, workloadClusterName, mgmtKubeconfigPath, faultNode.IP)
			AssertClusterStatusEventually(caseCtx, workloadClusterName, mgmtKubeconfigPath, ClusterStatusExpectation{
				Stage:         StatusStageRecovered,
				AllowedStates: []string{"Unhealthy"},
				AllowedStatus: []string{"Unhealthy", "Checking", "Recovering", "ScalingWorkerNodesUp", "ScalingWorkerNodesDown"},
			}, installTimeout)
		})
	})
})

func executeScaleOperationWithKubeconfig(ctx *CaseContext, role InstallRole, kind ClusterKind, spec ClusterSpec, op OperationType, kubeconfig string) {
	if op == OperationScaleOut {
		node := pickScaleOutNode(role, kind, spec)
		err := ctx.ClusterManager.ScaleOutNodeWithKubeconfig(ctx.ClusterName, node, kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "扩容应成功")
		ctx.LastScaleOutNodeIP = node.IP
		waitBKENodeState(ctx, ctx.ClusterName, kubeconfig, node.IP, "Ready")
		return
	}

	if op == OperationScaleIn {
		nodeIP := pickScaleInNodeIP(ctx, role, kind, spec)
		err := ctx.ClusterManager.ScaleInNodeWithKubeconfig(ctx.ClusterName, nodeIP, kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "缩容应成功")
		waitBKENodeDeleted(ctx, ctx.ClusterName, kubeconfig, nodeIP)
	}
}

func wireCommonHooks(tc *InstallationCase) {
	if tc.Prepare == nil {
		tc.Prepare = func(ctx *CaseContext) {
			By("准备基线集群")
			if tc.InstallRole == InstallRoleMgmt && tc.ClusterKind == ClusterKindWorkload {
				prepareMgmtThenInstallWorkload(ctx, tc)
				if tc.InjectFault != nil && tc.Fault != FaultNone && tc.FaultInjectPhase == FaultBeforeOperation {
					By("在基线拉起过程中注入故障")
					tc.InjectFault(ctx)
					tc.InjectFault = nil
				}
				waitClusterReady(ctx, ctx.ParentClusterName, "")
				waitClusterReady(ctx, tc.ClusterName, ctx.MgmtKubeconfigPath)
				return
			}
			prepareClusterWithParentKubeconfig(ctx, tc.ClusterName, tc.ClusterKind, tc.Spec, tc.HA, "", 0)
			if tc.InjectFault != nil && tc.Fault != FaultNone && tc.FaultInjectPhase == FaultBeforeOperation {
				By("在基线拉起过程中注入故障")
				tc.InjectFault(ctx)
				tc.InjectFault = nil
			}
			waitClusterReady(ctx, tc.ClusterName, "")
		}
	}

	if tc.Cleanup == nil {
		tc.Cleanup = func(ctx *CaseContext) {
			By("清理测试资源（仅清理配置文件，不执行集群卸载）")
			ctx.CleanupGeneratedConfigs()
		}
	}
}

func prepareMgmtThenInstallWorkload(ctx *CaseContext, tc *InstallationCase) {
	mgmtName := fmt.Sprintf("mgmt-parent-%s", tc.ClusterName)
	prepareClusterWithParentKubeconfig(ctx, mgmtName, ClusterKindMgmt, ClusterSpec1M1N, HANone, "", 0)

	checker := utils.NewClusterCheckerWithParentKubeconfig(ctx.LocalExecutor, mgmtName, "")
	mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
	Expect(err).NotTo(HaveOccurred(), "获取管理集群 kubeconfig 应成功")

	ctx.ParentClusterName = mgmtName
	ctx.MgmtKubeconfigPath = mgmtKubeconfigPath

	// 避免与管理集群节点冲突，业务集群从 offset=2 开始选节点
	prepareClusterWithParentKubeconfig(ctx, tc.ClusterName, ClusterKindWorkload, tc.Spec, tc.HA, mgmtKubeconfigPath, 2)
}

func prepareClusterWithParentKubeconfig(ctx *CaseContext, clusterName string, kind ClusterKind, spec ClusterSpec, ha HAType, parentKubeconfig string, offset int) {
	if kind == ClusterKindWorkload {
		waitWebhookReady(ctx, parentKubeconfig)
	}

	nodes := config.LoadTestNodesFromEnv()
	clusterNodes := pickNodesForSpec(nodes, spec, offset)
	cfg := buildClusterConfig(clusterName, kind, clusterNodes, ha)

	configPath, nodeConfigPath, err := ctx.ClusterManager.GetConfigGenerator().GenerateAndUpload(cfg)
	Expect(err).NotTo(HaveOccurred(), "生成配置文件应该成功")
	ctx.AddGeneratedConfigPaths(configPath, nodeConfigPath)

	err = ctx.ClusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, parentKubeconfig)
	Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")
}

func pickNodesForSpec(nodes []config.NodeInfo, spec ClusterSpec, offset int) []config.NodeInfo {
	switch spec {
	case ClusterSpec1M:
		Expect(len(nodes)).To(BeNumerically(">", offset), "1m 节点数量不足")
		master := nodes[offset]
		master.Role = []string{"master/node", "etcd"}
		return []config.NodeInfo{master}
	case ClusterSpec1M1N:
		Expect(len(nodes)).To(BeNumerically(">=", offset+2), "1m1n 节点数量不足")
		master := nodes[offset]
		master.Role = []string{"master/node", "etcd"}
		worker := nodes[offset+1]
		worker.Role = []string{"node"}
		return []config.NodeInfo{master, worker}
	case ClusterSpec3M1N:
		Expect(len(nodes)).To(BeNumerically(">=", offset+4), "3m1n 节点数量不足")
		m1 := nodes[offset]
		m1.Role = []string{"master/node", "etcd"}
		m2 := nodes[offset+1]
		m2.Role = []string{"master/node", "etcd"}
		m3 := nodes[offset+2]
		m3.Role = []string{"master/node", "etcd"}
		w1 := nodes[offset+3]
		w1.Role = []string{"node"}
		return []config.NodeInfo{m1, m2, m3, w1}
	case ClusterSpec3M2N:
		Expect(len(nodes)).To(BeNumerically(">=", offset+5), "3m2n 节点数量不足")
		m1 := nodes[offset]
		m1.Role = []string{"master/node", "etcd"}
		m2 := nodes[offset+1]
		m2.Role = []string{"master/node", "etcd"}
		m3 := nodes[offset+2]
		m3.Role = []string{"master/node", "etcd"}
		w1 := nodes[offset+3]
		w1.Role = []string{"node"}
		w2 := nodes[offset+4]
		w2.Role = []string{"node"}
		return []config.NodeInfo{m1, m2, m3, w1, w2}
	default:
		Fail("不支持的规格")
		return nil
	}
}

func buildClusterConfig(clusterName string, kind ClusterKind, nodes []config.NodeInfo, ha HAType) *config.BKEClusterConfig {
	var cfg *config.BKEClusterConfig
	if kind == ClusterKindMgmt {
		cfg = config.NewBKEClusterConfigForMgmt(clusterName, nodes)
	} else {
		cfg = config.NewDefaultBKEClusterConfig(clusterName, nodes)
	}

	if ha == HAEnabled {
		vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
		if kind == ClusterKindWorkload {
			vipHost, vipPort = config.LoadWorkloadClusterVIPFromEnv()
		}
		if vipHost == "" {
			vipHost = nodes[0].IP
		}
		cfg.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
	}
	return cfg
}

func pickScaleOutNode(role InstallRole, kind ClusterKind, spec ClusterSpec) config.NodeInfo {
	nodes := config.LoadTestNodesFromEnv()
	offset := 0
	if role == InstallRoleMgmt && kind == ClusterKindWorkload {
		offset = 2
	}

	var idx int
	if spec == ClusterSpec1M1N {
		idx = offset + 2
	} else {
		idx = offset + 5
	}
	Expect(len(nodes)).To(BeNumerically(">", idx), "扩容节点数量不足")
	n := nodes[idx]
	n.Role = []string{"node"}
	return n
}

func pickScaleInNodeIP(ctx *CaseContext, role InstallRole, kind ClusterKind, spec ClusterSpec) string {
	if ctx != nil && strings.TrimSpace(ctx.LastScaleOutNodeIP) != "" {
		return strings.TrimSpace(ctx.LastScaleOutNodeIP)
	}

	nodes := config.LoadTestNodesFromEnv()
	offset := 0
	if role == InstallRoleMgmt && kind == ClusterKindWorkload {
		offset = 2
	}

	var idx int
	if spec == ClusterSpec1M1N {
		idx = offset + 1
	} else {
		idx = offset + 3
	}
	Expect(len(nodes)).To(BeNumerically(">", idx), "缩容目标节点不足")
	return nodes[idx].IP
}

func pickFaultTargetNode(ctx *CaseContext, clusterName, kubeconfig string, role InstallRole, kind ClusterKind, spec ClusterSpec, fault FaultType) config.NodeInfo {
	nodes := config.LoadTestNodesFromEnv()

	if ctx != nil && strings.TrimSpace(clusterName) != "" {
		kubeconfigArg := ""
		if strings.TrimSpace(kubeconfig) != "" {
			kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfig)
		}
		cmd := fmt.Sprintf("%skubectl get bkenode -n bke-%s -o custom-columns=IP:.spec.ip,ROLE:.spec.role,STATE:.status.state --no-headers", kubeconfigArg, clusterName)
		res, err := ctx.SSHExecutor.Exec(cmd)
		if err == nil && res != nil && res.ExitCode == 0 {
			stdout := strings.TrimSpace(res.Stdout)
			if stdout != "" {
				nodeByIP := make(map[string]config.NodeInfo)
				for _, n := range nodes {
					nodeByIP[strings.TrimSpace(n.IP)] = n
				}

				workerCandidates := make([]config.NodeInfo, 0)
				masterCandidates := make([]config.NodeInfo, 0)

				for _, line := range strings.Split(stdout, "\n") {
					fields := strings.Fields(strings.TrimSpace(line))
					if len(fields) < 3 {
						continue
					}
					ip := strings.TrimSpace(fields[0])
					state := strings.TrimSpace(fields[len(fields)-1])
					roleText := strings.TrimSpace(strings.Join(fields[1:len(fields)-1], " "))
					if ip == "" || !strings.EqualFold(state, "Ready") {
						continue
					}
					n, ok := nodeByIP[ip]
					if !ok {
						continue
					}
					if strings.Contains(roleText, "master") {
						masterCandidates = append(masterCandidates, n)
					}
					if strings.Contains(roleText, "node") && !strings.Contains(roleText, "master") {
						n.Role = []string{"node"}
						workerCandidates = append(workerCandidates, n)
					}
				}

				sort.Slice(workerCandidates, func(i, j int) bool {
					return workerCandidates[i].IP < workerCandidates[j].IP
				})
				sort.Slice(masterCandidates, func(i, j int) bool {
					return masterCandidates[i].IP < masterCandidates[j].IP
				})

				if fault == FaultWorkerDown && len(workerCandidates) > 0 {
					return workerCandidates[0]
				}
				if fault == FaultMasterDown && len(masterCandidates) > 0 {
					return masterCandidates[0]
				}
			}
		}
	}

	// 回退到原有静态选择逻辑，兼容历史场景
	offset := 0
	if role == InstallRoleMgmt && kind == ClusterKindWorkload {
		offset = 2
	}

	if fault == FaultWorkerDown {
		idx := offset + 1
		if spec == ClusterSpec3M2N || spec == ClusterSpec3M1N {
			idx = offset + 3
		}
		Expect(len(nodes)).To(BeNumerically(">", idx), "worker 故障目标节点不足")
		n := nodes[idx]
		n.Role = []string{"node"}
		return n
	}

	idx := offset
	Expect(len(nodes)).To(BeNumerically(">", idx), "master 故障目标节点不足")
	n := nodes[idx]
	n.Role = []string{"master/node", "etcd"}
	return n
}

func injectStopKubeletFault(node config.NodeInfo) {
	By(fmt.Sprintf("注入故障: stop kubelet 节点 %s", node.IP))
	result, err := utils.ExecuteCommandOnNode(node, "systemctl stop kubelet")
	Expect(err).NotTo(HaveOccurred(), "stop kubelet 命令应执行成功")
	Expect(result).NotTo(BeNil(), "stop kubelet 命令结果不能为空")
	Expect(result.ExitCode).To(Equal(0), "stop kubelet 命令执行结果异常: %s", result.Stderr)
}

func recoverKubeletOnNode(node config.NodeInfo) {
	By(fmt.Sprintf("恢复故障: start kubelet 节点 %s", node.IP))
	result, err := utils.ExecuteCommandOnNode(node, "systemctl start kubelet")
	Expect(err).NotTo(HaveOccurred(), "start kubelet 命令应执行成功")
	Expect(result).NotTo(BeNil(), "start kubelet 命令结果不能为空")
	Expect(result.ExitCode).To(Equal(0), "start kubelet 命令执行结果异常: %s", result.Stderr)
}

func waitKubeletActive(node config.NodeInfo) {
	By(fmt.Sprintf("等待 kubelet 恢复 active: %s", node.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(node, "systemctl is-active kubelet")
		if err != nil || result == nil {
			return false
		}
		return strings.TrimSpace(result.Stdout) == "active"
	}, 3*time.Minute, 10*time.Second).Should(BeTrue(), "kubelet 应恢复为 active")
}

func collectClusterRuntimeStatus(ctx *CaseContext, clusterName, kubeconfig, stage string) {
	By(fmt.Sprintf("采集状态: %s", stage))

	phase, state, clusterStatus, err := ctx.ClusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfig)
	if err != nil {
		GinkgoWriter.Printf("[%s][BC] 获取状态失败: %v\n", stage, err)
	} else {
		GinkgoWriter.Printf("[%s][BC] phase=%s state=%s clusterStatus=%s\n", stage, phase, state, clusterStatus)
	}

	kubeconfigArg := ""
	if strings.TrimSpace(kubeconfig) != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfig)
	}

	bnCmd := fmt.Sprintf("%skubectl get bkenode -n bke-%s -o custom-columns=NAME:.metadata.name,STATE:.status.state,MESSAGE:.status.message,IP:.spec.ip,ROLE:.spec.role --no-headers", kubeconfigArg, clusterName)
	bnResult, bnErr := ctx.SSHExecutor.Exec(bnCmd)
	if bnErr != nil {
		GinkgoWriter.Printf("[%s][BN] 获取失败: %v\n", stage, bnErr)
	} else {
		GinkgoWriter.Printf("[%s][BN]\n%s\n", stage, strings.TrimSpace(bnResult.Stdout))
	}

	machineCmd := fmt.Sprintf("%skubectl get machine -A", kubeconfigArg)
	machineResult, machineErr := ctx.SSHExecutor.Exec(machineCmd)
	if machineErr != nil {
		GinkgoWriter.Printf("[%s][Machine] 获取失败: %v\n", stage, machineErr)
	} else {
		GinkgoWriter.Printf("[%s][Machine]\n%s\n", stage, strings.TrimSpace(machineResult.Stdout))
	}
}

func cleanupRemoteGeneratedYAMLs(ctx *CaseContext, clusterName string) {
	if strings.TrimSpace(clusterName) == "" {
		return
	}
	By(fmt.Sprintf("远端兜底清理 YAML: %s", clusterName))
	cmd := fmt.Sprintf("rm -f /bke/cluster/*%s* /bke/node/*%s*", clusterName, clusterName)
	_, _ = ctx.SSHExecutor.Exec(cmd)

	verifyCmd := fmt.Sprintf("sh -c 'ls -1 /bke/cluster/*%s* /bke/node/*%s* 2>/dev/null || true'", clusterName, clusterName)
	verifyResult, verifyErr := ctx.SSHExecutor.Exec(verifyCmd)
	if verifyErr != nil || verifyResult == nil {
		GinkgoWriter.Printf("verify cleanup for %s skipped, exec error: %v\n", clusterName, verifyErr)
		return
	}
	remaining := strings.TrimSpace(verifyResult.Stdout)
	if remaining == "" {
		GinkgoWriter.Printf("remote yaml cleanup success for cluster %s\n", clusterName)
		return
	}
	GinkgoWriter.Printf("remote yaml remains for cluster %s:\n%s\n", clusterName, remaining)
}

func waitClusterReady(ctx *CaseContext, clusterName, kubeconfig string) {
	By(fmt.Sprintf("等待基线集群 %s 就绪", clusterName))
	Eventually(func() bool {
		phase, state, clusterStatus, err := ctx.ClusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfig)
		if err != nil {
			GinkgoWriter.Printf("获取集群 %s 状态失败: %v\n", clusterName, err)
			return false
		}
		GinkgoWriter.Printf("[baseline %s] phase=%s state=%s clusterStatus=%s\n", clusterName, phase, state, clusterStatus)
		return clusterStatus == "Ready"
	}, installTimeout, pollInterval).Should(BeTrue(), "基线集群应先达到 Ready")
}

func waitBKEClusterCreated(ctx *CaseContext, clusterName string, kubeconfig string) {
	By(fmt.Sprintf("等待 BKECluster %s 创建成功", clusterName))

	ns := fmt.Sprintf("bke-%s", clusterName)
	bcName := fmt.Sprintf("bke-%s", clusterName)

	Eventually(func() bool {
		kubeconfigArg := ""
		if strings.TrimSpace(kubeconfig) != "" {
			kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfig)
		}

		cmd := fmt.Sprintf("%skubectl get bc -n %s %s", kubeconfigArg, ns, bcName)
		res, err := ctx.SSHExecutor.Exec(cmd)
		if err == nil && res != nil && res.ExitCode == 0 {
			return true
		}

		diagnoseCmd := fmt.Sprintf("sh -c '%skubectl get bn -n %s -o wide 2>&1; %skubectl get bc -A 2>&1; %skubectl get events -n %s --sort-by=.lastTimestamp 2>&1 | tail -50; %skubectl get endpoints -n cluster-system bke-webhook-service -o yaml 2>&1'",
			kubeconfigArg, ns,
			kubeconfigArg,
			kubeconfigArg, ns,
			kubeconfigArg,
		)
		diagRes, _ := ctx.SSHExecutor.Exec(diagnoseCmd)
		if diagRes != nil {
			fmt.Fprintf(GinkgoWriter, "[waitBKEClusterCreated] cluster=%s diagnose:\nstdout=%s\nstderr=%s\n", clusterName, diagRes.Stdout, diagRes.Stderr)
		}

		return false
	}, installTimeout, pollInterval).Should(BeTrue(), "BKECluster 应创建成功")
}

func waitBKENodeState(ctx *CaseContext, clusterName, kubeconfig, nodeIP, expectedState string) {
	By(fmt.Sprintf("等待节点 %s 状态为 %s", nodeIP, expectedState))
	isExpectedState := func() bool {
		kubeconfigArg := ""
		if strings.TrimSpace(kubeconfig) != "" {
			kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfig)
		}
		cmd := fmt.Sprintf("%skubectl get bkenode -n bke-%s -o custom-columns=IP:.spec.ip,STATE:.status.state --no-headers", kubeconfigArg, clusterName)
		result, err := ctx.SSHExecutor.Exec(cmd)
		if err != nil || result == nil || result.ExitCode != 0 {
			return false
		}
		stdout := strings.TrimSpace(result.Stdout)
		if stdout == "" {
			return false
		}
		lines := strings.Split(stdout, "\n")
		for _, line := range lines {
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			if strings.TrimSpace(fields[0]) == nodeIP {
				return strings.EqualFold(strings.TrimSpace(fields[1]), expectedState)
			}
		}
		return false
	}
	Eventually(isExpectedState, installTimeout, pollInterval).Should(BeTrue(), "节点应进入期望状态")
}

func waitBKEControllerManagerReady(ctx *CaseContext, kubeconfig string) {
	By("等待 bke-controller-manager 至少有一个实例 Running/Ready")
	Eventually(func() bool {
		kubeconfigArg := ""
		if strings.TrimSpace(kubeconfig) != "" {
			kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfig)
		}
		cmd := fmt.Sprintf("%skubectl get pod -n cluster-system -o jsonpath='{range .items[*]}{.metadata.name} {.status.phase} {.status.conditions[?(@.type==\"Ready\")].status}{\"\\n\"}{end}'", kubeconfigArg)
		result, err := ctx.SSHExecutor.Exec(cmd)
		if err != nil {
			GinkgoWriter.Printf("wait bke-controller-manager exec error: %v, cmd=%s\n", err, cmd)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("wait bke-controller-manager empty exec result, cmd=%s\n", cmd)
			return false
		}
		if result.ExitCode != 0 {
			GinkgoWriter.Printf("wait bke-controller-manager exit=%d stderr=%s stdout=%s\n", result.ExitCode, strings.TrimSpace(result.Stderr), strings.TrimSpace(result.Stdout))
			return false
		}
		stdout := strings.TrimSpace(result.Stdout)
		if stdout == "" {
			GinkgoWriter.Printf("wait bke-controller-manager no pod output yet\n")
			return false
		}
		GinkgoWriter.Printf("wait bke-controller-manager raw pods:\n%s\n", stdout)
		for _, line := range strings.Split(stdout, "\n") {
			fields := strings.Fields(strings.TrimSpace(line))
			if len(fields) < 3 {
				continue
			}
			if !strings.Contains(fields[0], "bke-controller-manager") {
				continue
			}
			if fields[1] == "Running" && strings.EqualFold(fields[2], "True") {
				GinkgoWriter.Printf("wait bke-controller-manager ready pod matched: %s\n", fields[0])
				return true
			}
		}
		GinkgoWriter.Printf("wait bke-controller-manager not ready yet\n")
		return false
	}, installTimeout, pollInterval).Should(BeTrue(), "bke-controller-manager 应恢复 Running/Ready")
}

func waitWebhookReady(ctx *CaseContext, kubeconfig string) {
	By("等待 webhook 服务与 endpoints 就绪")
	Eventually(func() bool {
		kubeconfigArg := ""
		if strings.TrimSpace(kubeconfig) != "" {
			kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfig)
		}

		svcCmd := fmt.Sprintf("%skubectl get svc -n cluster-system -o jsonpath='{range .items[*]}{.metadata.name}{\"\\n\"}{end}'", kubeconfigArg)
		svcRes, svcErr := ctx.SSHExecutor.Exec(svcCmd)
		if svcErr != nil || svcRes == nil || svcRes.ExitCode != 0 {
			return false
		}

		targetSvc := ""
		for _, line := range strings.Split(strings.TrimSpace(svcRes.Stdout), "\n") {
			name := strings.TrimSpace(line)
			if name == "" {
				continue
			}
			if strings.Contains(name, "webhook") || strings.Contains(name, "bke-controller-manager") {
				targetSvc = name
				break
			}
		}
		if targetSvc == "" {
			return false
		}

		epCmd := fmt.Sprintf("%skubectl get endpoints -n cluster-system %s -o jsonpath='{.subsets[*].addresses[*].ip}'", kubeconfigArg, targetSvc)
		epRes, epErr := ctx.SSHExecutor.Exec(epCmd)
		if epErr != nil || epRes == nil || epRes.ExitCode != 0 {
			return false
		}
		return strings.TrimSpace(epRes.Stdout) != ""
	}, installTimeout, pollInterval).Should(BeTrue(), "webhook 依赖服务应可用")
}

func ensureKubeconfigReachableOnReadyMaster(ctx *CaseContext, clusterName, bootstrapKubeconfig, kubeconfigPath string) string {
	By("确保管理集群 kubeconfig 指向 Ready master")
	if strings.TrimSpace(kubeconfigPath) == "" {
		return kubeconfigPath
	}

	checkReachable := func(path string) bool {
		cmd := fmt.Sprintf("KUBECONFIG=%s kubectl cluster-info --request-timeout=5s", path)
		res, err := ctx.SSHExecutor.Exec(cmd)
		return err == nil && res != nil && res.ExitCode == 0
	}
	if checkReachable(kubeconfigPath) {
		return kubeconfigPath
	}

	kubeconfigArg := ""
	if strings.TrimSpace(bootstrapKubeconfig) != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", bootstrapKubeconfig)
	}
	cmd := fmt.Sprintf("%skubectl get bkenode -n bke-%s -o custom-columns=IP:.spec.ip,ROLE:.spec.roles,STATE:.status.state --no-headers", kubeconfigArg, clusterName)
	res, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || res == nil || res.ExitCode != 0 {
		return kubeconfigPath
	}

	readyMasterIP := ""
	for _, line := range strings.Split(strings.TrimSpace(res.Stdout), "\n") {
		f := strings.Fields(line)
		if len(f) < 3 {
			continue
		}
		if strings.Contains(f[1], "master") && strings.EqualFold(f[2], "Ready") {
			readyMasterIP = f[0]
			break
		}
	}
	if readyMasterIP == "" {
		return kubeconfigPath
	}

	tmpPath := fmt.Sprintf("%s-ready-master", kubeconfigPath)
	rewriteCmd := fmt.Sprintf("sed -E 's#server: https://[^:]+:6443#server: https://%s:6443#g' %s > %s", readyMasterIP, kubeconfigPath, tmpPath)
	_, _ = ctx.SSHExecutor.Exec(rewriteCmd)
	ctx.AddGeneratedConfigPaths(tmpPath)
	if checkReachable(tmpPath) {
		GinkgoWriter.Printf("kubeconfig server rewrote to ready master: %s\n", readyMasterIP)
		return tmpPath
	}
	return kubeconfigPath
}

func waitBKENodeDeleted(ctx *CaseContext, clusterName, kubeconfig, nodeIP string) {
	By(fmt.Sprintf("等待节点 %s 完全缩容删除", nodeIP))
	Eventually(func() bool {
		kubeconfigArg := ""
		if strings.TrimSpace(kubeconfig) != "" {
			kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfig)
		}
		cmd := fmt.Sprintf("%skubectl get bkenode -n bke-%s -o custom-columns=IP:.spec.ip --no-headers", kubeconfigArg, clusterName)
		result, err := ctx.SSHExecutor.Exec(cmd)
		if err != nil || result == nil {
			return false
		}
		for _, line := range strings.Split(strings.TrimSpace(result.Stdout), "\n") {
			if strings.TrimSpace(line) == nodeIP {
				return false
			}
		}
		return true
	}, installTimeout, pollInterval).Should(BeTrue(), "缩容目标节点应被删除")
}

func abnormalDeleteMgmtCluster(ctx *CaseContext, mgmtClusterName, mgmtStatusKubeconfig string) {
	if ctx == nil || strings.TrimSpace(mgmtClusterName) == "" {
		GinkgoWriter.Printf("abnormalDeleteMgmtCluster skipped: invalid context or empty cluster name\n")
		return
	}
	GinkgoWriter.Printf("abnormalDeleteMgmtCluster start: mgmtClusterName=%s mgmtStatusKubeconfig=%s\n", mgmtClusterName, strings.TrimSpace(mgmtStatusKubeconfig))

	mgmtKubeconfig := strings.TrimSpace(mgmtStatusKubeconfig)
	if mgmtKubeconfig == "" {
		checker := utils.NewClusterCheckerWithParentKubeconfig(ctx.LocalExecutor, mgmtClusterName, "")
		path, err := checker.SaveKubeconfigToFile()
		if err != nil {
			GinkgoWriter.Printf("获取管理集群 kubeconfig 失败: %v\n", err)
		} else {
			mgmtKubeconfig = path
			ctx.AddGeneratedConfigPaths(path)
			GinkgoWriter.Printf("cleanup resolved mgmt kubeconfig: %s\n", path)
		}
	}

	mgmtNodes := selectAbnormalMgmtNodesForForceReset(ctx, mgmtClusterName)
	ctx.ClusterManager.CleanupManagementCluster(utils.ManagementClusterCleanupOptions{
		ClusterName:         mgmtClusterName,
		MgmtKubeconfigPath:  mgmtKubeconfig,
		MgmtNodes:           mgmtNodes,
		LocalExecutor:       ctx.LocalExecutor,
		SavedKubeconfigPath: mgmtKubeconfig,
	})
}

func selectAbnormalMgmtNodesForForceReset(ctx *CaseContext, mgmtClusterName string) []config.NodeInfo {
	envNodes := config.LoadTestNodesFromEnv()
	if len(envNodes) == 0 {
		return nil
	}

	fallbackByName := func() []config.NodeInfo {
		fallbackCount := 2
		lowerName := strings.ToLower(mgmtClusterName)
		if strings.Contains(lowerName, "3m2n") {
			fallbackCount = 5
		} else if strings.Contains(lowerName, "3m1n") {
			fallbackCount = 4
		}
		if fallbackCount > len(envNodes) {
			fallbackCount = len(envNodes)
		}
		return envNodes[:fallbackCount]
	}

	cmd := fmt.Sprintf("kubectl get bkenode -n bke-%s -o custom-columns=IP:.spec.ip --no-headers 2>/dev/null", mgmtClusterName)
	result, err := ctx.SSHExecutor.Exec(cmd)
	if err != nil || result == nil || result.ExitCode != 0 {
		GinkgoWriter.Printf("获取管理集群节点列表失败，按规格回退: err=%v\n", err)
		return fallbackByName()
	}

	ipSet := map[string]struct{}{}
	for _, line := range strings.Split(strings.TrimSpace(result.Stdout), "\n") {
		ip := strings.TrimSpace(line)
		if ip != "" {
			ipSet[ip] = struct{}{}
		}
	}
	if len(ipSet) == 0 {
		GinkgoWriter.Printf("管理集群 bkenode 结果为空，按规格回退\n")
		return fallbackByName()
	}

	selected := make([]config.NodeInfo, 0, len(ipSet))
	for _, n := range envNodes {
		if _, ok := ipSet[strings.TrimSpace(n.IP)]; ok {
			selected = append(selected, n)
		}
	}
	if len(selected) == 0 {
		GinkgoWriter.Printf("管理集群节点 IP 与测试节点未匹配，按规格回退\n")
		return fallbackByName()
	}
	return selected
}
