package installation

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
)

type InstallRole string

const (
	InstallRoleGuide InstallRole = "guide"
	InstallRoleMgmt  InstallRole = "management"
)

type ClusterKind string

const (
	ClusterKindMgmt     ClusterKind = "management"
	ClusterKindWorkload ClusterKind = "workload"
)

type ClusterSpec string

const (
	ClusterSpec1M   ClusterSpec = "1m"
	ClusterSpec1M1N ClusterSpec = "1m1n"
	ClusterSpec3M1N ClusterSpec = "3m1n"
	ClusterSpec3M2N ClusterSpec = "3m2n"
)

type HAType string

const (
	HANone    HAType = "none"
	HAEnabled HAType = "ha"
)

type OperationType string

const (
	OperationInstall           OperationType = "install"
	OperationScaleOut          OperationType = "scale_out"
	OperationScaleIn           OperationType = "scale_in"
	OperationUpgrade           OperationType = "upgrade"
	OperationUninstall         OperationType = "uninstall"
	OperationCreateSubCluster  OperationType = "create_sub_cluster"
	OperationDeleteSubCluster  OperationType = "delete_sub_cluster"
	OperationDeleteAbnormal    OperationType = "delete_abnormal_node"
	OperationRepairAbnormal    OperationType = "repair_abnormal_node"
)

type FaultType string

const (
	FaultNone       FaultType = "none"
	FaultWorkerDown FaultType = "worker_down"
	FaultMasterDown FaultType = "master_down"
)

type FaultInjectPhase string

const (
	FaultDuringOperation FaultInjectPhase = "during_operation"
	FaultBeforeOperation FaultInjectPhase = "before_operation"
)

type StatusStage string

const (
	StatusStageInProgress StatusStage = "in_progress"
	StatusStageSucceeded  StatusStage = "succeeded"
	StatusStageFailed     StatusStage = "failed"
	StatusStageRecovered  StatusStage = "recovered"
)

type ClusterStatusExpectation struct {
	Stage         StatusStage
	AllowedPhases []string 
	AllowedStates []string 
	AllowedStatus []string
}

type InstallationCase struct {
	ID       string
	Name     string

	InstallRole      InstallRole
	ClusterKind      ClusterKind
	Spec             ClusterSpec
	HA               HAType
	Operation        OperationType
	Fault             FaultType
	FaultInjectPhase  FaultInjectPhase
	WithPrePostScript bool
	CoDeployGuideMaster bool
	ClusterName string
	StatusExpectations []ClusterStatusExpectation
	Prepare func(ctx *CaseContext)
	InjectFault func(ctx *CaseContext)
	Execute func(ctx *CaseContext)
	Recover func(ctx *CaseContext)
	Cleanup func(ctx *CaseContext)
}

type CaseContext struct {
	ClusterName         string
	ParentClusterName   string
	MgmtKubeconfigPath  string
	GeneratedConfigPath []string
	LastScaleOutNodeIP  string

	SSHExecutor    *executor.SSHExecutor
	LocalExecutor  *executor.LocalExecutor
	ClusterManager *utils.ClusterManager
}

func RunInstallationCase(tc InstallationCase, ctx *CaseContext) {
	By(fmt.Sprintf("[CASE %s] %s", tc.ID, tc.Name))

	faultInjected := false
	injectFaultOnce := func() {
		if faultInjected || tc.InjectFault == nil || tc.Fault == FaultNone {
			return
		}
		tc.InjectFault(ctx)
		faultInjected = true
	}

	if tc.Prepare != nil {
		tc.Prepare(ctx)
	}
	if tc.FaultInjectPhase == FaultBeforeOperation {
		injectFaultOnce()
	}

	if tc.Execute != nil {
		tc.Execute(ctx)
	}

	if tc.FaultInjectPhase == FaultDuringOperation {
		injectFaultOnce()
	}

	for _, exp := range tc.StatusExpectations {
		AssertClusterStatusEventually(ctx, tc.ClusterName, ctx.MgmtKubeconfigPath, exp, installTimeout)
	}

	if tc.Recover != nil {
		tc.Recover(ctx)
	}

	if tc.Cleanup != nil {
		tc.Cleanup(ctx)
	}
}

func AssertClusterStatusEventually(
	ctx *CaseContext,
	clusterName string,
	kubeconfigPath string,
	exp ClusterStatusExpectation,
	timeout time.Duration,
) {
	By(fmt.Sprintf("校验状态阶段: %s", exp.Stage))
	Eventually(func() bool {
		phase, state, clusterStatus, err := ctx.ClusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
			return false
		}
		GinkgoWriter.Printf("[status] phase=%s state=%s clusterStatus=%s\n", phase, state, clusterStatus)

		if contains(exp.AllowedStates, "Healthy") && isClusterFailedState(state, clusterStatus) {
			Fail(fmt.Sprintf(failedStatesMsgTmpl, state, clusterStatus))
		}

		phaseOK := len(exp.AllowedPhases) == 0 || contains(exp.AllowedPhases, phase)
		stateOK := len(exp.AllowedStates) == 0 || contains(exp.AllowedStates, state)
		statusOK := len(exp.AllowedStatus) == 0 || contains(exp.AllowedStatus, clusterStatus)
		return phaseOK && stateOK && statusOK
	}, timeout, pollInterval).Should(BeTrue(), "状态断言失败: stage=%s", exp.Stage)
}

func AssertBNContains(ctx *CaseContext, clusterName string, expectedKeywords ...string) {
	By("校验 BN 侧状态")
	cmd := fmt.Sprintf("kubectl get bkenode -n bke-%s -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,REASON:.status.reason --no-headers", clusterName)
	result, err := ctx.SSHExecutor.Exec(cmd)
	Expect(err).NotTo(HaveOccurred(), "查询 BN 状态命令应该执行成功")
	Expect(result.ExitCode).To(Equal(0), "查询 BN 状态应该成功: %s", result.Stderr)

	out := strings.ToLower(result.Stdout)
	for _, k := range expectedKeywords {
		Expect(out).To(ContainSubstring(strings.ToLower(k)), "BN 输出应包含关键字: %s", k)
	}
}

func (ctx *CaseContext) AddGeneratedConfigPaths(paths ...string) {
	for _, p := range paths {
		if strings.TrimSpace(p) == "" {
			continue
		}
		ctx.GeneratedConfigPath = append(ctx.GeneratedConfigPath, p)
		GinkgoWriter.Printf("tracked generated config: %s\n", p)
	}
}

func (ctx *CaseContext) CleanupGeneratedConfigs() {
	for _, p := range ctx.GeneratedConfigPath {
		if strings.TrimSpace(p) == "" {
			continue
		}
		if err := ctx.ClusterManager.CleanupConfig(p); err != nil {
			GinkgoWriter.Printf("Failed to cleanup config file %s: %v\n", p, err)
		} else {
			GinkgoWriter.Printf("cleanup config file success: %s\n", p)
		}
	}
}

func contains(items []string, target string) bool {
	for _, i := range items {
		if i == target {
			return true
		}
	}
	return false
}
