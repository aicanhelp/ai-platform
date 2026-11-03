/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phases

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/conditions"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/intervention"
	_ "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// Constants for master initialization
const (
	// logInterval defines the frequency of log output during polling
	logInterval = 10
)

const (
	EnsureMasterInitName          confv1beta1.BKEClusterPhase = "EnsureMasterInit"
	MasterInitLogIntervalCount                                = 10 // 主节点初始化日志输出间隔计数
	MasterInitSleepSeconds                                    = 2  // 主节点初始化等待时间（秒）
	MasterInitPollIntervalSeconds                             = 1  // 主节点初始化轮询间隔（秒）
)

type EnsureMasterInit struct {
	phaseframe.BasePhase
}

func NewEnsureMasterInit(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureMasterInitName)
	return &EnsureMasterInit{BasePhase: base}
}
func (e *EnsureMasterInit) ExecutePreHook() error {
	return e.BasePhase.DefaultPreHook()
}

// ValidateMasterNodesParams 包含 validateMasterNodes 函数的参数
type ValidateMasterNodesParams struct {
	Ctx *phaseframe.PhaseContext
}

// validateMasterNodes 验证主节点
func (e *EnsureMasterInit) validateMasterNodes(params ValidateMasterNodesParams) (bkenode.Nodes, int, error) {
	nodeFetcher := params.Ctx.NodeFetcher()
	allNodes, _ := nodeFetcher.GetNodesForBKECluster(params.Ctx, params.Ctx.BKECluster)
	nodes := allNodes.Master()
	if len(nodes) == 0 {
		return nil, 0, errors.Errorf("no master node")
	}

	count := 0
	for _, node := range nodes {
		nodeStateFlag, _ := nodeFetcher.GetNodeStateFlagForCluster(params.Ctx, params.Ctx.BKECluster, node.IP, bkev1beta1.NodeEnvFlag)
		if !nodeStateFlag {
			count++
		}
	}

	if count == nodes.Length() {
		return nil, 0, errors.Errorf("all master node agent is not ready")
	}

	return nodes, count, nil
}

// SetupConditionAndRefreshParams 包含 setupConditionAndRefresh 函数的参数
type SetupConditionAndRefreshParams struct {
	Ctx *phaseframe.PhaseContext
}

// setupConditionAndRefresh 设置条件和刷新集群状态
func (e *EnsureMasterInit) setupConditionAndRefresh(params SetupConditionAndRefreshParams) error {
	// Clear any existing manual intervention mark (a new init round is starting).
	// Uses ClearInMemory to avoid a redundant SyncStatusUntilComplete; the
	// existing SyncStatusUntilComplete below will persist the change.
	intervention.ClearInMemory(params.Ctx.BKECluster)
	condition.ConditionMark(params.Ctx.BKECluster, bkev1beta1.ControlPlaneInitializedCondition, confv1beta1.ConditionFalse, constant.MasterNotInitReason, "Master still not init")

	if err := mergecluster.SyncStatusUntilComplete(params.Ctx.Client, params.Ctx.BKECluster); err != nil {
		return fmt.Errorf("failed to mark %q status false: %w", bkev1beta1.ControlPlaneInitializedCondition, err)
	}
	if err := params.Ctx.RefreshCtxBKECluster(); err != nil {
		return err
	}

	return nil
}

// WaitForInitCommandCompleteParams 包含 waitForInitCommandComplete 函数的参数
type WaitForInitCommandCompleteParams struct {
	Ctx                 *phaseframe.PhaseContext
	InitNodeIp          *string
	CommandCompleteFlag *bool
	PollCount           *int
}

// waitForInitCommandComplete 等待初始化命令完成
func (e *EnsureMasterInit) waitForInitCommandComplete(params WaitForInitCommandCompleteParams) (bool, error) {
	phaseContextFields131 := params.Ctx.Untie()
	_, c, bkeCluster, _, _ := phaseContextFields131.Context,
		phaseContextFields131.Client,
		phaseContextFields131.BKECluster,
		phaseContextFields131.Scheme,
		phaseContextFields131.Log

	// 需要拿到init的command
	initCommand, err := phaseutil.GetMasterInitCommand(params.Ctx.Context, c, bkeCluster)
	if err != nil {
		if strings.Contains(err.Error(), "command not found") {
			// 循环十次输出一次日志
			if *params.PollCount%MasterInitLogIntervalCount == 0 {
				log.Info(constant.MasterNotInitReason, "Waiting init command to be created, info:%v", err)
			}
			return false, nil
		}
		return false, err
	}

	if initCommand.Spec.NodeName != "" {
		*params.InitNodeIp = initCommand.Spec.NodeName
	} else {
		// only one k,v in initCommand.Spec.NodeSelector.MatchLabels
		for k, _ := range initCommand.Spec.NodeSelector.MatchLabels {
			*params.InitNodeIp = k
			break
		}
	}

	complete, successNodes, failedNodes := command.CheckCommandStatus(initCommand)
	if complete {
		if len(failedNodes) != 0 {
			// 使用通用的命令失败处理函数
			commandFailureParams := ProcessCommandFailureParams{
				Context:     params.Ctx.Context,
				Client:      c,
				BKECluster:  params.Ctx.BKECluster,
				InitCommand: initCommand,
				InitNodeIp:  params.InitNodeIp,
				FailedNodes: failedNodes,
				RefreshContext: func() error {
					return params.Ctx.RefreshCtxBKECluster()
				},
				Recorder: params.Ctx.Log.Recorder,
			}
			result := ProcessCommandFailure(commandFailureParams)
			if result.Done {
				return result.Success, result.Err
			} else {
				return result.Done, result.Err
			}
		}
		if len(successNodes) != 0 {
			*params.CommandCompleteFlag = true
			log.Info(constant.MasterNotInitReason, "Master node init command run success, success nodes: %v", successNodes)
		}
	} else {
		// Log output once every logInterval times
		if *params.PollCount%logInterval == 0 {
			log.Info(constant.MasterNotInitReason, "Master node init command not run complete, waiting...")
		}
		return false, nil
	}

	return true, nil
}

// WaitForMachineBootstrapParams 包含 waitForMachineBootstrap 函数的参数
type WaitForMachineBootstrapParams struct {
	Ctx       *phaseframe.PhaseContext
	PollCount *int
}

// waitForMachineBootstrap 等待机器引导
func (e *EnsureMasterInit) waitForMachineBootstrap(params WaitForMachineBootstrapParams) (bool, error) {
	phaseContextFields202 := params.Ctx.Untie()
	_, c, bkeCluster, _, _ := phaseContextFields202.Context,
		phaseContextFields202.Client,
		phaseContextFields202.BKECluster,
		phaseContextFields202.Scheme,
		phaseContextFields202.Log

	bkeMachine, err := phaseutil.GetControlPlaneInitBKEMachine(params.Ctx.Context, c, bkeCluster)
	if err != nil {
		if *params.PollCount%MasterInitLogIntervalCount == 0 {
			log.Debug(constant.MasterNotInitReason, "get init BKEMachine failed, err: %v", err)
		}
		return false, nil
	}
	if !bkeMachine.Status.Bootstrapped {
		return false, nil
	}
	return true, nil
}

// CheckClusterInitializedParams 包含 checkClusterInitialized 函数的参数
type CheckClusterInitializedParams struct {
	Ctx       *phaseframe.PhaseContext
	PollCount *int
}

// checkClusterInitialized 检查集群是否已初始化
func (e *EnsureMasterInit) checkClusterInitialized(params CheckClusterInitializedParams) (bool, error) {
	if err := params.Ctx.RefreshCtxCluster(); err != nil {
		return false, fmt.Errorf("Refresh ClusterAPI Cluster obj %q failed: %w", utils.ClientObjNS(params.Ctx.Cluster), err)
	}
	if err := params.Ctx.RefreshCtxBKECluster(); err != nil {
		return false, fmt.Errorf("Refresh BKECluster obj %q failed: %w", utils.ClientObjNS(params.Ctx.BKECluster), err)
	}

	if conditions.IsTrue(params.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
		params.Ctx.Log.Info(constant.MasterInitReason, "ClusterAPI Cluster obj already initialized")
		condition.ConditionMark(params.Ctx.BKECluster, bkev1beta1.ControlPlaneInitializedCondition, confv1beta1.ConditionTrue, "", "")
		return true, nil
	}

	if !conditions.IsTrue(params.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
		if *params.PollCount%MasterInitLogIntervalCount == 0 {
			params.Ctx.Log.Info(constant.MasterNotInitReason, "Waiting for ClusterAPI Cluster obj to be initialized")
		}
		// get Cluster obj
		return false, nil
	}

	return false, nil
}

// MasterInitPollParams 包含 masterInitPollFunc 函数的参数
type MasterInitPollParams struct {
	Ctx                 *phaseframe.PhaseContext
	Timeout             context.Context
	CommandCompleteFlag *bool
	MachineBootFlag     *bool
	InitNodeIp          *string
}

// CheckClusterInitializedStep 检查集群是否已初始化
func (e *EnsureMasterInit) checkClusterInitializedStep(params MasterInitPollParams, pollCount int) (bool, bool, error) {
	if err := params.Ctx.RefreshCtxCluster(); err != nil {
		return false, false, fmt.Errorf("Refresh ClusterAPI Cluster obj %q failed: %w", utils.ClientObjNS(params.Ctx.Cluster), err)
	}
	if err := params.Ctx.RefreshCtxBKECluster(); err != nil {
		return false, false, fmt.Errorf("Refresh BKECluster obj %q failed: %w", utils.ClientObjNS(params.Ctx.BKECluster), err)
	}

	if conditions.IsTrue(params.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
		log.Info(constant.MasterInitReason, "ClusterAPI Cluster obj already initialized")
		condition.ConditionMark(params.Ctx.BKECluster, bkev1beta1.ControlPlaneInitializedCondition, confv1beta1.ConditionTrue, "", "")
		return true, true, nil // done, success
	}

	return false, false, nil // continue, not done
}

// GetInitCommandStep 获取初始化命令
func (e *EnsureMasterInit) getInitCommandStep(params MasterInitPollParams, pollCount int) (*agentv1beta1.Command, bool, error) {
	phaseContextFields280 := params.Ctx.Untie()
	_, c, bkeCluster, _, _ := phaseContextFields280.Context,
		phaseContextFields280.Client,
		phaseContextFields280.BKECluster,
		phaseContextFields280.Scheme,
		phaseContextFields280.Log

	if *params.CommandCompleteFlag {
		return nil, false, nil // continue to next step
	}

	// 需要拿到init的command
	initCommand, err := phaseutil.GetMasterInitCommand(params.Timeout, c, bkeCluster)
	if err != nil {
		if strings.Contains(err.Error(), "command not found") {
			// 循环十次输出一次日志
			if pollCount%MasterInitLogIntervalCount == 0 {
				log.Info(constant.MasterNotInitReason, "Waiting init command to be created, info:%v", err)
			}
			return nil, false, nil
		}
		return nil, false, err
	}

	if initCommand.Spec.NodeName != "" {
		*params.InitNodeIp = initCommand.Spec.NodeName
	} else {
		// only one k,v in initCommand.Spec.NodeSelector.MatchLabels
		for k, _ := range initCommand.Spec.NodeSelector.MatchLabels {
			*params.InitNodeIp = k
			break
		}
	}

	return initCommand, true, nil // continue processing
}

// ProcessCommandFailure 处理命令失败情况
func (e *EnsureMasterInit) processCommandFailure(params MasterInitPollParams, c client.Client, initCommand *agentv1beta1.Command, failedNodes []string, pollCount int) (bool, bool, error) {

	// 使用通用的命令失败处理函数
	commandFailureParams := ProcessCommandFailureParams{
		Context:     params.Timeout,
		Client:      c,
		BKECluster:  params.Ctx.BKECluster,
		NodeFetcher: e.Ctx.NodeFetcher(),
		InitCommand: initCommand,
		InitNodeIp:  params.InitNodeIp,
		FailedNodes: failedNodes,
		RefreshContext: func() error {
			return params.Ctx.RefreshCtxBKECluster()
		},
		Recorder: e.Ctx.Log.Recorder,
	}
	result := ProcessCommandFailure(commandFailureParams)
	return result.Done, result.Success, result.Err
}

// ProcessCommandCompleteParams 包含 processCommandComplete 函数的参数
type ProcessCommandCompleteParams struct {
	MasterInitPollParams MasterInitPollParams
	InitCommand          *agentv1beta1.Command
	Complete             bool
	SuccessNodes         []string
	FailedNodes          []string
	PollCount            int
}

// ProcessCommandComplete 处理命令完成情况
func (e *EnsureMasterInit) processCommandComplete(params ProcessCommandCompleteParams) (bool, bool, error) {
	phaseContextFields345 := params.MasterInitPollParams.Ctx.Untie()
	_, c, _, _, _ := phaseContextFields345.Context,
		phaseContextFields345.Client,
		phaseContextFields345.BKECluster,
		phaseContextFields345.Scheme,
		phaseContextFields345.Log

	if params.Complete {
		if len(params.FailedNodes) != 0 {
			return e.processCommandFailure(params.MasterInitPollParams, c, params.InitCommand, params.FailedNodes, params.PollCount)
		}
		if len(params.SuccessNodes) != 0 {
			*params.MasterInitPollParams.CommandCompleteFlag = true
			log.Info(constant.MasterNotInitReason, "Master node init command run success, success nodes: %v", params.SuccessNodes)
		}
	} else {
		// 循环十次输出一次日志
		if params.PollCount%MasterInitLogIntervalCount == 0 {
			log.Info(constant.MasterNotInitReason, "Master node init command not run complete, waiting...")
		}
		return false, false, nil
	}

	return false, false, nil // continue to next step
}

// WaitForCommandCompleteStep 等待init command完成
func (e *EnsureMasterInit) waitForCommandCompleteStep(params MasterInitPollParams, pollCount int) (bool, bool, error) {
	// 获取初始化命令
	initCommand, shouldContinue, err := e.getInitCommandStep(params, pollCount)
	if !shouldContinue || err != nil {
		return false, false, err
	}

	// 如果命令为空，说明还在等待中
	if initCommand == nil {
		return false, false, nil
	}

	complete, successNodes, failedNodes := command.CheckCommandStatus(initCommand)
	commandCompleteParams := ProcessCommandCompleteParams{
		MasterInitPollParams: params,
		InitCommand:          initCommand,
		Complete:             complete,
		SuccessNodes:         successNodes,
		FailedNodes:          failedNodes,
		PollCount:            pollCount,
	}
	return e.processCommandComplete(commandCompleteParams)
}

// WaitForMachineBootstrapStep 等待bkeMachine被标记已经引导
func (e *EnsureMasterInit) waitForMachineBootstrapStep(params MasterInitPollParams, pollCount int) (bool, bool, error) {
	phaseContextFields393 := params.Ctx.Untie()
	_, c, bkeCluster, _, _ := phaseContextFields393.Context,
		phaseContextFields393.Client,
		phaseContextFields393.BKECluster,
		phaseContextFields393.Scheme,
		phaseContextFields393.Log

	if *params.MachineBootFlag {
		return false, false, nil // continue to next step
	}

	bkeMachine, err := phaseutil.GetControlPlaneInitBKEMachine(params.Timeout, c, bkeCluster)
	if err != nil {
		if pollCount%MasterInitLogIntervalCount == 0 {
			log.Debug(constant.MasterNotInitReason, "get init BKEMachine failed, err: %v", err)
		}
		return false, false, nil
	}
	if !bkeMachine.Status.Bootstrapped {
		return false, false, nil
	}
	*params.MachineBootFlag = true

	return false, false, nil // continue to next step
}

// CheckClusterFinalStep 检查集群最终状态
func (e *EnsureMasterInit) checkClusterFinalStep(params MasterInitPollParams, pollCount int) (bool, bool, error) {
	if !conditions.IsTrue(params.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
		if pollCount%MasterInitLogIntervalCount == 0 {
			log.Info(constant.MasterNotInitReason, "Waiting for ClusterAPI Cluster obj to be initialized")
		}
		// get Cluster obj
		return false, false, nil
	}

	return false, false, nil
}

// masterInitPollFunc 处理主节点初始化轮询逻辑
func (e *EnsureMasterInit) masterInitPollFunc(params MasterInitPollParams) func() (bool, error) {
	pollCount := 0
	return func() (bool, error) {
		pollCount++

		// Step 1: 检查集群是否已初始化
		done, success, err := e.checkClusterInitializedStep(params, pollCount)
		if done {
			return success, err
		}
		if err != nil {
			return false, err
		}

		// Step 2: 等待init command完成
		done, success, err = e.waitForCommandCompleteStep(params, pollCount)
		if done {
			return success, err
		}
		if err != nil {
			return false, err
		}

		// Step 3: 等待bkeMachine被标记已经引导
		done, success, err = e.waitForMachineBootstrapStep(params, pollCount)
		if done {
			return success, err
		}
		if err != nil {
			return false, err
		}

		// Step 4: 检查集群最终状态
		done, success, err = e.checkClusterFinalStep(params, pollCount)
		if done {
			return success, err
		}
		if err != nil {
			return false, err
		}

		return false, nil
	}
}

func (e *EnsureMasterInit) Execute() (ctrl.Result, error) {
	var err error

	// 设置条件和刷新集群状态
	setupParams := SetupConditionAndRefreshParams{
		Ctx: e.Ctx,
	}
	if err := e.setupConditionAndRefresh(setupParams); err != nil {
		return ctrl.Result{}, err
	}

	defer func() {
		// 在最后退出时，只要没有init成功，需要加上condition，来防止环境初始化command清除已经init完成的部分
		if derr := e.Ctx.RefreshCtxCluster(); derr != nil {
			err = fmt.Errorf("get ClusterAPI Cluster obj failed: %w", derr)
		}
		if e.Ctx.Cluster != nil && !conditions.IsTrue(e.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
			condition.ConditionMark(e.Ctx.BKECluster, bkev1beta1.ControlPlaneInitializedCondition, confv1beta1.ConditionFalse, constant.MasterNotInitReason, "Master still not init")
			if derr := mergecluster.SyncStatusUntilComplete(e.Ctx.Client, e.Ctx.BKECluster); derr != nil {
				err = fmt.Errorf("failed to mark %q status false: %w", bkev1beta1.ControlPlaneInitializedCondition, derr)
			}

		}
	}()

	timeOut, err := phaseutil.GetBootTimeOut(e.Ctx.BKECluster)
	if err != nil {
		log.Warn(constant.MasterNotInitReason, "Get boot timeout failed. err: %v", err)
	}

	ctx, cancel := context.WithTimeout(e.Ctx.Context, timeOut)
	defer cancel()

	commandCompleteFlag := false
	machineBootFlag := false
	initNodeIp := ""

	// 创建轮询参数
	pollParams := MasterInitPollParams{
		Ctx:                 e.Ctx,
		Timeout:             ctx,
		CommandCompleteFlag: &commandCompleteFlag,
		MachineBootFlag:     &machineBootFlag,
		InitNodeIp:          &initNodeIp,
	}

	err = wait.PollImmediateUntil(time.Duration(MasterInitPollIntervalSeconds)*time.Second, e.masterInitPollFunc(pollParams), ctx.Done())

	if err != nil {
		if errors.Is(err, wait.ErrWaitTimeout) {
			_ = intervention.Require(intervention.Params{
				BKECluster: e.Ctx.BKECluster,
				Client:     e.Ctx.Client,
				Recorder:   e.Ctx.Log.Recorder,
				Guidance:   intervention.MasterInitTimeout(timeOut),
			})
			return ctrl.Result{}, errors.Errorf("Wait master init failed")
		}
		return ctrl.Result{}, err
	}
	e.Ctx.Log.Info(constant.MasterInitReason, "Master node init success")
	return ctrl.Result{}, mergecluster.SyncStatusUntilComplete(e.Ctx.Client, e.Ctx.BKECluster)
}

func (e *EnsureMasterInit) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	needExecute := true
	if err := e.Ctx.RefreshCtxCluster(); err == nil {
		if conditions.IsTrue(e.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
			return false
		}
	}

	if needExecute {
		e.SetStatus(bkev1beta1.PhaseWaiting)
	}
	return needExecute
}
