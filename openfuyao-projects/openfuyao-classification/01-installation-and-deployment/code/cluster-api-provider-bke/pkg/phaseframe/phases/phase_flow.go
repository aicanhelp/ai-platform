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
	"runtime/debug"
	"strings"

	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"sigs.k8s.io/cluster-api/util"
	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	cvv1alpha1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/v1alpha1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/clusterversion"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type PhaseFlow struct {
	BKEPhases []phaseframe.Phase
	ctx       *phaseframe.PhaseContext
	// oldBKECluster record the old BKECluster
	oldBKECluster *bkev1beta1.BKECluster
	// newBKECluster record the current BKECluster
	newBKECluster *bkev1beta1.BKECluster
}

func init() {
	FullPhasesRegisFunc = append(FullPhasesRegisFunc, CommonPhases()...)
	FullPhasesRegisFunc = append(FullPhasesRegisFunc, DeployPhases()...)
	FullPhasesRegisFunc = append(FullPhasesRegisFunc, PostDeployPhases()...)
}

var FullPhasesRegisFunc []func(ctx *phaseframe.PhaseContext) phaseframe.Phase

// MaxPhaseStatusHistory 最大保留的phase状态历史数量
const MaxPhaseStatusHistory = 20

func NewPhaseFlow(ctx *phaseframe.PhaseContext) *PhaseFlow {
	return &PhaseFlow{
		ctx: ctx,
	}
}

// CalculatePhase is used to calculate the phase which need to be executed
func (p *PhaseFlow) CalculatePhase(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) error {
	phasesFuncs := p.determinePhasesFuncs()
	p.calculateAndAddPhases(old, new, phasesFuncs)
	return p.ReportPhaseStatus()
}

// determinePhasesFuncs determines which phases functions to use based on cluster state
func (p *PhaseFlow) determinePhasesFuncs() []func(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	if phaseutil.IsDeleteOrReset(p.ctx.BKECluster) {
		return DeletePhases()
	}
	return FullPhasesRegisFunc
}

// calculateAndAddPhases calculates the phases that need to be executed and adds them to the list
func (p *PhaseFlow) calculateAndAddPhases(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster, phasesFuncs []func(ctx *phaseframe.PhaseContext) phaseframe.Phase) {
	skipLegacyUpgrade := shouldSkipLegacyUpgrade(p.ctx)

	// 计算需要执行的阶段，并添加到status中
	for _, f := range phasesFuncs {
		phase := f(p.ctx)
		if p.skipPhaseAfterDeclarativeDAG(phase) {
			continue
		}
		if skipLegacyUpgrade && isLegacyClusterUpgradePhase(phase.Name()) {
			continue
		}
		if phase.NeedExecute(old, new) {
			p.BKEPhases = append(p.BKEPhases, phase)
		}
	}
}

func shouldSkipLegacyUpgrade(ctx *phaseframe.PhaseContext) bool {
	if ctx == nil || ctx.BKECluster == nil {
		return false
	}
	bkeCluster := ctx.BKECluster
	if bkeCluster.Status.ClusterStatus == bkev1beta1.ClusterUpgradeFailed {
		return true
	}
	if v, ok := condition.HasCondition(bkev1beta1.ClusterHealthyStateCondition, bkeCluster); ok && v != nil {
		if v.Status == confv1beta1.ConditionFalse && v.Message == string(bkev1beta1.UpgradeFailed) {
			return true
		}
	}
	if ctx.Client == nil {
		return false
	}
	cv, err := clusterversion.GetClusterVersionForBKECluster(ctx.Context, ctx.Client, bkeCluster)
	if err != nil || cv == nil {
		return false
	}
	desired := strings.TrimSpace(cv.Spec.DesiredVersion)
	current := strings.TrimSpace(cv.Status.CurrentVersion)
	return cv.Status.Phase == cvv1alpha1.ClusterVersionPhaseFailed &&
		desired != "" && desired != current
}

func isLegacyClusterUpgradePhase(name confv1beta1.BKEClusterPhase) bool {
	for _, phaseName := range ClusterUpgradePhaseNames() {
		if phaseName == name {
			return true
		}
	}
	for _, phaseName := range DeclarativeClusterUpgradePhaseNames() {
		if phaseName == name {
			return true
		}
	}
	return false
}

// skipPhaseAfterDeclarativeDAG avoids re-running inline upgrade phases already executed by the DAG.
func (p *PhaseFlow) skipPhaseAfterDeclarativeDAG(phase phaseframe.Phase) bool {
	if p.ctx == nil || !p.ctx.DeclarativeDAGCompleted {
		return false
	}
	return IsDeclarativeInlineUpgradePhase(phase.Name())
}

// ReportPhaseStatus is used to report the phase status
func (p *PhaseFlow) ReportPhaseStatus() error {
	if p.BKEPhases == nil || len(p.BKEPhases) == 0 {
		return nil
	}

	if p.ctx.BKECluster.Status.PhaseStatus != nil {
		p.processPhaseStatus()
	}

	// 上报
	waitPhaseCount, err := p.reportPhases()
	if err != nil {
		return err
	}

	p.ctx.Log.Debug("*****All of %d phases wait******", waitPhaseCount)

	//更新后记录oldBKECluster, newBKECluster
	return p.refreshOldAndNewBKECluster()
}

// processPhaseStatus processes the phase status by cleaning up succeeded phases and limiting the history
func (p *PhaseFlow) processPhaseStatus() {
	// 上报前先找到最后一个成功的
	var lastSuccessPhaseIndex int

	for i, phaseStatus := range p.ctx.BKECluster.Status.PhaseStatus {
		if phaseStatus.Status == bkev1beta1.PhaseSucceeded {
			lastSuccessPhaseIndex = i
		}
	}

	// 移除成功后除了失败的phase
	last := lastSuccessPhaseIndex + 1
	for last < len(p.ctx.BKECluster.Status.PhaseStatus) && p.ctx.BKECluster.Status.PhaseStatus[last].Status != bkev1beta1.PhaseFailed {
		last++
	}

	p.ctx.BKECluster.Status.PhaseStatus = p.ctx.BKECluster.Status.PhaseStatus[:last]

	// 最多保留MaxPhaseStatusHistory个phase，避免重复执行时，phaseStatus过多
	if len(p.ctx.BKECluster.Status.PhaseStatus) > MaxPhaseStatusHistory {
		p.ctx.BKECluster.Status.PhaseStatus = p.ctx.BKECluster.Status.PhaseStatus[len(p.ctx.BKECluster.Status.PhaseStatus)-MaxPhaseStatusHistory:]
	}
}

// reportPhases reports the phases and returns the count of waiting phases
func (p *PhaseFlow) reportPhases() (int, error) {
	waitPhaseCount := 0
	p.ctx.Log.Debug("*****Finish calculate phases****")
	for _, phase := range p.BKEPhases {
		if err := phase.Report("", true); err != nil {
			return 0, err
		}
		if phase.GetStatus() == bkev1beta1.PhaseWaiting {
			p.ctx.Log.Debug("phase %s    ->     %s", phase.Name(), bkev1beta1.PhaseWaiting)
			waitPhaseCount++
		}
	}
	return waitPhaseCount, nil
}

func (p *PhaseFlow) Execute() (ctrl.Result, error) {
	defer p.handlePanic()

	phases := p.determinePhases()

	go p.ctx.WatchBKEClusterStatus()

	return p.executePhases(phases)
}

// handlePanic handles panic recovery in the Execute function
func (p *PhaseFlow) handlePanic() {
	if e := recover(); e != nil {
		debug.PrintStack()
		if recoverErr, ok := e.(error); ok {
			log.Error("panic recovered", "error", recoverErr)
		}
	}
}

// determinePhases determines which phases need to be executed based on cluster state
func (p *PhaseFlow) determinePhases() confv1beta1.BKEClusterPhases {
	var phases confv1beta1.BKEClusterPhases

	// 从bkeCluster中获取需要执行的phase
	if phaseutil.IsDeleteOrReset(p.ctx.BKECluster) {
		phases = ClusterDeleteResetPhaseNames()
	} else {
		phases = p.getWaitingPhases()
	}
	return phases
}

// cleanupUnexecutedPhases cleans up unexecuted phases by setting their status to unknown
func (p *PhaseFlow) cleanupUnexecutedPhases(phases *confv1beta1.BKEClusterPhases) {
	if len(*phases) > 0 {
		for i, phase := range p.ctx.BKECluster.Status.PhaseStatus {
			if phase.Name.In(*phases) {
				p.ctx.BKECluster.Status.PhaseStatus[i].Status = bkev1beta1.PhaseUnknown
			}
		}
		if err := mergecluster.SyncStatusUntilComplete(p.ctx.Client, p.ctx.BKECluster); err != nil {
			return
		}
	}
}

// executePhases executes the determined phases and returns the result
func (p *PhaseFlow) executePhases(phases confv1beta1.BKEClusterPhases) (ctrl.Result, error) {
	var errs []error
	var err error
	var res ctrl.Result

	defer p.cleanupUnexecutedPhases(&phases)

	for _, phase := range p.BKEPhases {
		p.ctx.BindPhaseLogger(phase.Name())
		p.ctx.Log.NormalLogger.Debugf("waiting phases num: %d", len(phases))
		p.ctx.Log.NormalLogger.Infof("current phase name: %s", phase.Name())

		if phase.Name().In(phases) {
			// 移除该phasename从phases中
			phases.Remove(phase.Name())

			phase.RegisterPreHooks(
				calculatingClusterPreStatusByPhase,
				registerPhaseCName,
			)
			phase.RegisterPostHooks(calculatingClusterPostStatusByPhase)

			// 实时计算是否需要执行，避免前置的phase对BKECluster的状态修改导致后续phase不执行
			if phase.NeedExecute(p.oldBKECluster, p.newBKECluster) {

				// 执行前置hook，设置phase状态为running，设置开始时间,上报
				if err = phase.ExecutePreHook(); err != nil {
					return res, err
				}

				// 执行
				phaseResult, phaseErr := phase.Execute()
				if phaseErr != nil {
					err = phaseErr
					errs = append(errs, phaseErr)
				}
				res = util.LowestNonZeroResult(res, phaseResult)
			} else {
				phase.SetStatus(bkev1beta1.PhaseSkipped)
			}

		} else {
			phase.SetStatus(bkev1beta1.PhaseSkipped)
		}

		if phase.GetStatus() == bkev1beta1.PhaseSkipped {
			p.ctx.Log.Debug("********************************")
			p.ctx.Log.Debug("phase %s    ->     %s", phase.Name(), bkev1beta1.PhaseSkipped)
			p.ctx.Log.Debug("********************************")
			if err := phase.Report("", false); err != nil {
				return ctrl.Result{}, err
			}
			continue
		}

		// 执行后置hook，设置phase状态为success或者failed，设置结束时间,上报
		err = phase.ExecutePostHook(err)
		if err != nil {
			errs = append(errs, err)
		}

		logFinishWhenDeployFailed(p.ctx)

		if err = p.refreshOldAndNewBKECluster(); err != nil {
			errs = append(errs, err)
		}
		if len(errs) > 0 {
			err = kerrors.NewAggregate(errs)
			return res, err
		}
	}

	return res, nil
}

func (p *PhaseFlow) refreshOldAndNewBKECluster() error {
	oldBkeCluster, err := mergecluster.GetLastUpdatedBKECluster(p.ctx.BKECluster)
	if err != nil {
		return err
	}
	p.oldBKECluster = oldBkeCluster
	p.newBKECluster = p.ctx.BKECluster.DeepCopy()
	return nil
}

func (p *PhaseFlow) getWaitingPhases() confv1beta1.BKEClusterPhases {
	phases := confv1beta1.BKEClusterPhases{}
	for _, phase := range p.ctx.BKECluster.Status.PhaseStatus {
		if phase.Status == bkev1beta1.PhaseWaiting {
			phases.Add(phase.Name)
		}
	}
	return phases
}

func calculatingClusterPreStatusByPhase(phase phaseframe.Phase) error {
	ctx := phase.GetPhaseContext()

	if phase.Name() == EnsureClusterName {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterChecking)
		return nil
	}
	return calculateClusterStatusByPhase(phase, nil)
}

func calculatingClusterPostStatusByPhase(phase phaseframe.Phase, err error) error {
	defer func() {
		ctx := phase.GetPhaseContext()
		// 设置一个标识，配合状态记录器，从而忽略phase运行中的状态被重复记录
		if ctx.BKECluster.Status.ClusterStatus != bkev1beta1.ClusterUnknown {
			annotation.SetAnnotation(ctx.BKECluster, annotation.StatusRecordAnnotationKey, "")
		}
	}()
	return calculateClusterStatusByPhase(phase, err)
}

func calculateClusterStatusByPhase(phase phaseframe.Phase, err error) error {
	phaseName := phase.Name()
	ctx := phase.GetPhaseContext()

	switch {
	case phaseName.In(CustomSetStatusPhaseNames()):
		return nil
	case phaseName.In(ClusterInitPhaseNames()):
		handleClusterInitPhase(ctx, err)
	case phaseName.In(ClusterScaleMasterUpPhaseNames()):
		handleClusterScaleMasterUpPhase(ctx, err)
	case phaseName.In(ClusterScaleWorkerUpPhaseNames()):
		handleClusterScaleWorkerUpPhase(ctx, err)
	case phaseName.In(ClusterDeletePhaseNames()):
		handleClusterDeletePhase(ctx, err)
	case phaseName.In(ClusterPausedPhaseNames()):
		handleClusterPausedPhase(ctx, err)
	case phaseName.In(ClusterDryRunPhaseNames()):
		handleClusterDryRunPhase(ctx, err)
	case phaseName.In(ClusterAddonsPhaseNames()):
		handleClusterAddonsPhase(ctx, err)
	case phaseName.In(ClusterUpgradePhaseNames()) || phaseName.In(DeclarativeClusterUpgradePhaseNames()):
		handleClusterUpgradePhase(ctx, err)
	case phaseName.In(ClusterScaleMasterDownPhaseNames()):
		handleClusterScaleMasterDownPhase(ctx, err)
	case phaseName.In(ClusterScaleWorkerDownPhaseNames()):
		handleClusterScaleWorkerDownPhase(ctx, err)
	case phaseName.In(ClusterManagePhaseNames()):
		handleClusterManagePhase(ctx, err)
	default:
		ctx.Log.Debug("phase %s has no explicit status mapping, preserving current ClusterStatus", phaseName.String())
	}
	return nil
}

// handleClusterInitPhase 处理集群初始化阶段
func handleClusterInitPhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterInitializationFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterInitializing)
	}
}

// handleClusterScaleMasterUpPhase 处理集群Master扩容阶段
func handleClusterScaleMasterUpPhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterMasterScalingUpFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterMasterScalingUp)
	}
}

// handleClusterScaleWorkerUpPhase 处理集群Worker扩容阶段
func handleClusterScaleWorkerUpPhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterWorkerScalingUpFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterWorkerScalingUp)
	}
}

// handleClusterDeletePhase 处理集群删除阶段
func handleClusterDeletePhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterDeleteFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterDeleting)
	}
}

// handleClusterPausedPhase 处理集群暂停阶段
func handleClusterPausedPhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterPauseFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterPaused)
	}
}

// handleClusterDryRunPhase 处理集群DryRun阶段
func handleClusterDryRunPhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterDryRunFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterDryRun)
	}
}

// handleClusterAddonsPhase 处理集群插件阶段
func handleClusterAddonsPhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterDeployAddonFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterDeployingAddon)
	}
}

// handleClusterUpgradePhase 处理集群升级阶段
func handleClusterUpgradePhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterUpgradeFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterUpgrading)
	}
}

// handleClusterScaleMasterDownPhase 处理集群Master缩容阶段
func handleClusterScaleMasterDownPhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterMasterScalingDownFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterMasterScalingDown)
	}
}

// handleClusterScaleWorkerDownPhase 处理集群Worker缩容阶段
func handleClusterScaleWorkerDownPhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterWorkerScalingDownFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterWorkerScalingDown)
	}
}

// handleClusterManagePhase 处理集群纳管阶段
func handleClusterManagePhase(ctx *phaseframe.PhaseContext, err error) {
	if err != nil {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterManageFailed)
	} else {
		phaseutil.SetClusterStatus(ctx.BKECluster, bkev1beta1.ClusterManaging)
	}
}

func logFinishWhenDeployFailed(phaseContext *phaseframe.PhaseContext) {
	phaseContextFields506 := phaseContext.Untie()
	_, _, bkeCluster, _, log := phaseContextFields506.Context,
		phaseContextFields506.Client,
		phaseContextFields506.BKECluster,
		phaseContextFields506.Scheme,
		phaseContextFields506.Log
	if bkeCluster.Status.ClusterStatus == bkev1beta1.ClusterInitializationFailed {
		log.Finish(constant.ClusterDeployFailedReason, "Cluster deploy failed, process exit")
	}
}

func registerPhaseCName(phase phaseframe.Phase) error {
	cName := ConvertPhaseNameToCN(phase.Name().String())
	phase.SetCName(cName)
	return nil
}
