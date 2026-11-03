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

package phaseframe

import (
	"context"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/cluster-api/util/annotations"
	ctrl "sigs.k8s.io/controller-runtime"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	metricrecord "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/metrics/record"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
)

// BasePhase is the base implementation of Phase, you can use it to implement your own phase
type BasePhase struct {
	// PhaseName is the name of the phase
	PhaseName confv1beta1.BKEClusterPhase
	// PhaseCName is the Chinese name of the phase
	PhaseCName string

	// Ctx is the context of the phase, it contains the needed information for the phase to execute
	Ctx *PhaseContext
	// Status is the status of the phase
	Status confv1beta1.BKEClusterPhaseStatus
	// StartTime is the start time of the phase
	StartTime metav1.Time

	// CustomPreHookFunc is the custom pre hook function,
	// it will be executed after the default pre hook but before report when use DefaultPreHook
	CustomPreHookFuncs []func(p Phase) error

	// CustomPostHookFunc is the custom post hook function,
	// it will be executed after the default post hook but before report when use DefaultPostHook
	CustomPostHookFuncs []func(p Phase, err error) error
}

// NewBasePhase returns a new BasePhase,you can use it to implement your own phase
func NewBasePhase(ctx *PhaseContext, phaseName confv1beta1.BKEClusterPhase) BasePhase {
	if ctx == nil {
		ctx = NewReconcilePhaseCtx(context.Background())
	}
	if ctx.Log == nil {
		ctx.Log = bkev1beta1.NewBKELogger(nil, nil, ctx.BKECluster)
	}
	ctx.BindPhaseLogger(phaseName)
	return BasePhase{
		PhaseName:           phaseName,
		Ctx:                 ctx,
		CustomPreHookFuncs:  make([]func(p Phase) error, 0),
		CustomPostHookFuncs: make([]func(p Phase, err error) error, 0),
	}
}

// DefaultPreHook is the default implementation of ExecutePreHook, use on demand
func (b *BasePhase) DefaultPreHook() error {
	b.Ctx.BindPhaseLogger(b.PhaseName)

	// refresh bkecluster
	if err := b.Ctx.RefreshCtxBKECluster(); err != nil {
		return err
	}
	// refresh cluster, it's not necessary to refresh successfully
	_ = b.Ctx.RefreshCtxCluster()
	// set status and start time
	b.SetStatus(bkev1beta1.PhaseRunning)
	b.SetStartTime(metav1.Now())

	// run custom pre hook
	if b.CustomPreHookFuncs != nil && len(b.CustomPreHookFuncs) > 0 {
		for _, f := range b.CustomPreHookFuncs {
			if err := f(b); err != nil {
				return err
			}
		}
	}
	// report phase status
	return b.Report("", false)
}

// DefaultPostHook is the default implementation of ExecutePostHook, use on demand
func (b *BasePhase) DefaultPostHook(err error) error {

	if b.Name() != "EnsureDeleteOrReset" {
		defer metricrecord.PhaseDurationRecord(b.Ctx.BKECluster, b.CName(), b.StartTime.Time, err)
	}

	var msg string
	if err != nil {
		msg = err.Error()
	}
	if b.GetStatus() == bkev1beta1.PhaseSkipped {
		return b.Report(msg, false)
	}
	if err != nil {
		b.SetStatus(bkev1beta1.PhaseFailed)
		b.Ctx.Log.Debug("phase %q run failed: %v", b.Name(), err)
	} else {
		b.SetStatus(bkev1beta1.PhaseSucceeded)
		b.Ctx.Log.Debug("phase %q run succeeded", b.Name())
	}
	if b.CustomPostHookFuncs != nil && len(b.CustomPostHookFuncs) > 0 {
		for _, f := range b.CustomPostHookFuncs {
			if err := f(b, err); err != nil {
				return err
			}
		}
	}
	return b.Report(msg, false)
}

// checkCommonNeedExecute 检查通用的执行条件
func (b *BasePhase) checkCommonNeedExecute(new *bkev1beta1.BKECluster) bool {
	// 对于删除的BKECluster，不执行
	if !new.DeletionTimestamp.IsZero() {
		return false
	}
	// 对于暂停的BKECluster，不执行
	if new.Spec.Pause || annotations.HasPaused(new) {
		return false
	}
	// 对于DryRun的BKECluster，不执行
	if new.Spec.DryRun {
		return false
	}
	if strings.HasSuffix(string(new.Status.ClusterStatus), "Failed") {
		return false
	}
	return true
}

// DefaultNeedExecute is the default implementation of NeedExecute, use on demand
// it's only used for BKECluster type of 'bke'
func (b *BasePhase) DefaultNeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	// 检查通用条件
	if !b.checkCommonNeedExecute(new) {
		return false
	}

	// 对于不是BKECluster，并且没有完全控制的，不执行
	if !clusterutil.IsBKECluster(new) && !clusterutil.FullyControlled(new) {
		return false
	}

	return true
}

// NormalNeedExecute differs from DefaultNeedExecute is for all type of BKECluster
func (b *BasePhase) NormalNeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	// 检查通用条件
	return b.checkCommonNeedExecute(new)
}

// NeedExecute is the default implementation of NeedExecute.
// By comparing the status of old and new, determine whether the current stage needs to be executed
func (b *BasePhase) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	return b.DefaultNeedExecute(old, new)
}

// ExecutePreHook is the default implementation of ExecutePreHook.
// It is called before the execution of the current phase
func (b *BasePhase) ExecutePreHook() error {
	return b.DefaultPreHook()
}

// RegisterPreHooks is used to register a custom pre hook function
func (b *BasePhase) RegisterPreHooks(hooks ...func(p Phase) error) {
	b.CustomPreHookFuncs = append(b.CustomPreHookFuncs, hooks...)
}

// RegisterPostHooks is used to register a custom post hook function
func (b *BasePhase) RegisterPostHooks(hooks ...func(p Phase, err error) error) {
	b.CustomPostHookFuncs = append(b.CustomPostHookFuncs, hooks...)
}

// ExecutePostHook is the default implementation of ExecutePostHook.
// It is called after the execution of the current phase
func (b *BasePhase) ExecutePostHook(err error) error {
	return b.DefaultPostHook(err)
}

// Execute is the default implementation of Execute.
// It is called when the current phase is needed to be executed
func (b *BasePhase) Execute() (ctrl.Result, error) {
	panic("implement me")
}

// Name returns the name of the current phase
func (b *BasePhase) Name() confv1beta1.BKEClusterPhase {
	return b.PhaseName
}

func (b *BasePhase) CName() string {
	if b.PhaseCName == "" {
		return b.PhaseName.String()
	}
	return b.PhaseCName
}

func (b *BasePhase) SetCName(name string) {
	b.PhaseCName = name
}

// Report is used to report the status on the PhaseContext.BKECluster.Status.PhaseStatus
// todo bug -> 避免一个phase重复，应该更新开始时间和结束时间
func (b *BasePhase) Report(msg string, onlyRecord bool) error {
	phaseContextFields223 := b.Ctx.Untie()
	_, c, bkeCluster, _, log := phaseContextFields223.Context,
		phaseContextFields223.Client,
		phaseContextFields223.BKECluster,
		phaseContextFields223.Scheme,
		phaseContextFields223.Log

	// 没有状态不上报，说明不执行，也不需要在状态中展示
	if b.Status == "" {
		return nil
	}
	status := bkeCluster.Status.PhaseStatus

	defer func() {
		bkeCluster.Status.PhaseStatus = status
		if onlyRecord {
			return
		}
		if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
			log.NormalLogger.Warnf("Failed to update BKECluster status: %v", err)
		}
	}()

	// 根据不同状态处理报告
	switch b.Status {
	case bkev1beta1.PhaseSkipped:
		status = b.handleSkippedStatus(status, b.PhaseName)
	case bkev1beta1.PhaseWaiting:
		status = b.handleWaitingStatus(status, b.PhaseName)
	case bkev1beta1.PhaseRunning:
		status = b.handleRunningStatus(status, b.PhaseName, bkeCluster)
	default: // PhaseFailed or PhaseSucceeded
		status = b.handleCompletedStatus(status, b.PhaseName, msg)
	}
	return nil
}

// handleSkippedStatus handles the skipped status reporting
func (b *BasePhase) handleSkippedStatus(status []confv1beta1.PhaseState, phaseName confv1beta1.BKEClusterPhase) []confv1beta1.PhaseState {
	// 反向遍历status，如果已经存在，且是等待或者执行中，原地更新为跳过,且置0start-time和end-time
	for i := len(status) - 1; i >= 0; i-- {
		ps := *status[i].DeepCopy()
		if ps.Name == phaseName && (ps.Status == bkev1beta1.PhaseWaiting || ps.Status == bkev1beta1.PhaseRunning) {
			ps.Status = bkev1beta1.PhaseSkipped
			ps.StartTime = nil
			ps.EndTime = nil
			status[i] = ps
			return status
		}
	}
	ps := confv1beta1.PhaseState{
		Name:   phaseName,
		Status: bkev1beta1.PhaseSkipped,
	}
	status = append(status, ps)
	return status
}

// handleWaitingStatus handles the waiting status reporting
func (b *BasePhase) handleWaitingStatus(status []confv1beta1.PhaseState, phaseName confv1beta1.BKEClusterPhase) []confv1beta1.PhaseState {
	// 反向遍历status
	for i := len(status) - 1; i >= 0; i-- {
		ps := *status[i].DeepCopy()
		// 如果已经存在，且是执行失败，追加
		if ps.Name == phaseName && ps.Status == bkev1beta1.PhaseFailed {
			break
		}
		// 如果已经存在，且是执行成功，原地更新为等待或跳过,且置0start-time和end-time
		if ps.Name == phaseName && ps.Status == bkev1beta1.PhaseSucceeded {
			ps.Status = b.Status
			ps.StartTime = nil
			ps.EndTime = nil
			status[i] = ps
			return status
		}
		// 如果已经存在，且是等待或者执行中，原地更新为等待或跳过,且置0start-time和end-time
		if ps.Name == phaseName && (ps.Status == bkev1beta1.PhaseWaiting || ps.Status == bkev1beta1.PhaseRunning) {
			ps.Status = b.Status
			ps.StartTime = nil
			ps.EndTime = nil
			status[i] = ps
			return status
		}
	}

	ps := confv1beta1.PhaseState{
		Name:   phaseName,
		Status: b.Status,
	}
	status = append(status, ps)
	return status
}

// handleRunningStatus handles the running status reporting
func (b *BasePhase) handleRunningStatus(status []confv1beta1.PhaseState, phaseName confv1beta1.BKEClusterPhase, bkeCluster *bkev1beta1.BKECluster) []confv1beta1.PhaseState {
	// Phase 和 ClusterStatus 由 pre-hook calculatingClusterPreStatusByPhase 通过 SetClusterStatus 统一设置，
	// 此处不覆盖 Phase，以保持三字段一致性。仅更新 PhaseStatus 切片。
	// 找到已经存在的b.status，如果存在，原地更新
	for i := len(status) - 1; i >= 0; i-- {
		ps := *status[i].DeepCopy()
		if ps.Name == phaseName && (ps.Status == bkev1beta1.PhaseWaiting || ps.Status == bkev1beta1.PhaseRunning) {
			ps.Status = bkev1beta1.PhaseRunning
			ps.StartTime = &b.StartTime
			status[i] = ps
			return status
		}
	}
	// 没有找到，追加
	ps := confv1beta1.PhaseState{
		Name:      phaseName,
		Status:    bkev1beta1.PhaseRunning,
		StartTime: &b.StartTime,
	}
	status = append(status, ps)
	return status
}

// handleCompletedStatus handles the completed status reporting (failed or succeeded)
func (b *BasePhase) handleCompletedStatus(status []confv1beta1.PhaseState, phaseName confv1beta1.BKEClusterPhase, msg string) []confv1beta1.PhaseState {
	for i := len(status) - 1; i >= 0; i-- {
		ps := *status[i].DeepCopy()
		if ps.Name == phaseName && (ps.Status == bkev1beta1.PhaseRunning || ps.Status == bkev1beta1.PhaseWaiting) {
			ps.Status = b.Status
			ps.StartTime = &b.StartTime
			ps.EndTime = &metav1.Time{Time: time.Now()}
			ps.Message = msg

			status[i] = ps
			return status
		}
	}

	ps := confv1beta1.PhaseState{
		Name:      phaseName,
		Status:    b.Status,
		StartTime: &b.StartTime,
		EndTime:   &metav1.Time{Time: time.Now()},
		Message:   msg,
	}
	status = append(status, ps)
	return status
}

// SetStatus is used to set the status of the current phase
func (b *BasePhase) SetStatus(status confv1beta1.BKEClusterPhaseStatus) {
	b.Status = status
}

// GetStatus is used to get the status of the current phase
func (b *BasePhase) GetStatus() confv1beta1.BKEClusterPhaseStatus {
	return b.Status
}

// SetStartTime is used to set the startTime of the current phase
func (b *BasePhase) SetStartTime(startTime metav1.Time) {
	b.StartTime = startTime
}

// GetStartTime is used to get the startTime of the current phase
func (b *BasePhase) GetStartTime() metav1.Time {
	return b.StartTime
}

// GetPhaseContext is used to get the PhaseContext of the current phase
func (b *BasePhase) GetPhaseContext() *PhaseContext {
	return b.Ctx
}

// SetPhaseContext is used to set the PhaseContext of the current phase
func (b *BasePhase) SetPhaseContext(ctx *PhaseContext) {
	b.Ctx = ctx
}
