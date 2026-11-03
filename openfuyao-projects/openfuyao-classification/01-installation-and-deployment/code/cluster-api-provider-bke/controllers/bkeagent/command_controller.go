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

// package for controller bkeagent
package bkeagent

import (
	"context"
	"fmt"
	"sync"
	"time"

	apierr "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/util/workqueue"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkenet "gopkg.openfuyao.cn/cluster-api-provider-bke/common/utils/net"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/job/builtin/reset"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// CommandReconciler reconciles a Command object
type CommandReconciler struct {
	client.Client
	APIReader client.Reader
	Scheme    *runtime.Scheme
	Ctx       context.Context
	Job       job.Job
	NodeName  string
	NodeIP    string
}

const (
	// commandFinalizerName 与 agentv1beta1.CommandFinalizer 保持一致。
	commandFinalizerName = agentv1beta1.CommandFinalizer

	commandControllerName = "command"
)

const (
	defaultFastDelay       = 10 * time.Second
	defaultSlowDelay       = 60 * time.Second
	defaultMaxFastAttempts = 5
)

func commandLogger() *log.Logger {
	return log.ControllerLogger(commandControllerName)
}

// reconcileResult 封装 Reconcile 函数的返回结果
type reconcileResult struct {
	result ctrl.Result
	err    error
	done   bool
}

// continueReconcile 表示继续执行后续逻辑
func continueReconcile() reconcileResult {
	return reconcileResult{done: false}
}

// finishReconcile 表示应该立即返回
func finishReconcile(result ctrl.Result, err error) reconcileResult {
	return reconcileResult{result: result, err: err, done: true}
}

// finishWithRequeue 表示需要重新入队
func finishWithRequeue() reconcileResult {
	return reconcileResult{result: ctrl.Result{Requeue: true}, done: true}
}

// unwrap 解包返回 ctrl.Result 和 error
func (r reconcileResult) unwrap() (ctrl.Result, error) {
	return r.result, r.err
}

// +kubebuilder:rbac:groups=bkeagent.bocloud.com,resources=commands,verbs=get;list;watch
// +kubebuilder:rbac:groups=bkeagent.bocloud.com,resources=commands/status,verbs=get;update;patch
// Reconcile command
func (r *CommandReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// 1. 获取 Command 对象
	command, res := r.fetchCommand(ctx, req)
	if res.done {
		return res.result, res.err
	}

	// 2. Commands 为空检查
	if len(command.Spec.Commands) == 0 {
		commandLogger().Warnf("command is not configured, %s/%s", command.Namespace, command.Name)
		return ctrl.Result{}, nil
	}

	commandLogger().Infof("reconcile %s/%s resourceVersion: %s, generation: %d",
		command.Namespace, command.Name, command.GetResourceVersion(), command.GetGeneration())

	// 3. 初始化 Status
	if res := r.ensureStatusInitialized(command); res.done {
		return res.result, res.err
	}

	currentStatus := command.Status[r.commandStatusKey(command)]
	gid := fmt.Sprintf("%s/%s", command.Namespace, command.Name)

	// 4. 处理 Finalizer
	if res := r.handleFinalizer(command, gid); res.done {
		return res.result, res.err
	}

	// 5. 跳过已完成的命令
	if currentStatus.Phase == agentv1beta1.CommandComplete {
		return ctrl.Result{}, nil
	}

	// 6. 处理暂停逻辑
	if res := r.handleSuspend(command, currentStatus, gid); res.done {
		return res.result, res.err
	}

	// 7. 跳过旧版本任务
	if r.shouldSkipOldTask(command, gid) {
		return ctrl.Result{}, nil
	}

	// 8. 创建并启动任务
	return r.createAndStartTask(ctx, command, currentStatus, gid).unwrap()
}

// fetchCommand 获取 Command 对象
// - ctx: 上下文
// - req: reconcile 请求，包含 NamespacedName
// - *agentv1beta1.Command: 获取到的 Command 对象，如果不存在则为 nil
// - reconcileResult: 包含是否需要返回以及具体的返回值
func (r *CommandReconciler) fetchCommand(ctx context.Context,
	req ctrl.Request) (*agentv1beta1.Command, reconcileResult) {
	command := new(agentv1beta1.Command)
	if err := r.Get(ctx, req.NamespacedName, command); err != nil {
		if apierr.IsNotFound(err) {
			return nil, finishReconcile(ctrl.Result{}, nil)
		}
		commandLogger().Errorf("unable to fetch Command, %s", err.Error())
		return nil, finishReconcile(ctrl.Result{}, err)
	}
	return command, continueReconcile()
}

// ensureStatusInitialized 确保 Command 的 Status 已初始化
// - command: Command 对象
// - reconcileResult: 如果初始化失败需要返回，否则继续
func (r *CommandReconciler) ensureStatusInitialized(command *agentv1beta1.Command) reconcileResult {
	if command.Status == nil {
		command.Status = map[string]*agentv1beta1.CommandStatus{}
	}
	statusKey := r.commandStatusKey(command)
	if _, ok := command.Status[statusKey]; !ok {
		command.Status[statusKey] = &agentv1beta1.CommandStatus{
			Conditions:     []*agentv1beta1.Condition{},
			LastStartTime:  &metav1.Time{Time: time.Now()},
			CompletionTime: nil,
			Succeeded:      -1,
			Failed:         -1,
			Phase:          agentv1beta1.CommandRunning,
			Status:         metav1.ConditionUnknown,
		}
		if err := r.syncStatusUntilComplete(command); err != nil {
			commandLogger().Errorf("unable to update command status resourceVersion: %s error: %s",
				command.GetResourceVersion(), err.Error())
			return finishReconcile(ctrl.Result{}, nil)
		}
	}
	return continueReconcile()
}

// cleanupTask 清理指定 gid 的任务
func (r *CommandReconciler) cleanupTask(gid string) {
	if v, ok := r.Job.GetTask(gid); ok {
		v.SafeClose()
		r.Job.DeleteTask(gid)
	}
}

// ensureFinalizer checks the Command carries the bke-controller finalizer.
// Node bkeagent must not patch commands (spec/finalizers); missing finalizer is logged only.
func (r *CommandReconciler) ensureFinalizer(command *agentv1beta1.Command) reconcileResult {
	if controllerutil.ContainsFinalizer(command, commandFinalizerName) {
		return continueReconcile()
	}
	log.Warnf("command %s/%s missing %s (expected at create by bke-controller); continue without node-side add",
		command.Namespace, command.Name, commandFinalizerName)
	return continueReconcile()
}

// handleDeletion cleans up the local task when Command is deleting.
// Finalizer removal stays on bke-controller, which holds commands write permission.
func (r *CommandReconciler) handleDeletion(command *agentv1beta1.Command, gid string) reconcileResult {
	r.cleanupTask(gid)
	log.Infof("command %s local task cleaned on delete; finalizer removal is bke-controller responsibility", gid)
	return finishReconcile(ctrl.Result{}, nil)
}

// handleFinalizer 处理 Finalizer 的添加和删除
// - command: Command 对象
// - gid: 全局任务 ID
// - reconcileResult: 处理结果
func (r *CommandReconciler) handleFinalizer(command *agentv1beta1.Command, gid string) reconcileResult {
	// 非删除状态：确保 finalizer 存在
	if command.ObjectMeta.DeletionTimestamp.IsZero() {
		return r.ensureFinalizer(command)
	}
	return r.handleDeletion(command, gid)
}

// handleSuspend 处理命令暂停逻辑
// - command: Command 对象
// - currentStatus: 当前节点的状态
// - gid: 全局任务 ID
// - reconcileResult: 如果已暂停或需要暂停，返回 done=true
func (r *CommandReconciler) handleSuspend(command *agentv1beta1.Command,
	currentStatus *agentv1beta1.CommandStatus, gid string) reconcileResult {
	if !command.Spec.Suspend {
		return continueReconcile()
	}

	if currentStatus.Phase == agentv1beta1.CommandSuspend {
		return finishReconcile(ctrl.Result{}, nil)
	}

	commandLogger().Infof("%s has been suspended", gid)
	if v, ok := r.Job.GetTask(gid); ok {
		v.SafeClose()
		v.SetPhase(agentv1beta1.CommandSuspend)
	}

	// Statistical state
	countResult := agentv1beta1.ConditionCount(currentStatus.Conditions, len(command.Spec.Commands))
	currentStatus.Succeeded, currentStatus.Failed, currentStatus.Status, currentStatus.Phase =
		countResult.Succeeded, countResult.Failed, countResult.Status, countResult.Phase
	currentStatus.Phase = agentv1beta1.CommandSuspend

	if err := r.syncStatusUntilComplete(command); err != nil {
		commandLogger().Errorf("unable to update command status resourceVersion: %s error: %s",
			command.GetResourceVersion(), err.Error())
		return finishReconcile(ctrl.Result{}, err)
	}
	return finishReconcile(ctrl.Result{}, nil)
}

// shouldSkipOldTask 检查是否应该跳过旧版本任务
// - command: Command 对象
// - gid: 全局任务 ID
// - bool: 如果应该跳过返回 true
func (r *CommandReconciler) shouldSkipOldTask(command *agentv1beta1.Command, gid string) bool {
	if v, ok := r.Job.GetTask(gid); ok {
		// This value is increased by a spec change
		if command.GetGeneration() <= v.GetGeneration() {
			commandLogger().Infof("A later version task is being executed, command:%s resourceVersion:%s-%s, generation:%d<=%d",
				gid, command.GetResourceVersion(), v.GetResourceVersion(), command.GetGeneration(), v.GetGeneration())
			return true
		}
		v.SafeClose()
	}
	return false
}

// createAndStartTask 创建新任务并启动执行
// - ctx: 上下文
// - command: Command 对象
// - currentStatus: 当前节点的状态
// - gid: 全局任务 ID
// - reconcileResult: 任务创建结果
func (r *CommandReconciler) createAndStartTask(ctx context.Context, command *agentv1beta1.Command,
	currentStatus *agentv1beta1.CommandStatus, gid string) reconcileResult {
	// Task cancel root must NOT be reconcile ctx (dies when Reconcile returns).
	taskCtx, cancel := context.WithCancel(context.Background())
	if command.Spec.ActiveDeadlineSecond > 0 {
		deadline := time.Now().Add(time.Duration(command.Spec.ActiveDeadlineSecond) * time.Second)
		taskCtx, cancel = context.WithDeadline(context.Background(), deadline)
	}
	task := &job.Task{
		Ctx:                     taskCtx,
		Cancel:                  cancel,
		StopChan:                make(chan struct{}),
		Phase:                   agentv1beta1.CommandRunning,
		ResourceVersion:         command.ResourceVersion,
		Generation:              command.GetGeneration(),
		TTLSecondsAfterFinished: command.Spec.TTLSecondsAfterFinished,
		HasAddTimer:             false,
		Once:                    &sync.Once{},
	}
	r.Job.SetTask(gid, task)

	// The start time is reset each time the Reconcile function is entered
	currentStatus.LastStartTime = &metav1.Time{Time: time.Now()}
	currentStatus.CompletionTime = nil

	if err := r.syncStatusUntilComplete(command); err != nil {
		commandLogger().Errorf("unable to update command status resourceVersion: %s error: %s",
			command.GetResourceVersion(), err.Error())
		cancel()
		r.Job.DeleteTask(gid)
		return finishReconcile(ctrl.Result{}, nil)
	}

	go r.startTask(task, command)
	return finishReconcile(ctrl.Result{}, nil)
}

// SetupWithManager sets up the controller with the Manager.
func (r *CommandReconciler) SetupWithManager(mgr ctrl.Manager) error {
	if r.NodeIP == "" {
		r.NodeIP = resolveLocalNodeIP()
	}
	// Object-level TTL / Delete for Command is owned by bke-controller
	// (CommandTTLJanitor + PurgeCommand). Node agent only patches Status.

	return ctrl.NewControllerManagedBy(mgr).
		For(&agentv1beta1.Command{}, r.commandPredicateFn()).
		WithOptions(controller.Options{
			RateLimiter: workqueue.NewItemFastSlowRateLimiter(defaultFastDelay, defaultSlowDelay, defaultMaxFastAttempts),
		}).Complete(r)
}

// shouldReconcileCommand 判断是否应该触发 Reconcile
// eventType 用于日志标识事件来源 (CreateFunc/UpdateFunc)
func (r *CommandReconciler) shouldReconcileCommand(o *agentv1beta1.Command, eventType string) bool {
	if o == nil {
		return false
	}
	gid := fmt.Sprintf("%s/%s", o.Namespace, o.Name)
	if v, ok := r.Job.GetTask(gid); ok {
		// The value of this field will increase only if you update the spec
		if o.Generation <= v.GetGeneration() || o.ResourceVersion <= v.GetResourceVersion() {
			commandLogger().Debugf("%s: %d<=%d Only update status without Reconcile %s",
				eventType, o.Generation, v.GetGeneration(), gid)
			return false
		}
	}
	// todo 废弃Spec.NodeName，可承载信息太少
	if o.Spec.NodeName == r.NodeName {
		return true
	}
	return r.nodeMatchNodeSelector(o.Spec.NodeSelector)
}

func (r *CommandReconciler) commandPredicateFn() builder.Predicates {
	commandPredicateFn := builder.WithPredicates(
		predicate.Funcs{
			CreateFunc: func(e event.CreateEvent) bool {
				cmd, ok := e.Object.(*agentv1beta1.Command)
				if !ok {
					return false
				}
				return r.shouldReconcileCommand(cmd, "CreateFunc")
			},
			UpdateFunc: func(e event.UpdateEvent) bool {
				cmd, ok := e.ObjectNew.(*agentv1beta1.Command)
				if !ok {
					return false
				}
				return r.shouldReconcileCommand(cmd, "UpdateFunc")
			},
			DeleteFunc: func(e event.DeleteEvent) bool {
				return false
			},
			GenericFunc: func(e event.GenericEvent) bool {
				return false
			},
		},
	)
	return commandPredicateFn
}

// ========== startTask 辅助函数 ==========

// calculateStopTime 根据命令配置计算任务的截止时间
func calculateStopTime(lastStartTime time.Time, activeDeadlineSecond int) time.Time {
	if activeDeadlineSecond > 0 {
		return lastStartTime.Add(time.Duration(activeDeadlineSecond) * time.Second)
	}
	return lastStartTime.Add(time.Duration(agentv1beta1.DefaultActiveDeadlineSecond) * time.Second)
}

// isCommandCompleted 检查指定命令是否已经执行完成（含 Skip，视为成功完成）
func isCommandCompleted(conditions []*agentv1beta1.Condition, commandID string) bool {
	cond := agentv1beta1.GetCondition(conditions, &agentv1beta1.Condition{ID: commandID})
	if cond == nil {
		return false
	}
	return cond.Phase == agentv1beta1.CommandComplete || cond.Phase == agentv1beta1.CommandSkip
}

// newCondition 为执行命令创建新的 Condition 对象
func newCondition(execCommandID string) *agentv1beta1.Condition {
	return &agentv1beta1.Condition{
		ID:            execCommandID,
		Status:        metav1.ConditionUnknown,
		Phase:         agentv1beta1.CommandRunning,
		LastStartTime: &metav1.Time{Time: time.Now()},
		StdErr:        []string{},
		StdOut:        []string{},
		Count:         0,
	}
}

// executeByType 根据命令类型路由到相应的执行器。
// timeout > 0 时 Shell/K8s 走带超时杀进程路径；BuiltIn 消费 taskCtx（P3：kubeadm 等待点；P4：其余插件）。
// 未知类型返回错误，由上层记 Failed（P9），禁止静默成功。
func (r *CommandReconciler) executeByType(ctx context.Context, cmdType agentv1beta1.CommandType, command []string, timeout time.Duration) ([]string, error) {
	switch cmdType {
	case agentv1beta1.CommandBuiltIn:
		return r.Job.BuiltIn.Execute(ctx, command)
	case agentv1beta1.CommandKubernetes:
		return r.Job.K8s.Execute(ctx, timeout, command)
	case agentv1beta1.CommandShell:
		return r.Job.Shell.Execute(ctx, timeout, command)
	default:
		return nil, fmt.Errorf("unsupported command type: %s", cmdType)
	}
}

// commandExecutionResult 封装命令执行的结果
type commandExecutionResult struct {
	timedOut bool
}

// markTaskCancelled 任务取消时统一置 Failed 并记录取消原因
func markTaskCancelled(condition *agentv1beta1.Condition, cause error) {
	condition.Status = metav1.ConditionFalse
	condition.Phase = agentv1beta1.CommandFailed
	condition.StdErr = append(condition.StdErr, fmt.Sprintf("task cancelled: %v", cause))
}

// waitBackoffDelay 等待重试退避延迟，延迟本身受 deadline / cancel 约束。
// 返回 true 表示任务已取消，调用方应立即终止重试。
func waitBackoffDelay(taskCtx context.Context, execCommand agentv1beta1.ExecCommand,
	condition *agentv1beta1.Condition, remaining time.Duration) bool {
	delay := time.Duration(execCommand.BackoffDelay) * time.Second
	if delay > remaining {
		delay = remaining
	}
	timer := time.NewTimer(delay)
	select {
	case <-taskCtx.Done():
		timer.Stop()
		cancelCause := taskCtx.Err()
		if cancelCause == nil {
			// Done 已关闭时 Err 按 stdlib 契约必非 nil；防御异常实现兜底
			cancelCause = context.Canceled
		}
		markTaskCancelled(condition, cancelCause)
		return true
	case <-timer.C:
	}
	return false
}

// execOnceResult 单次执行的结果处置：done=true 时 timedOut 为最终结果，done=false 继续重试
type execOnceResult struct {
	done     bool
	timedOut bool
}

// executeOnce 执行单次命令并按 select 结果处置 condition。
// remaining 为本次执行的 ActiveDeadline 预算，同时约束 goroutine 与等待计时器。
func (r *CommandReconciler) executeOnce(taskCtx context.Context, execCommand agentv1beta1.ExecCommand,
	condition *agentv1beta1.Condition, remaining time.Duration) execOnceResult {
	type execOut struct {
		result []string
		err    error
	}
	ch := make(chan execOut, 1)
	go func(budget time.Duration) {
		res, err := r.executeByType(taskCtx, execCommand.Type, execCommand.Command, budget)
		if err != nil {
			// 显式判错后仍经 channel 上抛，由 select 分支统一处置
			commandLogger().Debugf("execute command %s type=%s failed: %v", condition.ID, execCommand.Type, err)
		}
		ch <- execOut{result: res, err: err}
	}(remaining)

	timer := time.NewTimer(remaining)
	select {
	case out := <-ch:
		if !timer.Stop() {
			<-timer.C
		}
		if out.err != nil {
			condition.Status = metav1.ConditionFalse
			condition.Phase = agentv1beta1.CommandFailed
			condition.StdErr = append(condition.StdErr, out.err.Error())
			// ctx 取消：不重试
			if taskCtx.Err() != nil {
				return execOnceResult{done: true, timedOut: true}
			}
			commandLogger().Errorf("Command exec failed: %s %s", condition.ID, out.err.Error())
			return execOnceResult{done: false}
		}
		condition.Status = metav1.ConditionTrue
		condition.Phase = agentv1beta1.CommandComplete
		condition.StdOut = append(condition.StdOut, out.result...)
		return execOnceResult{done: true, timedOut: false}
	case <-taskCtx.Done():
		if !timer.Stop() {
			<-timer.C
		}
		cancelCause := taskCtx.Err()
		if cancelCause == nil {
			// Done 已关闭时 Err 按 stdlib 契约必非 nil；防御异常实现兜底
			cancelCause = context.Canceled
		}
		markTaskCancelled(condition, cancelCause)
		commandLogger().Warnf("Command exec cancelled (id=%s type=%s): %v", condition.ID, execCommand.Type, cancelCause)
		return execOnceResult{done: true, timedOut: true}
	case <-timer.C:
		// 超时：标 Failed，不继续重试。BuiltIn 已持有 taskCtx，deadline/cancel 可打断 kubeadm 等待。
		markActiveDeadlineExceeded(condition)
		commandLogger().Warnf("Command exec hit active deadline (id=%s type=%s); status Failed, no further retries",
			condition.ID, execCommand.Type)
		return execOnceResult{done: true, timedOut: true}
	}
}

// executeWithRetry 执行单个命令，支持重试机制。
// 单次执行受 ActiveDeadline 剩余时间约束：超时写 Failed 并立即返回（不继续重试、不推进后续子命令）。
func (r *CommandReconciler) executeWithRetry(taskCtx context.Context, execCommand agentv1beta1.ExecCommand,
	condition *agentv1beta1.Condition, stopTime time.Time, backoffLimit int) commandExecutionResult {

	for backoffLimit >= 0 && condition.Count <= backoffLimit {
		if err := taskCtx.Err(); err != nil {
			markTaskCancelled(condition, err)
			return commandExecutionResult{timedOut: true}
		}
		remaining := time.Until(stopTime)
		if remaining <= 0 {
			markActiveDeadlineExceeded(condition)
			return commandExecutionResult{timedOut: true}
		}
		// 重试时添加延迟（延迟本身也受 deadline / cancel 约束）
		if execCommand.BackoffDelay != 0 && condition.Count > 0 {
			if waitBackoffDelay(taskCtx, execCommand, condition, remaining) {
				return commandExecutionResult{timedOut: true}
			}
			remaining = time.Until(stopTime)
			if remaining <= 0 {
				markActiveDeadlineExceeded(condition)
				return commandExecutionResult{timedOut: true}
			}
		}
		condition.LastStartTime = &metav1.Time{Time: time.Now()}
		condition.Count++

		out := r.executeOnce(taskCtx, execCommand, condition, remaining)
		if out.done {
			return commandExecutionResult{timedOut: out.timedOut}
		}
		// 单次失败且未取消：继续下一轮重试
	}
	return commandExecutionResult{timedOut: false}
}

func markActiveDeadlineExceeded(condition *agentv1beta1.Condition) {
	condition.Status = metav1.ConditionFalse
	condition.Phase = agentv1beta1.CommandFailed
	condition.StdErr = append(condition.StdErr, "active deadline exceeded")
}

// processCommandResult 封装单个命令处理的结果
type processCommandResult struct {
	shouldBreak bool
	syncError   error
}

// processExecCommand 处理单个执行命令的完整流程
func (r *CommandReconciler) processExecCommand(taskCtx context.Context, command *agentv1beta1.Command,
	execCommand agentv1beta1.ExecCommand, currentStatus *agentv1beta1.CommandStatus,
	stopTime time.Time) processCommandResult {

	condition := newCondition(execCommand.ID)
	currentStatus.Conditions = agentv1beta1.ReplaceCondition(currentStatus.Conditions, condition)

	result := r.executeWithRetry(taskCtx, execCommand, condition, stopTime, command.Spec.BackoffLimit)

	// BackoffIgnore→Skip：视为成功完成（P6），Status=True 以便 Wait/TTL 与成功同路径清理
	if !result.timedOut && condition.Status == metav1.ConditionFalse && execCommand.BackoffIgnore {
		condition.Phase = agentv1beta1.CommandSkip
		condition.Status = metav1.ConditionTrue
	}

	// 对象已删除：不写回 Status（避免删后回执）
	if !command.DeletionTimestamp.IsZero() {
		return processCommandResult{shouldBreak: true}
	}

	// 更新每个子指令执行后的状态
	if err := r.syncStatusUntilComplete(command); err != nil {
		return processCommandResult{syncError: err}
	}

	// 超时/取消或明确失败均停止后续指令；Skip 继续后续子命令
	shouldBreak := result.timedOut || condition.Phase == agentv1beta1.CommandFailed
	return processCommandResult{shouldBreak: shouldBreak}
}

// finalizeTaskStatus 统计并更新任务的最终状态
func (r *CommandReconciler) finalizeTaskStatus(command *agentv1beta1.Command,
	currentStatus *agentv1beta1.CommandStatus, gid string) error {

	countResult := agentv1beta1.ConditionCount(currentStatus.Conditions, len(command.Spec.Commands))
	currentStatus.Succeeded = countResult.Succeeded
	currentStatus.Failed = countResult.Failed
	currentStatus.Status = countResult.Status
	currentStatus.Phase = countResult.Phase
	currentStatus.CompletionTime = &metav1.Time{Time: time.Now()}

	command.Status[r.commandStatusKey(command)] = currentStatus
	if err := r.syncStatusUntilComplete(command); err != nil {
		return err
	}

	if v, ok := r.Job.GetTask(gid); ok {
		v.SetPhase(currentStatus.Phase)
		v.SafeClose()
	}
	return nil
}

func (r *CommandReconciler) startTask(task *job.Task, command *agentv1beta1.Command) {
	// Reset scope "shutdown" schedules delayed agent stop during the plugin.
	// Flush on exit is idempotent and covers failure catch-up paths that only Mark.
	defer func() {
		if flushErr := reset.FlushPendingAgentCleanup(); flushErr != nil {
			commandLogger().Warnf("flush pending bkeagent cleanup failed (best-effort): %v", flushErr)
		}
	}()

	gid := fmt.Sprintf("%s/%s", command.Namespace, command.Name)
	currentStatus := command.Status[r.commandStatusKey(command)]
	if currentStatus == nil {
		commandLogger().Errorf("command status missing for key %s, gid: %s", r.commandStatusKey(command), gid)
		return
	}
	stopTime := calculateStopTime(currentStatus.LastStartTime.Time, command.Spec.ActiveDeadlineSecond)
	taskCtx := task.Ctx
	if taskCtx == nil {
		taskCtx = context.Background()
	}

	terminated := false
	for _, execCommand := range command.Spec.Commands {
		// 检查是否收到停止/取消信号
		select {
		case <-taskCtx.Done():
			commandLogger().Warnf("Execution command terminated %s: %v", gid, taskCtx.Err())
			terminated = true
		default:
		}
		if terminated {
			return // 取消后不 finalize、不写已删对象回执
		}
		if !command.DeletionTimestamp.IsZero() {
			return
		}
		// 检查是否超时
		if stopTime.Before(time.Now()) {
			break
		}
		if isCommandCompleted(currentStatus.Conditions, execCommand.ID) {
			continue
		}

		result := r.processExecCommand(taskCtx, command, execCommand, currentStatus, stopTime)
		if result.syncError != nil {
			commandLogger().Errorf("unable to update command status resourceVersion: %s error: %s",
				command.ResourceVersion, result.syncError.Error())
			return
		}
		if result.shouldBreak {
			break
		}
	}

	if !command.DeletionTimestamp.IsZero() || taskCtx.Err() != nil {
		return
	}
	if err := r.finalizeTaskStatus(command, currentStatus, gid); err != nil {
		commandLogger().Errorf("unable to update command status resourceVersion: %s error: %s",
			command.ResourceVersion, err.Error())
	}
}

func (r *CommandReconciler) commandStatusKey(command *agentv1beta1.Command) string {
	ip := r.statusIPForCommand(command)
	if ip == "" {
		return r.NodeName
	}
	return fmt.Sprintf("%s/%s", r.NodeName, ip)
}

func (r *CommandReconciler) statusIPForCommand(command *agentv1beta1.Command) string {
	if command != nil {
		candidates := nodeSelectorCandidateIPs(command.Spec.NodeSelector)
		if len(candidates) > 0 {
			if ip, err := bkenet.ResolveLocalNodeIP(candidates...); err == nil && ip != "" {
				return ip
			}
		}
	}
	return r.NodeIP
}

func nodeSelectorCandidateIPs(selector *metav1.LabelSelector) []string {
	if selector == nil || len(selector.MatchLabels) == 0 {
		return nil
	}
	candidates := make([]string, 0, len(selector.MatchLabels))
	for ip := range selector.MatchLabels {
		candidates = append(candidates, ip)
	}
	return candidates
}

func (r *CommandReconciler) statusSyncTimeout() time.Duration {
	// Delayed agent stop is already scheduled in the shutdown phase. Cap
	// writeback retries just under that delay so management-cluster scale-down
	// can still patch Completed, while self-manage delete (API down) does not
	// block on the default multi-minute sync window.
	if reset.AgentCleanupPending() {
		return reset.WritebackTimeoutWhileCleanupPending()
	}
	return 5 * time.Minute
}

func (r *CommandReconciler) syncStatusUntilComplete(cmd *agentv1beta1.Command) error {
	return r.syncCommandStatus(cmd, r.statusSyncTimeout())
}

func (r *CommandReconciler) syncCommandStatus(cmd *agentv1beta1.Command, timeout time.Duration) (err error) {
	ctx, cancel := context.WithTimeout(r.Ctx, timeout)
	defer cancel()
	for {
		select {
		case <-r.Ctx.Done():
			return
		case <-ctx.Done():
			return fmt.Errorf("command status update failed to complete after %s", timeout)
		default:
		}
		// Execute concurrent tasks at different peaks.
		// When the number of concurrent tasks is greater than 100, a random value of 1-15 is preferred
		const sleepRangeLow = 1
		const sleepRangeHigh = 2
		time.Sleep(time.Duration(rand.IntnRange(sleepRangeLow, sleepRangeHigh)) * time.Second)
		obj := &agentv1beta1.Command{}
		// This refresh is a direct request to the Kube-apiserver, which is equivalent to refreshing the local Client-go cache
		err = r.APIReader.Get(r.Ctx, client.ObjectKey{Namespace: cmd.Namespace, Name: cmd.Name}, obj)
		if err != nil {
			if apierr.IsNotFound(err) {
				commandLogger().Warnf("Command resource %s-%s not found, skip sync", cmd.Namespace, cmd.Name)
				return nil
			}
			commandLogger().Errorf("Get command resource failed %s-%s error: %v", cmd.Namespace, cmd.Name, err)
			continue
		}
		if obj.Status == nil {
			obj.Status = map[string]*agentv1beta1.CommandStatus{}
		}
		objCopy := obj.DeepCopy()
		statusKey := r.commandStatusKey(cmd)
		objCopy.Status[statusKey] = cmd.Status[statusKey]
		//patch status
		err = r.Client.Status().Patch(r.Ctx, objCopy, client.MergeFrom(obj))
		if err != nil {
			commandLogger().Warnf("Update command resource failed %s-%s error: %v", cmd.Namespace, cmd.Name, err)
			continue
		}
		break
	}
	return nil
}

func (r *CommandReconciler) nodeMatchNodeSelector(s *metav1.LabelSelector) bool {
	if s == nil {
		return false
	}
	selector, err := metav1.LabelSelectorAsSelector(s)
	if err != nil {
		return false
	}
	// get node name from NodeSelector
	nodeName, found := selector.RequiresExactMatch(r.NodeName)
	if !found {
		nodeName = ""
	}
	if nodeName == r.NodeName {
		return true
	}

	// check whether any selector IP exists on local interfaces
	ips, err := bkenet.GetAllInterfaceIP()
	if err != nil {
		return false
	}
	for _, addr := range ips {
		for candidateIP := range s.MatchLabels {
			if !bkenet.AddressMatchesIP(addr, candidateIP) {
				continue
			}
			parsed := bkenet.ParseAddrIP(addr)
			if parsed == nil || parsed.IsLoopback() || parsed.IsLinkLocalUnicast() || parsed.IsLinkLocalMulticast() {
				continue
			}
			return true
		}
	}
	return false
}

// resolveLocalNodeIP returns the first non-loopback, non-link-local interface IP for status map keys.
// Called once during SetupWithManager so NodeIP is not mutated on the informer path.
func resolveLocalNodeIP() string {
	ip, err := bkenet.ResolveLocalNodeIP()
	if err != nil {
		return ""
	}
	return ip
}
