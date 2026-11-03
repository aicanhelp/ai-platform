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

package command

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/record"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util/patch"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	agentutils "gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

type BaseCommand struct {
	Ctx         context.Context
	Client      client.Client
	NameSpace   string
	Scheme      *runtime.Scheme
	OwnerObj    metav1.Object
	ClusterName string

	// RemoveAfterWait remove the command after wait
	RemoveAfterWait bool
	// Unique only one command can be stored in the cluster
	Unique bool

	// WaitTimeout the command wait timeout
	WaitTimeout time.Duration
	// WaitInterval the command wait interval
	WaitInterval time.Duration

	// Recorder optional; used to emit Owner Events on WaitTimeout (A8).
	Recorder record.EventRecorder

	commandName string
	// Command the command obj
	Command *agentv1beta1.Command
}

// TimeoutCaseResult encapsulates the result of handleTimeoutCase function
type TimeoutCaseResult struct {
	Err          error
	Complete     bool
	SuccessNodes []string
	FailedNodes  []string
}

// CommandNodes encapsulates the success and failed nodes
type CommandNodes struct {
	SuccessNodes []string
	FailedNodes  []string
}

// WaitCommandResult encapsulates the result of waitCommandComplete function
type WaitCommandResult struct {
	Err          error
	Reason       string
	Complete     bool
	SuccessNodes []string
	FailedNodes  []string
}

const (
	DefaultWaitTimeout  = 5 * time.Minute
	DefaultWaitInterval = 2 * time.Second

	// ActiveDeadlineWaitSkew is subtracted from WaitTimeout when linking ActiveDeadline.
	ActiveDeadlineWaitSkew = 60 * time.Second

	ReasonWaitTimeout       = "WaitTimeout"
	EventCommandWaitTimeout = "CommandWaitTimeout"
)

func (b *BaseCommand) validate() error {
	if b.Client == nil {
		return errors.New("client is nil")
	}
	if b.Scheme == nil {
		return errors.New("scheme is nil")
	}
	if b.NameSpace == "" {
		return errors.New("name space is empty")
	}
	return nil
}

// ValidateBkeCommand 验证BKE命令的通用函数，包括Nodes和BkeConfigName的验证
func ValidateBkeCommand(nodes interface{ Length() int }, bkeConfigName string, baseCommand *BaseCommand) error {
	if baseCommand.Client == nil {
		return errors.New("client is nil")
	}
	if lenNodes := nodes.Length(); lenNodes == 0 {
		return errors.New("env command except at least one node but got " + strconv.Itoa(lenNodes))
	}
	if bkeConfigName == "" {
		return errors.New("bkeConfigName is empty")
	}
	return baseCommand.validate()
}

// GenerateBkeConfigStr 生成BKE配置字符串
func GenerateBkeConfigStr(namespace, bkeConfigName string) string {
	return fmt.Sprintf("bkeConfig=%s:%s", namespace, bkeConfigName)
}

type Command interface {
	// Validate the command fields
	Validate() error
	// New create the command obj
	New() error
}

const (
	// BootstrapCommandNamePrefix all the bootstrap command Prefix,
	BootstrapCommandNamePrefix = "bootstrap-"

	// HACommandName the loadbalancer command name,
	HACommandName = "k8s-ha-deploy"

	// K8sEnvCommandName the k8s env command name,
	K8sEnvCommandName = "k8s-env-init"

	// K8sContainerdResetCommandName the k8s containerd reset command name,
	K8sContainerdResetCommandName = "k8s-containerd-reset"

	// K8sContainerdRedeployCommandName the k8s containerd redeploy command name,
	K8sContainerdRedeployCommandName = "k8s-containerd-redeploy"

	// K8sEnvDryRunCommandName the k8s env command name,
	K8sEnvDryRunCommandName = "k8s-env-dry-run"

	// K8sHostsCommandName the k8s hosts command name,
	K8sHostsCommandName = "k8s-hosts-generate"

	// K8sImagePrePullCommandName the k8s image pre pull command name,
	K8sImagePrePullCommandName = "k8s-image-pre-pull"

	// SwitchClusterCommandNamePrefix the switch cluster command name prefix,
	SwitchClusterCommandNamePrefix = "switch-cluster-"

	ResetNodeCommandNamePrefix = "reset-node-"

	UpgradeNodeCommandNamePrefix = "upgrade-node-"

	PingCommandNamePrefix = "ping-"

	CollectCertCommandNamePrefix = "collect-"
)

// two label used to select the Command,when reconcile
const (
	BKEClusterLabel = "bke.bocloud.com/cluster-command"
	BKEMachineLabel = "bke.bocloud.com/machine-command"

	MasterInitCommandLabel = "bke.bocloud.com/master-init-command"
	MasterJoinCommandLabel = "bke.bocloud.com/master-join-command"
	WorkerJoinCommandLabel = "bke.bocloud.com/worker-join-command"
)

const (
	// DefaultBackoffLimit max retry times
	DefaultBackoffLimit            = 3
	DefaultActiveDeadlineSecond    = 1000
	DefaultTTLSecondsAfterFinished = 600
)

// newCommand 创建命令对象
func (b *BaseCommand) newCommand(commandName, labelKey string, commandSpec *agentv1beta1.CommandSpec, customLabel ...string) error {
	if err := b.handleUniqueCommand(commandName); err != nil {
		return err
	}

	b.setCommandName(commandName)

	command := b.buildCommandObject(commandName, labelKey, commandSpec, customLabel)

	if err := b.setOwnerReference(command); err != nil {
		return err
	}

	return b.createCommand(command)
}

// uniqueCommandBase returns the stable identity of a command name.
// Only a trailing numeric segment is treated as a timestamp suffix.
func uniqueCommandBase(name string) string {
	idx := strings.LastIndex(name, "-")
	if idx <= 0 {
		return name
	}
	if _, err := strconv.ParseInt(name[idx+1:], 10, 64); err != nil {
		return name
	}
	return name[:idx]
}

func isSameUniqueCommandFamily(want, existing string) bool {
	return uniqueCommandBase(want) == uniqueCommandBase(existing)
}

// handleUniqueCommand 处理唯一性命令：设置了 Unique 时，创建新命令前清掉
// 同族（去掉尾部时间戳后同名）的所有已有命令（含失败/未完成对象，P5）。
// 注意两点失效模式都必须处理：
//  1. 只删一个会让重试轮次中新旧命令互相抵消，存量永不收敛，因此清全族；
//  2. 命令对象带 finalizer 且摘除职责在 bke-controller（agent 无 Commands patch
//     权限），裸 Delete 只会打出 deletionTimestamp 让对象永久 terminating——
//     后续每轮清理都会先命中这个僵尸并"幂等成功"，一个真存量都清不到。
//     因此删除后必须立即摘除 finalizer，对已 terminating 的僵尸则直接跳过。
//
// 失败 Command 回收策略（P5）：不设独立 Failed TTL；依赖本函数 Unique 覆盖清理，
// 以及运维人工标注/删除残留对象。成功对象仍由 CommandTTLJanitor 按 TTL 清理。
func (b *BaseCommand) handleUniqueCommand(commandName string) error {
	if !b.Unique {
		return nil
	}

	commandList := &agentv1beta1.CommandList{}
	if err := b.Client.List(b.Ctx, commandList, client.InNamespace(b.NameSpace)); err != nil {
		return errors.Wrapf(err, "failed to list command in namespace %s", b.NameSpace)
	}
	for i := range commandList.Items {
		command := &commandList.Items[i]
		if !isSameUniqueCommandFamily(commandName, command.Name) {
			continue
		}
		// Terminating commands are stuck on finalizer removal; skipping them
		// keeps them from consuming this cleanup round.
		if !command.DeletionTimestamp.IsZero() {
			continue
		}
		// Individual failures are logged instead of blocking the new command.
		if err := PurgeCommand(b.Ctx, b.Client, command); err != nil {
			log.Warnf("unique command purge of %s failed (continuing): %v", command.GetName(), err)
			continue
		}
	}
	return nil
}

// buildCommandObject 构建命令对象
func (b *BaseCommand) buildCommandObject(commandName, labelKey string, commandSpec *agentv1beta1.CommandSpec, customLabel []string) *agentv1beta1.Command {
	command := &agentv1beta1.Command{}
	command.SetGroupVersionKind(agentv1beta1.GroupVersion.WithKind("Command"))
	command.SetName(commandName)
	command.SetNamespace(b.NameSpace)
	command.Spec = *commandSpec.DeepCopy()
	b.applyActiveDeadlineLinkage(&command.Spec)

	// set labels
	labels := b.buildLabels(labelKey, customLabel)
	command.SetLabels(labels)
	controllerutil.AddFinalizer(command, agentv1beta1.CommandFinalizer)

	return command
}

// applyActiveDeadlineLinkage sets ActiveDeadlineSecond from WaitTimeout when unset (A3).
// Explicit positive values are respected.
func (b *BaseCommand) applyActiveDeadlineLinkage(spec *agentv1beta1.CommandSpec) {
	if spec == nil || spec.ActiveDeadlineSecond > 0 {
		return
	}
	spec.ActiveDeadlineSecond = b.linkedActiveDeadlineSecond()
}

func (b *BaseCommand) linkedActiveDeadlineSecond() int {
	wait := b.WaitTimeout
	if wait == 0 {
		wait = DefaultWaitTimeout
	}
	secs := int((wait - ActiveDeadlineWaitSkew).Seconds())
	if secs < 1 {
		return 1
	}
	return secs
}

// buildLabels 构建标签映射
func (b *BaseCommand) buildLabels(labelKey string, customLabel []string) map[string]string {
	labels := map[string]string{
		labelKey: "",
	}
	if b.ClusterName != "" {
		labels[clusterv1.ClusterNameLabel] = b.ClusterName
	}
	for _, label := range customLabel {
		labels[label] = ""
	}
	return labels
}

// setOwnerReference 设置所有者引用
func (b *BaseCommand) setOwnerReference(command *agentv1beta1.Command) error {
	if b.OwnerObj == nil {
		return nil
	}
	if err := controllerutil.SetControllerReference(b.OwnerObj, command, b.Scheme); err != nil {
		return errors.Wrapf(err, "failed to set controller reference for owner %s, controlled %s", b.OwnerObj.GetName(), command.Name)
	}
	return nil
}

// createCommand 创建命令
func (b *BaseCommand) createCommand(command *agentv1beta1.Command) error {
	if err := b.Client.Create(b.Ctx, command); err != nil {
		if apierrors.IsAlreadyExists(err) {
			log.Infof("command %s already exists, reusing existing one", command.Name)
			_, err := b.GetCommand()
			return err
		}
		return errors.Wrapf(err, "failed to create command %s", command.Name)
	}
	return nil
}

func (b *BaseCommand) setCommandName(commandName string) {
	b.commandName = commandName
}

func (b *BaseCommand) GetCommand() (*agentv1beta1.Command, error) {
	if b.commandName == "" {
		return nil, errors.New("command name is empty")
	}
	command := &agentv1beta1.Command{}
	if err := b.Client.Get(b.Ctx, client.ObjectKey{Namespace: b.NameSpace, Name: b.commandName}, command); err != nil {
		return nil, errors.Wrapf(err, "failed to get command %s", b.commandName)
	}
	b.Command = command
	return command, nil
}

// PurgeCommand 删除命令并摘除其 finalizer，保证对象真正消失。
// agent 无 Commands patch 权限、摘 finalizer 的职责在 bke-controller，
// 因此所有删除 Command 的路径都必须经过这里：裸 Delete 只会打出
// deletionTimestamp，对象将永久停留在 terminating 并堵住后续清理。
// 顺序为先 Delete（agent 观察到删除事件后清理本地任务）再摘 finalizer。
func PurgeCommand(ctx context.Context, c client.Client, command *agentv1beta1.Command) error {
	if err := c.Delete(ctx, command); err != nil && !apierrors.IsNotFound(err) {
		return errors.Wrapf(err, "failed to delete command %s", command.GetName())
	}
	stale := &agentv1beta1.Command{}
	if err := c.Get(ctx, client.ObjectKeyFromObject(command), stale); err != nil {
		return nil // already gone
	}
	if !controllerutil.ContainsFinalizer(stale, agentv1beta1.CommandFinalizer) {
		return nil
	}
	patchHelper, err := patch.NewHelper(stale, c)
	if err != nil {
		return errors.Wrapf(err, "failed to create patch helper for command %s", stale.GetName())
	}
	controllerutil.RemoveFinalizer(stale, agentv1beta1.CommandFinalizer)
	if err := patchHelper.Patch(ctx, stale); err != nil {
		return errors.Wrapf(err, "failed to remove finalizer from command %s", stale.GetName())
	}
	return nil
}

func (b *BaseCommand) deleteCommand() error {
	if b.Command == nil {
		return nil
	}
	return PurgeCommand(b.Ctx, b.Client, b.Command)
}

// waitCommandComplete 等待命令完成
func (b *BaseCommand) waitCommandComplete() (error, bool, CommandNodes) {
	result := b.waitCommandCompleteWithStruct()
	return result.Err, result.Complete, CommandNodes{
		SuccessNodes: result.SuccessNodes,
		FailedNodes:  result.FailedNodes,
	}
}

// waitCommandCompleteWithStruct 等待命令完成，返回结构体
func (b *BaseCommand) waitCommandCompleteWithStruct() WaitCommandResult {
	if b.commandName == "" {
		return WaitCommandResult{
			Err:          errors.New("command name is empty"),
			Complete:     false,
			SuccessNodes: nil,
			FailedNodes:  nil,
		}
	}
	if b.WaitInterval == 0 {
		b.WaitInterval = DefaultWaitInterval
	}
	if b.WaitTimeout == 0 {
		b.WaitTimeout = DefaultWaitTimeout
	}

	var complete bool
	var successNodes []string
	var failedNodes []string
	var reason string

	// 等待命令完成
	err := b.waitForCommandCompletion(&complete, &successNodes, &failedNodes)

	// 处理超时情况：留证后立即 Purge（与 RemoveAfterWait 解耦）
	if errors.Is(err, wait.ErrWaitTimeout) {
		timeoutResult := b.handleTimeoutCase(complete, successNodes, failedNodes)
		err = timeoutResult.Err
		complete = timeoutResult.Complete
		successNodes = timeoutResult.SuccessNodes
		failedNodes = timeoutResult.FailedNodes
		reason = ReasonWaitTimeout
		b.recordWaitTimeoutEvidence(failedNodes, successNodes)
		if deleteErr := b.deleteCommand(); deleteErr != nil {
			log.Warnf("purge command %s after %s failed: %v", b.commandName, ReasonWaitTimeout, deleteErr)
		}
		if err == nil {
			err = fmt.Errorf("%s: command=%s failedNodes=%v", ReasonWaitTimeout, b.commandName, failedNodes)
		}
	} else {
		// 成功路径：仍由 RemoveAfterWait 控制是否删除
		if b.RemoveAfterWait {
			if deleteErr := b.deleteCommand(); deleteErr != nil {
				log.Warnf("delete command %s failed: %v", b.commandName, deleteErr)
			}
		}
	}

	return WaitCommandResult{
		Err:          err,
		Reason:       reason,
		Complete:     complete,
		SuccessNodes: successNodes,
		FailedNodes:  failedNodes,
	}
}

// waitForCommandCompletion 内部函数：等待命令执行完成
func (b *BaseCommand) waitForCommandCompletion(complete *bool, successNodes, failedNodes *[]string) error {
	ctxTimeout, cancel := context.WithTimeout(b.Ctx, b.WaitTimeout)
	defer cancel()

	// seen 记录是否成功读取过命令对象，用于区分两类 NotFound。
	seen := false
	err := wait.PollImmediateUntil(b.WaitInterval, func() (bool, error) {
		command, err := b.GetCommand()
		if err != nil {
			// NotFound 分两类：
			//  1. 从未见过的 NotFound：Create 直写 apiserver、Get 走 informer
			//     缓存，首次轮询会撞上"watch 事件未送达"的缓存空窗，属瞬时
			//     读写竞态，容忍并继续轮询，交由 WaitTimeout 兜底。
			//  2. 曾经见过后的 NotFound：命令被中途删除（外部清理或并发
			//     Purge），继续轮询只会傻等满 WaitTimeout；立即终止并把
			//     NotFound 原样传给调用方，避免误入超时分支（handleTimeoutCase
			//     / 留证 / 超时语义）。
			if apierrors.IsNotFound(errors.Cause(err)) {
				if !seen {
					return false, nil
				}
				return false, err
			}
			// 其他错误（网络抖动等）容忍，继续轮询
			return false, nil
		}
		seen = true
		*complete, *successNodes, *failedNodes = CheckCommandStatus(command)
		if *complete {
			return true, nil
		}
		return false, nil
	}, ctxTimeout.Done())

	return err
}

// handleTimeoutCase 处理命令执行超时的情况
func (b *BaseCommand) handleTimeoutCase(originalComplete bool, originalSuccessNodes, originalFailedNodes []string) TimeoutCaseResult {
	// 超时不返回错误
	var err error
	complete := originalComplete
	successNodes := append([]string(nil), originalSuccessNodes...)
	failedNodes := append([]string(nil), originalFailedNodes...)

	command, cmdErr := b.GetCommand()
	if cmdErr != nil {
		return TimeoutCaseResult{
			Err:          cmdErr,
			Complete:     complete,
			SuccessNodes: successNodes,
			FailedNodes:  failedNodes,
		}
	}

	successNodeSet := make(map[string]struct{}, len(successNodes))
	for _, node := range successNodes {
		successNodeSet[agentutils.GetNodeIPFromCommandWaitResult(node)] = struct{}{}
	}
	failedNodeSet := make(map[string]struct{}, len(failedNodes))
	for _, node := range failedNodes {
		failedNodeSet[agentutils.GetNodeIPFromCommandWaitResult(node)] = struct{}{}
	}

	// 检查超时后仍未成功的目标节点，并追加到失败列表
	for nodeName := range command.Spec.NodeSelector.MatchLabels {
		if _, exists := successNodeSet[nodeName]; exists {
			continue
		}
		if _, exists := failedNodeSet[nodeName]; exists {
			continue
		}
		failedNodes = append(failedNodes, nodeName)
		failedNodeSet[nodeName] = struct{}{}
	}

	return TimeoutCaseResult{
		Err:          err,
		Complete:     complete,
		SuccessNodes: successNodes,
		FailedNodes:  failedNodes,
	}
}

// ClusterNameLabelSelectorRequirement return the label selector requirement
// Deprecated
func (b *BaseCommand) ClusterNameLabelSelectorRequirement() metav1.LabelSelectorRequirement {
	return metav1.LabelSelectorRequirement{
		Key:      agentutils.ClusterNameLabelKey,
		Operator: metav1.LabelSelectorOpIn,
		Values:   []string{b.ClusterName},
	}
}

func GenerateDefaultCommandSpec() *agentv1beta1.CommandSpec {
	return &agentv1beta1.CommandSpec{
		NodeName:     "",
		Suspend:      false,
		Commands:     []agentv1beta1.ExecCommand{},
		BackoffLimit: DefaultBackoffLimit,
		// ActiveDeadlineSecond left 0 so buildCommandObject links it to WaitTimeout (A3).
		ActiveDeadlineSecond:    0,
		TTLSecondsAfterFinished: DefaultTTLSecondsAfterFinished,
		NodeSelector:            &metav1.LabelSelector{},
	}
}

// recordWaitTimeoutEvidence writes structured logs and an Owner Event before Purge (A8).
func (b *BaseCommand) recordWaitTimeoutEvidence(failedNodes, successNodes []string) {
	uid := ""
	activeDeadline := 0
	ns := b.NameSpace
	if b.Command != nil {
		uid = string(b.Command.GetUID())
		activeDeadline = b.Command.Spec.ActiveDeadlineSecond
		if ns == "" {
			ns = b.Command.GetNamespace()
		}
	}
	log.Warnf("command=%s uid=%s reason=%s waitTimeout=%s activeDeadline=%d failedNodes=%v successNodes=%v",
		b.commandName, uid, ReasonWaitTimeout, b.WaitTimeout, activeDeadline, failedNodes, successNodes)

	if b.Recorder == nil || b.OwnerObj == nil {
		return
	}
	obj, ok := b.OwnerObj.(runtime.Object)
	if !ok {
		return
	}
	b.Recorder.Eventf(obj, corev1.EventTypeWarning, EventCommandWaitTimeout,
		"command %s/%s uid=%s waitTimeout=%s activeDeadline=%d failedNodes=%v",
		ns, b.commandName, uid, b.WaitTimeout, activeDeadline, failedNodes)
}

// ValidateCommand validate the command
func ValidateCommand(c *agentv1beta1.Command) error {
	if c.Spec.NodeName == "" && c.Spec.NodeSelector.String() == "" {
		return errors.New("not a valid command,at least provide a node name or NodeSelector")
	}
	// todo: validate this command
	return nil
}

// CheckCommandStatus check all the command is success or not at agentv1beta1.Status
// if all command exec completed,return true , successNodes and failedNodes
// else return false.
// Skip 与成功同属可完成态（P6）：节点 Phase=Skip 或 Status=True 计入 successNodes。
func CheckCommandStatus(c *agentv1beta1.Command) (complete bool, successNodes []string, failedNodes []string) {
	if len(c.Status) == 0 || c.Status == nil {
		return
	}

	complete = true
	if c.Spec.Suspend {
		complete = false
		return
	}
	count := 0

	for nodeName, commandStatus := range c.Status {
		count++
		switch {
		case commandStatus.Phase == agentv1beta1.CommandSuspend:
			complete = false
			return
		case commandStatus.Phase == agentv1beta1.CommandRunning:
			complete = false
			return
		case commandStatus.Phase == agentv1beta1.CommandFailed:
			failedNodes = append(failedNodes, nodeName)
		case commandStatus.Phase == agentv1beta1.CommandSkip:
			// Skip 视为成功完成，可走 Wait/TTL/删除
			successNodes = append(successNodes, nodeName)
		case commandStatus.Status != metav1.ConditionTrue:
			failedNodes = append(failedNodes, nodeName)
		case commandStatus.Failed > 0:
			failedNodes = append(failedNodes, nodeName)
		default:
			successNodes = append(successNodes, nodeName)
		}
	}
	if len(successNodes)+len(failedNodes) != count {
		complete = false
	}

	if count == 0 || len(c.Spec.NodeSelector.MatchLabels) != count {
		complete = false
	}

	return
}

// IsOwnerRefCommand returns a bool ,if object is owner of the command
func IsOwnerRefCommand(object metav1.Object, command agentv1beta1.Command) bool {
	for _, ref := range command.GetOwnerReferences() {
		if ref.UID == object.GetUID() {
			return true
		}
	}
	return false
}

func getNodeSelector(nodes bkenode.Nodes) *metav1.LabelSelector {
	nodeSelector := &metav1.LabelSelector{}
	for _, node := range nodes {
		metav1.AddLabelToSelector(nodeSelector, node.IP, node.IP)
	}
	return nodeSelector
}
