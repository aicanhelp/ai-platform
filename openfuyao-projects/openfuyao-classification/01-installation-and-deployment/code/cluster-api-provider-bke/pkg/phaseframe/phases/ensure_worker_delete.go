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
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	kubedrain "k8s.io/kubectl/pkg/drain"
	"k8s.io/utils/pointer"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/cluster-api/util/conditions"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureWorkerDeleteName          confv1beta1.BKEClusterPhase = "EnsureWorkerDelete"
	WorkerDeleteRequeueAfterSeconds                             = 10 // 工作节点删除重排队时间（秒）
	WorkerDeleteWaitTimeoutMinutes                              = 4  // 工作节点删除等待超时时间（分钟）
	WorkerDeletePollIntervalSeconds                             = 2  // 工作节点删除轮询间隔时间（秒）
)

type EnsureWorkerDelete struct {
	phaseframe.BasePhase
	machinesAndNodesToDelete     map[string]phaseutil.MachineAndNode
	machinesAndNodesToWaitDelete map[string]phaseutil.MachineAndNode
	mockClient                   kubernetes.Interface
}

func NewEnsureWorkerDelete(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureWorkerDeleteName)
	return &EnsureWorkerDelete{
		BasePhase:                    base,
		machinesAndNodesToWaitDelete: make(map[string]phaseutil.MachineAndNode),
		machinesAndNodesToDelete:     make(map[string]phaseutil.MachineAndNode),
	}
}

func (e *EnsureWorkerDelete) Execute() (ctrl.Result, error) {
	return e.reconcileWorkerDelete()
}

func (e *EnsureWorkerDelete) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	// First try legacy mode with appointment annotation
	nodes := phaseutil.GetNeedDeleteWorkerNodes(e.Ctx, e.Ctx.Client, new)
	if nodes.Length() > 0 {
		e.SetStatus(bkev1beta1.PhaseWaiting)
		return true
	}

	targetNodes, ok := getDeleteTargetNodesIfDeployed(e.Ctx, new)
	if !ok {
		return false
	}

	nodes = phaseutil.GetNeedDeleteWorkerNodesWithTargetNodes(e.Ctx, e.Ctx.Client, new, targetNodes)
	if nodes.Length() == 0 {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

// getTargetClusterNodes gets nodes from the target k8s cluster.
func (e *EnsureWorkerDelete) getTargetClusterNodes(bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
	return GetTargetClusterNodes(e.Ctx.Context, e.Ctx.Client, bkeCluster)
}

// kubeClient returns the kubernetes client used to clean up legacy pods.
// In production this is obtained from the remote cluster client;
// tests may inject a fake clientset via mockClient.
func (e *EnsureWorkerDelete) kubeClient() kubernetes.Interface {
	if e.mockClient != nil {
		return e.mockClient
	}
	return nil
}

// DrainNodesParams 包含 drainNodes 函数的参数
type DrainNodesParams struct {
	Ctx                    context.Context
	Client                 client.Client
	BKECluster             *bkev1beta1.BKECluster
	MachineToNodeDeleteMap map[string]phaseutil.MachineAndNode
	ClientSet              kubernetes.Interface
	DynamicClient          dynamic.Interface
	Log                    *bkev1beta1.BKELogger
}

// DrainNodesResult 包含 drainNodes 函数的返回结果
type DrainNodesResult struct {
	UpdatedMachineToNodeDeleteMap map[string]phaseutil.MachineAndNode
	CanNotDeleteMachinesAndNodes  map[string]phaseutil.MachineAndNode
}

// drainNodes 对节点进行驱逐
func (e *EnsureWorkerDelete) drainNodes(params DrainNodesParams) DrainNodesResult {
	clientSet, dynamicClient, _ := params.ClientSet, params.DynamicClient, params.Log
	drainer := phaseutil.NewDrainer(params.Ctx, clientSet, dynamicClient, true, params.Log)

	canNotDeleteMachinesAndNodes := map[string]phaseutil.MachineAndNode{}
	machineToNodeDeleteMap := params.MachineToNodeDeleteMap
	client, _, _ := kube.GetTargetClusterClient(params.Ctx, params.Client, params.BKECluster)
	nodeFetcher := e.Ctx.NodeFetcher()
	// 缩容前对节点进行驱逐(dry-run)，代替cluster-api的驱逐功能为了获取日志，将驱逐成功的node保留到machineToNodeDeleteMap
	for machineName, machineAndNode := range machineToNodeDeleteMap {
		remoteNode, err := phaseutil.GetRemoteNodeByBKENode(params.Ctx, client, machineAndNode.Node)
		if err != nil {
			if apierrors.IsNotFound(err) {
				// 找不到也需要删除
				params.Log.Warn(constant.WorkerDeleteFailedReason, "Node %s not found in remote cluster, delete directly", phaseutil.NodeInfo(machineAndNode.Node))
				continue
			}
			//todo 这种情况需要考虑下，先让后面能删除的先删除
			nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, machineAndNode.Node.IP, bkev1beta1.NodeDeleteFailed, "Failed to get node from remote cluster")
			canNotDeleteMachinesAndNodes[machineName] = machineAndNode
			delete(machineToNodeDeleteMap, machineName)
			params.Log.Warn(constant.WorkerDeleteFailedReason, "unable to get remote node %q, err: %v", phaseutil.NodeInfo(machineAndNode.Node), err)
		}

		// drain
		podsList, errs := drainer.GetPodsForDeletion(remoteNode.Name)
		if errs != nil {
			nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, machineAndNode.Node.IP, bkev1beta1.NodeDeleteFailed, "Failed to get need evict pods")
			params.Log.Warn(constant.WorkerDeleteFailedReason, "unable to drain remote node %q, err: %v", phaseutil.NodeInfo(machineAndNode.Node), errs)
			canNotDeleteMachinesAndNodes[machineName] = machineAndNode
			delete(machineToNodeDeleteMap, machineName)
		}

		var podsWaitForEvict []string
		for _, pod := range podsList.Pods() {
			podsWaitForEvict = append(podsWaitForEvict, utils.ClientObjNS(&pod))
		}
		params.Log.Info(constant.WorkerDeletingReason, "All of %d pods need to be drained from node %s, pods: %v", len(podsList.Pods()), remoteNode.Name, strings.Join(podsWaitForEvict, ", "))

		if err := kubedrain.RunNodeDrain(drainer, remoteNode.Name); err != nil {
			// todo 驱逐失败时则不删除了报错就好了，先让后面能删除的先删除
			nodeFetcher.SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, machineAndNode.Node.IP, bkev1beta1.NodeDeleteFailed, "Failed to drain remote node")
			params.Log.Warn(constant.WorkerDeleteFailedReason, "unable to drain remote node %q, err: %v", phaseutil.NodeInfo(machineAndNode.Node), err)
			canNotDeleteMachinesAndNodes[machineName] = machineAndNode
			delete(machineToNodeDeleteMap, machineName)
		}
	}

	return DrainNodesResult{
		UpdatedMachineToNodeDeleteMap: machineToNodeDeleteMap,
		CanNotDeleteMachinesAndNodes:  canNotDeleteMachinesAndNodes,
	}
}

// MarkMachinesForDeletionParams 包含 markMachinesForDeletion 函数的参数
type MarkMachinesForDeletionParams struct {
	Ctx                          context.Context
	Client                       client.Client
	BKECluster                   *bkev1beta1.BKECluster
	MachineToNodeDeleteMap       map[string]phaseutil.MachineAndNode
	CanNotDeleteMachinesAndNodes map[string]phaseutil.MachineAndNode
	Log                          *bkev1beta1.BKELogger
}

// MarkMachinesForDeletionResult 包含 markMachinesForDeletion 函数的返回结果
type MarkMachinesForDeletionResult struct {
	FinalMachineToNodeDeleteMap       map[string]phaseutil.MachineAndNode
	FinalCanNotDeleteMachinesAndNodes map[string]phaseutil.MachineAndNode
}

// markMachinesForDeletion 标记需要删除的机器
func (e *EnsureWorkerDelete) markMachinesForDeletion(params MarkMachinesForDeletionParams) MarkMachinesForDeletionResult {
	// 标记需要删除的节点关联的machine,将标记成功的machine保留到machineToNodeDeleteMap
	finalMachineToNodeDeleteMap := params.MachineToNodeDeleteMap
	finalCanNotDeleteMachinesAndNodes := params.CanNotDeleteMachinesAndNodes

	for machineName, machineAndNode := range finalMachineToNodeDeleteMap {
		machine := machineAndNode.Machine
		params.Log.Info(constant.WorkerDeletingReason,
			"Mark machine for deletion, %s", formatMachineAndNodeInfo(machineName, machineAndNode))
		if err := phaseutil.MarkMachineForDeletion(params.Ctx, params.Client, machine); err != nil {
			params.Log.Warn(constant.WorkerDeleteFailedReason, "Can't delete node %s", phaseutil.NodeInfo(machineAndNode.Node))
			params.Log.Warn(constant.WorkerDeleteFailedReason, "Mark machine %s for deletion failed. err: %v", utils.ClientObjNS(machine), err)
			finalCanNotDeleteMachinesAndNodes[machineName] = machineAndNode
			delete(finalMachineToNodeDeleteMap, machineName)
			e.Ctx.NodeFetcher().SetNodeStateWithMessageForCluster(params.Ctx, params.BKECluster, machineAndNode.Node.IP, bkev1beta1.NodeDeleteFailed, "Failed to mark acssociated machine for deletion")
		}
	}

	return MarkMachinesForDeletionResult{
		FinalMachineToNodeDeleteMap:       finalMachineToNodeDeleteMap,
		FinalCanNotDeleteMachinesAndNodes: finalCanNotDeleteMachinesAndNodes,
	}
}

// InitialSetupParams 包含 initialSetup 函数的参数
type InitialSetupParams struct {
	Ctx         context.Context
	Client      client.Client
	BKECluster  *bkev1beta1.BKECluster
	Cluster     *clusterv1.Cluster
	Log         *bkev1beta1.BKELogger
	TargetNodes bkenode.Nodes // Nodes from target k8s cluster for BKENode deletion detection
}

// InitialSetupResult 包含 initialSetup 函数的返回结果
type InitialSetupResult struct {
	NodeMappingResult ProcessNodeMachineMappingResult
	Scope             *phaseutil.ClusterAPIObjs
	CurrentReplicas   *int32
	Error             error
}

// initialSetup 执行初始设置和准备工作
func (e *EnsureWorkerDelete) initialSetup(params InitialSetupParams) InitialSetupResult {
	// First try legacy mode, then BKENode deletion mode
	nodes := phaseutil.GetNeedDeleteWorkerNodes(params.Ctx, params.Client, params.BKECluster)
	if nodes.Length() == 0 && params.TargetNodes != nil {
		nodes = phaseutil.GetNeedDeleteWorkerNodesWithTargetNodes(params.Ctx, params.Client, params.BKECluster, params.TargetNodes)
	}

	// 在节点确定点输出待删除节点列表，便于排查缩容目标问题
	params.Log.Info(constant.WorkerDeletingReason,
		"Determined %d worker node(s) to be deleted: [%s]",
		nodes.Length(), formatNodeInfos(nodes))

	params.Log.Info(constant.WorkerDeletingReason, "Start delete worker nodes process")
	params.Log.Info(constant.WorkerDeletingReason, "Check whether the node has been associated with a Machine to avoid duplicate deletion")

	// 处理节点和机器的映射关系
	nodeMappingParams := ProcessNodeMachineMappingParams{
		Ctx:               params.Ctx,
		Client:            params.Client,
		BKECluster:        params.BKECluster,
		Nodes:             nodes,
		Log:               params.Log,
		NodeDeletedReason: constant.WorkerDeletedReason,
		NodeJoinedReason:  constant.WorkerJoinedReason,
	}
	nodeMappingResult, err := ProcessNodeMachineMapping(nodeMappingParams)
	if err != nil {
		return InitialSetupResult{Error: err}
	}

	// for wait
	e.machinesAndNodesToWaitDelete = nodeMappingResult.WaitDeleteMap

	// 如果没有需要删除的节点，直接返回
	if nodeMappingResult.NodesCount == 0 {
		params.Log.Info(constant.WorkerDeleteSucceedReason, "No worker nodes need to be deleted")
		return InitialSetupResult{Error: nil}
	}

	params.Log.Info(constant.WorkerDeletingReason, "%d nodes need to deleted, nodes: %v", nodeMappingResult.NodesCount, strings.Join(nodeMappingResult.NodesInfos, ", "))

	// 获取 MachineDeployment
	scope, err := phaseutil.GetClusterAPIAssociateObjs(params.Ctx, params.Client, params.Cluster)
	if err != nil || scope.MachineDeployment == nil {
		// cluster api object error, no need to continue
		if err != nil {
			return InitialSetupResult{Error: fmt.Errorf("Get cluster-api associate objs failed: %w", err)}
		}
		return InitialSetupResult{Error: fmt.Errorf("Get cluster-api associate objs failed")}
	}

	// 暂停 MachineDeployment的运行，以便我们能设置注释指定删除某些节点
	params.Log.Debug("pause machine deployment")
	if err = phaseutil.PauseClusterAPIObj(params.Ctx, params.Client, scope.MachineDeployment); err != nil {
		return InitialSetupResult{Error: fmt.Errorf("Pause MachineDeployment failed: %w", err)}
	}
	params.Log.Debug("Pause MachineDeployment success")

	specCopy := scope.MachineDeployment.Spec.DeepCopy()
	currentReplicas := specCopy.Replicas

	return InitialSetupResult{
		NodeMappingResult: nodeMappingResult,
		Scope:             scope,
		CurrentReplicas:   currentReplicas,
		Error:             nil,
	}
}

// ProcessDrainAndMarkParams 包含 processDrainAndMark 函数的参数
type ProcessDrainAndMarkParams struct {
	Ctx               context.Context
	Client            client.Client
	BKECluster        *bkev1beta1.BKECluster
	NodeMappingResult ProcessNodeMachineMappingResult
	Scope             *phaseutil.ClusterAPIObjs
	Log               *bkev1beta1.BKELogger
}

// ProcessDrainAndMarkResult 包含 processDrainAndMark 函数的返回结果
type ProcessDrainAndMarkResult struct {
	MarkResult MarkMachinesForDeletionResult
	Scope      *phaseutil.ClusterAPIObjs
}

// processDrainAndMark 处理节点驱逐和标记删除
func (e *EnsureWorkerDelete) processDrainAndMark(params ProcessDrainAndMarkParams) ProcessDrainAndMarkResult {
	// 对节点进行驱逐
	clientSet, dynamicClient, _ := kube.GetTargetClusterClient(params.Ctx, params.Client, params.BKECluster)
	drainParams := DrainNodesParams{
		Ctx:                    params.Ctx,
		Client:                 params.Client,
		BKECluster:             params.BKECluster,
		MachineToNodeDeleteMap: params.NodeMappingResult.DeleteMap,
		ClientSet:              clientSet,
		DynamicClient:          dynamicClient,
		Log:                    params.Log,
	}
	drainResult := e.drainNodes(drainParams)

	// 标记需要删除的机器
	markParams := MarkMachinesForDeletionParams{
		Ctx:                          params.Ctx,
		Client:                       params.Client,
		BKECluster:                   params.BKECluster,
		MachineToNodeDeleteMap:       drainResult.UpdatedMachineToNodeDeleteMap,
		CanNotDeleteMachinesAndNodes: drainResult.CanNotDeleteMachinesAndNodes,
		Log:                          params.Log,
	}
	markResult := e.markMachinesForDeletion(markParams)

	return ProcessDrainAndMarkResult{
		MarkResult: markResult,
		Scope:      params.Scope,
	}
}

// FinalizeDeletionParams 包含 finalizeDeletion 函数的参数
type FinalizeDeletionParams struct {
	Ctx             context.Context
	Client          client.Client
	BKECluster      *bkev1beta1.BKECluster
	MarkResult      MarkMachinesForDeletionResult
	Scope           *phaseutil.ClusterAPIObjs
	CurrentReplicas *int32
	Log             *bkev1beta1.BKELogger
}

// FinalizeDeletionResult 包含 finalizeDeletion 函数的返回结果
type FinalizeDeletionResult struct {
	Result ctrl.Result
	Error  error
}

// finalizeDeletion 完成删除操作的最后步骤
func (e *EnsureWorkerDelete) finalizeDeletion(params FinalizeDeletionParams) FinalizeDeletionResult {
	// 检查是否有无法删除的节点
	req := ctrl.Result{}
	if len(params.MarkResult.FinalCanNotDeleteMachinesAndNodes) > 0 {
		req = ctrl.Result{RequeueAfter: time.Duration(WorkerDeleteRequeueAfterSeconds) * time.Second}
	}

	// 如果没有需要删除的节点，直接返回
	if len(params.MarkResult.FinalMachineToNodeDeleteMap) == 0 {
		params.Log.Info(constant.WorkerDeleteFailedReason, "Some nodes cannot be completely deleted")
		return FinalizeDeletionResult{
			Result: req,
			Error:  errors.Errorf("some nodes cannot be completely deleted"),
		}
	}

	e.machinesAndNodesToDelete = params.MarkResult.FinalMachineToNodeDeleteMap

	// 缩容MachineDeployment的副本数，以便删除节点
	exceptReplicas := *params.CurrentReplicas - int32(len(params.MarkResult.FinalMachineToNodeDeleteMap))
	// 无论如何 md的副本数都不能为负数
	if exceptReplicas < 0 {
		exceptReplicas = 0
	}
	params.Log.Info(constant.WorkerDeletingReason, "Scale down MachineDeployment replicas to %d.", exceptReplicas)

	// Update MachineDeployment replicas
	err := phaseutil.UpdateMachineDeploymentReplicas(params.Ctx, params.Client, params.Scope.MachineDeployment, exceptReplicas)
	if err != nil {
		return FinalizeDeletionResult{
			Result: req,
			Error:  fmt.Errorf("Scale down MachineDeployment replicas failed: %w", err),
		}
	}

	// Resume MachineDeployment
	if err = phaseutil.ResumeClusterAPIObj(params.Ctx, params.Client, params.Scope.MachineDeployment); err != nil {
		return FinalizeDeletionResult{
			Result: req,
			Error:  fmt.Errorf("Resume MachineDeployment failed: %w", err),
		}
	}

	return FinalizeDeletionResult{
		Result: req,
		Error:  nil,
	}
}

func (e *EnsureWorkerDelete) reconcileWorkerDelete() (ctrl.Result, error) {
	phaseContextFields426 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields426.Context,
		phaseContextFields426.Client,
		phaseContextFields426.BKECluster,
		phaseContextFields426.Scheme,
		phaseContextFields426.Log

	// Get target cluster nodes for BKENode deletion detection
	targetNodes, targetErr := e.getTargetClusterNodes(bkeCluster)
	if targetErr != nil {
		log.Debug("scale-in: Failed to get target cluster nodes: %v", targetErr)
		// Continue with nil targetNodes, will fall back to legacy mode
	}

	// 执行初始设置和准备工作
	initialSetupParams := InitialSetupParams{
		Ctx:         ctx,
		Client:      c,
		BKECluster:  bkeCluster,
		Cluster:     e.Ctx.Cluster,
		Log:         log,
		TargetNodes: targetNodes,
	}
	initialResult := e.initialSetup(initialSetupParams)

	// 如果初始设置失败或没有节点需要删除，直接返回
	if initialResult.Error != nil {
		return ctrl.Result{}, initialResult.Error
	}
	// 如果没有需要删除的节点，直接返回
	if initialResult.NodeMappingResult.NodesCount == 0 {
		return ctrl.Result{}, nil
	}
	// 设置回滚逻辑
	scope := initialResult.Scope
	currentReplicas := initialResult.CurrentReplicas
	err := error(nil)
	defer func() {
		if err != nil {
			e.rollbackMachineDeployment(ctx, c, scope.MachineDeployment, *currentReplicas, log)
		}
	}()

	drainMarkParams := ProcessDrainAndMarkParams{
		Ctx:               ctx,
		Client:            c,
		BKECluster:        bkeCluster,
		NodeMappingResult: initialResult.NodeMappingResult,
		Scope:             scope,
		Log:               log,
	}
	drainMarkResult := e.processDrainAndMark(drainMarkParams)
	finalizeParams := FinalizeDeletionParams{
		Ctx:             ctx,
		Client:          c,
		BKECluster:      bkeCluster,
		MarkResult:      drainMarkResult.MarkResult,
		Scope:           scope,
		CurrentReplicas: currentReplicas,
		Log:             log,
	}
	finalizeResult := e.finalizeDeletion(finalizeParams)
	if finalizeResult.Error != nil {
		err = finalizeResult.Error
		return finalizeResult.Result, finalizeResult.Error
	}

	err = e.waitWorkerDelete()
	return finalizeResult.Result, err
}

func (e *EnsureWorkerDelete) rollbackMachineDeployment(ctx context.Context, c client.Client, md *clusterv1.MachineDeployment, replicas int32, log *bkev1beta1.BKELogger) {
	log.Debug(constant.WorkerDeleteFailedReason+
		": Rollback: scale up MachineDeployment replicas to %d.", replicas)
	if rollbackErr := phaseutil.UpdateMachineDeploymentReplicas(ctx, c, md, replicas); rollbackErr != nil {
		log.Warn(constant.WorkerDeleteFailedReason, "Rollback MachineDeployment replicas failed. err: %v", rollbackErr)
	}
	if rollbackErr := phaseutil.ResumeClusterAPIObj(ctx, c, md); rollbackErr != nil {
		log.Warn(constant.WorkerDeleteFailedReason, "Resume MachineDeployment failed. err: %v", rollbackErr)
	}
}

// PrepareMachinesAndNodesToWaitDeleteParams 包含 prepareMachinesAndNodesToWaitDelete 函数的参数
type PrepareMachinesAndNodesToWaitDeleteParams struct {
	MachinesAndNodesToWaitDelete map[string]phaseutil.MachineAndNode
	MachinesAndNodesToDelete     map[string]phaseutil.MachineAndNode
}

// prepareMachinesAndNodesToWaitDelete 合并待删除的机器和节点列表
func prepareMachinesAndNodesToWaitDelete(params PrepareMachinesAndNodesToWaitDeleteParams) map[string]phaseutil.MachineAndNode {
	machinesAndNodesToWaitDelete := params.MachinesAndNodesToWaitDelete
	if len(params.MachinesAndNodesToDelete) != 0 && params.MachinesAndNodesToDelete != nil {
		for k, v := range params.MachinesAndNodesToDelete {
			machinesAndNodesToWaitDelete[k] = v
		}
	}
	return machinesAndNodesToWaitDelete
}

// prepareMachinesAndNodesToWaitDelete 合并待删除的机器和节点列表
func (e *EnsureWorkerDelete) prepareMachinesAndNodesToWaitDelete() map[string]phaseutil.MachineAndNode {
	params := PrepareMachinesAndNodesToWaitDeleteParams{
		MachinesAndNodesToWaitDelete: e.machinesAndNodesToWaitDelete,
		MachinesAndNodesToDelete:     e.machinesAndNodesToDelete,
	}
	return prepareMachinesAndNodesToWaitDelete(params)
}

func (e *EnsureWorkerDelete) waitWorkerDelete() error {
	machinesAndNodesToWaitDelete := e.prepareMachinesAndNodesToWaitDelete()
	if len(machinesAndNodesToWaitDelete) == 0 || machinesAndNodesToWaitDelete == nil {
		return nil
	}

	phaseContextFields535 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields535.Context,
		phaseContextFields535.Client,
		phaseContextFields535.BKECluster,
		phaseContextFields535.Scheme,
		phaseContextFields535.Log

	// 等待机器删除
	waitParams := WaitMachinesDeleteParams{
		Ctx:                          ctx,
		Client:                       c,
		BKECluster:                   bkeCluster,
		MachinesAndNodesToWaitDelete: machinesAndNodesToWaitDelete,
		Log:                          log,
	}
	waitResult := e.waitForMachinesDelete(waitParams)
	if waitResult.Error != nil {
		return waitResult.Error
	}

	// 处理成功删除的节点
	processParams := ProcessSuccessfulDeletionsParams{
		Ctx:                ctx,
		Client:             c,
		BKECluster:         bkeCluster,
		SuccessDeletedNode: waitResult.SuccessDeletedNode,
		Log:                log,
	}
	return e.processSuccessfulDeletions(processParams)
}

// WaitMachinesDeleteParams 包含 waitForMachinesDelete 函数的参数
type WaitMachinesDeleteParams struct {
	Ctx                          context.Context
	Client                       client.Client
	BKECluster                   *bkev1beta1.BKECluster
	MachinesAndNodesToWaitDelete map[string]phaseutil.MachineAndNode
	Log                          *bkev1beta1.BKELogger
}

// WaitMachinesDeleteResult 包含 waitForMachinesDelete 函数的返回结果
type WaitMachinesDeleteResult struct {
	SuccessDeletedNode map[string]confv1beta1.Node
	Error              error
}

// waitForMachinesDelete 等待机器删除
func (e *EnsureWorkerDelete) waitForMachinesDelete(params WaitMachinesDeleteParams) WaitMachinesDeleteResult {
	successDeletedNode := map[string]confv1beta1.Node{}
	ctxTimeout, cancel := context.WithTimeout(params.Ctx, time.Duration(WorkerDeleteWaitTimeoutMinutes)*time.Minute)
	defer cancel()
	err := wait.PollImmediateUntil(time.Duration(WorkerDeletePollIntervalSeconds)*time.Second, func() (done bool, err error) {
		for machineName, machineWithNode := range params.MachinesAndNodesToWaitDelete {
			if _, ok := successDeletedNode[machineName]; ok {
				continue
			}
			machine := machineWithNode.Machine
			if err = params.Client.Get(params.Ctx, util.ObjectKey(machine), machine); err != nil {
				if apierrors.IsNotFound(err) {
					params.Log.Info(constant.WorkerDeleteSucceedReason, "Machine %s has been deleted", utils.ClientObjNS(machine))
					successDeletedNode[machineName] = machineWithNode.Node
					continue
				}
				return false, fmt.Errorf("Get machine %s failed: %w", utils.ClientObjNS(machine), err)
			}

			drainCondition := conditions.Get(machine, clusterv1.DrainingSucceededCondition)
			if drainCondition != nil && drainCondition.Status == corev1.ConditionFalse {
				if drainCondition.Reason == clusterv1.DrainingFailedReason {
					params.Log.Warn(drainCondition.Reason, "node: %s %s", phaseutil.NodeInfo(machineWithNode.Node), drainCondition.Message)
				} else {
					params.Log.Info(drainCondition.Reason, "node: %s %s", phaseutil.NodeInfo(machineWithNode.Node), drainCondition.Message)
				}
			}

			volumeDetachCondition := conditions.Get(machine, clusterv1.VolumeDetachSucceededCondition)
			if volumeDetachCondition != nil && volumeDetachCondition.Status == corev1.ConditionFalse {
				params.Log.Info(volumeDetachCondition.Reason, "node: %s %s", phaseutil.NodeInfo(machineWithNode.Node), volumeDetachCondition.Message)
			}

			nodeHealthyCondition := conditions.Get(machine, clusterv1.MachineNodeHealthyCondition)
			if nodeHealthyCondition != nil && nodeHealthyCondition.Status == corev1.ConditionFalse {
				if nodeHealthyCondition.Reason == clusterv1.DeletionFailedReason {
					params.Log.Warn(nodeHealthyCondition.Reason, "node: %s %s", phaseutil.NodeInfo(machineWithNode.Node), nodeHealthyCondition.Message)
				} else {
					params.Log.Info(nodeHealthyCondition.Reason, "node: %s %s", phaseutil.NodeInfo(machineWithNode.Node), nodeHealthyCondition.Reason)
				}
			}

		}
		if len(successDeletedNode) != len(params.MachinesAndNodesToWaitDelete) {
			return false, nil
		}
		return true, nil
	}, ctxTimeout.Done())

	if errors.Is(err, wait.ErrWaitTimeout) {
		return WaitMachinesDeleteResult{
			Error: errors.Errorf("Wait worker node delete failed"),
		}
	}
	if err != nil {
		return WaitMachinesDeleteResult{
			Error: err,
		}
	}
	params.Log.Info(constant.WorkerDeleteSucceedReason, "Worker nodes delete success")

	return WaitMachinesDeleteResult{
		SuccessDeletedNode: successDeletedNode,
		Error:              nil,
	}
}

// ProcessSuccessfulDeletionsParams 包含 processSuccessfulDeletions 函数的参数
type ProcessSuccessfulDeletionsParams struct {
	Ctx                context.Context
	Client             client.Client
	BKECluster         *bkev1beta1.BKECluster
	SuccessDeletedNode map[string]confv1beta1.Node
	Log                *bkev1beta1.BKELogger
}

// processSuccessfulDeletions 处理成功删除的节点
func (e *EnsureWorkerDelete) processSuccessfulDeletions(params ProcessSuccessfulDeletionsParams) error {
	if len(params.SuccessDeletedNode) != 0 {
		params.Log.Info(constant.WorkerDeletedReason, "Attempt to clean the legacy daemonset pod of the removed node")
		var clientSet kubernetes.Interface
		if cs := e.kubeClient(); cs != nil {
			clientSet = cs
		} else {
			remoteClient, err := kube.NewRemoteClientByBKECluster(params.Ctx, params.Client, e.Ctx.BKECluster)
			if err != nil {
				params.Log.Warn(constant.WorkerDeletedReason, "Get remote client failed. err: %v", err)
				return nil
			}
			ks, _ := remoteClient.KubeClient()
			clientSet = ks
		}
		for _, node := range params.SuccessDeletedNode {
			// 清理节点上的 Pod
			if err := e.cleanupNodePods(params.Ctx, clientSet, params.BKECluster, node, params.Log); err != nil {
				params.Log.Warn(constant.WorkerDeletedReason, "Cleanup pods on node %s failed. err: %v", node.Hostname, err)
				continue
			}
		}
		return mergecluster.SyncStatusUntilComplete(params.Client, params.BKECluster)
	}
	return nil
}

// CleanupNodePodsParams 包含 cleanupNodePods 函数的参数
type CleanupNodePodsParams struct {
	Ctx        context.Context
	ClientSet  kubernetes.Interface
	BKECluster *bkev1beta1.BKECluster
	Node       confv1beta1.Node
	Log        *bkev1beta1.BKELogger
}

// cleanupNodePods 清理节点上的 Pod
func (e *EnsureWorkerDelete) cleanupNodePods(ctx context.Context, clientSet kubernetes.Interface, bkeCluster *bkev1beta1.BKECluster, node confv1beta1.Node, log *bkev1beta1.BKELogger) error {
	// 顺便从bkecluster中删除节点
	e.Ctx.NodeFetcher().DeleteBKENodeForCluster(e.Ctx, bkeCluster, node.IP)
	// 将节点从状态管理器中删除
	statusmanage.DefaultStatusManager().RemoveSingleNodeStatusCache(bkeCluster, node.IP)

	nodeName := node.Hostname
	// list all pods in the node
	pods, err := clientSet.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
	})
	if err != nil {
		return fmt.Errorf("list pods in node %s failed: %w", nodeName, err)
	}
	for _, pod := range pods.Items {
		// force delete the pod
		err = clientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{
			GracePeriodSeconds: pointer.Int64(0),
		})
		if err != nil {
			return fmt.Errorf("delete pod %s failed: %w", utils.ClientObjNS(&pod), err)
		}
	}
	return nil
}
