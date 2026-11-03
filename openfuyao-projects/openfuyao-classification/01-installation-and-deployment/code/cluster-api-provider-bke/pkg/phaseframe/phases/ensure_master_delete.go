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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/utils/pointer"
	"sigs.k8s.io/cluster-api/util"
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
	EnsureMasterDeleteName              confv1beta1.BKEClusterPhase = "EnsureMasterDelete"
	WaitMasterDeleteTimeoutMinutes                                  = 4 // 等待主节点删除的超时时间（分钟）
	WaitMasterDeletePollIntervalSeconds                             = 2 // 等待主节点删除的轮询间隔（秒）
)

type EnsureMasterDelete struct {
	phaseframe.BasePhase
	machinesAndNodesToDelete     map[string]phaseutil.MachineAndNode
	machinesAndNodesToWaitDelete map[string]phaseutil.MachineAndNode
	mockClient                   kubernetes.Interface
}

func NewEnsureMasterDelete(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureMasterDeleteName)
	return &EnsureMasterDelete{
		BasePhase:                    base,
		machinesAndNodesToWaitDelete: make(map[string]phaseutil.MachineAndNode),
		machinesAndNodesToDelete:     make(map[string]phaseutil.MachineAndNode),
	}
}

func (e *EnsureMasterDelete) Execute() (ctrl.Result, error) {
	if err := e.reconcileMasterDelete(); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, e.waitMasterDelete()
}

func (e *EnsureMasterDelete) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	// First try legacy mode with appointment annotation
	nodes := phaseutil.GetNeedDeleteMasterNodes(e.Ctx, e.Ctx.Client, new)
	if nodes.Length() > 0 {
		e.SetStatus(bkev1beta1.PhaseWaiting)
		return true
	}

	targetNodes, ok := getDeleteTargetNodesIfDeployed(e.Ctx, new)
	if !ok {
		return false
	}

	nodes = phaseutil.GetNeedDeleteMasterNodesWithTargetNodes(e.Ctx, e.Ctx.Client, new, targetNodes)
	if nodes.Length() == 0 {
		return false
	}
	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

// getTargetClusterNodes gets nodes from the target k8s cluster.
func (e *EnsureMasterDelete) getTargetClusterNodes(bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
	return GetTargetClusterNodes(e.Ctx.Context, e.Ctx.Client, bkeCluster)
}

// kubeClient returns the kubernetes client used to clean up legacy pods.
// In production this is obtained from the remote cluster client;
// tests may inject a fake clientset via mockClient.
func (e *EnsureMasterDelete) kubeClient() kubernetes.Interface {
	if e.mockClient != nil {
		return e.mockClient
	}
	return nil
}

func (e *EnsureMasterDelete) reconcileMasterDelete() error {
	phaseContextFields113 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields113.Context,
		phaseContextFields113.Client,
		phaseContextFields113.BKECluster,
		phaseContextFields113.Scheme,
		phaseContextFields113.Log

	// Get target cluster nodes for BKENode deletion detection
	targetNodes, targetErr := e.getTargetClusterNodes(bkeCluster)
	if targetErr != nil {
		log.Debug("scale-in: Failed to get target cluster nodes: %v", targetErr)
		// Continue with nil targetNodes, will fall back to legacy mode
	}

	// First try legacy mode, then BKENode deletion mode
	nodes := phaseutil.GetNeedDeleteMasterNodes(ctx, c, bkeCluster)
	if nodes.Length() == 0 && targetNodes != nil {
		nodes = phaseutil.GetNeedDeleteMasterNodesWithTargetNodes(ctx, c, bkeCluster, targetNodes)
	}

	// 在节点确定点输出待删除节点列表，便于排查缩容目标问题
	log.Info(constant.MasterDeletingReason,
		"Determined %d master node(s) to be deleted: [%s]",
		nodes.Length(), formatNodeInfos(nodes))

	log.Info(constant.MasterDeletingReason, "Start delete master nodes process")
	log.Info(constant.MasterDeletingReason, "Check whether the node has been associated with a Machine to avoid duplicate deletion")

	// 处理节点和机器的映射关系
	params := ProcessNodeMachineMappingParams{
		Ctx:               ctx,
		Client:            c,
		BKECluster:        bkeCluster,
		Nodes:             nodes,
		Log:               log,
		NodeDeletedReason: constant.MasterDeletedReason,
		NodeJoinedReason:  constant.MasterJoinedReason,
	}
	result, err := ProcessNodeMachineMapping(params)
	if err != nil {
		return err
	}

	// 如果没有需要删除的节点，直接返回
	if result.NodesCount == 0 {
		log.Info(constant.MasterDeletedReason, "No master nodes need to be deleted")
		return nil
	}

	e.machinesAndNodesToWaitDelete = result.WaitDeleteMap

	log.Info(constant.MasterDeletingReason, "%d nodes need to deleted, nodes: %v", result.NodesCount, strings.Join(result.NodesInfos, ", "))

	// 暂停并缩容控制平面
	pauseParams := PauseAndScaleDownControlPlaneParams{
		Ctx:        ctx,
		Client:     c,
		BKECluster: bkeCluster,
		DeleteMap:  result.DeleteMap,
		Log:        log,
	}
	return e.pauseAndScaleDownControlPlane(pauseParams)
}

// PauseAndScaleDownControlPlaneParams 包含 pauseAndScaleDownControlPlane 方法的参数
type PauseAndScaleDownControlPlaneParams struct {
	Ctx        context.Context
	Client     client.Client
	BKECluster *bkev1beta1.BKECluster
	DeleteMap  map[string]phaseutil.MachineAndNode
	Log        *bkev1beta1.BKELogger
}

// pauseAndScaleDownControlPlane 暂停并缩容控制平面
func (e *EnsureMasterDelete) pauseAndScaleDownControlPlane(params PauseAndScaleDownControlPlaneParams) error {
	ctx := params.Ctx
	c := params.Client
	_ = params.BKECluster // bkeCluster parameter is not used in this function
	deleteMap := params.DeleteMap
	log := params.Log
	log.Debug("get kubeadm control plane")
	scope, err := phaseutil.GetClusterAPIAssociateObjs(ctx, c, e.Ctx.Cluster)
	if err != nil || scope.KubeadmControlPlane == nil {
		// cluster api object error, no need to continue
		if err != nil {
			return fmt.Errorf("Get cluster-api associate objs failed: %w", err)
		}
		return fmt.Errorf("Get cluster-api associate objs failed")
	}

	// 暂停 KubeadmControlPlane的运行，以便我们能设置注释指定删除某些节点
	log.Debug("pause kubeadm control plane")
	if err = phaseutil.PauseClusterAPIObj(ctx, c, scope.KubeadmControlPlane); err != nil {
		return fmt.Errorf("Pause KubeadmControlPlane failed: %w", err)
	}
	log.Info(constant.MasterDeletingReason, "Pause KubeadmControlPlane success")

	specCopy := scope.KubeadmControlPlane.Spec.DeepCopy()
	currentReplicas := specCopy.Replicas
	// 如果节点删除过程中出现异常，需要将节点数量恢复到删除前的状态
	defer func() {
		if err != nil {
			log.Info(constant.MasterDeleteFailedReason, "Scale up KubeadmControlPlane replicas to %d.", currentReplicas)
			if rollbackErr := phaseutil.UpdateKubeadmControlPlaneReplicas(ctx, c, scope.KubeadmControlPlane, *currentReplicas); rollbackErr != nil {
				log.Warn(constant.MasterDeleteFailedReason, "Rollback KubeadmControlPlane replicas failed. err: %v", rollbackErr)
			}
			if err = phaseutil.ResumeClusterAPIObj(ctx, c, scope.KubeadmControlPlane); err != nil {
				log.Warn(constant.MasterDeleteFailedReason, "Resume KubeadmControlPlane failed. err: %v", err)
			}
		}
	}()

	// 标记需要删除的节点关联的machine
	log.Debug("mark machine for deletion")
	for _, machineAndNode := range deleteMap {
		machine := machineAndNode.Machine
		if err = phaseutil.MarkMachineForDeletion(ctx, c, machine); err != nil {
			log.Warn(constant.MasterDeleteFailedReason, "Can't delete node %s", phaseutil.NodeInfo(machineAndNode.Node))
			log.Warn(constant.MasterDeleteFailedReason, "Mark machine %s for deletion failed. err: %v", utils.ClientObjNS(machine), err)
			delete(deleteMap, machine.Name)
		}
	}

	// 如果没有需要删除的节点，直接返回
	if len(deleteMap) == 0 {
		log.Info(constant.MasterDeleteFailedReason, "Some nodes cannot be completely deleted")
		return nil
	}
	e.machinesAndNodesToDelete = deleteMap

	// 缩容KubeadmControlPlane的副本数，以便删除节点
	exceptReplicas := *currentReplicas - int32(len(deleteMap))
	// 无论如何，副本数不能小于1
	if exceptReplicas < 1 {
		exceptReplicas = 1
	}
	log.Info(constant.MasterDeletingReason, "Scale down KubeadmControlPlane replicas to %d.", exceptReplicas)

	// Update KubeadmControlPlane replicas
	if err = phaseutil.UpdateKubeadmControlPlaneReplicas(ctx, c, scope.KubeadmControlPlane, exceptReplicas); err != nil {
		return fmt.Errorf("Scale down KubeadmControlPlane replicas failed: %w", err)
	}

	// Resume KubeadmControlPlane
	if err = phaseutil.ResumeClusterAPIObj(ctx, c, scope.KubeadmControlPlane); err != nil {
		return fmt.Errorf("Resume KubeadmControlPlane failed: %w", err)
	}

	return nil
}

// prepareMachinesAndNodesToWaitDelete 合并待删除的机器和节点列表
func (e *EnsureMasterDelete) prepareMachinesAndNodesToWaitDelete() map[string]phaseutil.MachineAndNode {
	machinesAndNodesToWaitDelete := e.machinesAndNodesToWaitDelete
	if len(e.machinesAndNodesToDelete) != 0 && e.machinesAndNodesToDelete != nil {
		for k, v := range e.machinesAndNodesToDelete {
			machinesAndNodesToWaitDelete[k] = v
		}
	}
	return machinesAndNodesToWaitDelete
}

// WaitForMachinesDeleteParams 等待机器删除完成函数的参数
type WaitForMachinesDeleteParams struct {
	Ctx                          context.Context
	Client                       client.Client
	MachinesAndNodesToWaitDelete map[string]phaseutil.MachineAndNode
	Log                          *bkev1beta1.BKELogger
}

// waitForMachinesDelete 等待机器删除完成
func (e *EnsureMasterDelete) waitForMachinesDelete(params WaitForMachinesDeleteParams) (map[string]confv1beta1.Node, error) {
	ctx := params.Ctx
	c := params.Client
	machinesAndNodesToWaitDelete := params.MachinesAndNodesToWaitDelete
	log := params.Log
	successDeletedNode := map[string]confv1beta1.Node{}
	ctxTimeout, cancel := context.WithTimeout(ctx, time.Duration(WaitMasterDeleteTimeoutMinutes)*time.Minute)
	defer cancel()
	err := wait.PollImmediateUntil(time.Duration(WaitMasterDeletePollIntervalSeconds)*time.Second, func() (done bool, err error) {
		for machineName, machineWithNode := range machinesAndNodesToWaitDelete {
			if _, ok := successDeletedNode[machineName]; ok {
				continue
			}
			machine := machineWithNode.Machine
			if err = c.Get(ctx, util.ObjectKey(machine), machine); err != nil {
				if apierrors.IsNotFound(err) {
					log.Info(constant.MasterDeleteSucceedReason, "Machine %s has been deleted", utils.ClientObjNS(machine))
					successDeletedNode[machineName] = machineWithNode.Node
					continue
				}
				return false, fmt.Errorf("Get machine %s failed: %w", utils.ClientObjNS(machine), err)
			}
		}
		if len(successDeletedNode) != len(machinesAndNodesToWaitDelete) {
			return false, nil
		}
		return true, nil
	}, ctxTimeout.Done())

	return successDeletedNode, err
}

// waitMasterDelete wait for master node to be deleted.
// todo 这块代码与隔壁waitWorkerDelete几乎一样，单独抽出来作为一个函数调用
func (e *EnsureMasterDelete) waitMasterDelete() error {
	machinesAndNodesToWaitDelete := e.prepareMachinesAndNodesToWaitDelete()
	if len(machinesAndNodesToWaitDelete) == 0 || machinesAndNodesToWaitDelete == nil {
		return nil
	}

	phaseContextFields314 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields314.Context,
		phaseContextFields314.Client,
		phaseContextFields314.BKECluster,
		phaseContextFields314.Scheme,
		phaseContextFields314.Log

	params := WaitForMachinesDeleteParams{
		Ctx:                          ctx,
		Client:                       c,
		MachinesAndNodesToWaitDelete: machinesAndNodesToWaitDelete,
		Log:                          log,
	}
	successDeletedNode, err := e.waitForMachinesDelete(params)
	if err != nil {
		if errors.Is(err, wait.ErrWaitTimeout) {
			return errors.Errorf("Wait master node delete failed")
		}
		return err
	}

	if len(successDeletedNode) > 0 {
		log.Info(constant.MasterDeleteSucceedReason, "Master nodes delete success")
	}

	if len(successDeletedNode) != 0 {
		cleanupParams := CleanupDeletedNodePodsParams{
			Ctx:                ctx,
			Client:             c,
			BKECluster:         bkeCluster,
			SuccessDeletedNode: successDeletedNode,
		}
		return e.cleanupDeletedNodePods(cleanupParams)
	}
	return nil
}

// CleanupDeletedNodePodsParams 清理已删除节点的 Pod 函数的参数
type CleanupDeletedNodePodsParams struct {
	Ctx                context.Context
	Client             client.Client
	BKECluster         *bkev1beta1.BKECluster
	SuccessDeletedNode map[string]confv1beta1.Node
}

// cleanupDeletedNodePods 清理已删除节点的 Pod
func (e *EnsureMasterDelete) cleanupDeletedNodePods(params CleanupDeletedNodePodsParams) error {
	ctx := params.Ctx
	c := params.Client
	bkeCluster := params.BKECluster
	successDeletedNode := params.SuccessDeletedNode
	phaseContextFields362 := e.Ctx.Untie()
	log := phaseContextFields362.Log // 获取日志实例
	log.Info(constant.MasterDeletedReason, "Attempt to clean the legacy daemonset pod of the removed node")
	var clientSet kubernetes.Interface
	if cs := e.kubeClient(); cs != nil {
		clientSet = cs
	} else {
		remoteClient, err := kube.NewRemoteClientByBKECluster(ctx, c, e.Ctx.BKECluster)
		if err != nil {
			log.Warn(constant.MasterDeletedReason, "Get remote client failed. err: %v", err)
			return nil
		}
		cs, _ := remoteClient.KubeClient()
		clientSet = cs
	}
	for _, node := range successDeletedNode {
		// 顺便从bkecluster中删除节点
		e.Ctx.NodeFetcher().DeleteBKENodeForCluster(params.Ctx, bkeCluster, node.IP)
		// 将节点从状态管理器中删除
		statusmanage.DefaultStatusManager().RemoveSingleNodeStatusCache(bkeCluster, node.IP)

		nodeName := node.Hostname
		// list all pods in the node
		pods, err := clientSet.CoreV1().Pods("").List(ctx, metav1.ListOptions{
			FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
		})
		if err != nil {
			log.Warn(constant.MasterDeletedReason, "List pods in node %s failed. err: %v", nodeName, err)
			continue
		}
		for _, pod := range pods.Items {
			// force delete the pod
			err = clientSet.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{
				GracePeriodSeconds: pointer.Int64(0),
			})
			if err != nil {
				log.Warn(constant.MasterDeletedReason, "Delete pod %s failed. err: %v", utils.ClientObjNS(&pod), err)
				continue
			}
		}
	}
	return mergecluster.SyncStatusUntilComplete(c, bkeCluster)
}
