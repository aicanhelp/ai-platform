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

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
)

const (
	EnsureMasterJoinName confv1beta1.BKEClusterPhase = "EnsureMasterJoin"
	// LogOutputInterval 控制日志输出的轮询间隔
	LogOutputInterval = 10
)

type EnsureMasterJoin struct {
	phaseframe.BasePhase
	nodesToJoin bkenode.Nodes
}

func NewEnsureMasterJoin(ctx *phaseframe.PhaseContext) phaseframe.Phase {
	base := phaseframe.NewBasePhase(ctx, EnsureMasterJoinName)
	return &EnsureMasterJoin{BasePhase: base}
}

func (e *EnsureMasterJoin) Execute() (ctrl.Result, error) {
	if err := e.reconcileMasterJoin(); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

func (e *EnsureMasterJoin) NeedExecute(old *bkev1beta1.BKECluster, new *bkev1beta1.BKECluster) bool {
	if !e.BasePhase.DefaultNeedExecute(old, new) {
		return false
	}

	masterInited := false
	// cluster没有初始化的condition，不需要执行
	if err := e.Ctx.RefreshCtxCluster(); err == nil {
		if conditions.IsTrue(e.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
			masterInited = true
		}
	}
	// Use NodeFetcher to get BKENodes from API server
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(e.Ctx, new)
	if err != nil {
		return false
	}
	nodes := phaseutil.GetNeedJoinMasterNodesWithBKENodes(new, bkeNodes)

	// 第一种情况：首次创建集群，此时masterInited为false,nodes=1,返回false
	if !masterInited && len(nodes) == 1 {
		return false
	}
	// 第二种情况：集群已经初始化，此时masterInited为true,nodes=0,返回false
	if masterInited && len(nodes) == 0 {
		return false
	}
	// 第三种情况：master没有初始化，此时masterInited为false,nodes=0,返回false
	if !masterInited && len(nodes) == 0 {
		return false
	}

	e.SetStatus(bkev1beta1.PhaseWaiting)
	return true
}

// MasterJoinParams 包含master join操作的基本参数
type MasterJoinParams struct {
	Ctx        context.Context
	Client     client.Client
	BKECluster *bkev1beta1.BKECluster
	Log        *bkev1beta1.BKELogger
}

// MasterJoinScaleParams 包含master join操作的缩放参数
type MasterJoinScaleParams struct {
	Ctx        context.Context
	Client     client.Client
	BKECluster *bkev1beta1.BKECluster
	Log        *bkev1beta1.BKELogger
	NodesCount int
}

func (e *EnsureMasterJoin) reconcileMasterJoin() error {
	phaseContextFields115 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields115.Context,
		phaseContextFields115.Client,
		phaseContextFields115.BKECluster,
		phaseContextFields115.Scheme,
		phaseContextFields115.Log

	// 检查Agent状态和集群初始化状态
	preconditionParams := MasterJoinParams{
		Ctx:        ctx,
		Client:     c,
		BKECluster: bkeCluster,
		Log:        log,
	}
	if err := e.checkPreconditions(preconditionParams); err != nil {
		return err
	}

	// 获取需要加入的节点并检查是否已关联到Machine
	nodesCount, nodesInfos, err := e.getJoinableNodes(preconditionParams)
	if err != nil {
		return err
	}

	if nodesCount == 0 {
		log.Info(constant.MasterJoinedReason, "All nodes have been joined, skip join node.")
		if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster); err != nil {
			return err
		}
		return nil
	}

	// 处理博云集群的特殊配置
	if clusterutil.IsBocloudCluster(bkeCluster) {
		log.Info(constant.MasterJoiningReason, "bocloud cluster need to distribute kube-proxy kubeconfig")
		if err := phaseutil.DistributeKubeProxyKubeConfig(ctx, c, bkeCluster, e.nodesToJoin, log); err != nil {
			return err
		}
	}

	log.Info(constant.MasterJoiningReason, "%d nodes will join, nodes: %v", nodesCount, strings.Join(nodesInfos, ", "))

	// 调整KubeadmControlPlane副本数并等待节点加入
	scaleParams := MasterJoinScaleParams{
		Ctx:        ctx,
		Client:     c,
		BKECluster: bkeCluster,
		Log:        log,
		NodesCount: nodesCount,
	}
	return e.scaleAndJoinMasterNodes(scaleParams)
}

// checkPreconditions 检查前置条件
func (e *EnsureMasterJoin) checkPreconditions(params MasterJoinParams) error {
	if err := e.ensureMasterAgentsReady(params); err != nil {
		return err
	}

	if conditions.IsFalse(e.Ctx.Cluster, clusterv1.ControlPlaneInitializedCondition) {
		params.Log.Info(constant.MasterJoinFailedReason, "master is not initialized, skip join master nodes process")
	}
	return nil
}

// ensureMasterAgentsReady verifies that every master node's bkeagent is ready.
// Worker agent failures (e.g. NeedSkip) must not block master join.
func (e *EnsureMasterJoin) ensureMasterAgentsReady(params MasterJoinParams) error {
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(params.Ctx, params.BKECluster)
	if err != nil {
		return fmt.Errorf("failed to get BKENodes for master agent check: %w", err)
	}
	return masterAgentsReadyError(bkeNodes, bkeNodes.ToNodes().Master())
}

func masterAgentsReadyError(bkeNodes bkev1beta1.BKENodes, masters bkenode.Nodes) error {
	if masters.Length() == 0 {
		return fmt.Errorf("no master node")
	}

	notReady := make([]string, 0, masters.Length())
	for _, node := range masters {
		if bkeNodes.GetNodeStateFlag(node.IP, bkev1beta1.NodeAgentReadyFlag) {
			continue
		}
		notReady = append(notReady, node.IP)
	}
	if len(notReady) > 0 {
		return fmt.Errorf("master agent is not ready: %s", strings.Join(notReady, ", "))
	}
	return nil
}

// getJoinableNodes 获取可加入的节点
func (e *EnsureMasterJoin) getJoinableNodes(params MasterJoinParams) (int, []string, error) {
	// Use NodeFetcher to get BKENodes from API server
	bkeNodes, err := e.Ctx.NodeFetcher().GetBKENodesWrapperForCluster(params.Ctx, params.BKECluster)
	if err != nil {
		params.Log.Warn(constant.MasterJoiningReason, "failed to get BKENodes: %v", err)
		return 0, nil, nil
	}
	nodes := phaseutil.GetNeedJoinMasterNodesWithBKENodes(params.BKECluster, bkeNodes)
	e.nodesToJoin = make(bkenode.Nodes, 0)

	params.Log.Info(constant.MasterJoiningReason, "Start join master nodes process")
	params.Log.Info(constant.MasterJoiningReason, "Check whether the node has been associated with a Machine to avoid duplicate creation")

	var nodesInfos []string
	nodesCount := 0
	for _, node := range nodes {
		// 正常应该是找不到关联的machine的
		if _, err := phaseutil.NodeToMachine(params.Ctx, params.Client, params.BKECluster, node); err == nil {
			params.Log.Info(constant.MasterJoinedReason, "Node already exists, skip join node. node: %v", node.Hostname)
			e.Ctx.NodeFetcher().MarkNodeStateFlagForCluster(params.Ctx, params.BKECluster, node.IP, bkev1beta1.NodeBootFlag)
		} else {
			nodesInfos = append(nodesInfos, phaseutil.NodeInfo(node))
			nodesCount++
			e.nodesToJoin = append(e.nodesToJoin, node)
		}
	}
	return nodesCount, nodesInfos, nil
}

// scaleAndJoinMasterNodes 调整KubeadmControlPlane副本数并等待节点加入
func (e *EnsureMasterJoin) scaleAndJoinMasterNodes(params MasterJoinScaleParams) error {
	scope, err := phaseutil.GetClusterAPIAssociateObjs(params.Ctx, params.Client, e.Ctx.Cluster)
	if err != nil || scope.KubeadmControlPlane == nil {
		// cluster api object error, no need to continue
		return fmt.Errorf("Get cluster-api associate objs failed: %w", err)
	}

	specCopy := scope.KubeadmControlPlane.Spec.DeepCopy()
	currentReplicas := specCopy.Replicas
	// 如果节点加入过程中出现异常，需要将节点数量恢复到加入前的状态
	defer func() {
		if err != nil {
			params.Log.Info(constant.MasterJoinFailedReason, "Scale down KubeadmControlPlane replicas to %d.", currentReplicas)
			if rollbackErr := phaseutil.UpdateKubeadmControlPlaneReplicas(params.Ctx, params.Client, scope.KubeadmControlPlane, *currentReplicas); rollbackErr != nil {
				params.Log.Warn(constant.MasterJoinFailedReason, "Back up KubeadmControlPlane replicas failed. err: %v", rollbackErr)
			}
			if err = phaseutil.ResumeClusterAPIObj(params.Ctx, params.Client, scope.KubeadmControlPlane); err != nil {
				params.Log.Warn(constant.MasterJoinFailedReason, "Resume KubeadmControlPlane failed. err: %v", err)
			}
		}
	}()

	bkeNodes, _ := e.Ctx.NodeFetcher().GetNodesForBKECluster(params.Ctx, params.BKECluster)
	masterNodes := bkeNodes.Master()

	exceptReplicas := *currentReplicas + int32(params.NodesCount)
	// 无论如何不能超过bkecluster的master数量
	if exceptReplicas > int32(masterNodes.Length()) {
		exceptReplicas = int32(masterNodes.Length())
	}

	params.Log.Info(constant.MasterJoiningReason, "Scale up KubeadmControlPlane replicas %d to %d", *currentReplicas, exceptReplicas)

	if err = phaseutil.UpdateKubeadmControlPlaneReplicas(params.Ctx, params.Client, scope.KubeadmControlPlane, exceptReplicas); err != nil {
		return fmt.Errorf("Scale up KubeadmControlPlane replicas failed: %w", err)
	}

	if err = phaseutil.ResumeClusterAPIObj(params.Ctx, params.Client, scope.KubeadmControlPlane); err != nil {
		return fmt.Errorf("Resume KubeadmControlPlane failed: %w", err)
	}

	if err = e.waitMasterJoin(params.NodesCount); err != nil {
		return fmt.Errorf("Wait worker join failed: %w", err)
	}

	return nil
}

func (e *EnsureMasterJoin) waitMasterJoin(nodesCount int) error {
	phaseContextFields283 := e.Ctx.Untie()
	ctx, c, bkeCluster, _, log := phaseContextFields283.Context,
		phaseContextFields283.Client,
		phaseContextFields283.BKECluster,
		phaseContextFields283.Scheme,
		phaseContextFields283.Log
	successJoinNode := map[int]confv1beta1.Node{}

	timeOut, err := phaseutil.GetBootTimeOut(e.Ctx.BKECluster)
	if err != nil {
		log.Warn(constant.MasterNotInitReason, "Get boot timeout failed. err: %v", err)
	}

	waitTime := time.Duration(nodesCount) * timeOut
	ctxTimeout, cancel := context.WithTimeout(ctx, waitTime)
	defer cancel()

	err = waitForNodesJoin(WaitForNodesJoinParams{
		Ctx:             ctx,
		Client:          c,
		BKECluster:      bkeCluster,
		NodesToJoin:     e.nodesToJoin,
		Log:             log,
		Timeout:         ctxTimeout,
		SuccessJoinNode: successJoinNode,
	})

	if errors.Is(err, wait.ErrWaitTimeout) {
		return errors.Errorf("Wait master join failed")
	}
	if err != nil {
		return err
	}
	log.Info(constant.MasterJoinSucceedReason, "Master nodes join success")
	return e.Ctx.RefreshCtxBKECluster()
}

// WaitForNodesJoinParams 包含等待节点加入函数的参数
type WaitForNodesJoinParams struct {
	Ctx             context.Context
	Client          client.Client
	BKECluster      *bkev1beta1.BKECluster
	NodesToJoin     bkenode.Nodes
	Log             *bkev1beta1.BKELogger
	Timeout         context.Context
	SuccessJoinNode map[int]confv1beta1.Node
}

// waitForNodesJoin 等待节点加入的公共函数
func waitForNodesJoin(params WaitForNodesJoinParams) error {
	pollCount := 0
	err := wait.PollImmediateUntil(1*time.Second, func() (done bool, err error) {
		pollCount++
		for i, node := range params.NodesToJoin {
			if _, ok := params.SuccessJoinNode[i]; ok {
				continue
			}
			machine, err := phaseutil.NodeToMachine(params.Ctx, params.Client, params.BKECluster, node)
			if err != nil {
				continue
			}
			if machine.Status.NodeRef != nil {
				params.Log.Info(constant.MasterJoinSucceedReason, "Master node join success. node: %v", phaseutil.NodeInfo(node))
				params.SuccessJoinNode[i] = node
			}
		}
		if len(params.SuccessJoinNode) != len(params.NodesToJoin) {
			if pollCount%LogOutputInterval == 0 {
				params.Log.Info(constant.MasterJoiningReason, "Wait master join. success: %d, total: %d", len(params.SuccessJoinNode), len(params.NodesToJoin))
			}
			return false, nil
		}

		params.Log.Info(constant.MasterJoiningReason, "Wait master join. success: %d, total: %d", len(params.SuccessJoinNode), len(params.NodesToJoin))
		return true, nil
	}, params.Timeout.Done())

	return err
}
