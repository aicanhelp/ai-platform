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
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/record"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/constant"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/intervention"
	labelhelper "gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/label"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// ProcessNodeMachineMappingResult 包含节点机器映射处理的结果
type ProcessNodeMachineMappingResult struct {
	DeleteMap     map[string]phaseutil.MachineAndNode
	WaitDeleteMap map[string]phaseutil.MachineAndNode
	NodesInfos    []string
	NodesCount    int
}

// ProcessNodeMachineMappingParams 包含 processNodeMachineMapping 方法的参数
type ProcessNodeMachineMappingParams struct {
	Ctx               context.Context
	Client            client.Client
	BKECluster        *bkev1beta1.BKECluster
	NodeFetcher       *nodeutil.NodeFetcher
	Nodes             bkenode.Nodes
	Log               *bkev1beta1.BKELogger
	NodeDeletedReason string
	NodeJoinedReason  string
}

// formatNodeInfos 返回节点列表的可读描述（hostname/IP，逗号分隔）
func formatNodeInfos(nodes bkenode.Nodes) string {
	infos := make([]string, 0, len(nodes))
	for _, node := range nodes {
		infos = append(infos, phaseutil.NodeInfo(node))
	}
	return strings.Join(infos, ", ")
}

// formatMachineAndNodeInfo 返回 Machine 及其对应节点的可读描述，用于缩容链路日志
func formatMachineAndNodeInfo(machineName string, machineAndNode phaseutil.MachineAndNode) string {
	providerID := ""
	phase := "Unknown"
	if machineAndNode.Machine != nil {
		if machineAndNode.Machine.Spec.ProviderID != nil {
			providerID = *machineAndNode.Machine.Spec.ProviderID
		}
		phase = machineAndNode.Machine.Status.Phase
	}
	return fmt.Sprintf("machine=%s, node=%s, providerID=%s, phase=%s",
		machineName, phaseutil.NodeInfo(machineAndNode.Node), providerID, phase)
}

// ProcessNodeMachineMapping 处理节点和机器的映射关系
func ProcessNodeMachineMapping(params ProcessNodeMachineMappingParams) (ProcessNodeMachineMappingResult, error) {
	ctx := params.Ctx
	c := params.Client
	bkeCluster := params.BKECluster
	nodes := params.Nodes
	var nodesInfos []string
	nodesCount := 0
	// machineName -> machineAndNode
	machineToNodeDeleteMap := make(map[string]phaseutil.MachineAndNode)
	machineToNodeWaitDeleteMap := make(map[string]phaseutil.MachineAndNode)

	// 遍历所有需要删除的节点，如果节点已经被删除了，需要从status中删除
	params.Log.Debug("get associated machine for nodes, to avoid duplicate deletion")
	for _, node := range nodes {
		// 正常应该是能够找到machine的，如果找不到，说明节点已经被删除了，需要从status中删除
		if machine, err := phaseutil.NodeToMachine(ctx, c, bkeCluster, node); err != nil {
			cleanupParams := CleanupOrphanedNodeParams{
				Ctx:               ctx,
				Client:            c,
				BKECluster:        bkeCluster,
				Node:              node,
				NodeFetcher:       params.NodeFetcher,
				Log:               params.Log,
				NodeDeletedReason: params.NodeDeletedReason,
			}
			if err := cleanupOrphanedNode(cleanupParams); err != nil {
				return ProcessNodeMachineMappingResult{}, err
			}
		} else {
			if machine.Status.Phase == string(clusterv1.MachinePhaseDeleting) {
				params.Log.Info(params.NodeDeletedReason, "node %s is in deleting phase (machine %s), skip delete it",
					phaseutil.NodeInfo(node), machine.Name)
				machineToNodeWaitDeleteMap[machine.Name] = phaseutil.MachineAndNode{Machine: machine, Node: node}
				continue
			}
			if machine.Status.Phase == string(clusterv1.MachinePhaseDeleted) {
				params.Log.Info(params.NodeDeletedReason, "node %s is in deleted phase (machine %s), skip delete it",
					phaseutil.NodeInfo(node), machine.Name)
				continue
			}
			nodesInfos = append(nodesInfos, phaseutil.NodeInfo(node))
			nodesCount++
			machineToNodeDeleteMap[machine.Name] = phaseutil.MachineAndNode{
				Machine: machine,
				Node:    node,
			}
		}
	}

	// 输出节点与 Machine 的映射关系，便于排查缩容删除目标的问题
	for machineName, machineAndNode := range machineToNodeDeleteMap {
		params.Log.Info(params.NodeDeletedReason,
			"Node mapped to machine, will be deleted: %s", formatMachineAndNodeInfo(machineName, machineAndNode))
	}
	for machineName, machineAndNode := range machineToNodeWaitDeleteMap {
		params.Log.Info(params.NodeDeletedReason,
			"Node mapped to machine, machine is already deleting: %s", formatMachineAndNodeInfo(machineName, machineAndNode))
	}

	return ProcessNodeMachineMappingResult{
		DeleteMap:     machineToNodeDeleteMap,
		WaitDeleteMap: machineToNodeWaitDeleteMap,
		NodesInfos:    nodesInfos,
		NodesCount:    nodesCount,
	}, nil
}

// CleanupOrphanedNodeParams 清理孤儿节点的参数
type CleanupOrphanedNodeParams struct {
	Ctx               context.Context
	Client            client.Client
	BKECluster        *bkev1beta1.BKECluster
	Node              confv1beta1.Node
	NodeFetcher       *nodeutil.NodeFetcher
	Log               *bkev1beta1.BKELogger
	NodeDeletedReason string
}

// cleanupOrphanedNode 清理无关联 Machine 的孤儿节点
func cleanupOrphanedNode(params CleanupOrphanedNodeParams) error {
	ctx := params.Ctx
	c := params.Client
	bkeCluster := params.BKECluster
	node := params.Node
	nodeFetcher := params.NodeFetcher
	log := params.Log
	nodeDeletedReason := params.NodeDeletedReason

	// 二次确认：删除决策必须以目标集群自身的 BKENode 一致性为准，而不是发起方本地的
	// （可能过期的）状态。若目标集群仍存在纳管该节点的活跃 BKENode（非 Deleting），
	// 说明"本地查无 Machine"是陈旧或第三方视角的误判（典型场景：自纳管后引导集群
	// 残留的控制器以过期状态调谐管理集群，把新加入的节点误判为孤儿），此时禁止删除
	// 目标集群的 Node，避免误删刚加入的节点。
	if skip, reason := shouldSkipOrphanNodeCleanup(ctx, c, bkeCluster, node.IP); skip {
		log.Warn(nodeDeletedReason,
			"Node %s: skip orphan cleanup: %s",
			phaseutil.NodeInfo(node), reason)
		return nil
	}

	log.Warn(nodeDeletedReason,
		"Node %s has not been associated with a Machine, will delete it from target cluster",
		phaseutil.NodeInfo(node))
	// Delete the Node from target cluster if it still exists
	if node.Hostname != "" {
		if err := DeleteTargetClusterNode(ctx, c, bkeCluster, node.Hostname); err != nil {
			log.Warn(nodeDeletedReason,
				"Failed to delete Node %s from target cluster: %v", node.Hostname, err)
		}
	}
	// 设置 Deleting 状态后再删除 BKENode，避免 webhook 拦截 master 节点
	if nodeFetcher != nil {
		if err := nodeFetcher.SetNodeStateWithMessageForCluster(ctx, bkeCluster, node.IP,
			confv1beta1.NodeDeleting, "Orphaned node cleanup"); err != nil {
			return fmt.Errorf("set BKENode %s to Deleting state failed: %w", node.IP, err)
		}
		if err := nodeFetcher.DeleteBKENodeForCluster(ctx, bkeCluster, node.IP); err != nil {
			log.Warn(nodeDeletedReason, "Delete BKENode for %s failed, err: %v", node.IP, err)
		}
	}
	// 将节点从状态管理器中删除
	statusmanage.DefaultStatusManager().RemoveSingleNodeStatusCache(bkeCluster, node.IP)
	// remove node from AppointmentDeletedNodesAnnotationKey
	patchFunc := func(cluster *bkev1beta1.BKECluster) {
		phaseutil.RemoveAppointmentDeletedNodes(cluster, node.IP)
	}
	if err := mergecluster.SyncStatusUntilComplete(c, bkeCluster, patchFunc); err != nil {
		return fmt.Errorf("Sync status failed: %w", err)
	}
	return nil
}

// ProcessCommandFailureParams 包含处理命令失败的参数
type ProcessCommandFailureParams struct {
	Context        context.Context
	Client         client.Client
	BKECluster     *bkev1beta1.BKECluster
	NodeFetcher    *nodeutil.NodeFetcher
	InitCommand    *agentv1beta1.Command
	InitNodeIp     *string
	FailedNodes    []string
	RefreshContext func() error
	Recorder       record.EventRecorder
}

// ProcessCommandFailureResult 包含处理命令失败的结果
type ProcessCommandFailureResult struct {
	Done    bool
	Success bool
	Err     error
}

// ProcessCommandFailure 处理命令失败情况
func ProcessCommandFailure(params ProcessCommandFailureParams) ProcessCommandFailureResult {
	// 等两秒
	time.Sleep(time.Duration(MasterInitSleepSeconds) * time.Second)
	if err := params.RefreshContext(); err != nil {
		return ProcessCommandFailureResult{Done: false, Success: false, Err: fmt.Errorf("Refresh BKECluster obj %q failed: %w", utils.ClientObjNS(params.BKECluster), err)}
	}

	nodeHasFailedFlag, _ := params.NodeFetcher.GetNodeStateFlagForCluster(params.Context, params.BKECluster, *params.InitNodeIp, bkev1beta1.NodeFailedFlag)
	if nodeHasFailedFlag {
		_ = params.NodeFetcher.SetNodeStateWithMessageForCluster(params.Context, params.BKECluster, *params.InitNodeIp, bkev1beta1.NodeBootStrapFailed, "")
		//删除command
		if err := command.PurgeCommand(params.Context, params.Client, params.InitCommand); err != nil {
			log.Warn(constant.MasterNotInitReason, "Delete init command failed, err: %v", err)
			return ProcessCommandFailureResult{Done: false, Success: false, Err: nil}
		}
		// Notify manual intervention requirement
		if params.Recorder != nil {
			_ = intervention.Require(intervention.Params{
				BKECluster: params.BKECluster,
				Client:     params.Client,
				Recorder:   params.Recorder,
				Guidance:   intervention.NodeBootstrapTerminal(*params.InitNodeIp, params.FailedNodes),
			})
		}
		return ProcessCommandFailureResult{Done: true, Success: false, Err: errors.Errorf("master node init failed, failed nodes: %v", params.FailedNodes)}
	}

	log.Warn(constant.MasterNotInitReason, "Master node init failed, failed nodes: %v", params.FailedNodes)
	log.Info(constant.MasterNotInitReason, "Trying to init master node again...")
	// get command owner bkeMachine
	ownerBkeMachineName := params.InitCommand.OwnerReferences[0].Name
	// update owner bkeMachine status for bootstrap again

	key := client.ObjectKey{
		Name:      ownerBkeMachineName,
		Namespace: params.InitCommand.Namespace,
	}

	ownerBkeMachine := &bkev1beta1.BKEMachine{}
	var getFailed bool
	if err := phaseutil.RetryOnConflict(func() error {
		if err := params.Client.Get(params.Context, key, ownerBkeMachine); err != nil {
			getFailed = true
			return fmt.Errorf("Get init command owner bkeMachine failed: %w", err)
		}
		getFailed = false
		labelhelper.RemoveBKEMachineLabel(ownerBkeMachine, bkenode.MasterNodeRole)
		return params.Client.Update(params.Context, ownerBkeMachine)
	}); err != nil {
		if !getFailed {
			err = fmt.Errorf("Update init command owner bkeMachine failed: %w", err)
		}
		return ProcessCommandFailureResult{Done: false, Success: false, Err: err}
	}
	return ProcessCommandFailureResult{Done: false, Success: false, Err: errors.Errorf("master node init command run failed, failed nodes: %v", params.FailedNodes)}
}

// GetTargetClusterNodes gets nodes from the target k8s cluster.
// This is a shared function used by EnsureWorkerDelete and EnsureMasterDelete.
func GetTargetClusterNodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
	clientset, _, err := kube.GetTargetClusterClient(ctx, c, bkeCluster)
	if err != nil {
		return nil, err
	}

	nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var nodes bkenode.Nodes
	for _, n := range nodeList.Items {
		node := confv1beta1.Node{
			Hostname: n.Name,
			IP:       kube.GetNodeIP(&n),
			Role:     phaseutil.GetNodeRolesFromK8sNode(&n),
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

// deleteTargetClusterNode deletes a Node from the target cluster by name.
// Returns nil if the Node is not found (already deleted).
func DeleteTargetClusterNode(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeName string) error {
	if nodeName == "" {
		return nil
	}
	clientset, _, err := kube.GetTargetClusterClient(ctx, c, bkeCluster)
	if err != nil {
		return errors.Wrapf(err, "failed to get target cluster client")
	}
	if err := clientset.CoreV1().Nodes().Delete(ctx, nodeName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		return errors.Wrapf(err, "failed to delete Node %s from target cluster", nodeName)
	}
	return nil
}

// bkenodeGVR is the GroupVersionResource of the BKENode custom resource.
var bkenodeGVR = schema.GroupVersionResource{
	Group:    confv1beta1.BKENodeGVK.Group,
	Version:  confv1beta1.BKENodeGVK.Version,
	Resource: "bkenodes",
}

func init() {
	// Guard against silent fail-open if BKENode GVK/resource constants drift.
	expected := schema.GroupVersionResource{
		Group:    "bke.bocloud.com",
		Version:  "v1beta1",
		Resource: "bkenodes",
	}
	if bkenodeGVR != expected || confv1beta1.BKENodeGVK.Kind != "BKENode" {
		panic(fmt.Sprintf("bkenodeGVR/GVK mismatch: gvr=%s kind=%s (expected gvr=%s kind=BKENode)",
			bkenodeGVR.String(), confv1beta1.BKENodeGVK.Kind, expected.String()))
	}
}

// isBKENodeCRDUnavailable reports whether err means the target cluster has no BKENode API
// (business clusters). Only these cases stay fail-open for orphan cleanup.
func isBKENodeCRDUnavailable(err error) bool {
	if err == nil {
		return false
	}
	return apierrors.IsNotFound(err) || meta.IsNoMatchError(err)
}

// shouldSkipOrphanNodeCleanup decides whether orphan Node cleanup must be skipped.
//
// skip=true when:
//   - the target cluster still has an active (non-Deleting) BKENode for nodeIP; or
//   - the target BKENode status cannot be confirmed safely (fail-closed on transient /
//     unexpected errors), to avoid mis-deleting nodes during management-cluster scale-up.
//
// skip=false when:
//   - no matching active BKENode exists; or
//   - the target has no BKENode CRD (fail-open, preserve business-cluster behavior).
func shouldSkipOrphanNodeCleanup(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, nodeIP string) (bool, string) {
	if bkeCluster == nil || nodeIP == "" {
		return false, ""
	}
	_, dynamicClient, err := kube.GetTargetClusterClient(ctx, c, bkeCluster)
	if err != nil {
		return true, fmt.Sprintf("fail-closed: cannot get target cluster client: %v", err)
	}
	labelSelector := fmt.Sprintf("%s=%s", clusterv1.ClusterNameLabel, bkeCluster.Name)
	unstructuredList, err := dynamicClient.Resource(bkenodeGVR).Namespace(bkeCluster.Namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		if isBKENodeCRDUnavailable(err) {
			log.Warnf("orphan cleanup fail-open: BKENode API unavailable on target cluster (gvr=%s, cluster=%s/%s, nodeIP=%s): %v",
				bkenodeGVR.String(), bkeCluster.Namespace, bkeCluster.Name, nodeIP, err)
			return false, ""
		}
		return true, fmt.Sprintf("fail-closed: list BKENode failed (gvr=%s): %v", bkenodeGVR.String(), err)
	}
	bkeNodeList := &confv1beta1.BKENodeList{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredList.UnstructuredContent(), bkeNodeList); err != nil {
		return true, fmt.Sprintf("fail-closed: convert BKENodeList failed (gvr=%s): %v", bkenodeGVR.String(), err)
	}
	for i := range bkeNodeList.Items {
		if bkeNodeList.Items[i].Spec.IP == nodeIP && bkeNodeList.Items[i].Status.State != confv1beta1.NodeDeleting {
			return true, "active BKENode found on target cluster"
		}
	}
	return false, ""
}
