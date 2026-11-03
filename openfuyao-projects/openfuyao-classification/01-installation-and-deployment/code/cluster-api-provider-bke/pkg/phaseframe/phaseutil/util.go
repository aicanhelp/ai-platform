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

package phaseutil

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"

	hashversion "github.com/hashicorp/go-version"
	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	healthconfig "gopkg.openfuyao.cn/cluster-api-provider-bke/config/health"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/bkeagent/cluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/clusterutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// DefaultRightVersionFields defines the default number of fields in a normalized version string.
const DefaultRightVersionFields = 4

// OnceFunc provides a sync.Once instance for one-time initialization.
var OnceFunc sync.Once

// NodeInfo returns a formatted string representation of a node (hostname/IP).
func NodeInfo(node confv1beta1.Node) string {
	if node.Hostname == "" {
		return node.IP
	}
	if node.IP == "" {
		return node.Hostname
	}
	return fmt.Sprintf("%s/%s", node.Hostname, node.IP)
}

// NodeRoleString returns a semicolon-separated string of node roles.
func NodeRoleString(node confv1beta1.Node) string {
	return strings.Join(node.Role, ";")
}

// IsMasterNode checks if the node has the master role.
func IsMasterNode(node *confv1beta1.Node) bool {
	for _, role := range node.Role {
		if role == bkenode.MasterNodeRole {
			return true
		}
	}
	return false
}

// IsWorkerNode checks if the node has the worker role.
func IsWorkerNode(node *confv1beta1.Node) bool {
	for _, role := range node.Role {
		if role == bkenode.WorkerNodeRole {
			return true
		}
	}
	return false
}

// IsEtcdNode checks if the node has the etcd role.
func IsEtcdNode(node *confv1beta1.Node) bool {
	for _, role := range node.Role {
		if role == bkenode.EtcdNodeRole {
			return true
		}
	}
	return false
}

// GetBKEAllNodesFromNodesStatus extracts all BKE nodes from NodesStates.
// Note: This function uses GetBKENodesFromCluster which requires local kubeconfig.
// In controller context, use GetBKEAllNodesFromNodesStatusWithBKENodes instead.
func GetBKEAllNodesFromNodesStatus(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	var nodes bkenode.Nodes
	bkenodes := GetBKENodesFromCluster(bkeCluster)
	for _, bkenode := range bkenodes {
		nodes = append(nodes, bkenode.ToNode())
	}
	return nodes
}

// GetBKEAllNodesFromNodesStatusWithBKENodes extracts all BKE nodes from pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetBKEAllNodesFromNodesStatusWithBKENodes(bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	var nodes bkenode.Nodes
	for _, bkenode := range bkeNodes {
		nodes = append(nodes, bkenode.ToNode())
	}
	return nodes
}

// GetBKENodesFromNodesStatus extracts BKE nodes from NodesStates.
// Note: This function uses GetBKENodesFromCluster which requires local kubeconfig.
// In controller context, use GetBKENodesFromNodesStatusWithBKENodes instead.
func GetBKENodesFromNodesStatus(bkeCluster *bkev1beta1.BKECluster) bkev1beta1.BKENodes {
	var nodes bkev1beta1.BKENodes
	bkenodes := GetBKENodesFromCluster(bkeCluster)
	for _, bkenode := range bkenodes {
		if bkenode.Status.NeedSkip {
			continue
		}
		nodes = append(nodes, bkenode)
	}
	return nodes
}

// GetBKENodesFromNodesStatusWithBKENodes extracts BKE nodes from pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetBKENodesFromNodesStatusWithBKENodes(bkeNodes bkev1beta1.BKENodes) bkev1beta1.BKENodes {
	var nodes bkev1beta1.BKENodes
	for _, bkenode := range bkeNodes {
		if bkenode.Status.NeedSkip {
			continue
		}
		nodes = append(nodes, bkenode)
	}
	return nodes
}

// ConvertELBNodesToBKENodes converts ELB node IPs to BKE node objects.
func ConvertELBNodesToBKENodes(elbNodes []string, src bkenode.Nodes) bkenode.Nodes {
	var nodes bkenode.Nodes
	for _, ip := range elbNodes {
		for _, node := range src {
			if node.IP == ip {
				nodes = append(nodes, node)
			}
		}
	}
	return nodes
}

type NodeFilterOption func(*nodeFilterConfig)

func WithExcludeAppointmentNodes() NodeFilterOption {
	return func(c *nodeFilterConfig) {
		c.excludeAppointment = true
	}
}

// WithBKENodes allows passing pre-fetched BKENodes to avoid calling GetBKENodesFromCluster.
func WithBKENodes(bkeNodes bkev1beta1.BKENodes) NodeFilterOption {
	return func(c *nodeFilterConfig) {
		c.bkeNodes = bkeNodes
	}
}

type nodeFilterConfig struct {
	excludeAppointment bool
	bkeNodes           bkev1beta1.BKENodes
}

type NodePredicate func(ip string, nodeState *confv1beta1.BKENode) bool

// filterNodes 公共核心函数
func filterNodes(
	bkeCluster *bkev1beta1.BKECluster,
	predicate NodePredicate,
	opts ...NodeFilterOption,
) bkenode.Nodes {
	var cfg nodeFilterConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	var nodes bkenode.Nodes
	var bkenodes bkev1beta1.BKENodes
	if cfg.bkeNodes != nil {
		bkenodes = cfg.bkeNodes
	} else {
		bkenodes = GetBKENodesFromCluster(bkeCluster)
	}
	for _, bkenode := range bkenodes {
		if bkenodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeFailedFlag) {
			continue
		}
		if bkenodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeDeletingFlag) {
			continue
		}
		if bkenode.Status.NeedSkip {
			continue
		}

		if predicate(bkenode.Spec.IP, &bkenode) {
			nodes = append(nodes, bkenode.ToNode())
		}
	}

	if !cfg.excludeAppointment {
		return nodes
	}

	appointmentNodes := GetAppointmentAddNodes(bkeCluster)
	if appointmentNodes.Length() == 0 {
		return nodes
	}

	var finalNodes bkenode.Nodes
	for _, node := range nodes {
		if appointmentNodes.Filter(bkenode.FilterOptions{"IP": node.IP}).Length() == 0 {
			finalNodes = append(finalNodes, node)
		}
	}

	return finalNodes
}

// GetNeedPushAgentNodes returns nodes that need agent push.
// In controller context, use GetNeedPushAgentNodesWithBKENodes instead.
func GetNeedPushAgentNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return filterNodes(bkeCluster,
		func(ip string, bn *confv1beta1.BKENode) bool {
			return !GetNodeStateFlag(bn, ip, bkev1beta1.NodeAgentPushedFlag)
		},
		WithExcludeAppointmentNodes(),
	)
}

// GetNeedPushAgentNodesWithBKENodes returns nodes that need agent push using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedPushAgentNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	return filterNodes(bkeCluster,
		func(ip string, bn *confv1beta1.BKENode) bool {
			return !GetNodeStateFlag(bn, ip, bkev1beta1.NodeAgentPushedFlag)
		},
		WithExcludeAppointmentNodes(),
		WithBKENodes(bkeNodes),
	)
}

// GetAgentPushedNodes 获取已推送agent的节点
// In controller context, use GetAgentPushedNodesWithBKENodes instead.
func GetAgentPushedNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	bkeNodes := GetBKENodesFromCluster(bkeCluster)
	var nodes bkenode.Nodes
	for _, bkeNode := range bkeNodes {
		if bkeNodes.GetNodeStateFlag(bkeNode.Spec.IP, bkev1beta1.NodeAgentPushedFlag) {
			nodes = append(nodes, bkeNode.ToNode())
		}
	}
	return nodes
}

// GetAgentPushedNodesWithBKENodes 获取已推送agent的节点，使用预获取的 BKENodes
// Use this in controller context where local kubeconfig is not available.
func GetAgentPushedNodesWithBKENodes(bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	var nodes bkenode.Nodes
	for _, bkeNode := range bkeNodes {
		if bkeNodes.GetNodeStateFlag(bkeNode.Spec.IP, bkev1beta1.NodeAgentPushedFlag) {
			nodes = append(nodes, bkeNode.ToNode())
		}
	}
	return nodes
}

// GetNeedInitEnvNodes returns nodes that need environment initialization.
// In controller context, use GetNeedInitEnvNodesWithBKENodes instead.
func GetNeedInitEnvNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return filterNodes(bkeCluster,
		func(ip string, bn *confv1beta1.BKENode) bool {
			return !GetNodeStateFlag(bn, ip, bkev1beta1.NodeEnvFlag)
		},
		WithExcludeAppointmentNodes(),
	)
}

// GetNeedInitEnvNodesWithBKENodes returns nodes that need environment initialization using pre-fetched BKENodes.
func GetNeedInitEnvNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	return filterNodes(bkeCluster,
		func(ip string, bn *confv1beta1.BKENode) bool {
			return !GetNodeStateFlag(bn, ip, bkev1beta1.NodeEnvFlag)
		},
		WithExcludeAppointmentNodes(),
		WithBKENodes(bkeNodes),
	)
}

// GetNeedPostProcessNodes returns nodes that need postprocess execution.
// Deprecated: In controller context, use GetNeedPostProcessNodesWithBKENodes instead.
func GetNeedPostProcessNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return filterNodes(bkeCluster,
		func(ip string, bn *confv1beta1.BKENode) bool {
			return GetNodeStateFlag(bn, ip, bkev1beta1.NodeBootFlag) &&
				!GetNodeStateFlag(bn, ip, bkev1beta1.NodePostProcessFlag)
		},
		WithExcludeAppointmentNodes(),
	)
}

// GetNeedPostProcessNodesWithBKENodes returns nodes that need postprocess execution, using pre-fetched BKENodes.
func GetNeedPostProcessNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	return filterNodes(bkeCluster,
		func(ip string, bn *confv1beta1.BKENode) bool {
			return GetNodeStateFlag(bn, ip, bkev1beta1.NodeBootFlag) &&
				!GetNodeStateFlag(bn, ip, bkev1beta1.NodePostProcessFlag)
		},
		WithExcludeAppointmentNodes(),
		WithBKENodes(bkeNodes),
	)
}

// ----------join--------------------

// GetNeedJoinNodes 获取需要加入集群的节点
// In controller context, use GetNeedJoinNodesWithBKENodes instead.
func GetNeedJoinNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return filterNodes(bkeCluster,
		func(ip string, bn *confv1beta1.BKENode) bool {
			return !GetNodeStateFlag(bn, ip, bkev1beta1.NodeBootFlag) &&
				!GetNodeStateFlag(bn, ip, bkev1beta1.MasterInitFlag)
		},
	)
}

// GetNeedJoinNodesWithBKENodes 获取需要加入集群的节点，使用预获取的 BKENodes
func GetNeedJoinNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	return filterNodes(bkeCluster,
		func(ip string, bn *confv1beta1.BKENode) bool {
			return !GetNodeStateFlag(bn, ip, bkev1beta1.NodeBootFlag) &&
				!GetNodeStateFlag(bn, ip, bkev1beta1.MasterInitFlag)
		},
		WithBKENodes(bkeNodes),
	)
}

// GetNeedJoinMasterNodes returns master nodes that need to join the cluster.
// In controller context, use GetNeedJoinMasterNodesWithBKENodes instead.
func GetNeedJoinMasterNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	needAddNodes := GetNeedJoinNodes(bkeCluster).Master()
	if needAddNodes.Length() == 0 {
		return nil
	}

	appointmentNodes := GetAppointmentAddNodes(bkeCluster)
	if appointmentNodes.Length() > 0 {
		return ComputeFinalAddNodes(needAddNodes, appointmentNodes)
	}

	return needAddNodes
}

// GetNeedJoinMasterNodesWithBKENodes returns master nodes that need to join the cluster using pre-fetched BKENodes.
func GetNeedJoinMasterNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	needAddNodes := GetNeedJoinNodesWithBKENodes(bkeCluster, bkeNodes).Master()
	if needAddNodes.Length() == 0 {
		return nil
	}

	appointmentNodes := GetAppointmentAddNodesWithBKENodes(bkeCluster, bkeNodes)
	if appointmentNodes.Length() > 0 {
		return ComputeFinalAddNodes(needAddNodes, appointmentNodes)
	}

	return needAddNodes
}

// GetNeedJoinWorkerNodes returns worker nodes that need to join the cluster.
// In controller context, use GetNeedJoinWorkerNodesWithBKENodes instead.
func GetNeedJoinWorkerNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	needAddNodes := GetNeedJoinNodes(bkeCluster).Worker()
	if needAddNodes.Length() == 0 {
		return nil
	}

	appointmentNodes := GetAppointmentAddNodes(bkeCluster)
	if appointmentNodes.Length() > 0 {
		return ComputeFinalAddNodes(needAddNodes, appointmentNodes)
	}

	return needAddNodes
}

// GetNeedJoinWorkerNodesWithBKENodes returns worker nodes that need to join the cluster using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedJoinWorkerNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	needAddNodes := GetNeedJoinNodesWithBKENodes(bkeCluster, bkeNodes).Worker()
	if needAddNodes.Length() == 0 {
		return nil
	}

	appointmentNodes := GetAppointmentAddNodesWithBKENodes(bkeCluster, bkeNodes)
	if appointmentNodes.Length() > 0 {
		return ComputeFinalAddNodes(needAddNodes, appointmentNodes)
	}

	return needAddNodes
}

// GetAppointmentAddNodes returns nodes scheduled for addition via annotation.
func GetAppointmentAddNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	v, found := annotation.HasAnnotation(bkeCluster, annotation.AppointmentAddNodesAnnotationKey)
	if !found {
		return nil
	}
	nodesIP := strings.Split(v, ",")

	var nodes bkenode.Nodes
	statusNodes := GetBKENodesFromNodesStatus(bkeCluster)
	for _, bkenode := range statusNodes {
		if utils.ContainsString(nodesIP, bkenode.Spec.IP) {
			nodes = append(nodes, bkenode.ToNode())
		}
	}
	return nodes
}

// GetAppointmentAddNodesWithBKENodes returns nodes scheduled for addition via annotation using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetAppointmentAddNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	v, found := annotation.HasAnnotation(bkeCluster, annotation.AppointmentAddNodesAnnotationKey)
	if !found {
		return nil
	}
	nodesIP := strings.Split(v, ",")

	var nodes bkenode.Nodes
	statusNodes := GetBKENodesFromNodesStatusWithBKENodes(bkeNodes)
	for _, bkenode := range statusNodes {
		if utils.ContainsString(nodesIP, bkenode.Spec.IP) {
			nodes = append(nodes, bkenode.ToNode())
		}
	}
	return nodes
}

// ComputeFinalAddNodes computes the final set of nodes to be added.
func ComputeFinalAddNodes(needAddNodes bkenode.Nodes, appointmentNodes bkenode.Nodes) bkenode.Nodes {
	var nodes bkenode.Nodes
	for _, node := range needAddNodes {
		if appointmentNodes.Filter(bkenode.FilterOptions{"IP": node.IP}).Length() == 0 {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// ----------delete--------------------

// GetNeedDeleteNodes returns nodes that need to be deleted from the cluster.
// It compares nodes currently deployed in the target cluster (from k8s Node list)
// with the expected nodes (from BKENode resources).
// Nodes that exist in target cluster but not in BKENode resources are marked for deletion.
func GetNeedDeleteNodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	var nodes bkenode.Nodes
	fetcher := nodeutil.NewNodeFetcher(c)

	// Get expected nodes from BKENode resources (this is what user wants)
	specNodes, err := fetcher.GetNodes(ctx, bkeCluster.Namespace, bkeCluster.Name)
	if err != nil {
		log.Warnf("Failed to fetch BKENodes for cluster %s: %v", bkeCluster.Name, err)
		return nil
	}

	// Get deployed nodes from BKENode resources with State=Ready
	// These represent nodes that have been successfully deployed to the target cluster
	nodeStates, err := fetcher.GetNodeStates(ctx, bkeCluster.Namespace, bkeCluster.Name)
	if err != nil {
		log.Warnf("Failed to get node states for cluster %s: %v", bkeCluster.Name, err)
		return nil
	}

	// Only consider nodes with Ready state as "deployed"
	var deployedNodes bkenode.Nodes
	for _, ns := range nodeStates {
		if ns.State == confv1beta1.NodeReady {
			deployedNodes = append(deployedNodes, ns.Node)
		}
	}

	// Compare: nodes in deployedNodes but not in specNodes need to be deleted
	// Note: After BKENode resource is deleted, it won't appear in specNodes,
	// but the node in target cluster still exists and needs to be removed
	nodeTs, ok := bkenode.CompareBKEConfigNode(deployedNodes, specNodes)
	if !ok {
		return nil
	}
	for _, node := range nodeTs {
		if node.Operate == bkenode.RemoveNode {
			nodes = append(nodes, *node.Node)
		}
	}
	return nodes
}

// GetNeedDeleteWorkerNodes returns worker nodes that need to be deleted.
// It supports two modes:
// 1. Appointment mode: User sets annotation to specify which nodes to delete (legacy mode)
// 2. BKENode deletion mode: Compare target cluster nodes with BKENode resources to find deleted nodes
func GetNeedDeleteWorkerNodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	// First try appointment mode (legacy)
	appointmentNodes := GetAppointmentDeletedNodes(bkeCluster)
	if appointmentNodes.Length() > 0 {
		needDepleteNodes := GetNeedDeleteNodes(ctx, c, bkeCluster).Worker()
		if needDepleteNodes.Length() == 0 {
			return nil
		}
		return ComputeFinalDeleteNodes(needDepleteNodes, appointmentNodes)
	}

	// BKENode deletion mode is not supported here due to import cycle.
	// Use GetNeedDeleteWorkerNodesWithTargetNodes instead when target cluster nodes are available.
	return nil
}

// GetNeedDeleteWorkerNodesWithTargetNodes returns worker nodes that need to be deleted.
// targetNodes should be the list of nodes currently in the target k8s cluster.
func GetNeedDeleteWorkerNodesWithTargetNodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, targetNodes bkenode.Nodes) bkenode.Nodes {
	// First try appointment mode (legacy)
	appointmentNodes := GetAppointmentDeletedNodes(bkeCluster)
	if appointmentNodes.Length() > 0 {
		needDepleteNodes := GetNeedDeleteNodes(ctx, c, bkeCluster).Worker()
		if needDepleteNodes.Length() == 0 {
			return nil
		}
		return ComputeFinalDeleteNodes(needDepleteNodes, appointmentNodes)
	}

	// BKENode deletion mode: find nodes that exist in target cluster but not in BKENode resources
	needDeleteNodes := GetNeedDeleteNodesFromTargetNodes(ctx, c, bkeCluster, targetNodes)
	return needDeleteNodes.Worker()
}

// GetNeedDeleteMasterNodes returns master nodes that need to be deleted.
func GetNeedDeleteMasterNodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	// First try appointment mode (legacy)
	appointmentNodes := GetAppointmentDeletedNodes(bkeCluster)
	if appointmentNodes.Length() > 0 {
		needDepleteNodes := GetNeedDeleteNodes(ctx, c, bkeCluster).Master()
		if needDepleteNodes.Length() == 0 {
			return nil
		}
		return ComputeFinalDeleteNodes(needDepleteNodes, appointmentNodes)
	}

	// BKENode deletion mode is not supported here due to import cycle.
	// Use GetNeedDeleteMasterNodesWithTargetNodes instead when target cluster nodes are available.
	return nil
}

// GetNeedDeleteMasterNodesWithTargetNodes returns master nodes that need to be deleted.
// targetNodes should be the list of nodes currently in the target k8s cluster.
func GetNeedDeleteMasterNodesWithTargetNodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, targetNodes bkenode.Nodes) bkenode.Nodes {
	// First try appointment mode (legacy)
	appointmentNodes := GetAppointmentDeletedNodes(bkeCluster)
	if appointmentNodes.Length() > 0 {
		needDepleteNodes := GetNeedDeleteNodes(ctx, c, bkeCluster).Master()
		if needDepleteNodes.Length() == 0 {
			return nil
		}
		return ComputeFinalDeleteNodes(needDepleteNodes, appointmentNodes)
	}

	// BKENode deletion mode
	needDeleteNodes := GetNeedDeleteNodesFromTargetNodes(ctx, c, bkeCluster, targetNodes)
	return needDeleteNodes.Master()
}

// GetNeedDeleteNodesFromTargetNodes compares nodes in target k8s cluster with BKENode resources.
// Returns nodes that exist in target cluster but have no corresponding BKENode resource.
// targetNodes should be obtained from the target k8s cluster's node list.
func GetNeedDeleteNodesFromTargetNodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, targetNodes bkenode.Nodes) bkenode.Nodes {
	if targetNodes == nil || len(targetNodes) == 0 {
		return nil
	}

	fetcher := nodeutil.NewNodeFetcher(c)

	// Get expected nodes from BKENode resources
	specNodes, err := fetcher.GetNodes(ctx, bkeCluster.Namespace, bkeCluster.Name)
	if err != nil {
		log.Warnf("Failed to fetch BKENodes for cluster %s: %v", bkeCluster.Name, err)
		return nil
	}

	// Compare: nodes in targetNodes but not in specNodes need to be deleted
	var needDeleteNodes bkenode.Nodes
	for _, targetNode := range targetNodes {
		found := false
		for _, specNode := range specNodes {
			if targetNode.IP == specNode.IP {
				found = true
				break
			}
		}
		if !found {
			needDeleteNodes = append(needDeleteNodes, targetNode)
		}
	}
	return needDeleteNodes
}

// GetNodeRolesFromK8sNode extracts roles from k8s node labels.
func GetNodeRolesFromK8sNode(node *corev1.Node) []string {
	var roles []string
	if _, ok := node.Labels["node-role.kubernetes.io/control-plane"]; ok {
		roles = append(roles, bkenode.MasterNodeRole)
	}
	if _, ok := node.Labels["node-role.kubernetes.io/master"]; ok {
		if !utils.ContainsString(roles, bkenode.MasterNodeRole) {
			roles = append(roles, bkenode.MasterNodeRole)
		}
	}
	roles = append(roles, bkenode.WorkerNodeRole)
	return roles
}

// GetAppointmentDeletedNodes returns nodes scheduled for deletion via annotation.
func GetAppointmentDeletedNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	v, found := annotation.HasAnnotation(bkeCluster, annotation.AppointmentDeletedNodesAnnotationKey)
	if !found {
		return nil
	}
	nodesIP := strings.Split(v, ",")

	var nodes bkenode.Nodes
	statusNodes := GetNodesFromCluster(bkeCluster)
	for _, node := range statusNodes {
		if utils.ContainsString(nodesIP, node.IP) {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// RemoveAppointmentDeletedNodes removes a node from the appointment deletion list.
func RemoveAppointmentDeletedNodes(bkeCluster *bkev1beta1.BKECluster, nodeIP string) {
	v, found := annotation.HasAnnotation(bkeCluster, annotation.AppointmentDeletedNodesAnnotationKey)
	if !found {
		return
	}
	nodesIP := strings.Split(v, ",")

	// Find the index of the node to remove
	indexToRemove := -1
	for i, ip := range nodesIP {
		if ip == nodeIP {
			indexToRemove = i
			break
		}
	}

	if indexToRemove != -1 {
		nodesIP = append(nodesIP[:indexToRemove], nodesIP[indexToRemove+1:]...)
	}

	annotation.SetAnnotation(bkeCluster, annotation.AppointmentDeletedNodesAnnotationKey, strings.Join(nodesIP, ","))
}

// ComputeFinalDeleteNodes computes the final set of nodes to be deleted.
func ComputeFinalDeleteNodes(needDepleteNodes bkenode.Nodes, appointmentNodes bkenode.Nodes) bkenode.Nodes {
	var nodes bkenode.Nodes
	for _, node := range needDepleteNodes {
		if appointmentNodes.Filter(bkenode.FilterOptions{"IP": node.IP}).Length() != 0 {
			nodes = append(nodes, node)
		}
	}
	return nodes
}

// ----------upgrade--------------------

// NormalizeVersion normalizes a version string to vX.Y.Z format, removing leading zeros.
func NormalizeVersion(version string) (string, error) {
	// Remove possible "v" prefix and convert to lowercase
	clean := strings.TrimPrefix(strings.ToLower(version), "v")

	// Use regex to extract number parts
	re := regexp.MustCompile(`(\d+)\.?(\d*)\.?(\d*)`)
	matches := re.FindStringSubmatch(clean)

	if len(matches) < DefaultRightVersionFields {
		return "", fmt.Errorf("invalid version format: %s", version)
	}

	parts := make([]string, DefaultRightVersionFields-1)
	for i := 1; i < DefaultRightVersionFields; i++ {
		part := matches[i]
		if part == "" {
			part = "0"
		}

		num, err := strconv.Atoi(part)
		if err != nil {
			return "", fmt.Errorf("invalid number in version: %s", part)
		}
		parts[i-1] = strconv.Itoa(num)
	}

	return "v" + strings.Join(parts, "."), nil
}

// GetNeedUpgradeComponentNodes returns nodes that need component upgrade.
func GetNeedUpgradeComponentNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	if bkeCluster.Status.OpenFuyaoVersion == "" {
		return nil
	}
	newConfig := bkeCluster.Spec.ClusterConfig
	if !NeedUpgrade(bkeCluster.Status.OpenFuyaoVersion, newConfig.Cluster.OpenFuyaoVersion) {
		return nil
	}
	var nodes bkenode.Nodes
	statusNodes := GetBKENodesFromNodesStatus(bkeCluster)
	for _, bkenode := range statusNodes {
		if statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeFailedFlag) {
			continue
		}
		nodes = append(nodes, bkenode.ToNode())
	}

	return nodes
}

// GetNeedUpgradeComponentNodesWithBKENodes returns nodes that need component upgrade using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedUpgradeComponentNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	if bkeCluster.Status.OpenFuyaoVersion == "" {
		return nil
	}
	newConfig := bkeCluster.Spec.ClusterConfig
	if !NeedUpgrade(bkeCluster.Status.OpenFuyaoVersion, newConfig.Cluster.OpenFuyaoVersion) {
		return nil
	}
	var nodes bkenode.Nodes
	statusNodes := GetBKENodesFromNodesStatusWithBKENodes(bkeNodes)
	for _, bkenode := range statusNodes {
		if statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeFailedFlag) {
			continue
		}
		nodes = append(nodes, bkenode.ToNode())
	}

	return nodes
}

// GetNeedUpgradeContainerdNodesWithBKENodes returns nodes that need containerd upgrade using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedUpgradeContainerdNodesWithBKENodes(
	bkeCluster *bkev1beta1.BKECluster,
	bkeNodes bkev1beta1.BKENodes,
) bkenode.Nodes {
	if bkeCluster == nil || bkeCluster.Spec.ClusterConfig == nil {
		return nil
	}
	if bkeCluster.Status.ContainerdVersion == "" {
		return nil
	}
	if !NeedUpgrade(bkeCluster.Status.ContainerdVersion, bkeCluster.Spec.ClusterConfig.Cluster.ContainerdVersion) {
		return nil
	}
	return filterNonFailedNodes(GetBKENodesFromNodesStatusWithBKENodes(bkeNodes))
}

// GetNeedUpgradeNodes returns nodes that need to be upgraded.
// In controller context, use GetNeedUpgradeNodesWithBKENodes instead.
func GetNeedUpgradeNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	if bkeCluster.Status.OpenFuyaoVersion == "" {
		return nil
	}

	return compareVersionAndGetNodes(
		bkeCluster,
		bkeCluster.Status.OpenFuyaoVersion,
		bkeCluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion,
	)
}

// GetNeedUpgradeNodesWithBKENodes returns nodes that need to be upgraded using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedUpgradeNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	if bkeCluster.Status.OpenFuyaoVersion == "" {
		return nil
	}

	return compareVersionAndGetNodesWithBKENodes(
		bkeCluster,
		bkeCluster.Status.OpenFuyaoVersion,
		bkeCluster.Spec.ClusterConfig.Cluster.OpenFuyaoVersion,
		bkeNodes,
	)
}

// GetNeedUpgradeK8sNodes returns nodes that need Kubernetes upgrade.
// Note: This function uses GetBKENodesFromNodesStatus which requires local kubeconfig.
// In controller context, use GetNeedUpgradeK8sNodesWithBKENodes instead.
func GetNeedUpgradeK8sNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return compareVersionAndGetNodes(
		bkeCluster,
		bkeCluster.Status.KubernetesVersion,
		bkeCluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
	)
}

// GetNeedUpgradeK8sNodesWithBKENodes returns nodes that need Kubernetes upgrade using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedUpgradeK8sNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	return compareVersionAndGetNodesWithBKENodes(
		bkeCluster,
		bkeCluster.Status.KubernetesVersion,
		bkeCluster.Spec.ClusterConfig.Cluster.KubernetesVersion,
		bkeNodes,
	)
}

// GetNeedUpgradeMasterNodes returns master nodes that need to be upgraded.
// In controller context, use GetNeedUpgradeMasterNodesWithBKENodes instead.
func GetNeedUpgradeMasterNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return GetNeedUpgradeK8sNodes(bkeCluster).Master()
}

// GetNeedUpgradeMasterNodesWithBKENodes returns master nodes that need to be upgraded using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedUpgradeMasterNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	return GetNeedUpgradeK8sNodesWithBKENodes(bkeCluster, bkeNodes).Master()
}

// GetNeedUpgradeWorkerNodes returns worker nodes that need to be upgraded.
// In controller context, use GetNeedUpgradeWorkerNodesWithBKENodes instead.
func GetNeedUpgradeWorkerNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return GetNeedUpgradeK8sNodes(bkeCluster).Worker()
}

// GetNeedUpgradeWorkerNodesWithBKENodes returns worker nodes that need to be upgraded using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedUpgradeWorkerNodesWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	return GetNeedUpgradeK8sNodesWithBKENodes(bkeCluster, bkeNodes).Worker()
}

// ----------loadbalance--------------------

func mergeNodesToMap(nodeLists ...bkenode.Nodes) bkenode.Nodes {
	nodesMap := make(map[string]confv1beta1.Node)
	for _, nodeList := range nodeLists {
		for _, node := range nodeList {
			nodesMap[node.IP] = node
		}
	}
	var nodes bkenode.Nodes
	for _, node := range nodesMap {
		nodes = append(nodes, node)
	}
	return nodes
}

// GetNeedLoadBalanceNodes returns nodes that need load balancing.
// In controller context, use GetNeedLoadBalanceNodesWithBKENodes instead.
func GetNeedLoadBalanceNodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return mergeNodesToMap(
		GetNeedJoinMasterNodes(bkeCluster),
		GetNeedDeleteMasterNodes(ctx, c, bkeCluster),
	)
}

// GetNeedLoadBalanceNodesWithBKENodes returns nodes that need load balancing using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedLoadBalanceNodesWithBKENodes(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	return mergeNodesToMap(
		GetNeedJoinMasterNodesWithBKENodes(bkeCluster, bkeNodes),
		GetNeedDeleteMasterNodes(ctx, c, bkeCluster),
	)
}

// ----------bke-config--------------------

// GetBKEConfigCMData retrieves BKE configuration data from ConfigMap.
func GetBKEConfigCMData(ctx context.Context, c client.Client) (map[string]string, error) {
	config := &corev1.ConfigMap{}
	err := c.Get(ctx, clusterutil.BKEConfigCmKey(), config)
	if err != nil {
		return nil, err
	}
	return config.Data, nil
}

// GetRemoteBKEConfigCM retrieves BKE configuration ConfigMap from remote cluster.
func GetRemoteBKEConfigCM(ctx context.Context, clientSet *kubernetes.Clientset) (*corev1.ConfigMap, error) {
	config, err := clientSet.CoreV1().ConfigMaps(clusterutil.BKEConfigCmKey().Namespace).Get(ctx, clusterutil.BKEConfigCmKey().Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return config, nil
}

// MigrateBKEConfigCM migrates BKE configuration ConfigMap to remote cluster.
func MigrateBKEConfigCM(ctx context.Context, c client.Client, clientSet *kubernetes.Clientset) error {
	config := &corev1.ConfigMap{}
	err := c.Get(ctx, clusterutil.BKEConfigCmKey(), config)
	if err != nil {
		return err
	}
	remoteConfig := &corev1.ConfigMap{}
	remoteConfig.Name = clusterutil.BKEConfigCmKey().Name
	remoteConfig.Namespace = clusterutil.BKEConfigCmKey().Namespace
	remoteConfig.Data = config.Data

	// create ns
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: remoteConfig.Namespace,
		},
	}

	_, err = clientSet.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	if err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return fmt.Errorf("create remote bke config ns %q failed: %w", remoteConfig.Namespace, err)
		}
	}

	_, err = clientSet.CoreV1().ConfigMaps(remoteConfig.Namespace).Create(ctx, remoteConfig, metav1.CreateOptions{})
	if err != nil {
		if apierrors.IsAlreadyExists(err) {
			err = RetryOnConflict(func() error {
				existingConfigMap, err := clientSet.CoreV1().ConfigMaps(remoteConfig.Namespace).Get(ctx, remoteConfig.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				existingConfigMap.Data = remoteConfig.Data
				_, err = clientSet.CoreV1().ConfigMaps(remoteConfig.Namespace).Update(ctx, existingConfigMap, metav1.UpdateOptions{})
				return err
			})
			if err != nil {
				return fmt.Errorf("update remote bke config cm failed: %w", err)
			}
			return nil
		}
		return fmt.Errorf("create remote bke config cm failed: %w", err)
	}
	return nil
}

// EnsureLocalHealthCheckConfigCM ensures the controller cluster has the default health check ConfigMap.
func EnsureLocalHealthCheckConfigCM(ctx context.Context, c client.Client) error {
	if err := ensureHealthCheckConfigNamespace(ctx, c); err != nil {
		return err
	}
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      healthconfig.Name,
			Namespace: healthconfig.Namespace,
		},
		Data: map[string]string{healthconfig.DataKey: healthconfig.DefaultConfig},
	}
	if err := c.Create(ctx, cm); err != nil {
		if apierrors.IsAlreadyExists(err) {
			// 已存在时保留用户配置，避免每次调谐都把 ConfigMap 覆盖回默认值。
			return nil
		}
		return err
	}
	return nil
}

func ensureHealthCheckConfigNamespace(ctx context.Context, c client.Client) error {
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: healthconfig.Namespace}}
	if err := c.Create(ctx, ns); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	return nil
}

// GetRemoteHealthCheckConfigCM retrieves health check ConfigMap from remote cluster.
func GetRemoteHealthCheckConfigCM(ctx context.Context, clientSet *kubernetes.Clientset) (*corev1.ConfigMap, error) {
	if clientSet == nil {
		return nil, errors.New("remote kubernetes clientset is nil")
	}
	config, err := clientSet.CoreV1().ConfigMaps(healthconfig.Namespace).Get(ctx, healthconfig.Name, metav1.GetOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	return config, nil
}

// MigrateHealthCheckConfigCM migrates health check ConfigMap to remote cluster.
func MigrateHealthCheckConfigCM(ctx context.Context, c client.Client, clientSet *kubernetes.Clientset) error {
	if clientSet == nil {
		return errors.New("remote kubernetes clientset is nil")
	}
	if err := EnsureLocalHealthCheckConfigCM(ctx, c); err != nil {
		return fmt.Errorf("ensure local health check config cm failed: %w", err)
	}
	config := &corev1.ConfigMap{}
	key := client.ObjectKey{Namespace: healthconfig.Namespace, Name: healthconfig.Name}
	if err := c.Get(ctx, key, config); err != nil {
		return err
	}
	remoteConfig := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      healthconfig.Name,
			Namespace: healthconfig.Namespace,
		},
		Data: config.Data,
	}

	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: healthconfig.Namespace}}
	if _, err := clientSet.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create remote health check config ns %q failed: %w", healthconfig.Namespace, err)
	}

	_, err := clientSet.CoreV1().ConfigMaps(healthconfig.Namespace).Create(ctx, remoteConfig, metav1.CreateOptions{})
	if err == nil {
		return nil
	}
	if !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create remote health check config cm failed: %w", err)
	}
	return RetryOnConflict(func() error {
		existingConfigMap, getErr := clientSet.CoreV1().ConfigMaps(healthconfig.Namespace).Get(ctx, healthconfig.Name, metav1.GetOptions{})
		if getErr != nil {
			return getErr
		}
		existingConfigMap.Data = remoteConfig.Data
		_, updateErr := clientSet.CoreV1().ConfigMaps(healthconfig.Namespace).Update(ctx, existingConfigMap, metav1.UpdateOptions{})
		return updateErr
	})
}
func MigratePatchConfigCM(ctx context.Context, c client.Client, clientSet *kubernetes.Clientset) error {
	patchNamespace := "openfuyao-patch"

	configMapList := &corev1.ConfigMapList{}
	if err := c.List(ctx, configMapList, &client.ListOptions{Namespace: patchNamespace}); err != nil {
		return errors.Wrapf(err, "failed to list ConfigMaps in namespace %q", patchNamespace)
	}

	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: patchNamespace}}
	if _, err := clientSet.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return errors.Wrapf(err, "create remote namespace %q failed", patchNamespace)
		}
	}

	for _, cm := range configMapList.Items {
		if !strings.HasPrefix(cm.Name, "cm.") {
			continue
		}

		cmCopy := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      cm.Name,
				Namespace: patchNamespace,
			},
			Data: cm.Data,
		}

		_, err := clientSet.CoreV1().ConfigMaps(patchNamespace).Create(ctx, cmCopy, metav1.CreateOptions{})
		if err == nil {
			continue
		}

		if !apierrors.IsAlreadyExists(err) {
			return errors.Wrapf(err, "create remote ConfigMap %s/%s failed", patchNamespace, cm.Name)
		}

		if updateErr := RetryOnConflict(func() error {
			existingConfigMap, err := clientSet.CoreV1().ConfigMaps(patchNamespace).Get(ctx, cmCopy.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			existingConfigMap.Data = cmCopy.Data
			_, err = clientSet.CoreV1().ConfigMaps(patchNamespace).Update(ctx, existingConfigMap, metav1.UpdateOptions{})
			return err
		}); updateErr != nil {
			return errors.Wrapf(updateErr, "update remote ConfigMap %s/%s failed", patchNamespace, cm.Name)
		}
	}

	return nil
}

// ----------bootstrap--------------------

// GetReadyBootstrapNodes returns nodes that are ready for bootstrap.
// Note: This function uses GetBKENodesFromNodesStatus which requires local kubeconfig.
// In controller context, use GetReadyBootstrapNodesWithBKENodes instead.
func GetReadyBootstrapNodes(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	var nodes bkenode.Nodes
	statusNodes := GetBKENodesFromNodesStatus(bkeCluster)
	for _, bkenode := range statusNodes {
		// agent is ready and env is ready and boot is not
		if statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeAgentReadyFlag) &&
			statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeEnvFlag) &&
			!statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeBootFlag) {
			nodes = append(nodes, bkenode.ToNode())
		}
	}
	return nodes
}

// GetReadyBootstrapNodesWithBKENodes returns nodes that are ready for bootstrap using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetReadyBootstrapNodesWithBKENodes(bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	var nodes bkenode.Nodes
	statusNodes := GetBKENodesFromNodesStatusWithBKENodes(bkeNodes)
	for _, bkenode := range statusNodes {
		// agent is ready and env is ready and boot is not
		if statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeAgentReadyFlag) &&
			statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeEnvFlag) &&
			!statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeBootFlag) {
			nodes = append(nodes, bkenode.ToNode())
		}
	}
	return nodes
}

// GetRemoteNodeByBKENode retrieves a Kubernetes node by BKE node information.
func GetRemoteNodeByBKENode(ctx context.Context, clientSet *kubernetes.Clientset, node confv1beta1.Node) (*corev1.Node, error) {
	return clientSet.CoreV1().Nodes().Get(ctx, node.Hostname, metav1.GetOptions{})
}

// GetClientURLByIP formats an etcd client URL from an IP address.
func GetClientURLByIP(ip string) string {
	return fmt.Sprintf("https://%s:2379", ip)
}

// NeedUpgrade could compare version with of.x
func NeedUpgrade(oldV, newV string) bool {
	oldBase, oldOf := splitVersion(oldV)
	newBase, newOf := splitVersion(newV)

	oldVer, err1 := hashversion.NewVersion(oldBase)
	newVer, err2 := hashversion.NewVersion(newBase)
	if err1 != nil || err2 != nil {
		return false
	}

	cmp := newVer.Compare(oldVer)
	if cmp > 0 {
		return true
	} else if cmp < 0 {
		return false
	}

	return newOf > oldOf
}

func splitVersion(v string) (string, int) {
	if strings.HasPrefix(v, "v") {
		v = v[1:]
	}

	ofIndex := strings.Index(v, "-of.")
	if ofIndex == -1 {
		return "v" + v, 0
	}

	baseVersion := "v" + v[:ofIndex]
	ofStr := v[ofIndex+len("-of."):]

	ofNum := 0
	if num, err := strconv.Atoi(ofStr); err == nil {
		ofNum = num
	}

	return baseVersion, ofNum
}

// GetNeedUpgradeEtcds returns etcds that need to be upgraded.
// Note: This function uses GetBKENodesFromNodesStatus which requires local kubeconfig.
// In controller context, use GetNeedUpgradeEtcdsWithBKENodes instead.
func GetNeedUpgradeEtcds(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	return compareVersionAndGetNodes(
		bkeCluster,
		bkeCluster.Status.EtcdVersion,
		bkeCluster.Spec.ClusterConfig.Cluster.EtcdVersion,
	).Etcd()
}

// GetNeedUpgradeEtcdsWithBKENodes returns etcds that need to be upgraded using pre-fetched BKENodes.
// Use this in controller context where local kubeconfig is not available.
func GetNeedUpgradeEtcdsWithBKENodes(bkeCluster *bkev1beta1.BKECluster, bkeNodes bkev1beta1.BKENodes) bkenode.Nodes {
	if bkeCluster.Spec.ClusterConfig == nil {
		return nil
	}
	if !NeedUpgrade(bkeCluster.Status.EtcdVersion, bkeCluster.Spec.ClusterConfig.Cluster.EtcdVersion) {
		return nil
	}
	return filterNonFailedNodes(GetBKENodesFromNodesStatusWithBKENodes(bkeNodes)).Etcd()
}

func filterNonFailedNodes(statusNodes bkev1beta1.BKENodes) bkenode.Nodes {
	var nodes bkenode.Nodes
	for _, bkenode := range statusNodes {
		if statusNodes.GetNodeStateFlag(bkenode.Spec.IP, bkev1beta1.NodeFailedFlag) {
			continue
		}
		nodes = append(nodes, bkenode.ToNode())
	}
	return nodes
}

// compareVersionAndGetNodes is a helper function to compare versions and return nodes that need upgrading.
// this func is used to compare version like vx.y.z // vx.y // vx.y-rc.z // vx.y.z-of.a
// Note: This function uses GetBKENodesFromNodesStatus which requires local kubeconfig.
// In controller context, use compareVersionAndGetNodesWithBKENodes instead.
func compareVersionAndGetNodes(
	bkeCluster *bkev1beta1.BKECluster,
	oldVersionStr, newVersionStr string,
) bkenode.Nodes {
	if !NeedUpgrade(oldVersionStr, newVersionStr) {
		return nil
	}
	return filterNonFailedNodes(GetBKENodesFromNodesStatus(bkeCluster))
}

// compareVersionAndGetNodesWithBKENodes is a helper function to compare versions and return nodes that need upgrading.
// Uses pre-fetched BKENodes for controller context where local kubeconfig is not available.
func compareVersionAndGetNodesWithBKENodes(
	bkeCluster *bkev1beta1.BKECluster,
	oldVersionStr, newVersionStr string,
	bkeNodes bkev1beta1.BKENodes,
) bkenode.Nodes {
	if !NeedUpgrade(oldVersionStr, newVersionStr) {
		return nil
	}
	return filterNonFailedNodes(GetBKENodesFromNodesStatusWithBKENodes(bkeNodes))
}

func GetNodesFromCluster(bkeCluster *bkev1beta1.BKECluster) bkenode.Nodes {
	bkeNodes := bkenode.Nodes{}
	nodesData, err := cluster.GetNodesData(bkeCluster.Namespace, bkeCluster.Name)
	if err != nil {
		// Return empty nodes - phases using this in NeedExecute should use NodeFetcher instead
		log.Debugf("GetNodesFromCluster failed (expected in controller context): %v", err)
		return bkeNodes
	}
	return nodesData
}

func GetBKENodesFromCluster(bkeCluster *bkev1beta1.BKECluster) bkev1beta1.BKENodes {
	nodes := GetNodesFromCluster(bkeCluster)
	return bkenode.ConvertNodesToBKENodes(nodes, bkeCluster.Namespace, bkeCluster.Name)
}

func GetNodeStateFlag(node *confv1beta1.BKENode, nodeIP string, flag int) bool {
	if node.Spec.IP == nodeIP {
		return node.Status.StateCode&flag != 0
	}
	return false
}

// HasNodesNeedingPhase checks if any nodes need a specific phase by checking StateCode flag.
// Returns true if at least one node lacks the specified flag and is not failed/deleting/skipped.
func HasNodesNeedingPhase(bkeNodes bkev1beta1.BKENodes, flag int) bool {
	for _, bn := range bkeNodes {
		// Check if node doesn't have the required flag set
		if bn.Status.StateCode&flag == 0 {
			// Also check not failed, not deleting, not need skip
			if bn.Status.StateCode&bkev1beta1.NodeFailedFlag == 0 &&
				bn.Status.StateCode&bkev1beta1.NodeDeletingFlag == 0 &&
				!bn.Status.NeedSkip {
				return true
			}
		}
	}
	return false
}
