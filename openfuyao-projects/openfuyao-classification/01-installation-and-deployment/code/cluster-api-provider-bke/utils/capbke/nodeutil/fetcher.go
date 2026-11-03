/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package nodeutil

import (
	"context"

	"github.com/pkg/errors"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

const (
	// ClusterNameLabel is the label key used to associate BKENodes with a BKECluster
	ClusterNameLabel = "cluster.x-k8s.io/cluster-name"
)

// NodeFetcher provides methods to fetch BKENode resources from the cluster.
// It uses controller-runtime client which benefits from informer caching.
type NodeFetcher struct {
	client client.Client
}

// NewNodeFetcher creates a new NodeFetcher instance.
func NewNodeFetcher(c client.Client) *NodeFetcher {
	return &NodeFetcher{client: c}
}

// FetchResult contains the result of fetching BKENodes for a cluster.
type FetchResult struct {
	// BKENodes is the list of raw BKENode CRD objects
	BKENodes []confv1beta1.BKENode
	// Nodes is the converted Nodes type for business logic use
	Nodes bkenode.Nodes
}

// FetchNodesForCluster fetches all BKENodes associated with a BKECluster.
// It uses the cluster.x-k8s.io/cluster-name label to find associated nodes.
// Returns both the raw BKENode objects and the converted Nodes for flexibility.
func (f *NodeFetcher) FetchNodesForCluster(ctx context.Context, namespace, clusterName string) (*FetchResult, error) {
	bkeNodeList := &confv1beta1.BKENodeList{}

	if err := f.client.List(ctx, bkeNodeList,
		client.InNamespace(namespace),
		client.MatchingLabels{ClusterNameLabel: clusterName},
	); err != nil {
		return nil, errors.Wrapf(err, "failed to list BKENodes for cluster %s/%s", namespace, clusterName)
	}

	// Convert to Nodes type and apply defaults
	nodes := bkenode.ConvertBKENodeListToNodes(bkeNodeList)
	nodes = bkenode.SetDefaultsForNodes(nodes)

	return &FetchResult{
		BKENodes: bkeNodeList.Items,
		Nodes:    nodes,
	}, nil
}

// FetchNodesForBKECluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) FetchNodesForBKECluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (*FetchResult, error) {
	return f.FetchNodesForCluster(ctx, bkeCluster.Namespace, bkeCluster.Name)
}

// GetNodes fetches and returns only the Nodes (business model) for a cluster.
// This is the most commonly used method for existing code that expects bkenode.Nodes.
func (f *NodeFetcher) GetNodes(ctx context.Context, namespace, clusterName string) (bkenode.Nodes, error) {
	result, err := f.FetchNodesForCluster(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}
	return result.Nodes, nil
}

// GetNodesForBKECluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) GetNodesForBKECluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
	return f.GetNodes(ctx, bkeCluster.Namespace, bkeCluster.Name)
}

// GetBKENodes fetches and returns only the raw BKENode CRD objects.
// Useful when you need to access or update the BKENode resources directly.
func (f *NodeFetcher) GetBKENodes(ctx context.Context, namespace, clusterName string) ([]confv1beta1.BKENode, error) {
	result, err := f.FetchNodesForCluster(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}
	return result.BKENodes, nil
}

// GetBKENodesForBKECluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) GetBKENodesForBKECluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) ([]confv1beta1.BKENode, error) {
	return f.GetBKENodes(ctx, bkeCluster.Namespace, bkeCluster.Name)
}

// GetNodeByIP finds a specific node by its IP address.
func (f *NodeFetcher) GetNodeByIP(ctx context.Context, namespace, clusterName, ip string) (*confv1beta1.BKENode, error) {
	nodes, err := f.GetBKENodes(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}
	for i := range nodes {
		if nodes[i].Spec.IP == ip {
			return &nodes[i], nil
		}
	}
	return nil, errors.Errorf("node with IP %s not found in cluster %s/%s", ip, namespace, clusterName)
}

// UpdateNodeStatus updates the status of a BKENode.
func (f *NodeFetcher) UpdateNodeStatus(ctx context.Context, bkeNode *confv1beta1.BKENode) error {
	return f.client.Status().Update(ctx, bkeNode)
}

// GetNodesFromClient is a standalone function for use without creating a NodeFetcher instance.
// Useful in places where you don't want to manage a fetcher lifecycle.
func GetNodesFromClient(ctx context.Context, c client.Client, namespace, clusterName string) (bkenode.Nodes, error) {
	fetcher := NewNodeFetcher(c)
	return fetcher.GetNodes(ctx, namespace, clusterName)
}

// GetNodesForBKEClusterFromClient is a convenience function.
func GetNodesForBKEClusterFromClient(ctx context.Context, c client.Client, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
	return GetNodesFromClient(ctx, c, bkeCluster.Namespace, bkeCluster.Name)
}

// GetBKENodesFromClient returns raw BKENode objects.
func GetBKENodesFromClient(ctx context.Context, c client.Client, namespace, clusterName string) ([]confv1beta1.BKENode, error) {
	fetcher := NewNodeFetcher(c)
	return fetcher.GetBKENodes(ctx, namespace, clusterName)
}

// NodeStateInfo represents the state information extracted from a BKENode.
// This replaces the old confv1beta1.NodeState that was stored in BKECluster.Status.NodesStatus.
type NodeStateInfo struct {
	Node      confv1beta1.Node     // Use confv1beta1.Node for compatibility
	BKENode   *confv1beta1.BKENode // Reference to original BKENode for updates
	State     confv1beta1.NodeState
	StateCode int
	Message   string
	NeedSkip  bool
}

// GetNodeStates fetches BKENodes and extracts their state information.
// This replaces the pattern of accessing bkeCluster.Status.NodesStatus.
func (f *NodeFetcher) GetNodeStates(ctx context.Context, namespace, clusterName string) ([]NodeStateInfo, error) {
	bkeNodes, err := f.GetBKENodes(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}

	states := make([]NodeStateInfo, 0, len(bkeNodes))
	for i := range bkeNodes {
		bkeNode := &bkeNodes[i]
		states = append(states, NodeStateInfo{
			Node:      bkeNode.ToNode(),
			BKENode:   bkeNode,
			State:     bkeNode.Status.State,
			StateCode: bkeNode.Status.StateCode,
			Message:   bkeNode.Status.Message,
			NeedSkip:  bkeNode.Status.NeedSkip,
		})
	}
	return states, nil
}

// GetNodeStatesForBKECluster is a convenience method.
func (f *NodeFetcher) GetNodeStatesForBKECluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) ([]NodeStateInfo, error) {
	return f.GetNodeStates(ctx, bkeCluster.Namespace, bkeCluster.Name)
}

// GetDeletingNodes returns nodes that have a deletionTimestamp set (being deleted).
// These are nodes where the user has deleted the BKENode resource but it hasn't been
// fully removed yet (e.g., waiting for finalizer to be removed after cleanup).
func (f *NodeFetcher) GetDeletingNodes(ctx context.Context, namespace, clusterName string) (bkenode.Nodes, error) {
	bkeNodes, err := f.GetBKENodes(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}

	var deletingNodes bkenode.Nodes
	for i := range bkeNodes {
		if bkeNodes[i].DeletionTimestamp != nil {
			deletingNodes = append(deletingNodes, bkeNodes[i].ToNode())
		}
	}
	return deletingNodes, nil
}

// GetDeletingNodesForBKECluster is a convenience method.
func (f *NodeFetcher) GetDeletingNodesForBKECluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
	return f.GetDeletingNodes(ctx, bkeCluster.Namespace, bkeCluster.Name)
}

// FilterNodesByState filters nodes by their state.
func FilterNodesByState(states []NodeStateInfo, targetState confv1beta1.NodeState) []NodeStateInfo {
	var filtered []NodeStateInfo
	for _, s := range states {
		if s.State == targetState {
			filtered = append(filtered, s)
		}
	}
	return filtered
}

// GetReadyNodes returns nodes that are in Ready state.
func (f *NodeFetcher) GetReadyNodes(ctx context.Context, namespace, clusterName string) (bkenode.Nodes, error) {
	states, err := f.GetNodeStates(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}

	var nodes bkenode.Nodes
	for _, s := range states {
		if s.State == confv1beta1.NodeReady {
			nodes = append(nodes, s.Node)
		}
	}
	return nodes, nil
}

// GetNodesExcludingSkipped returns nodes that are not marked as NeedSkip.
// This replaces phaseutil.GetBKENodesFromNodesStatus which filtered out skipped nodes.
func (f *NodeFetcher) GetNodesExcludingSkipped(ctx context.Context, namespace, clusterName string) (bkenode.Nodes, error) {
	states, err := f.GetNodeStates(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}

	var nodes bkenode.Nodes
	for _, s := range states {
		if !s.NeedSkip {
			nodes = append(nodes, s.Node)
		}
	}
	return nodes, nil
}

// GetAllNodes returns all nodes including skipped ones.
// This replaces phaseutil.GetBKEAllNodesFromNodesStatus.
func (f *NodeFetcher) GetAllNodes(ctx context.Context, namespace, clusterName string) (bkenode.Nodes, error) {
	return f.GetNodes(ctx, namespace, clusterName)
}

// GetReadyBootstrapNodes returns nodes that are ready for bootstrap operations.
// This replaces phaseutil.GetReadyBootstrapNodes.
func (f *NodeFetcher) GetReadyBootstrapNodes(ctx context.Context, namespace, clusterName string) (bkenode.Nodes, error) {
	bkeNodes, err := f.GetBKENodes(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}

	bkeNodesList := bkev1beta1.BKENodes(bkeNodes)
	var nodes bkenode.Nodes
	for _, bkeNode := range bkeNodes {
		// agent is ready and env is ready and boot is not
		if bkeNodesList.GetNodeStateFlag(bkeNode.Spec.IP, bkev1beta1.NodeAgentReadyFlag) &&
			bkeNodesList.GetNodeStateFlag(bkeNode.Spec.IP, bkev1beta1.NodeEnvFlag) &&
			!bkeNodesList.GetNodeStateFlag(bkeNode.Spec.IP, bkev1beta1.NodeBootFlag) {
			nodes = append(nodes, bkeNode.ToNode())
		}
	}
	return nodes, nil
}

// CompareNodes compares current nodes with desired nodes and returns differences.
// This helps with detecting node additions/removals.
func (f *NodeFetcher) CompareNodes(ctx context.Context, namespace, clusterName string, desiredNodes bkenode.Nodes) (added, removed bkenode.Nodes, err error) {
	currentNodes, err := f.GetNodes(ctx, namespace, clusterName)
	if err != nil {
		return nil, nil, err
	}

	// Find added nodes (in desired but not in current)
	for _, desired := range desiredNodes {
		found := false
		for _, current := range currentNodes {
			if current.IP == desired.IP {
				found = true
				break
			}
		}
		if !found {
			added = append(added, desired)
		}
	}

	// Find removed nodes (in current but not in desired)
	for _, current := range currentNodes {
		found := false
		for _, desired := range desiredNodes {
			if desired.IP == current.IP {
				found = true
				break
			}
		}
		if !found {
			removed = append(removed, current)
		}
	}

	return added, removed, nil
}

type nodeStateUpdate struct {
	namespace   string
	clusterName string
	ip          string
	state       confv1beta1.NodeState
	message     string
}

func (f *NodeFetcher) updateBKENodeState(ctx context.Context, update nodeStateUpdate) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		bkeNode, err := f.GetNodeByIP(ctx, update.namespace, update.clusterName, update.ip)
		if err != nil {
			return err
		}

		bkeNode.Status.State = update.state
		bkeNode.Status.Message = update.message

		return f.UpdateNodeStatus(ctx, bkeNode)
	})
}

// SetNodeNeedSkip marks a node as needing to be skipped.
func (f *NodeFetcher) SetNodeNeedSkip(ctx context.Context, namespace, clusterName, ip string, needSkip bool) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
		if err != nil {
			return err
		}

		bkeNode.Status.NeedSkip = needSkip
		return f.UpdateNodeStatus(ctx, bkeNode)
	})
}

// MarkNodeStateFlag sets a flag on the node's StateCode.
func (f *NodeFetcher) MarkNodeStateFlag(ctx context.Context, namespace, clusterName, ip string, flag int) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
		if err != nil {
			return err
		}

		bkeNode.Status.StateCode |= flag

		return f.UpdateNodeStatus(ctx, bkeNode)
	})
}

// UnmarkNodeStateFlag clears a flag from the node's StateCode.
func (f *NodeFetcher) UnmarkNodeStateFlag(ctx context.Context, namespace, clusterName, ip string, flag int) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
		if err != nil {
			return err
		}

		bkeNode.Status.StateCode &= ^flag

		return f.UpdateNodeStatus(ctx, bkeNode)
	})
}

// GetNodeStateNeedSkip returns whether the node should be skipped.
func (f *NodeFetcher) GetNodeStateNeedSkip(ctx context.Context, namespace, clusterName, ip string) (bool, error) {
	bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
	if err != nil {
		// If node not found, don't skip (conservative approach)
		return false, err
	}
	return bkeNode.Status.NeedSkip, nil
}

// GetBKENodesWrapper returns a BKENodes wrapper for node state operations.
// This is useful when you need to perform multiple operations on nodes.
func (f *NodeFetcher) GetBKENodesWrapper(ctx context.Context, namespace, clusterName string) (bkev1beta1.BKENodes, error) {
	bkeNodes, err := f.GetBKENodes(ctx, namespace, clusterName)
	if err != nil {
		return nil, err
	}
	return bkev1beta1.NewBKENodes(bkeNodes), nil
}

// GetBKENodesWrapperForCluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) GetBKENodesWrapperForCluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, error) {
	return f.GetBKENodesWrapper(ctx, bkeCluster.Namespace, bkeCluster.Name)
}

// UpdateModifiedNodes updates all nodes that have been modified (have NodeStateNeedRecord flag).
func (f *NodeFetcher) UpdateModifiedNodes(ctx context.Context, nodes bkev1beta1.BKENodes) error {
	modifiedNodes := nodes.GetModifiedNodes()
	for i := range modifiedNodes {
		if err := f.UpdateNodeStatus(ctx, &modifiedNodes[i]); err != nil {
			return err
		}
	}
	nodes.ClearRecordFlags()
	return nil
}

// GetNodeCount returns the number of nodes in a cluster.
func (f *NodeFetcher) GetNodeCount(ctx context.Context, namespace, clusterName string) (int, error) {
	bkeNodes, err := f.GetBKENodes(ctx, namespace, clusterName)
	if err != nil {
		return 0, err
	}
	return len(bkeNodes), nil
}

// GetNodeCountForCluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) GetNodeCountForCluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (int, error) {
	return f.GetNodeCount(ctx, bkeCluster.Namespace, bkeCluster.Name)
}

// DeleteBKENode deletes a BKENode by IP.
// This is equivalent to the old RemoveNodeState behavior.
func (f *NodeFetcher) DeleteBKENode(ctx context.Context, namespace, clusterName, ip string) error {
	bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
	if err != nil {
		// If node not found, consider it already deleted
		return nil
	}

	return f.client.Delete(ctx, bkeNode)
}

// DeleteBKENodeForCluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) DeleteBKENodeForCluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster, ip string) error {
	return f.DeleteBKENode(ctx, bkeCluster.Namespace, bkeCluster.Name, ip)
}

// SetNodeStateWithMessageForCluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) SetNodeStateWithMessageForCluster(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	ip string,
	state confv1beta1.NodeState,
	message string,
) error {
	return f.updateBKENodeState(ctx, nodeStateUpdate{
		namespace: bkeCluster.Namespace, clusterName: bkeCluster.Name,
		ip: ip, state: state, message: message,
	})
}

// UpdateNodeStatusByIP atomically updates node status with conflict retry.
// The updater can mutate multiple status fields in one API update.
func (f *NodeFetcher) UpdateNodeStatusByIP(
	ctx context.Context,
	namespace, clusterName, ip string,
	updater func(status *confv1beta1.BKENodeStatus),
) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
		if err != nil {
			return err
		}
		updater(&bkeNode.Status)
		return f.UpdateNodeStatus(ctx, bkeNode)
	})
}

// UpdateNodeStatusByIPForCluster is a convenience method for UpdateNodeStatusByIP.
func (f *NodeFetcher) UpdateNodeStatusByIPForCluster(
	ctx context.Context,
	bkeCluster *bkev1beta1.BKECluster,
	ip string,
	updater func(status *confv1beta1.BKENodeStatus),
) error {
	return f.UpdateNodeStatusByIP(ctx, bkeCluster.Namespace, bkeCluster.Name, ip, updater)
}

// MarkNodeStateFlagForCluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) MarkNodeStateFlagForCluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster, ip string, flag int) error {
	return f.MarkNodeStateFlag(ctx, bkeCluster.Namespace, bkeCluster.Name, ip, flag)
}

// UnmarkNodeStateFlagForCluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) UnmarkNodeStateFlagForCluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster, ip string, flag int) error {
	return f.UnmarkNodeStateFlag(ctx, bkeCluster.Namespace, bkeCluster.Name, ip, flag)
}

// GetNodeStateFlag checks if a flag is set on the node's StateCode.
func (f *NodeFetcher) GetNodeStateFlag(ctx context.Context, namespace, clusterName, ip string, flag int) (bool, error) {
	bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
	if err != nil {
		return false, err
	}
	return bkeNode.Status.StateCode&flag != 0, nil
}

// GetNodeStateFlagForCluster is a convenience method that takes a BKECluster object.
func (f *NodeFetcher) GetNodeStateFlagForCluster(ctx context.Context, bkeCluster *bkev1beta1.BKECluster, ip string, flag int) (bool, error) {
	return f.GetNodeStateFlag(ctx, bkeCluster.Namespace, bkeCluster.Name, ip, flag)
}

// SetBKENodeStateMessage updates the state message of a BKENode by IP.
func (f *NodeFetcher) SetBKENodeStateMessage(ctx context.Context, namespace, clusterName, ip string, message string) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
		if err != nil {
			return err
		}

		bkeNode.Status.Message = message

		return f.UpdateNodeStatus(ctx, bkeNode)
	})
}

// SetSkipNodeErrorForWorker updates the needskip of a BKENode by IP if the node is a worker.
func (f *NodeFetcher) SetSkipNodeErrorForWorker(ctx context.Context, namespace, clusterName, ip string) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		bkeNode, err := f.GetNodeByIP(ctx, namespace, clusterName, ip)
		if err != nil {
			return err
		}
		if utils.ContainsString(bkeNode.Spec.Role, node.WorkerNodeRole) {
			bkeNode.Status.NeedSkip = true
		}
		return f.UpdateNodeStatus(ctx, bkeNode)
	})
}
