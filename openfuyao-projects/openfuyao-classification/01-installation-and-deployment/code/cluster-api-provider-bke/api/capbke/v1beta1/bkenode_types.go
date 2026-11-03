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

package v1beta1

import (
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

// BKENodes is a wrapper for []confv1beta1.BKENode that provides helper methods
type BKENodes []confv1beta1.BKENode

// NewBKENodes creates a BKENodes wrapper from a slice of BKENode
func NewBKENodes(nodes []confv1beta1.BKENode) BKENodes {
	if len(nodes) == 0 {
		return BKENodes{}
	}
	return BKENodes(nodes)
}

// NewBKENodesFromList creates a BKENodes wrapper from a BKENodeList
func NewBKENodesFromList(nodeList *confv1beta1.BKENodeList) BKENodes {
	if nodeList == nil {
		return BKENodes{}
	}
	return BKENodes(nodeList.Items)
}

// GetNodeByIP returns the BKENode with the specified IP
func (nodes BKENodes) GetNodeByIP(nodeIP string) *confv1beta1.BKENode {
	for i := range nodes {
		if nodes[i].Spec.IP == nodeIP {
			return &nodes[i]
		}
	}
	return nil
}

// GetNodeState returns the state of the node with the specified IP
func (nodes BKENodes) GetNodeState(nodeIP string) confv1beta1.NodeState {
	for _, n := range nodes {
		if n.Spec.IP == nodeIP {
			return n.Status.State
		}
	}
	return ""
}

// SetNodeStateWithMessage sets the state and message for the node with the specified IP
func (nodes BKENodes) SetNodeStateWithMessage(nodeIP string, state confv1beta1.NodeState, msg string) {
	for i := range nodes {
		if nodes[i].Spec.IP == nodeIP {
			nodes[i].Status.State = state
			nodes[i].Status.Message = msg
			nodes[i].Status.StateCode |= NodeStateNeedRecord
			break
		}
	}
}

// SetSkipNodeError marks a worker node as needing to be skipped
func (nodes BKENodes) SetSkipNodeError(nodeIP string) {
	for i := range nodes {
		if nodes[i].Spec.IP == nodeIP {
			// just set worker node needSkip
			if utils.ContainsString(nodes[i].Spec.Role, node.WorkerNodeRole) {
				nodes[i].Status.NeedSkip = true
			}
			break
		}
	}
}

// SetNodeState sets the state for the node with the specified IP
func (nodes BKENodes) SetNodeState(nodeIP string, state confv1beta1.NodeState) {
	for i := range nodes {
		if nodes[i].Spec.IP == nodeIP {
			nodes[i].Status.State = state
			nodes[i].Status.StateCode |= NodeStateNeedRecord
			break
		}
	}
}

// SetNodeStateMessage sets the message for the node with the specified IP
func (nodes BKENodes) SetNodeStateMessage(nodeIP string, msg string) {
	for i := range nodes {
		if nodes[i].Spec.IP == nodeIP {
			nodes[i].Status.Message = msg
			break
		}
	}
}

// RemoveNode removes the node with the specified IP from the slice
func (nodes *BKENodes) RemoveNode(nodeIP string) {
	foundIndex := -1
	for i, n := range *nodes {
		if n.Spec.IP == nodeIP {
			foundIndex = i
			break
		}
	}
	if foundIndex != -1 {
		*nodes = append((*nodes)[:foundIndex], (*nodes)[foundIndex+1:]...)
	}
}

// MarkNodeStateFlag sets a flag on the node's StateCode
func (nodes BKENodes) MarkNodeStateFlag(nodeIP string, flag int) {
	for i := range nodes {
		if nodes[i].Spec.IP == nodeIP {
			nodes[i].Status.StateCode |= flag
			break
		}
	}
}

// UnmarkNodeStateFlag clears a flag from the node's StateCode
func (nodes BKENodes) UnmarkNodeStateFlag(nodeIP string, flag int) {
	for i := range nodes {
		if nodes[i].Spec.IP == nodeIP {
			nodes[i].Status.StateCode &= ^flag
			break
		}
	}
}

// GetNodeStateFlag checks if a flag is set on the node's StateCode
func (nodes BKENodes) GetNodeStateFlag(nodeIP string, flag int) bool {
	for _, n := range nodes {
		if n.Spec.IP == nodeIP {
			return n.Status.StateCode&flag != 0
		}
	}
	return false
}

// GetNodeStateNeedSkip returns whether the node should be skipped
func (nodes BKENodes) GetNodeStateNeedSkip(nodeIP string) bool {
	for _, n := range nodes {
		if n.Spec.IP == nodeIP {
			return n.Status.NeedSkip
		}
	}
	return false
}

// SetNodeStateNeedSkip sets the NeedSkip flag for the node
func (nodes BKENodes) SetNodeStateNeedSkip(nodeIP string, needSkip bool) {
	for i := range nodes {
		if nodes[i].Spec.IP == nodeIP {
			nodes[i].Status.NeedSkip = needSkip
			break
		}
	}
}

// ToNodes converts BKENodes to the legacy node.Nodes type
func (nodes BKENodes) ToNodes() node.Nodes {
	return node.ConvertBKENodesToNodes(nodes)
}

// Length returns the number of nodes
func (nodes BKENodes) Length() int {
	return len(nodes)
}

// DeepCopy creates a deep copy of BKENodes
func (nodes BKENodes) DeepCopy() BKENodes {
	if len(nodes) == 0 {
		return nil
	}
	out := make(BKENodes, len(nodes))
	for i := range nodes {
		nodes[i].DeepCopyInto(&out[i])
	}
	return out
}

// GetModifiedNodes returns a list of nodes that have the NodeStateNeedRecord flag set
// These nodes need to be updated in the API server
func (nodes BKENodes) GetModifiedNodes() []confv1beta1.BKENode {
	var modified []confv1beta1.BKENode
	for _, n := range nodes {
		if n.Status.StateCode&NodeStateNeedRecord != 0 {
			modified = append(modified, n)
		}
	}
	return modified
}

// ClearRecordFlags clears the NodeStateNeedRecord flag from all nodes
func (nodes BKENodes) ClearRecordFlags() {
	for i := range nodes {
		nodes[i].Status.StateCode &= ^NodeStateNeedRecord
	}
}

func init() {
	SchemeBuilder.Register(&confv1beta1.BKENode{}, &confv1beta1.BKENodeList{})
}
