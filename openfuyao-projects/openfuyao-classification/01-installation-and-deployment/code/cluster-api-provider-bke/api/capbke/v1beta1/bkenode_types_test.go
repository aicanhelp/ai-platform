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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

func createTestBKENodes() BKENodes {
	return BKENodes{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node-1",
				Namespace: "test",
			},
			Spec: confv1beta1.BKENodeSpec{
				IP:       "192.168.1.1",
				Hostname: "node1",
				Role:     []string{node.MasterNodeRole, node.EtcdNodeRole},
			},
			Status: confv1beta1.BKENodeStatus{
				State:     NodeReady,
				StateCode: 0,
				Message:   "",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-node-2",
				Namespace: "test",
			},
			Spec: confv1beta1.BKENodeSpec{
				IP:       "192.168.1.2",
				Hostname: "node2",
				Role:     []string{node.WorkerNodeRole},
			},
			Status: confv1beta1.BKENodeStatus{
				State:     NodeNotReady,
				StateCode: 0,
				Message:   "",
			},
		},
	}
}

func TestNewBKENodes(t *testing.T) {
	t.Run("nil input", func(t *testing.T) {
		nodes := NewBKENodes(nil)
		if len(nodes) != 0 {
			t.Errorf("expected empty BKENodes, got %d nodes", len(nodes))
		}
	})

	t.Run("valid input", func(t *testing.T) {
		input := []confv1beta1.BKENode{
			{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}},
		}
		nodes := NewBKENodes(input)
		if len(nodes) != 1 {
			t.Errorf("expected 1 node, got %d", len(nodes))
		}
	})
}

func TestNewBKENodesFromList(t *testing.T) {
	t.Run("nil input", func(t *testing.T) {
		nodes := NewBKENodesFromList(nil)
		if len(nodes) != 0 {
			t.Errorf("expected empty BKENodes, got %d nodes", len(nodes))
		}
	})

	t.Run("valid input", func(t *testing.T) {
		list := &confv1beta1.BKENodeList{
			Items: []confv1beta1.BKENode{
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.1"}},
				{Spec: confv1beta1.BKENodeSpec{IP: "192.168.1.2"}},
			},
		}
		nodes := NewBKENodesFromList(list)
		if len(nodes) != 2 {
			t.Errorf("expected 2 nodes, got %d", len(nodes))
		}
	})
}

func TestBKENodes_GetNodeByIP(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("existing IP", func(t *testing.T) {
		n := nodes.GetNodeByIP("192.168.1.1")
		if n == nil {
			t.Error("expected to find node, got nil")
		}
		if n.Spec.Hostname != "node1" {
			t.Errorf("expected hostname 'node1', got '%s'", n.Spec.Hostname)
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		n := nodes.GetNodeByIP("192.168.1.100")
		if n != nil {
			t.Errorf("expected nil, got node %v", n)
		}
	})
}

func TestBKENodes_GetNodeState(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("existing IP", func(t *testing.T) {
		state := nodes.GetNodeState("192.168.1.1")
		if state != NodeReady {
			t.Errorf("expected state '%s', got '%s'", NodeReady, state)
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		state := nodes.GetNodeState("192.168.1.100")
		if state != "" {
			t.Errorf("expected empty state, got '%s'", state)
		}
	})
}

func TestBKENodes_SetNodeState(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("existing IP", func(t *testing.T) {
		nodes.SetNodeState("192.168.1.1", NodeUpgrading)
		n := nodes.GetNodeByIP("192.168.1.1")
		if n.Status.State != NodeUpgrading {
			t.Errorf("expected state '%s', got '%s'", NodeUpgrading, n.Status.State)
		}
		if n.Status.StateCode&NodeStateNeedRecord == 0 {
			t.Error("expected NodeStateNeedRecord flag to be set")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		// Should not panic
		nodes.SetNodeState("192.168.1.100", NodeUpgrading)
	})
}

func TestBKENodes_SetNodeStateWithMessage(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("existing IP", func(t *testing.T) {
		nodes.SetNodeStateWithMessage("192.168.1.1", confv1beta1.NodeFailed, "test error message")
		n := nodes.GetNodeByIP("192.168.1.1")
		if n.Status.State != confv1beta1.NodeFailed {
			t.Errorf("expected state '%s', got '%s'", confv1beta1.NodeFailed, n.Status.State)
		}
		if n.Status.Message != "test error message" {
			t.Errorf("expected message 'test error message', got '%s'", n.Status.Message)
		}
		if n.Status.StateCode&NodeStateNeedRecord == 0 {
			t.Error("expected NodeStateNeedRecord flag to be set")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		// Should not panic
		nodes.SetNodeStateWithMessage("192.168.1.100", confv1beta1.NodeFailed, "test")
	})
}

func TestBKENodes_SetNodeStateMessage(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("existing IP", func(t *testing.T) {
		nodes.SetNodeStateMessage("192.168.1.1", "new message")
		n := nodes.GetNodeByIP("192.168.1.1")
		if n.Status.Message != "new message" {
			t.Errorf("expected message 'new message', got '%s'", n.Status.Message)
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		// Should not panic
		nodes.SetNodeStateMessage("192.168.1.100", "new message")
	})
}

func TestBKENodes_SetSkipNodeError(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("worker node", func(t *testing.T) {
		nodes.SetSkipNodeError("192.168.1.2") // worker node
		n := nodes.GetNodeByIP("192.168.1.2")
		if !n.Status.NeedSkip {
			t.Error("expected NeedSkip to be true for worker node")
		}
	})

	t.Run("master node", func(t *testing.T) {
		nodes.SetSkipNodeError("192.168.1.1") // master node
		n := nodes.GetNodeByIP("192.168.1.1")
		if n.Status.NeedSkip {
			t.Error("expected NeedSkip to be false for master node")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		// Should not panic
		nodes.SetSkipNodeError("192.168.1.100")
	})
}

func TestBKENodes_RemoveNode(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("existing IP", func(t *testing.T) {
		originalLen := nodes.Length()
		nodes.RemoveNode("192.168.1.1")
		if nodes.Length() != originalLen-1 {
			t.Errorf("expected %d nodes, got %d", originalLen-1, nodes.Length())
		}
		if nodes.GetNodeByIP("192.168.1.1") != nil {
			t.Error("expected node to be removed")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		nodes := createTestBKENodes()
		originalLen := nodes.Length()
		nodes.RemoveNode("192.168.1.100")
		if nodes.Length() != originalLen {
			t.Errorf("expected %d nodes, got %d", originalLen, nodes.Length())
		}
	})
}

func TestBKENodes_MarkNodeStateFlag(t *testing.T) {
	nodes := createTestBKENodes()
	testFlag := NodeAgentPushedFlag

	t.Run("existing IP", func(t *testing.T) {
		nodes.MarkNodeStateFlag("192.168.1.1", testFlag)
		n := nodes.GetNodeByIP("192.168.1.1")
		if n.Status.StateCode&testFlag == 0 {
			t.Error("expected flag to be set")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		// Should not panic
		nodes.MarkNodeStateFlag("192.168.1.100", testFlag)
	})
}

func TestBKENodes_UnmarkNodeStateFlag(t *testing.T) {
	nodes := createTestBKENodes()
	testFlag := NodeAgentPushedFlag

	// First set the flag
	nodes.MarkNodeStateFlag("192.168.1.1", testFlag)

	t.Run("existing IP", func(t *testing.T) {
		nodes.UnmarkNodeStateFlag("192.168.1.1", testFlag)
		n := nodes.GetNodeByIP("192.168.1.1")
		if n.Status.StateCode&testFlag != 0 {
			t.Error("expected flag to be cleared")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		// Should not panic
		nodes.UnmarkNodeStateFlag("192.168.1.100", testFlag)
	})
}

func TestBKENodes_GetNodeStateFlag(t *testing.T) {
	nodes := createTestBKENodes()
	testFlag := NodeAgentPushedFlag

	t.Run("flag not set", func(t *testing.T) {
		result := nodes.GetNodeStateFlag("192.168.1.1", testFlag)
		if result {
			t.Error("expected false when flag is not set")
		}
	})

	t.Run("flag set", func(t *testing.T) {
		nodes.MarkNodeStateFlag("192.168.1.1", testFlag)
		result := nodes.GetNodeStateFlag("192.168.1.1", testFlag)
		if !result {
			t.Error("expected true when flag is set")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		result := nodes.GetNodeStateFlag("192.168.1.100", testFlag)
		if result {
			t.Error("expected false for non-existing IP")
		}
	})
}

func TestBKENodes_GetNodeStateNeedSkip(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("NeedSkip false", func(t *testing.T) {
		result := nodes.GetNodeStateNeedSkip("192.168.1.1")
		if result {
			t.Error("expected false")
		}
	})

	t.Run("NeedSkip true", func(t *testing.T) {
		nodes.SetNodeStateNeedSkip("192.168.1.1", true)
		result := nodes.GetNodeStateNeedSkip("192.168.1.1")
		if !result {
			t.Error("expected true")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		result := nodes.GetNodeStateNeedSkip("192.168.1.100")
		if result {
			t.Error("expected false for non-existing IP")
		}
	})
}

func TestBKENodes_SetNodeStateNeedSkip(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("set to true", func(t *testing.T) {
		nodes.SetNodeStateNeedSkip("192.168.1.1", true)
		n := nodes.GetNodeByIP("192.168.1.1")
		if !n.Status.NeedSkip {
			t.Error("expected NeedSkip to be true")
		}
	})

	t.Run("set to false", func(t *testing.T) {
		nodes.SetNodeStateNeedSkip("192.168.1.1", false)
		n := nodes.GetNodeByIP("192.168.1.1")
		if n.Status.NeedSkip {
			t.Error("expected NeedSkip to be false")
		}
	})

	t.Run("non-existing IP", func(t *testing.T) {
		// Should not panic
		nodes.SetNodeStateNeedSkip("192.168.1.100", true)
	})
}

func TestBKENodes_ToNodes(t *testing.T) {
	nodes := createTestBKENodes()
	legacyNodes := nodes.ToNodes()

	if len(legacyNodes) != len(nodes) {
		t.Errorf("expected %d nodes, got %d", len(nodes), len(legacyNodes))
	}

	if legacyNodes[0].IP != nodes[0].Spec.IP {
		t.Errorf("expected IP '%s', got '%s'", nodes[0].Spec.IP, legacyNodes[0].IP)
	}
}

func TestBKENodes_Length(t *testing.T) {
	nodes := createTestBKENodes()
	if nodes.Length() != 2 {
		t.Errorf("expected 2, got %d", nodes.Length())
	}

	emptyNodes := BKENodes{}
	if emptyNodes.Length() != 0 {
		t.Errorf("expected 0, got %d", emptyNodes.Length())
	}
}

func TestBKENodes_DeepCopy(t *testing.T) {
	nodes := createTestBKENodes()
	copied := nodes.DeepCopy()

	if len(copied) != len(nodes) {
		t.Errorf("expected %d nodes, got %d", len(nodes), len(copied))
	}

	// Modify original and verify copy is unchanged
	nodes.SetNodeState("192.168.1.1", confv1beta1.NodeFailed)
	if copied.GetNodeState("192.168.1.1") == confv1beta1.NodeFailed {
		t.Error("expected deep copy to be independent")
	}

	// Test nil case
	var nilNodes BKENodes
	nilCopied := nilNodes.DeepCopy()
	if nilCopied != nil {
		t.Error("expected nil for nil input")
	}
}

func TestBKENodes_GetModifiedNodes(t *testing.T) {
	nodes := createTestBKENodes()

	t.Run("no modified nodes", func(t *testing.T) {
		modified := nodes.GetModifiedNodes()
		if len(modified) != 0 {
			t.Errorf("expected 0 modified nodes, got %d", len(modified))
		}
	})

	t.Run("with modified nodes", func(t *testing.T) {
		nodes.SetNodeState("192.168.1.1", NodeUpgrading) // This sets NodeStateNeedRecord
		modified := nodes.GetModifiedNodes()
		if len(modified) != 1 {
			t.Errorf("expected 1 modified node, got %d", len(modified))
		}
		if modified[0].Spec.IP != "192.168.1.1" {
			t.Errorf("expected IP '192.168.1.1', got '%s'", modified[0].Spec.IP)
		}
	})
}

func TestBKENodes_ClearRecordFlags(t *testing.T) {
	nodes := createTestBKENodes()

	// Set record flags on some nodes
	nodes.SetNodeState("192.168.1.1", NodeUpgrading)
	nodes.SetNodeState("192.168.1.2", confv1beta1.NodeFailed)

	// Verify flags are set
	if len(nodes.GetModifiedNodes()) != 2 {
		t.Error("expected 2 modified nodes before clearing")
	}

	// Clear flags
	nodes.ClearRecordFlags()

	// Verify flags are cleared
	if len(nodes.GetModifiedNodes()) != 0 {
		t.Error("expected 0 modified nodes after clearing")
	}
}
