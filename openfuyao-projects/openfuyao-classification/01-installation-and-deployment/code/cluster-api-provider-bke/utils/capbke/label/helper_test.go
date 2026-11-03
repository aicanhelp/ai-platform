/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package label

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

type mockObject struct {
	client.Object
	labels map[string]string
}

func (m *mockObject) GetLabels() map[string]string {
	return m.labels
}

func (m *mockObject) SetLabels(labels map[string]string) {
	m.labels = labels
}

func TestSetLabel(t *testing.T) {
	obj := &mockObject{}
	SetLabel(obj, "test-key", "test-value")
	if obj.labels["test-key"] != "test-value" {
		t.Errorf("expected test-value, got %s", obj.labels["test-key"])
	}
}

func TestSetLabel_NilLabels(t *testing.T) {
	obj := &mockObject{labels: nil}
	SetLabel(obj, "test-key", "test-value")
	if obj.labels == nil {
		t.Error("labels should not be nil after SetLabel")
	}
	if obj.labels["test-key"] != "test-value" {
		t.Errorf("expected test-value, got %s", obj.labels["test-key"])
	}
}

func TestSetBKEMachineLabel_MasterNode(t *testing.T) {
	obj := &mockObject{}
	SetBKEMachineLabel(obj, node.MasterNodeRole, "master-node-1")
	if obj.labels[MasterNodeHost] != "master-node-1" {
		t.Errorf("expected master-node-1, got %s", obj.labels[MasterNodeHost])
	}
}

func TestSetBKEMachineLabel_WorkerNode(t *testing.T) {
	obj := &mockObject{}
	SetBKEMachineLabel(obj, node.WorkerNodeRole, "worker-node-1")
	if obj.labels[WorkerNodeHost] != "worker-node-1" {
		t.Errorf("expected worker-node-1, got %s", obj.labels[WorkerNodeHost])
	}
}

func TestCheckBKEMachineLabel_WithMasterLabel(t *testing.T) {
	obj := &mockObject{labels: map[string]string{MasterNodeHost: "master-node-1"}}
	value, ok := CheckBKEMachineLabel(obj)
	if !ok {
		t.Error("expected true, got false")
	}
	if value != "master-node-1" {
		t.Errorf("expected master-node-1, got %s", value)
	}
}

func TestCheckBKEMachineLabel_WithWorkerLabel(t *testing.T) {
	obj := &mockObject{labels: map[string]string{WorkerNodeHost: "worker-node-1"}}
	value, ok := CheckBKEMachineLabel(obj)
	if !ok {
		t.Error("expected true, got false")
	}
	if value != "worker-node-1" {
		t.Errorf("expected worker-node-1, got %s", value)
	}
}

func TestCheckBKEMachineLabel_NoLabel(t *testing.T) {
	obj := &mockObject{labels: nil}
	_, ok := CheckBKEMachineLabel(obj)
	if ok {
		t.Error("expected false, got true")
	}
}

func TestCheckBKEMachineLabel_EmptyLabels(t *testing.T) {
	obj := &mockObject{labels: map[string]string{}}
	_, ok := CheckBKEMachineLabel(obj)
	if ok {
		t.Error("expected false, got true")
	}
}

func TestRemoveBKEMachineLabel_MasterNode(t *testing.T) {
	obj := &mockObject{labels: map[string]string{MasterNodeHost: "master-node-1"}}
	RemoveBKEMachineLabel(obj, node.MasterNodeRole)
	if _, ok := obj.labels[MasterNodeHost]; ok {
		t.Error("label should be removed")
	}
}

func TestRemoveBKEMachineLabel_WorkerNode(t *testing.T) {
	obj := &mockObject{labels: map[string]string{WorkerNodeHost: "worker-node-1"}}
	RemoveBKEMachineLabel(obj, node.WorkerNodeRole)
	if _, ok := obj.labels[WorkerNodeHost]; ok {
		t.Error("label should be removed")
	}
}

func TestRemoveBKEMachineLabel_NilLabels(t *testing.T) {
	obj := &mockObject{labels: nil}
	RemoveBKEMachineLabel(obj, node.MasterNodeRole)
}

func TestIsMasterNode(t *testing.T) {
	node := &corev1.Node{}
	node.Labels = map[string]string{NodeRoleMasterLabel: ""}
	if !IsMasterNode(node) {
		t.Error("expected true, got false")
	}
}

func TestIsMasterNode_False(t *testing.T) {
	node := &corev1.Node{}
	if IsMasterNode(node) {
		t.Error("expected false, got true")
	}
}

func TestIsWorkerNode(t *testing.T) {
	node := &corev1.Node{}
	node.Labels = map[string]string{NodeRoleNodeLabel: ""}
	if !IsWorkerNode(node) {
		t.Error("expected true, got false")
	}
}

func TestIsWorkerNode_False(t *testing.T) {
	node := &corev1.Node{}
	if IsWorkerNode(node) {
		t.Error("expected false, got true")
	}
}

func TestSetMasterRoleLabel(t *testing.T) {
	node := &corev1.Node{}
	SetMasterRoleLabel(node)
	if node.Labels[NodeRoleMasterLabel] != "" {
		t.Errorf("expected empty value, got %s", node.Labels[NodeRoleMasterLabel])
	}
	if node.Labels[NodeRoleControlPlaneLabelKey] != NodeRoleControlPlaneLabelValue {
		t.Errorf("expected %s, got %s", NodeRoleControlPlaneLabelValue, node.Labels[NodeRoleControlPlaneLabelKey])
	}
}

func TestSetMasterRoleLabel_ExistingLabel(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{NodeRoleMasterLabel: "existing"},
		},
	}
	SetMasterRoleLabel(node)
	if node.Labels[NodeRoleMasterLabel] != "existing" {
		t.Error("existing label should not be overwritten")
	}
}

func TestSetWorkerRoleLabel(t *testing.T) {
	node := &corev1.Node{}
	SetWorkerRoleLabel(node)
	if node.Labels[NodeRoleNodeLabel] != "" {
		t.Errorf("expected empty value, got %s", node.Labels[NodeRoleNodeLabel])
	}
	if node.Labels[NodeRoleWorkerLabelKey] != NodeRoleWorkerLabelValue {
		t.Errorf("expected %s, got %s", NodeRoleWorkerLabelValue, node.Labels[NodeRoleWorkerLabelKey])
	}
}

func TestSetWorkerRoleLabel_ExistingLabel(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{NodeRoleNodeLabel: "existing"},
		},
	}
	SetWorkerRoleLabel(node)
	if node.Labels[NodeRoleNodeLabel] != "existing" {
		t.Error("existing label should not be overwritten")
	}
}

func TestSetBocVersionLabel(t *testing.T) {
	node := &corev1.Node{}
	SetBocVersionLabel(node, "v1.0.0")
	if node.Labels[BocVersionLabelKey] != "v1.0.0" {
		t.Errorf("expected v1.0.0, got %s", node.Labels[BocVersionLabelKey])
	}
}

func TestSetBocVersionLabel_UpdateExisting(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{BocVersionLabelKey: "v0.9.0"},
		},
	}
	SetBocVersionLabel(node, "v1.0.0")
	if node.Labels[BocVersionLabelKey] != "v1.0.0" {
		t.Errorf("expected v1.0.0, got %s", node.Labels[BocVersionLabelKey])
	}
}

func TestSetBocVersionLabel_SameVersion(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{BocVersionLabelKey: "v1.0.0"},
		},
	}
	SetBocVersionLabel(node, "v1.0.0")
	if node.Labels[BocVersionLabelKey] != "v1.0.0" {
		t.Errorf("expected v1.0.0, got %s", node.Labels[BocVersionLabelKey])
	}
}

func TestHasLabel(t *testing.T) {
	obj := &mockObject{labels: map[string]string{"test-key": "test-value"}}
	if !HasLabel(obj, "test-key") {
		t.Error("expected true, got false")
	}
}

func TestHasLabel_NotFound(t *testing.T) {
	obj := &mockObject{labels: map[string]string{}}
	if HasLabel(obj, "test-key") {
		t.Error("expected false, got true")
	}
}

func TestHasLabel_NilLabels(t *testing.T) {
	obj := &mockObject{labels: nil}
	if HasLabel(obj, "test-key") {
		t.Error("expected false, got true")
	}
}

func TestIsLabelEqual(t *testing.T) {
	obj := &mockObject{labels: map[string]string{"test-key": "test-value"}}
	if !IsLabelEqual(obj, "test-key", "test-value") {
		t.Error("expected true, got false")
	}
}

func TestIsLabelEqual_NotEqual(t *testing.T) {
	obj := &mockObject{labels: map[string]string{"test-key": "other-value"}}
	if IsLabelEqual(obj, "test-key", "test-value") {
		t.Error("expected false, got true")
	}
}

func TestIsLabelEqual_KeyNotFound(t *testing.T) {
	obj := &mockObject{labels: map[string]string{}}
	if IsLabelEqual(obj, "test-key", "test-value") {
		t.Error("expected false, got true")
	}
}

func TestIsLabelEqual_NilLabels(t *testing.T) {
	obj := &mockObject{labels: nil}
	if IsLabelEqual(obj, "test-key", "test-value") {
		t.Error("expected false, got true")
	}
}
