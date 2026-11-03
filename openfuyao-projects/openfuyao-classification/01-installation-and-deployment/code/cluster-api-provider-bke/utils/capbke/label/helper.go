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

package label

import (
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
)

func SetLabel(obj client.Object, key, value string) {
	labels := obj.GetLabels()
	if labels == nil {
		labels = map[string]string{}
	}
	labels[key] = value
	obj.SetLabels(labels)
}

// SetBKEMachineLabel set the WorkerNodeHost or MasterNodeHost label of BKE machine by role
func SetBKEMachineLabel(bkeMachine client.Object, role string, value string) {
	label := WorkerNodeHost
	if role == bkenode.MasterNodeRole {
		label = MasterNodeHost
	}

	labels := bkeMachine.GetLabels()
	if labels == nil {
		labels = make(map[string]string)
	}

	labels[label] = value
	bkeMachine.SetLabels(labels)
}

// CheckBKEMachineLabel returns true if the WorkerNodeHost or MasterNodeHost label of BKE machine is set,and return the value
func CheckBKEMachineLabel(bkeMachine client.Object) (string, bool) {
	labels := bkeMachine.GetLabels()
	if labels == nil {
		return "", false
	}

	if value, ok := labels[WorkerNodeHost]; ok {
		return value, true
	}
	if value, ok := labels[MasterNodeHost]; ok {
		return value, true
	}

	return "", false
}

// RemoveBKEMachineLabel remove the WorkerNodeHost or MasterNodeHost label of BKE machine by role
func RemoveBKEMachineLabel(bkeMachine client.Object, role string) {
	label := WorkerNodeHost
	if role == bkenode.MasterNodeRole {
		label = MasterNodeHost
	}

	labels := bkeMachine.GetLabels()
	if labels == nil {
		return
	}
	delete(labels, label)
	bkeMachine.SetLabels(labels)
}

func IsMasterNode(node *corev1.Node) bool {
	return HasLabel(node, NodeRoleMasterLabel)
}

func IsWorkerNode(node *corev1.Node) bool {
	return HasLabel(node, NodeRoleNodeLabel)
}

func SetMasterRoleLabel(node *corev1.Node) {
	labels := node.GetLabels()
	if labels == nil {
		labels = make(map[string]string)
	}

	isAddLabel := false
	if _, ok := labels[NodeRoleMasterLabel]; !ok {
		labels[NodeRoleMasterLabel] = ""
		isAddLabel = true
	}
	if _, ok := labels[NodeRoleControlPlaneLabelKey]; !ok {
		labels[NodeRoleControlPlaneLabelKey] = NodeRoleControlPlaneLabelValue
		isAddLabel = true
	}

	if isAddLabel {
		node.SetLabels(labels)
	}
}

func SetWorkerRoleLabel(node *corev1.Node) {
	labels := node.GetLabels()
	if labels == nil {
		labels = make(map[string]string)
	}

	isAddLabel := false
	if _, ok := labels[NodeRoleNodeLabel]; !ok {
		labels[NodeRoleNodeLabel] = ""
		isAddLabel = true
	}
	if _, ok := labels[NodeRoleWorkerLabelKey]; !ok {
		labels[NodeRoleWorkerLabelKey] = NodeRoleWorkerLabelValue
		isAddLabel = true
	}

	if isAddLabel {
		node.SetLabels(labels)
	}
}

func SetBocVersionLabel(node *corev1.Node, version string) {
	labels := node.GetLabels()
	if labels == nil {
		labels = make(map[string]string)
	}
	if v, ok := labels[BocVersionLabelKey]; !ok || v != version {
		labels[BocVersionLabelKey] = version
		node.SetLabels(labels)
	}
}

func HasLabel(obj client.Object, key string) bool {
	labels := obj.GetLabels()
	if labels == nil {
		return false
	}

	_, ok := labels[key]
	return ok
}

// Determine whether the tag is included
func IsLabelEqual(obj client.Object, key, value string) bool {
	labels := obj.GetLabels()
	if labels == nil {
		return false
	}

	if v, ok := labels[key]; ok && v == value {
		return true
	}
	return false
}
