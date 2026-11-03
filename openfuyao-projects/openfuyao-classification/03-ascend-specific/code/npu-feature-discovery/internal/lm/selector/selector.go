// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package selector provides functionality to generate labels for Kubernetes nodes
// based on their role (e.g., control plane or worker) and their hardware features.
package selector

import (
	corev1 "k8s.io/api/core/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

const (
	controlPlaneRoleLabelKey = "node-role.kubernetes.io/control-plane"
	// MasterLabelKey is the key used for labeling master nodes in the Kubernetes cluster.
	MasterLabelKey = "masterselector"
	// WorkerLabelKey is the key used for labeling worker nodes in the Kubernetes cluster.
	WorkerLabelKey = "workerselector"
	// WorkerRoleLabelKey is the label key for worker role nodes.
	WorkerRoleLabelKey   = "node-role.kubernetes.io/worker"
	masterLabelValue     = "dls-master-node"
	workerLabelValue     = "dls-worker-node"
	workerRoleLabelValue = "worker"
)

// NewSelectorLabeler generates labels based on the role of the node (control-plane or worker)
// and whether the node has an NPU device attached. Control plane nodes are labeled with a master
// label, while worker nodes are labeled as worker nodes, and those with NPUs are specially marked.
func NewSelectorLabeler(node *corev1.Node) common.Labeler {
	result := common.Labels{}
	isControlPlane := hasControlPlaneLabel(node.Labels)
	hasNPU := utils.HasNPUDeviceLabels(node.Labels)

	if isControlPlane {
		result[MasterLabelKey] = masterLabelValue
	}
	if !isControlPlane || hasNPU {
		result[WorkerRoleLabelKey] = workerRoleLabelValue
		result[WorkerLabelKey] = workerLabelValue
	}

	return result
}

func hasControlPlaneLabel(labels map[string]string) bool {
	_, ok := labels[controlPlaneRoleLabelKey]
	return ok
}
