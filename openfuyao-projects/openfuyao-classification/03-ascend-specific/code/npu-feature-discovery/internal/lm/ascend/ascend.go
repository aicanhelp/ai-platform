// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package ascend provides functionality to generate a label indicating the presence of NPU devices
// on a Kubernetes node. It helps in identifying nodes that have NPU devices attached.
package ascend

import (
	corev1 "k8s.io/api/core/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

const (
	// NPUPresentLabelKey is the key used to label a Kubernetes node indicating the presence of an NPU.
	NPUPresentLabelKey = "openfuyao.com/npu.present"
)

// NewAscendPresenceLabeler generates a label for indicating the presence of an NPU device
// on a given Kubernetes node. It checks the node's labels for the presence of NPU devices and
// returns the appropriate label and a boolean indicating whether the node has an NPU.
func NewAscendPresenceLabeler(node *corev1.Node) (common.Labeler, bool) {
	result := common.Labels{}

	hasNPU := utils.HasNPUDeviceLabels(node.Labels)

	if hasNPU {
		result[NPUPresentLabelKey] = ""
	}

	return result, hasNPU
}
