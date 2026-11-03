// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package accelerator provides functionality to generate labels related to accelerators
// (such as NPUs) for Kubernetes nodes.
package accelerator

import (
	corev1 "k8s.io/api/core/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

const (
	// AcceleratorLabelKey is the key used for the accelerator label in Kubernetes nodes.
	AcceleratorLabelKey = "accelerator"
)

// NewAcceleratorLabeler returns a Labeler that can be used to generate accelerator-related labels
// for a given Kubernetes node. It checks the node's labels for known NPU device types and applies
// the corresponding label. If no known device type is found, it assigns a default "unknown" label.
func NewAcceleratorLabeler(node *corev1.Node) common.Labeler {
	for key := range node.Labels {
		if value, ok := common.NPUDeviceType(key); ok {
			return common.Labels{AcceleratorLabelKey: value}
		}
	}

	return common.Labels{AcceleratorLabelKey: "unknown"}
}
