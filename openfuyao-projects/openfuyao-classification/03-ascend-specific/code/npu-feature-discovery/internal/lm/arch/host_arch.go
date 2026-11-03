// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package arch provides functionality to generate labels related to the host architecture
// (e.g., x86, ARM) for Kubernetes nodes.
package arch

import (
	corev1 "k8s.io/api/core/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

const (
	// HostArchLabelKey is the key used for labeling the host architecture in Kubernetes nodes.
	HostArchLabelKey = "host-arch"
)

// NewHostArchLabeler returns a Labeler that generates a label for the host architecture of
// the provided Kubernetes node. It checks the node's architecture label and assigns a label
// corresponding to either x86 or ARM architecture, defaulting to "unknown" if not present.
func NewHostArchLabeler(node *corev1.Node) common.Labeler {
	result := common.Labels{}

	arch, ok := node.Labels["kubernetes.io/arch"]
	if !ok {
		return common.Labels{HostArchLabelKey: "unknown"}
	}

	if arch == "amd64" {
		result[HostArchLabelKey] = "huawei-x86"
	} else if arch == "arm64" {
		result[HostArchLabelKey] = "huawei-arm"
	} else {
		result[HostArchLabelKey] = arch
	}

	return result
}
