// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package serverusage provides functionality to generate labels for Kubernetes nodes
// based on the server usage or type of hardware accelerator, such as Atlas cards.
// It helps in categorizing nodes that are designated for specific use cases, such as inference.
package serverusage

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

const (
	// ServerUsageLabelKey is the key used to label the server usage type in Kubernetes nodes.
	ServerUsageLabelKey = "server-usage"
)

// NewServerLabeler generates a label for a given Kubernetes node based on the card type.
// If the node has a supported accelerator card (e.g., Atlas 800I A2 or Atlas 800I A3),
// it will be labeled for inference use.
func NewServerLabeler(node *corev1.Node) (common.Labeler, error) {
	result := common.Labels{}

	card, err := utils.GetCardInfo(node)
	if err != nil {
		return result, fmt.Errorf("failed to get card info for node %s: %v", node.Name, err)
	}

	supportedCards := map[string]bool{
		"Atlas 800I A2": true,
		"Atlas 800I A3": true,
	}

	if supportedCards[card] {
		result[ServerUsageLabelKey] = "infer"
	}

	return result, nil
}
