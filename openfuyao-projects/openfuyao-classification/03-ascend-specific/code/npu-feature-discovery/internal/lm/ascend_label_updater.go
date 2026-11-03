// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package lm provides functionality for managing and updating Kubernetes node labels.
// It includes utilities for adding, removing, and updating labels based on node attributes
// like container runtime, server usage, accelerator type, and more.
package lm

import (
	"fmt"
	"strings"

	"github.com/urfave/cli/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/accelerator"
	"openfuyao.com/npu-feature-discovery/internal/lm/acceleratortype"
	"openfuyao.com/npu-feature-discovery/internal/lm/arch"
	"openfuyao.com/npu-feature-discovery/internal/lm/ascend"
	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/lm/network"
	"openfuyao.com/npu-feature-discovery/internal/lm/runtime"
	"openfuyao.com/npu-feature-discovery/internal/lm/selector"
	"openfuyao.com/npu-feature-discovery/internal/lm/serverusage"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

// Output updates the labels of a Kubernetes node based on the provided labels.
// If the labels are modified, it updates the node in the Kubernetes cluster.
func Output(c *cli.Context, labels common.Labels) error {
	client, node, err := utils.GetNode(c)
	if err != nil {
		return err
	}

	nodeCopy := node.DeepCopy()
	updated := false

	removeLabelKeys := labelsToRemove(labels, node.Labels)
	updated = removeLabels(nodeCopy.Labels, removeLabelKeys)

	for key, value := range labels {
		updated = addLabel(nodeCopy.Labels, key, value) || updated
	}

	if updated {
		if _, err = client.CoreV1().Nodes().Update(c.Context, nodeCopy, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("update node labels: %w", err)
		}
	}
	return nil
}

func labelsToRemove(labels common.Labels, currentLabels common.Labels) []string {
	var removeLabelKeys []string

	removeLabelKeys = append(removeLabelKeys, getLabelsToRemoveForMissing(labels)...)

	removeLabelKeys = append(removeLabelKeys, getRDMADeviceLabelsToRemove(labels, currentLabels)...)

	return removeLabelKeys
}

func getLabelsToRemoveForMissing(labels common.Labels) []string {
	var removeLabelKeys []string

	labelConditions := map[string]func() bool{
		runtime.RuntimeLabelKey:                 isLabelMissing(labels, runtime.RuntimeLabelKey),
		selector.MasterLabelKey:                 isLabelMissing(labels, selector.MasterLabelKey),
		selector.WorkerLabelKey:                 isLabelMissing(labels, selector.WorkerLabelKey),
		selector.WorkerRoleLabelKey:             isLabelMissing(labels, selector.WorkerRoleLabelKey),
		ascend.NPUPresentLabelKey:               isLabelMissing(labels, ascend.NPUPresentLabelKey),
		arch.HostArchLabelKey:                   isLabelMissing(labels, arch.HostArchLabelKey),
		accelerator.AcceleratorLabelKey:         isLabelMissing(labels, accelerator.AcceleratorLabelKey),
		acceleratortype.AcceleratorTypeLabelKey: isLabelMissing(labels, acceleratortype.AcceleratorTypeLabelKey),
		serverusage.ServerUsageLabelKey:         isLabelMissing(labels, serverusage.ServerUsageLabelKey),
		network.CardSideRoCEBandwidthLabelKey:   isLabelMissing(labels, network.CardSideRoCEBandwidthLabelKey),
	}

	for key, condition := range labelConditions {
		if condition() {
			removeLabelKeys = append(removeLabelKeys, key)
		}
	}
	return removeLabelKeys
}

func isLabelMissing(labels common.Labels, key string) func() bool {
	return func() bool {
		_, exists := labels[key]
		return !exists
	}
}

func getRDMADeviceLabelsToRemove(labels common.Labels, currentLabels common.Labels) []string {
	var removeLabelKeys []string

	for key := range currentLabels {
		if strings.HasPrefix(key, network.NodeSideRDMADeviceLabelKey) ||
			strings.HasPrefix(key, network.NodeSideRDMADevicePCIeLabelKey) {
			if _, exists := labels[key]; !exists {
				removeLabelKeys = append(removeLabelKeys, key)
			}
		}
	}

	return removeLabelKeys
}

func addLabel(labels map[string]string, key, value string) bool {
	if labels == nil {
		return false
	}
	if strings.HasPrefix(key, network.NodeSideRDMADeviceLabelKey) ||
		strings.HasPrefix(key, network.NodeSideRDMADevicePCIeLabelKey) {

		if v, ok := labels[key]; ok && v == value {
			return false
		}

		labels[key] = value
		return true
	}

	if v, ok := labels[key]; ok && v == value {
		return false
	}

	labels[key] = value
	return true
}

func removeLabels(labels map[string]string, removeLabelKeys []string) bool {
	removed := false

	for _, key := range removeLabelKeys {
		if _, ok := labels[key]; ok {
			removed = true
			delete(labels, key)
		}
	}
	return removed
}
