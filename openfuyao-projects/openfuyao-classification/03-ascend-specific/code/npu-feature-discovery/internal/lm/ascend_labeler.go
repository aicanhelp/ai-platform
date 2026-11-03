// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package lm

import (
	"strings"

	"github.com/urfave/cli/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

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

// DevicePluginResourcePrefix is the resource name prefix advertised by the NPU device plugin.
const DevicePluginResourcePrefix = "huawei.com/"

// NewAscendLabeler generates a set of labels for a Kubernetes node based on its hardware configuration.
// This function collects various labels, including runtime information, device types, network configurations,
// and accelerator information (including Ascend NPUs), and merges them into a final set of labels.
func NewAscendLabeler(c *cli.Context) (common.Labeler, error) {
	_, node, err := utils.GetNode(c)
	if err != nil {
		return nil, err
	}

	var labelers common.Labelers

	runtimeLabeler, err := runtime.NewRuntimeLabeler(node)
	if err != nil {
		return nil, err
	}

	labelers = append(labelers, runtimeLabeler)

	labelers = append(labelers, selector.NewSelectorLabeler(node))

	ascendPresenceLabeler, hasAscend := ascend.NewAscendPresenceLabeler(node)
	if !hasAscend {
		return common.Merge(labelers...), nil
	}
	labelers = append(labelers, ascendPresenceLabeler)

	if !checkDevicePlugin(node) {
		return common.Merge(labelers...), nil
	}

	labelers = append(labelers,
		arch.NewHostArchLabeler(node),
		accelerator.NewAcceleratorLabeler(node),
	)

	labelers = appendLabeler(labelers, acceleratortype.NewAcceleratorTypeLabeler, node)
	labelers = appendLabeler(labelers, serverusage.NewServerLabeler, node)
	labelers = appendLabeler(labelers, network.NewCardSideRoCECardBandwidthLabeler, node)
	labelers = appendLabeler(labelers, network.NewNodeSideRDMADeviceLabeler, node)
	labelers = appendLabeler(labelers, network.NewNodeSideRDMADevicePCIeLabeler, node)

	return common.Merge(labelers...), nil
}

func appendLabeler(labelers []common.Labeler, labelerFunc func(*corev1.Node) (common.Labeler, error),
	node *corev1.Node) []common.Labeler {
	if labeler, err := labelerFunc(node); err != nil {
		klog.Error(err)
	} else {
		labelers = append(labelers, labeler)
	}
	return labelers
}

func checkDevicePlugin(node *corev1.Node) bool {
	for resourceName, resourceQuantity := range node.Status.Capacity {
		if strings.HasPrefix(resourceName.String(), DevicePluginResourcePrefix) {
			if resourceQuantity.Sign() > 0 {
				return true
			}
		}
	}
	klog.Info("No device plugin resources found or resource quantity is 0.")
	return false
}
