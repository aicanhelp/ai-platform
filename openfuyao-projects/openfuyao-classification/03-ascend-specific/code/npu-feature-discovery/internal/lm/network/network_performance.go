// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package network provides functionality for generating labels related to network devices,
// specifically RDMA devices and PCIe bandwidth information. It helps in labeling Kubernetes
// nodes based on their RDMA capabilities, PCIe bandwidth, and so on.
package network

import (
	"fmt"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

const (
	// CardSideRoCEBandwidthLabelKey is the key used to label the card-side RoCE bandwidth for a node.
	CardSideRoCEBandwidthLabelKey = "card-side-roce-bandwidth"
	// NodeSideRDMADeviceLabelKey is the prefix for labeling RDMA device speed for a node.
	NodeSideRDMADeviceLabelKey = "node-side-rdma-speed-"
	// NodeSideRDMADevicePCIeLabelKey is the prefix for labeling PCIe details related to RDMA devices.
	NodeSideRDMADevicePCIeLabelKey = "node-side-rdma-pcie-"
	// NodeSideRDMADevicePCIeLnkCapSpeed is the prefix for labeling PCIe link cap speed label for RDMA devices.
	NodeSideRDMADevicePCIeLnkCapSpeed = NodeSideRDMADevicePCIeLabelKey + "lnk-cap-speed-"
	// NodeSideRDMADevicePCIeLnkCapWidth is the prefix for labeling PCIe link cap width label for RDMA devices.
	NodeSideRDMADevicePCIeLnkCapWidth = NodeSideRDMADevicePCIeLabelKey + "lnk-cap-width-"
	// NodeSideRDMADevicePCIeLnkStaSpeed is the prefix for labeling PCIe link sta speed label for RDMA devices.
	NodeSideRDMADevicePCIeLnkStaSpeed = NodeSideRDMADevicePCIeLabelKey + "lnk-sta-speed-"
	// NodeSideRDMADevicePCIeLnkStaWidth is the prefix for labeling PCIe link sta width label for RDMA devices.
	NodeSideRDMADevicePCIeLnkStaWidth = NodeSideRDMADevicePCIeLabelKey + "lnk-sta-width-"
)

const (
	pciAddressPartIndex   = 3
	lnkCapLnkStaPartIndex = 4
)

// NewNodeSideRDMADevicePCIeLabeler generates labels for RDMA device PCIe link speed and width details
// for the provided Kubernetes node. It retrieves PCIe information for each RDMA device and labels it accordingly.
func NewNodeSideRDMADevicePCIeLabeler(node *corev1.Node) (common.Labeler, error) {
	result := common.Labels{}

	deviceNames, err := getRDMADeviceNames()
	if err != nil {
		return result, fmt.Errorf("failed to get RDMA device names for node %s: %v", node.Name, err)
	}

	for _, deviceName := range deviceNames {
		pciAddress, err := getPCIDeviceAddress(deviceName)
		if err != nil {
			klog.Errorf("failed to get PCI address for RDMA device %s: %v", deviceName, err)
			continue
		}

		linkInfo, err := getPCIBandwidth(pciAddress)
		if err != nil {
			klog.Errorf("failed to get PCI bandwidth info for device %s: %v", deviceName, err)
			continue
		}

		result[NodeSideRDMADevicePCIeLnkCapSpeed+deviceName] = linkInfo["LnkCapSpeed"]
		result[NodeSideRDMADevicePCIeLnkCapWidth+deviceName] = linkInfo["LnkCapWidth"]
		result[NodeSideRDMADevicePCIeLnkStaSpeed+deviceName] = linkInfo["LnkStaSpeed"]
		result[NodeSideRDMADevicePCIeLnkStaWidth+deviceName] = linkInfo["LnkStaWidth"]
	}

	return result, nil
}

// NewCardSideRoCECardBandwidthLabeler generates a label for the card-side RoCE bandwidth on the provided node.
// It retrieves the server's RoCE bandwidth information and labels the node accordingly.
func NewCardSideRoCECardBandwidthLabeler(node *corev1.Node) (common.Labeler, error) {
	result := common.Labels{}

	server, err := utils.GetCardInfo(node)
	if err != nil {
		return result, fmt.Errorf("failed to get card info for node %s: %v", node.Name, err)
	}

	bandwidth, ok := common.ServerRoCEBandwidth(server)
	if !ok {
		result[CardSideRoCEBandwidthLabelKey] = "unknown"
	} else {
		result[CardSideRoCEBandwidthLabelKey] = bandwidth
	}

	return result, nil
}

// NewNodeSideRDMADeviceLabeler generates labels for the speed of each RDMA device on the provided node.
// It retrieves the speed of each RDMA device and labels the node with the appropriate information.
func NewNodeSideRDMADeviceLabeler(node *corev1.Node) (common.Labeler, error) {
	result := common.Labels{}

	deviceNames, err := getRDMADeviceNames()
	if err != nil {
		return result, fmt.Errorf("failed to get RDMA device names for node %s: %v", node.Name, err)
	}

	for _, deviceName := range deviceNames {
		speed, err := getRDMADeviceSpeed(deviceName)
		if err != nil {
			klog.Errorf("failed to get RDMA device speed for %s: %v", deviceName, err)
			continue
		}

		labelKey := NodeSideRDMADeviceLabelKey + deviceName
		result[labelKey] = speed
	}

	return result, nil
}

func getPCIBandwidth(pciAddress string) (map[string]string, error) {
	cmdOutput, err := utils.ExecuteCommand(fmt.Sprintf("/usr/bin/lspci -vv -s %s", pciAddress))
	if err != nil {
		return nil, fmt.Errorf("failed to execute lspci for PCI address %s: %v", pciAddress, err)
	}

	linkInfo := make(map[string]string)

	re := regexp.MustCompile(`(?i)(LnkCap|LnkSta):.*?Speed\s+([0-9.]+[A-Za-z/]+).*?Width\s+(x\d+)`)

	matches := re.FindAllStringSubmatch(cmdOutput, -1)

	for _, match := range matches {
		if len(match) >= lnkCapLnkStaPartIndex {
			speed := strings.ReplaceAll(match[2], "/", "p")
			width := match[3]

			if match[1] == "LnkCap" {
				linkInfo["LnkCapSpeed"] = speed
				linkInfo["LnkCapWidth"] = width
			} else {
				linkInfo["LnkStaSpeed"] = speed
				linkInfo["LnkStaWidth"] = width
			}
		}
	}

	if len(linkInfo) == 0 {
		return nil, fmt.Errorf("PCIe link info not found for PCI address %s", pciAddress)
	}

	return linkInfo, nil
}

func getPCIDeviceAddress(deviceName string) (string, error) {
	cmdOutput, err := utils.ExecuteCommand("/usr/sbin/udevadm info /sys/class/net/" + deviceName)
	if err != nil {
		return "", fmt.Errorf("failed to execute udevadm for device %s: %v", deviceName, err)
	}

	for _, line := range strings.Split(cmdOutput, "\n") {
		if strings.HasPrefix(line, "P:") {
			parts := strings.Split(line, "/")
			if len(parts) > pciAddressPartIndex {
				return parts[len(parts)-pciAddressPartIndex], nil
			}
		}
	}

	return "", fmt.Errorf("PCI address not found for device %s", deviceName)
}

func getRDMADeviceSpeed(deviceName string) (string, error) {
	cmdOutput, err := utils.ExecuteCommand(fmt.Sprintf("/usr/sbin/ethtool %s", deviceName))
	if err != nil {
		return "", fmt.Errorf("failed to execute ethtool for device %s: %v", deviceName, err)
	}

	re := regexp.MustCompile(`Speed:\s*(\d+\S+)`)
	match := re.FindStringSubmatch(cmdOutput)

	if len(match) > 1 {
		speed := strings.ReplaceAll(match[1], "/", "p")
		return speed, nil
	}

	return "", fmt.Errorf("speed information not found for device %s", deviceName)
}

func getRDMADeviceNames() ([]string, error) {
	cmdOutput, err := utils.ExecuteCommand("/usr/sbin/rdma link")
	if err != nil {
		return nil, err
	}

	lines := strings.Split(cmdOutput, "\n")

	var devices []string

	for _, line := range lines {
		if strings.Contains(line, "state ACTIVE") && strings.Contains(line, "physical_state LINK_UP") {
			parts := strings.Fields(line)
			if len(parts) > 0 {
				deviceName := parts[len(parts)-1]
				devices = append(devices, deviceName)
			}
		}
	}

	if len(devices) == 0 {
		return nil, fmt.Errorf("no active RDMA devices found with link up")
	}

	return devices, nil
}
