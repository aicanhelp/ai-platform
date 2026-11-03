// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package network

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/npu-feature-discovery/internal/utils"
)

func TestNewNodeSideRDMADevicePCIeLabelerSuccessSingleDevice(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev xxxxxx`, nil
		}
		if callCount == 2 {
			return `P: /devices/pci0000:00/0000:00:01.0/0000:01:00.0/net/xxxxxx`, nil
		}
		return `LnkCap:	Port #0, Speed 8GT/s, Width x16, ASPM L0s L1
		LnkSta:	Speed 8GT/s, Width x16, TrErr- Train- SlotClk+`, nil
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADevicePCIeLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADevicePCIeLabeler() err = %v", err)
	}
	labels := labeler.Labels()

	expectedLabels := map[string]string{
		NodeSideRDMADevicePCIeLnkCapSpeed + "xxxxxx": "8GTps",
		NodeSideRDMADevicePCIeLnkCapWidth + "xxxxxx": "x16",
		NodeSideRDMADevicePCIeLnkStaSpeed + "xxxxxx": "8GTps",
		NodeSideRDMADevicePCIeLnkStaWidth + "xxxxxx": "x16",
	}

	for key, expectedValue := range expectedLabels {
		if labels[key] != expectedValue {
			t.Errorf("NewNodeSideRDMADevicePCIeLabeler() labels[%q] = %q, want %q",
				key, labels[key], expectedValue)
		}
	}
	if len(labels) != len(expectedLabels) {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() got %d labels, want %d",
			len(labels), len(expectedLabels))
	}
}

func TestNewNodeSideRDMADevicePCIeLabelerSuccessMultipleDevices(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `link mlx5_0: state ACTIVE physical_state LINK_UP netdev xxxxxx
link mlx5_1: state ACTIVE physical_state LINK_UP netdev yyyyyy`, nil
		}
		if callCount == 2 {
			return `P: /devices/pci0000:00/0000:00:01.0/0000:01:00.0/net/xxxxxx`, nil
		}
		if callCount == 3 {
			return `LnkCap: Speed 8GT/s, Width x16
			LnkSta: Speed 8GT/s, Width x16`, nil
		}
		if callCount == 4 {
			return `P: /devices/pci0000:00/0000:00:02.0/0000:02:00.0/net/yyyyyy`, nil
		}
		return `LnkCap: Speed 16GT/s, Width x8
		LnkSta: Speed 16GT/s, Width x8`, nil
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADevicePCIeLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADevicePCIeLabeler() err = %v", err)
	}
	labels := labeler.Labels()

	if labels[NodeSideRDMADevicePCIeLnkCapSpeed+"xxxxxx"] != "8GTps" {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() mlx5_0 LnkCapSpeed = %q, want 8GTps",
			labels[NodeSideRDMADevicePCIeLnkCapSpeed+"mlx5_0"])
	}
	if labels[NodeSideRDMADevicePCIeLnkCapWidth+"xxxxxx"] != "x16" {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() mlx5_0 LnkCapWidth = %q, want x16",
			labels[NodeSideRDMADevicePCIeLnkCapWidth+"mlx5_0"])
	}

	if labels[NodeSideRDMADevicePCIeLnkCapSpeed+"yyyyyy"] != "16GTps" {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() mlx5_1 LnkCapSpeed = %q, want 16GTps",
			labels[NodeSideRDMADevicePCIeLnkCapSpeed+"mlx5_1"])
	}
	if labels[NodeSideRDMADevicePCIeLnkCapWidth+"yyyyyy"] != "x8" {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() mlx5_1 LnkCapWidth = %q, want x8",
			labels[NodeSideRDMADevicePCIeLnkCapWidth+"mlx5_1"])
	}

	if len(labels) != 8 {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() got %d labels, want 8", len(labels))
	}
}

func TestNewNodeSideRDMADevicePCIeLabelerGetRDMADeviceNamesFails(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("rdma link failed")
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	_, err := NewNodeSideRDMADevicePCIeLabeler(node)
	if err == nil {
		t.Error("NewNodeSideRDMADevicePCIeLabeler() expected error when getRDMADeviceNames fails")
	}
	if !strings.Contains(err.Error(), "failed to get RDMA device names") {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() err = %v, want message containing "+
			"'failed to get RDMA device names'", err)
	}
	if !strings.Contains(err.Error(), "test-node") {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() err should contain node name, got %v", err)
	}
}

func TestNewNodeSideRDMADevicePCIeLabelerPCIAddressFails(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `link mlx5_0: state ACTIVE physical_state LINK_UP netdev xxxxxx`, nil
		}
		return "", fmt.Errorf("failed to get PCI address")
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADevicePCIeLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADevicePCIeLabeler() err = %v", err)
	}
	labels := labeler.Labels()
	if len(labels) != 0 {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() should return empty labels when PCI address fails, "+
			"got %v", labels)
	}
}

func TestNewNodeSideRDMADevicePCIeLabeler_BandwidthFails(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `1: mlx5_0: state ACTIVE physical_state LINK_UP`, nil
		}
		if callCount == 2 {
			return `P: /devices/pci0000:00/0000:00:01.0/0000:01:00.0/net/mlx5_0`, nil
		}
		// Bandwidth lookup fails
		return "", fmt.Errorf("failed to get bandwidth")
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADevicePCIeLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADevicePCIeLabeler() err = %v", err)
	}
	labels := labeler.Labels()
	// Should return empty labels when bandwidth fails
	if len(labels) != 0 {
		t.Errorf("NewNodeSideRDMADevicePCIeLabeler() should return empty labels when bandwidth fails, got %v", labels)
	}
}

func TestNewCardSideRoCECardBandwidthLabelerSuccess(t *testing.T) {
	nodeName := "test-node"
	patch := gomonkey.ApplyFunc(utils.GetCardInfo, func(node *corev1.Node) (string, error) {
		return "Atlas 800I A2", nil
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
	labeler, err := NewCardSideRoCECardBandwidthLabeler(node)
	if err != nil {
		t.Fatalf("NewCardSideRoCECardBandwidthLabeler() err = %v", err)
	}
	labels := labeler.Labels()
	if labels[CardSideRoCEBandwidthLabelKey] != "200Gbps" {
		t.Errorf("NewCardSideRoCECardBandwidthLabeler() label = %q, want 200Gbps",
			labels[CardSideRoCEBandwidthLabelKey])
	}
}

func TestNewCardSideRoCECardBandwidthLabelerUnknownServer(t *testing.T) {
	nodeName := "test-node"
	patch := gomonkey.ApplyFunc(utils.GetCardInfo, func(node *corev1.Node) (string, error) {
		return "UnknownServer", nil
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
	labeler, err := NewCardSideRoCECardBandwidthLabeler(node)
	if err != nil {
		t.Fatalf("NewCardSideRoCECardBandwidthLabeler() err = %v", err)
	}
	labels := labeler.Labels()
	if labels[CardSideRoCEBandwidthLabelKey] != "unknown" {
		t.Errorf("NewCardSideRoCECardBandwidthLabeler() label = %q, want unknown",
			labels[CardSideRoCEBandwidthLabelKey])
	}
}

func TestNewCardSideRoCECardBandwidthLabelerGetCardInfoFails(t *testing.T) {
	nodeName := "test-node"
	patch := gomonkey.ApplyFunc(utils.GetCardInfo, func(node *corev1.Node) (string, error) {
		return "", fmt.Errorf("failed to get card info")
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
	_, err := NewCardSideRoCECardBandwidthLabeler(node)
	if err == nil {
		t.Error("NewCardSideRoCECardBandwidthLabeler() expected error when GetCardInfo fails")
	}
	if !strings.Contains(err.Error(), "card info") {
		t.Errorf("NewCardSideRoCECardBandwidthLabeler() err = %v, want message containing 'card info'", err)
	}
}

func TestNewNodeSideRDMADeviceLabelerSuccessSingleDevice(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `link mlx5_0 state ACTIVE physical_state LINK_UP netdev xxxxxx`, nil
		}
		return `Settings for xxxxxx:
	Speed: 100000Mb/s
	Duplex: Full`, nil
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADeviceLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADeviceLabeler() err = %v", err)
	}
	labels := labeler.Labels()

	expectedKey := NodeSideRDMADeviceLabelKey + "xxxxxx"
	if labels[expectedKey] != "100000Mbps" {
		t.Errorf("NewNodeSideRDMADeviceLabeler() labels[%q] = %q, want 100000Mbps",
			expectedKey, labels[expectedKey])
	}
	if len(labels) != 1 {
		t.Errorf("NewNodeSideRDMADeviceLabeler() got %d labels, want 1", len(labels))
	}
}

func TestNewNodeSideRDMADeviceLabelerSuccessMultipleDevices(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `link mlx5_0 state ACTIVE physical_state LINK_UP netdev xxxxxx
link mlx5_1 state ACTIVE physical_state LINK_UP netdev yyyyyy
link mlx5_2 state ACTIVE physical_state LINK_UP zzzzzz`, nil
		}
		if strings.Contains(cmd, "xxxxxx") {
			return `Speed: 100000Mb/s`, nil
		}
		if strings.Contains(cmd, "yyyyyy") {
			return `Speed: 25000Mb/s`, nil
		}
		return `Speed: 100Gb/s`, nil
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADeviceLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADeviceLabeler() err = %v", err)
	}
	labels := labeler.Labels()

	// Check all three devices have labels
	if labels[NodeSideRDMADeviceLabelKey+"xxxxxx"] != "100000Mbps" {
		t.Errorf("NewNodeSideRDMADeviceLabeler() mlx5_0 speed = %q, want 100000Mbps",
			labels[NodeSideRDMADeviceLabelKey+"mlx5_0"])
	}
	if labels[NodeSideRDMADeviceLabelKey+"yyyyyy"] != "25000Mbps" {
		t.Errorf("NewNodeSideRDMADeviceLabeler() mlx5_1 speed = %q, want 25000Mbps",
			labels[NodeSideRDMADeviceLabelKey+"mlx5_1"])
	}
	if labels[NodeSideRDMADeviceLabelKey+"zzzzzz"] != "100Gbps" {
		t.Errorf("NewNodeSideRDMADeviceLabeler() mlx5_2 speed = %q, want 100Gbps",
			labels[NodeSideRDMADeviceLabelKey+"mlx5_2"])
	}
	if len(labels) != 3 {
		t.Errorf("NewNodeSideRDMADeviceLabeler() got %d labels, want 3", len(labels))
	}
}

func TestNewNodeSideRDMADeviceLabelerGetRDMADeviceNamesFails(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("rdma link failed")
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	_, err := NewNodeSideRDMADeviceLabeler(node)
	if err == nil {
		t.Error("NewNodeSideRDMADeviceLabeler() expected error when getRDMADeviceNames fails")
	}
	if !strings.Contains(err.Error(), "failed to get RDMA device names") {
		t.Errorf("NewNodeSideRDMADeviceLabeler() err = %v, want message containing 'failed to get RDMA device names'", err)
	}
	if !strings.Contains(err.Error(), "test-node") {
		t.Errorf("NewNodeSideRDMADeviceLabeler() err should contain node name, got %v", err)
	}
}

func TestNewNodeSideRDMADeviceLabelerSpeedFails(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `link mlx5_0 state ACTIVE physical_state LINK_UP netdev xxxxxxx`, nil
		}
		// Speed lookup fails
		return "", fmt.Errorf("ethtool failed")
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADeviceLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADeviceLabeler() err = %v", err)
	}
	labels := labeler.Labels()

	if len(labels) != 0 {
		t.Errorf("NewNodeSideRDMADeviceLabeler() should return empty labels when speed fails, got %v", labels)
	}
}

func TestNewNodeSideRDMADeviceLabelerAllDevicesFail(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `link mlx5_0 state ACTIVE physical_state LINK_UP netdev xxxxxx
link mlx5_1 state ACTIVE physical_state LINK_UP netdev yyyyyy`, nil
		}
		// All speed lookups fail
		return "", fmt.Errorf("ethtool failed")
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADeviceLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADeviceLabeler() err = %v", err)
	}
	labels := labeler.Labels()
	// Should return empty labels when all devices fail
	if len(labels) != 0 {
		t.Errorf("NewNodeSideRDMADeviceLabeler() should return empty labels when all devices fail, got %v",
			labels)
	}
}

func TestNewNodeSideRDMADeviceLabelerNoSpeedInfo(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return `link mlx5_0 state ACTIVE physical_state LINK_UP netdev xxxxxx`, nil
		}
		// No speed information in output
		return `Settings for xxxxxx:
	Duplex: Full`, nil
	})
	defer patch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	labeler, err := NewNodeSideRDMADeviceLabeler(node)
	if err != nil {
		t.Fatalf("NewNodeSideRDMADeviceLabeler() err = %v", err)
	}
	labels := labeler.Labels()
	// Should skip device when speed info not found
	if len(labels) != 0 {
		t.Errorf("NewNodeSideRDMADeviceLabeler() should skip device when speed info not found, got %v", labels)
	}
}

func TestGetPCIBandwidthSuccess(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `Capabilities: [100 v1] Express Endpoint, MSI 00
		LnkCap:	Port #0, Speed 8GT/s, Width x16, ASPM L0s L1, Exit Latency L0s <512ns, L1 <4us
		LnkSta:	Speed 8GT/s, Width x16, TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-`, nil
	})
	defer patch.Reset()

	info, err := getPCIBandwidth("0000:01:00.0")
	if err != nil {
		t.Fatalf("getPCIBandwidth() err = %v", err)
	}
	// Note: regex replaces "/" with "p" in speed
	if info["LnkCapSpeed"] != "8GTps" {
		t.Errorf("getPCIBandwidth() LnkCapSpeed = %q, want 8GTps", info["LnkCapSpeed"])
	}
	if info["LnkCapWidth"] != "x16" {
		t.Errorf("getPCIBandwidth() LnkCapWidth = %q, want x16", info["LnkCapWidth"])
	}
	if info["LnkStaSpeed"] != "8GTps" {
		t.Errorf("getPCIBandwidth() LnkStaSpeed = %q, want 8GTps", info["LnkStaSpeed"])
	}
	if info["LnkStaWidth"] != "x16" {
		t.Errorf("getPCIBandwidth() LnkStaWidth = %q, want x16", info["LnkStaWidth"])
	}
}

func TestGetPCIBandwidthCommandFails(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("lspci failed")
	})
	defer patch.Reset()

	_, err := getPCIBandwidth("0000:01:00.0")
	if err == nil {
		t.Error("getPCIBandwidth() expected error when command fails")
	}
}

func TestGetPCIBandwidthNoLinkInfo(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `Capabilities: [100 v1] Express Endpoint`, nil
	})
	defer patch.Reset()

	_, err := getPCIBandwidth("0000:01:00.0")
	if err == nil {
		t.Error("getPCIBandwidth() expected error when link info not found")
	}
}

func TestGetPCIDeviceAddressSuccess(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `P: /devices/pci0000:00/0000:00:01.0/0000:01:00.0/net/xxxxxx
M: xxxxxx`, nil
	})
	defer patch.Reset()

	address, err := getPCIDeviceAddress("xxxxxx")
	if err != nil {
		t.Fatalf("getPCIDeviceAddress() err = %v", err)
	}
	if address != "0000:01:00.0" {
		t.Errorf("getPCIDeviceAddress() = %q, want 0000:01:00.0", address)
	}
}

func TestGetPCIDeviceAddressNoPLine(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `M: xxxxxx`, nil
	})
	defer patch.Reset()

	_, err := getPCIDeviceAddress("xxxxxx")
	if err == nil {
		t.Error("getPCIDeviceAddress() expected error when P: line not found")
	}
}

func TestGetPCIDeviceAddressCommandFails(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("udevadm failed")
	})
	defer patch.Reset()

	_, err := getPCIDeviceAddress("xxxxxx")
	if err == nil {
		t.Error("getPCIDeviceAddress() expected error when command fails")
	}
}

func TestGetPCIDeviceAddressShortPath(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `P: /devices/pci0000:00`, nil
	})
	defer patch.Reset()

	_, err := getPCIDeviceAddress("xxxxxx")
	if err == nil {
		t.Error("getPCIDeviceAddress() expected error when path too short")
	}
}

func TestGetRDMADeviceSpeedSuccess(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `Settings for xxxxxx:
	Speed: 100000Mb/s
	Duplex: Full`, nil
	})
	defer patch.Reset()

	speed, err := getRDMADeviceSpeed("xxxxxx")
	if err != nil {
		t.Fatalf("getRDMADeviceSpeed() err = %v", err)
	}
	if speed != "100000Mbps" {
		t.Errorf("getRDMADeviceSpeed() = %q, want 100000Mbps", speed)
	}
}

func TestGetRDMADeviceSpeedWithSlash(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `Speed: 25Gb/s`, nil
	})
	defer patch.Reset()

	speed, err := getRDMADeviceSpeed("xxxxxx")
	if err != nil {
		t.Fatalf("getRDMADeviceSpeed() err = %v", err)
	}
	if speed != "25Gbps" {
		t.Errorf("getRDMADeviceSpeed() = %q, want 25Gbps", speed)
	}
}

func TestGetRDMADeviceSpeedCommandFails(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("ethtool failed")
	})
	defer patch.Reset()

	_, err := getRDMADeviceSpeed("xxxxxx")
	if err == nil {
		t.Error("getRDMADeviceSpeed() expected error when command fails")
	}
}

func TestGetRDMADeviceSpeedNoSpeedInfo(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `Settings for xxxxxx:
	Duplex: Full`, nil
	})
	defer patch.Reset()

	_, err := getRDMADeviceSpeed("xxxxxx")
	if err == nil {
		t.Error("getRDMADeviceSpeed() expected error when speed info not found")
	}
}

func TestGetRDMADeviceNamesSuccess(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev xxxxxx
link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev yyyyyy
link mlx5_2/1 state DOWN physical_state LINK_DOWN netdev zzzzzz`, nil
	})
	defer patch.Reset()

	devices, err := getRDMADeviceNames()
	if err != nil {
		t.Fatalf("getRDMADeviceNames() err = %v", err)
	}
	expected := []string{"xxxxxx", "yyyyyy"}
	if !reflect.DeepEqual(devices, expected) {
		t.Errorf("getRDMADeviceNames() = %v, want %v", devices, expected)
	}
}

func TestGetRDMADeviceNamesNoActiveDevices(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `link mlx5_0/1 state DOWN physical_state LINK_DOWN
link mlx5_1/1 state ACTIVE physical_state LINK_DOWN`, nil
	})
	defer patch.Reset()

	_, err := getRDMADeviceNames()
	if err == nil {
		t.Error("getRDMADeviceNames() expected error when no active devices")
	}
	if !strings.Contains(err.Error(), "no active RDMA devices") {
		t.Errorf("getRDMADeviceNames() err = %v, want message containing 'no active RDMA devices'", err)
	}
}

func TestGetRDMADeviceNamesCommandFails(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("rdma link failed")
	})
	defer patch.Reset()

	_, err := getRDMADeviceNames()
	if err == nil {
		t.Error("getRDMADeviceNames() expected error when command fails")
	}
}

func TestGetRDMADeviceNamesEmptyOutput(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", nil
	})
	defer patch.Reset()

	_, err := getRDMADeviceNames()
	if err == nil {
		t.Error("getRDMADeviceNames() expected error when output is empty")
	}
}
