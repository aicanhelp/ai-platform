// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package common

const (
	nfdLabelPrefix    = "feature.node.kubernetes.io/"
	npuDeviceClassId  = "1200" // Processing accelerators
	npuDeviceVendorId = "19e5"
	// NpuDeviceLabelPrefix is the label prefix for NPU devices, used to identify NPUs in Kubernetes.
	NpuDeviceLabelPrefix = nfdLabelPrefix + "pci-" + npuDeviceClassId + "_" + npuDeviceVendorId + "_"
	// NpuDeviceLabelValue is the value used to label NPU devices.
	NpuDeviceLabelValue = "true"
)

var npuDeviceTypes = map[string]string{
	NpuDeviceLabelPrefix + "d100.present": "huawei-Ascend310",
	NpuDeviceLabelPrefix + "d500.present": "huawei-Ascend310P",
	NpuDeviceLabelPrefix + "d801.present": "huawei-Ascend910",
	NpuDeviceLabelPrefix + "d802.present": "huawei-Ascend910",
	NpuDeviceLabelPrefix + "d803.present": "huawei-Ascend910",
	NpuDeviceLabelPrefix + "d806.present": "huawei-AscendA5",
}

// NPUDeviceType returns the device name associated with an NPU device label.
func NPUDeviceType(label string) (string, bool) {
	deviceType, ok := npuDeviceTypes[label]
	return deviceType, ok
}

// IsServerDeviceID reports whether deviceID belongs to the specified server model.
func IsServerDeviceID(server, deviceID string) bool {
	switch server {
	case "Atlas_800":
		return deviceID == "0x02" || deviceID == "0x27" || deviceID == "0x21" || deviceID == "0x24" || deviceID == "0x28"
	case "Atlas_800_A2":
		return deviceID == "0x30" || deviceID == "0x31" || deviceID == "0x32" || deviceID == "0x34" || deviceID == "0x38"
	case "Atlas_900_A2_PoD":
		return deviceID == "0x30" || deviceID == "0x31" || deviceID == "0x32" || deviceID == "0x34"
	case "Atlas_200T_A2_Box16":
		return deviceID == "0x50" || deviceID == "0x51" || deviceID == "0x53" || deviceID == "0x52"
	case "Atlas_300T":
		return deviceID == "0x01" || deviceID == "0x03" || deviceID == "0x06"
	case "Atlas_300T_A2":
		return deviceID == "0x10" || deviceID == "0x13" || deviceID == "0x12" || deviceID == "0x11"
	case "Atlas_200T_A3_Box8":
		return deviceID == "0xb1"
	case "Atlas_800I_A3":
		return deviceID == "0xb3"
	case "Atlas_900_A3_SuperPod":
		return deviceID == "0xb0" || deviceID == "0xb1" || deviceID == "0xb2"
	default:
		return false
	}
}

// PCIIdentity represents the identity of a PCI device,
// including its vendor, device, subsystem vendor, and subsystem device IDs.
type PCIIdentity struct {
	Vendor          string
	Device          string
	SubsystemVendor string
	SubsystemDevice string
}

// ResolveCardServerName returns the server model associated with a PCI identity and architecture.
func ResolveCardServerName(identity PCIIdentity, archKey string) string {
	if archKey == "UNKNOWN" {
		return ""
	}

	if matchesPCI(identity, "0x19e5", "0xd100", "0x0200", "0x0100") {
		return resolveD100ServerName(archKey)
	}
	if matchesPCI(identity, "0x19e5", "0xd801", "0x0200", "0x0100") {
		return "A300T-9000"
	}
	if isA900T(identity) {
		return "A900T"
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x6000") {
		return "A300t-a2"
	}
	if matchesPCI(identity, "0x19e5", "0xd500", "0x0200", "0x0100") {
		return "A300i-pro"
	}
	if matchesPCI(identity, "0x19e5", "0xd500", "0x0200", "0x0110") {
		return "A300i-duo"
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x4000") {
		return "A300i-a2"
	}
	if matchesPCI(identity, "0x19e5", "0xd105", "0x0200", "0x0100") ||
		matchesPCI(identity, "0x19e5", "0xd107", "0x0000", "0x0000") {
		return "A200i-a2"
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3002") ||
		matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3004") ||
		matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3005") {
		return "Atlas 800I A2"
	}
	if matchesPCI(identity, "0x19e5", "0xd803", "0x19e5", "0x3000") ||
		matchesPCI(identity, "0x19e5", "0xd803", "0x19e5", "0x3001") ||
		matchesPCI(identity, "0x19e5", "0xd803", "0x19e5", "0x3002") {
		return "Atlas 900 A3 SuperPod"
	}
	if matchesPCI(identity, "0x19e5", "0xd803", "0x19e5", "0x0100") ||
		matchesPCI(identity, "0x19e5", "0xd803", "0x19e5", "0x3003") {
		return "Atlas 800I A3"
	}
	return ""
}

func resolveD100ServerName(archKey string) string {
	if archKey == "x86_64" {
		return "A300-3010"
	}
	if archKey == "aarch64" {
		return "A300-3000"
	}
	return ""
}

func matchesPCI(identity PCIIdentity, vendor, device, subsystemVendor, subsystemDevice string) bool {
	return identity.Vendor == vendor && identity.Device == device &&
		identity.SubsystemVendor == subsystemVendor && identity.SubsystemDevice == subsystemDevice
}

func isA900T(identity PCIIdentity) bool {
	if matchesPCI(identity, "0x19e5", "0xd802", "0x0200", "0x0100") {
		return true
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3000") {
		return true
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3001") {
		return true
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3003") {
		return true
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3400") {
		return true
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3401") {
		return true
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3402") {
		return true
	}
	if matchesPCI(identity, "0x19e5", "0xd802", "0x19e5", "0x3403") {
		return true
	}
	return false
}

// ServerRoCEBandwidth returns the RoCE bandwidth associated with a server model.
func ServerRoCEBandwidth(server string) (string, bool) {
	switch server {
	case "Atlas 800I A2":
		return "200Gbps", true
	case "Atlas 900 A3 SuperPod", "Atlas 800I A3":
		return "400Gbps", true
	default:
		return "", false
	}
}
