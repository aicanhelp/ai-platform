// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package acceleratortype provides functionality to generate accelerator-related labels for
// Kubernetes nodes, specifically related to the type of accelerator present.
package acceleratortype

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"

	"openfuyao.com/npu-feature-discovery/internal/lm/common"
	"openfuyao.com/npu-feature-discovery/internal/utils"
)

const (
	fullCardNums    = 8
	halfCardNums    = 4
	lengthThreshold = 3
	oneCardNum      = 1
)

const (
	// AcceleratorTypeLabelKey is the key used for labeling the accelerator type in Kubernetes nodes.
	AcceleratorTypeLabelKey = "accelerator-type"
)

type boardIDTypeRule struct {
	serverKeys      []string
	acceleratorType string
}

var boardIDTypeRules = []boardIDTypeRule{
	{
		serverKeys:      []string{"Atlas_800_A2", "Atlas_900_A2_PoD"},
		acceleratorType: "module-910b-8",
	},
	{
		serverKeys:      []string{"Atlas_200T_A2_Box16"},
		acceleratorType: "module-910b-16",
	},
	{
		serverKeys:      []string{"Atlas_300T"},
		acceleratorType: "card",
	},
	{
		serverKeys:      []string{"Atlas_300T_A2"},
		acceleratorType: "card-910b-2",
	},
	{
		serverKeys:      []string{"Atlas_200T_A3_Box8", "Atlas_800I_A3"},
		acceleratorType: "module-a3-16",
	},
	{
		serverKeys:      []string{"Atlas_900_A3_SuperPod"},
		acceleratorType: "module-a3-16-super-pod",
	},
}

// NewAcceleratorTypeLabeler returns a Labeler that generates a label for the accelerator type
// of the provided Kubernetes node. It fetches the NPU information, including the number of cards
// and device IDs, and assigns an accelerator type label accordingly.
func NewAcceleratorTypeLabeler(node *corev1.Node) (common.Labeler, error) {
	result := common.Labels{}

	cardName, err := utils.GetCardInfo(node)
	if err != nil {
		return result, fmt.Errorf("failed to get card info for node %s: %v", node.Name, err)
	}

	info, err := getNPUInfo()
	cardNums := info.cardNums
	npuID := info.npuID
	chipID := info.chipID
	if err != nil {
		return result, fmt.Errorf("failed to get NPU info for node %s: %v", node.Name, err)
	}

	boardID, err := getBoardID(npuID, chipID)
	if err != nil {
		return result, fmt.Errorf("failed to get board ID for node %s: %v", node.Name, err)
	}

	acceleratorType := getAcceleratorType(cardName, boardID, cardNums)
	if acceleratorType != "" {
		result[AcceleratorTypeLabelKey] = acceleratorType
		return result, nil
	}

	return result, fmt.Errorf("no accelerator type found for node %s", node.Name)
}

func getAcceleratorType(cardName, boardID string, cardNums int) string {
	if acceleratorType := acceleratorTypeByCardName(cardName); acceleratorType != "" {
		return acceleratorType
	}

	return acceleratorTypeByBoardID(boardID, cardNums)
}

func acceleratorTypeByCardName(cardName string) string {
	switch cardName {
	case "Atlas 900 A3 SuperPod":
		return "module-a3-16-super-pod"
	case "Atlas 800I A3":
		return "module-a3-16"
	default:
		return ""
	}
}

func acceleratorTypeByBoardID(boardID string, cardNums int) string {
	if common.IsServerDeviceID("Atlas_800", boardID) {
		return atlas800AcceleratorType(cardNums)
	}

	for _, rule := range boardIDTypeRules {
		if containsAnyServerDeviceID(rule.serverKeys, boardID) {
			return rule.acceleratorType
		}
	}

	return ""
}

func atlas800AcceleratorType(cardNums int) string {
	switch cardNums {
	case fullCardNums:
		return "module"
	case halfCardNums:
		return "half"
	default:
		return ""
	}
}

func containsAnyServerDeviceID(serverKeys []string, boardID string) bool {
	for _, serverKey := range serverKeys {
		if common.IsServerDeviceID(serverKey, boardID) {
			return true
		}
	}

	return false
}

func getBoardID(npuID, chipID string) (string, error) {
	cmdOutput, err := utils.ExecuteCommand(fmt.Sprintf("/usr/local/sbin/npu-smi info -t board -i %s -c %s",
		npuID, chipID))
	if err != nil {
		return "", err
	}

	var boardID string
	for _, line := range strings.Split(cmdOutput, "\n") {
		if parsedBoardID, ok := parseBoardIDLine(line); ok {
			boardID = parsedBoardID
			break
		}
	}

	return strings.ToLower(boardID), nil
}

func parseBoardIDLine(line string) (string, bool) {
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, "Board ID") {
		return "", false
	}

	_, value, found := strings.Cut(line, ":")
	if !found {
		return "", false
	}

	return strings.TrimSpace(value), true
}

type npuInfo struct {
	cardNums int
	npuID    string
	chipID   string
}

func getNPUInfo() (npuInfo, error) {
	output, err := utils.ExecuteCommand("/usr/local/sbin/npu-smi info -m")
	if err != nil {
		return npuInfo{cardNums: 0, npuID: "", chipID: ""}, err
	}

	info := npuInfo{cardNums: 0, npuID: "0", chipID: "0"}

	for _, line := range strings.Split(output, "\n") {
		line = strings.ReplaceAll(line, "\t", " ")
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if !strings.Contains(line, "Ascend") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) < lengthThreshold {
			continue
		}

		info.cardNums++
		if info.cardNums == oneCardNum {
			info.npuID = parts[0]
			info.chipID = parts[1]
		}
	}

	return info, nil
}
