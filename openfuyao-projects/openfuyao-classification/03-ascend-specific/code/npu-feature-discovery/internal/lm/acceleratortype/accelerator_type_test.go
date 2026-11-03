// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package acceleratortype

import (
	"fmt"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"openfuyao.com/npu-feature-discovery/internal/utils"
)

type argsGetAcceleratorType struct {
	cardName string
	boardID  string
	cardNums int
}
type testParamGetAcceleratorType struct {
	name string
	args argsGetAcceleratorType
	want string
}

func genTestParamGetAcceleratorType(name string,
	cardName, boardID string, cardNums int, want string,
) testParamGetAcceleratorType {
	return testParamGetAcceleratorType{
		name: name,
		args: argsGetAcceleratorType{
			cardName: cardName,
			boardID:  boardID,
			cardNums: cardNums,
		},
		want: want,
	}
}

func TestGetAcceleratorType(t *testing.T) {
	cardNumSingle := 1
	cardNumHalf := 4
	cardNumModule := 8
	tests := []testParamGetAcceleratorType{
		genTestParamGetAcceleratorType(
			"half", "", "0x27", cardNumHalf, "half"),
		genTestParamGetAcceleratorType(
			"module", "", "0x27", cardNumModule, "module"),
		genTestParamGetAcceleratorType(
			"module-910b-8", "", "0x30", cardNumSingle, "module-910b-8"),
		genTestParamGetAcceleratorType(
			"module-910b-16", "", "0x51", cardNumSingle, "module-910b-16"),
		genTestParamGetAcceleratorType(
			"card", "", "0x01", cardNumSingle, "card"),
		genTestParamGetAcceleratorType(
			"card-910b-2", "", "0x10", cardNumSingle, "card-910b-2"),
		genTestParamGetAcceleratorType(
			"module-a3-16 Atlas_200T_A3_Box8", "", "0xb1", cardNumSingle, "module-a3-16"),
		genTestParamGetAcceleratorType(
			"module-a3-16 Atlas_800I_A3", "", "0xb3", cardNumSingle, "module-a3-16"),
		genTestParamGetAcceleratorType(
			"module-a3-16-super-pod Atlas_900_A3_SuperPod by board id", "", "0xb0", cardNumSingle, "module-a3-16-super-pod"),
		genTestParamGetAcceleratorType(
			"module-a3-16-super-pod Atlas 900 A3 SuperPod by card info first", "Atlas 900 A3 SuperPod", "0xb1", cardNumSingle, "module-a3-16-super-pod"),
		genTestParamGetAcceleratorType(
			"module-a3-16 Atlas 800I A3 by card info first", "Atlas 800I A3", "0xb0", cardNumSingle, "module-a3-16"),
		genTestParamGetAcceleratorType(
			"Atlas_800 other device IDs", "", "0x02", cardNumModule, "module"),
		genTestParamGetAcceleratorType(
			"Atlas_800 0x21 half", "", "0x21", cardNumHalf, "half"),
		genTestParamGetAcceleratorType(
			"Atlas_900_A2_PoD", "", "0x34", cardNumSingle, "module-910b-8"),
		genTestParamGetAcceleratorType(
			"Atlas_300T 0x03", "", "0x03", cardNumSingle, "card"),
		genTestParamGetAcceleratorType(
			"Atlas_300T_A2 0x11", "", "0x11", cardNumSingle, "card-910b-2"),
		genTestParamGetAcceleratorType(
			"Atlas_200T_A2_Box16 0x52", "", "0x52", cardNumSingle, "module-910b-16"),
		genTestParamGetAcceleratorType(
			"unknown board ID returns empty", "", "0xff", cardNumSingle, ""),
		genTestParamGetAcceleratorType(
			"empty board ID", "", "", cardNumSingle, ""),
		genTestParamGetAcceleratorType(
			"Atlas_800 with 1 card", "", "0x27", cardNumSingle, ""),
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getAcceleratorType(tt.args.cardName, tt.args.boardID, tt.args.cardNums); got != tt.want {
				t.Errorf("getAcceleratorType() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestContainsAnyServerDeviceID(t *testing.T) {
	tests := []struct {
		name       string
		serverKeys []string
		boardID    string
		want       bool
	}{
		{"item present", []string{"Atlas_800_A2"}, "0x30", true},
		{"item absent", []string{"Atlas_800_A2"}, "0x51", false},
		{"unknown server", []string{"unknown"}, "0x30", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := containsAnyServerDeviceID(tt.serverKeys, tt.boardID); got != tt.want {
				t.Errorf("containsAnyServerDeviceID() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetNPUInfoFail(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("command failed")
	})
	defer patch.Reset()
	tests := []struct {
		name               string
		want0CommandFail   int
		want1CommandFail   string
		want2CommandFail   string
		wantErrCommandFail bool
	}{
		{
			name:               "npu-smi info command failed",
			want0CommandFail:   0,
			want1CommandFail:   "",
			want2CommandFail:   "",
			wantErrCommandFail: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, err := getNPUInfo()
			got := info.cardNums
			got1 := info.npuID
			got2 := info.chipID
			if (err != nil) != tt.wantErrCommandFail {
				t.Errorf("getCardNumsAndNPUInfo() error = %v, wantErr %v", err, tt.wantErrCommandFail)
				return
			}
			if got != tt.want0CommandFail {
				t.Errorf("getCardNumsAndNPUInfo() got = %v, want %v", got, tt.want0CommandFail)
			}
			if got1 != tt.want1CommandFail {
				t.Errorf("getCardNumsAndNPUInfo() got1 = %v, want %v", got1, tt.want1CommandFail)
			}
			if got2 != tt.want2CommandFail {
				t.Errorf("getCardNumsAndNPUInfo() got2 = %v, want %v", got2, tt.want2CommandFail)
			}
		})
	}
}

func TestGetNPUInfoSuccess(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `NPU ID Chip ID Chip Logic ID Chip Name
 0 0 0 Ascend 310B1
 0 1 - Mcu`, nil
	})
	defer patch.Reset()
	tests := []struct {
		name                  string
		want0CommandSuccess   int
		want1CommandSuccess   string
		want2CommandSuccess   string
		wantErrCommandSuccess bool
	}{
		{
			name:                  "npu-smi info command succeed",
			want0CommandSuccess:   1,
			want1CommandSuccess:   "0",
			want2CommandSuccess:   "0",
			wantErrCommandSuccess: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, err := getNPUInfo()
			got := info.cardNums
			got1 := info.npuID
			got2 := info.chipID
			if (err != nil) != tt.wantErrCommandSuccess {
				t.Errorf("getCardNumsAndNPUInfo() error = %v, wantErr %v", err, tt.wantErrCommandSuccess)
				return
			}
			if got != tt.want0CommandSuccess {
				t.Errorf("getCardNumsAndNPUInfo() got = %v, want %v", got, tt.want0CommandSuccess)
			}
			if got1 != tt.want1CommandSuccess {
				t.Errorf("getCardNumsAndNPUInfo() got1 = %v, want %v", got1, tt.want1CommandSuccess)
			}
			if got2 != tt.want2CommandSuccess {
				t.Errorf("getCardNumsAndNPUInfo() got2 = %v, want %v", got2, tt.want2CommandSuccess)
			}
		})
	}
}

func TestGetBoardIDSuccess(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return `some header
 Board ID    : 0x27
 other line`, nil
	})
	defer patch.Reset()
	got, err := getBoardID("0", "0")
	if err != nil {
		t.Fatalf("getBoardID() err = %v", err)
	}
	if got != "0x27" {
		t.Errorf("getBoardID() = %q, want 0x27", got)
	}
}

func TestGetBoardIDSuccessOutputLowercased(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return " Board ID : 0XB1 \n", nil
	})
	defer patch.Reset()
	got, err := getBoardID("0", "0")
	if err != nil {
		t.Fatalf("getBoardID() err = %v", err)
	}
	if got != "0xb1" {
		t.Errorf("getBoardID() = %q, want 0xb1 (lowercased)", got)
	}
}

func TestGetBoardIDIgnoresMainBoardID(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return " Main Board ID : 0x18 \n Board ID : 0xb0 \n", nil
	})
	defer patch.Reset()

	got, err := getBoardID("0", "0")
	if err != nil {
		t.Fatalf("getBoardID() err = %v", err)
	}
	if got != "0xb0" {
		t.Errorf("getBoardID() = %q, want 0xb0", got)
	}
}

func TestGetBoardIDFail(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("npu-smi failed")
	})
	defer patch.Reset()
	_, err := getBoardID("0", "0")
	if err == nil {
		t.Error("getBoardID() expected error")
	}
}

func TestNewAcceleratorTypeLabelerSuccess(t *testing.T) {
	nodeName := "test-node"
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return ` 0 0 0 Ascend 310B1
 0 1 - Mcu
 1 0 0 Ascend 310B1
 1 1 - Mcu
 2 0 0 Ascend 310B1
 2 1 - Mcu
 3 0 0 Ascend 310B1
 3 1 - Mcu`, nil
		}
		return " Board ID : 0x27 \n", nil
	})
	defer patch.Reset()
	cardPatch := gomonkey.ApplyFunc(utils.GetCardInfo, func(node *corev1.Node) (string, error) {
		return "UNKNOWN", nil
	})
	defer cardPatch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
	labeler, err := NewAcceleratorTypeLabeler(node)
	if err != nil {
		t.Fatalf("NewAcceleratorTypeLabeler() err = %v", err)
	}
	labels := labeler.Labels()
	if labels[AcceleratorTypeLabelKey] != "half" {
		t.Errorf("NewAcceleratorTypeLabeler() label = %q, want half (0x27 + 4 cards)", labels[AcceleratorTypeLabelKey])
	}
}

func TestNewAcceleratorTypeLabelerA3SuperPodByCardInfoFirst(t *testing.T) {
	nodeName := "test-node"
	callCount := 0
	execPatch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return ` 0 0 0 Ascend 910A3
 0 1 - Mcu`, nil
		}
		return " Board ID : 0xb1 \n", nil
	})
	defer execPatch.Reset()

	cardPatch := gomonkey.ApplyFunc(utils.GetCardInfo, func(node *corev1.Node) (string, error) {
		return "Atlas 900 A3 SuperPod", nil
	})
	defer cardPatch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}}
	labeler, err := NewAcceleratorTypeLabeler(node)
	if err != nil {
		t.Fatalf("NewAcceleratorTypeLabeler() err = %v", err)
	}
	labels := labeler.Labels()
	if labels[AcceleratorTypeLabelKey] != "module-a3-16-super-pod" {
		t.Errorf("NewAcceleratorTypeLabeler() label = %q, want module-a3-16-super-pod",
			labels[AcceleratorTypeLabelKey])
	}
}

func TestNewAcceleratorTypeLabelerGetNPUInfoFails(t *testing.T) {
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		return "", fmt.Errorf("npu-smi info failed")
	})
	defer patch.Reset()
	cardPatch := gomonkey.ApplyFunc(utils.GetCardInfo, func(node *corev1.Node) (string, error) {
		return "UNKNOWN", nil
	})
	defer cardPatch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	_, err := NewAcceleratorTypeLabeler(node)
	if err == nil {
		t.Error("NewAcceleratorTypeLabeler() expected error when getNPUInfo fails")
	}
	if !strings.Contains(err.Error(), "NPU info") {
		t.Errorf("NewAcceleratorTypeLabeler() err = %v, want message containing 'NPU info'", err)
	}
}

func TestNewAcceleratorTypeLabelerGetBoardIDFails(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return ` 0 0 0 Ascend 310B1
 0 1 - Mcu`, nil
		}
		return "", fmt.Errorf("npu-smi board failed")
	})
	defer patch.Reset()
	cardPatch := gomonkey.ApplyFunc(utils.GetCardInfo, func(node *corev1.Node) (string, error) {
		return "UNKNOWN", nil
	})
	defer cardPatch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	_, err := NewAcceleratorTypeLabeler(node)
	if err == nil {
		t.Error("NewAcceleratorTypeLabeler() expected error when getBoardID fails")
	}
	if !strings.Contains(err.Error(), "board") {
		t.Errorf("NewAcceleratorTypeLabeler() err = %v, want message containing 'board'", err)
	}
}

func TestNewAcceleratorTypeLabelerNoAcceleratorTypeForNode(t *testing.T) {
	callCount := 0
	patch := gomonkey.ApplyFunc(utils.ExecuteCommand, func(cmd string) (string, error) {
		callCount++
		if callCount == 1 {
			return ` 0 0 0 Ascend 310B1
 0 1 - Mcu`, nil
		}
		return " Board ID : 0xff \n", nil
	})
	defer patch.Reset()
	cardPatch := gomonkey.ApplyFunc(utils.GetCardInfo, func(node *corev1.Node) (string, error) {
		return "UNKNOWN", nil
	})
	defer cardPatch.Reset()

	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "test-node"}}
	_, err := NewAcceleratorTypeLabeler(node)
	if err == nil {
		t.Error("NewAcceleratorTypeLabeler() expected error when accelerator type is unknown")
	}
	if !strings.Contains(err.Error(), "no accelerator type") {
		t.Errorf("NewAcceleratorTypeLabeler() err = %v, want message containing 'no accelerator type'", err)
	}
}
