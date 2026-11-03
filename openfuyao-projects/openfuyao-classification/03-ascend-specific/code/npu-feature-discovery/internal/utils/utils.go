// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package utils provides utility functions for interacting with the Kubernetes environment
// and the system, including functions for retrieving node information, interacting with
// PCI devices, and executing system commands.
package utils

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/urfave/cli/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubernetes2 "k8s.io/client-go/kubernetes"

	"openfuyao.com/npu-feature-discovery/internal/kubernetes"
	"openfuyao.com/npu-feature-discovery/internal/lm/common"
)

const (
	// Unknown 无效的NPU型号
	unknown = "UNKNOWN"
	// ClassAccelerator NPU芯片Class ID
	classAccelerator = "0x1200"
	// ClassBridge PCI桥Class ID
	classBridge       = "0x0604"
	arrLength         = 4
	sysPCIDevicesPath = "/sys/bus/pci/devices"
)

// GetEnvVar retrieves the value of an environment variable by name.
// If the environment variable is not set, it returns an error.
func GetEnvVar(name string) (string, error) {
	value := os.Getenv(name)
	if value == "" {
		return "", fmt.Errorf("%s environment variable is not set", name)
	}
	return value, nil
}

// GetNode retrieves the Kubernetes client and the node information based on the
// environment variable "NODE_NAME".
// Returns an error if either the client or node cannot be retrieved.
func GetNode(c *cli.Context) (kubernetes2.Interface, *corev1.Node, error) {
	client, err := kubernetes.GetKubernetesClient()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get Kubernetes client: %v", err)
	}

	nodeName, err := GetEnvVar("NODE_NAME")
	if err != nil {
		return nil, nil, err
	}

	node, err := client.CoreV1().Nodes().Get(c.Context, nodeName, metav1.GetOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get node %s: %w", nodeName, err)
	}

	return client, node, nil
}

// HasNPUDeviceLabels checks if the node's labels contain NPU device labels.
func HasNPUDeviceLabels(labels map[string]string) bool {
	for key, value := range labels {
		_, ok := common.NPUDeviceType(key)
		if ok && value == common.NpuDeviceLabelValue {
			return true
		}
	}
	return false
}

// GetCardInfo retrieves the card information from the system's PCI devices for a given node.
// It identifies the card based on its class ID and returns the card's name.
// Returns "UNKNOWN" if no valid card is found.
//
//go:noinline
func GetCardInfo(node *corev1.Node) (string, error) {
	return getCardInfo(node, sysPCIDevicesPath)
}

func getCardInfo(node *corev1.Node, devicesPath string) (string, error) {
	tmpValue := unknown
	entries, err := os.ReadDir(devicesPath)
	if err != nil {
		return unknown, err
	}

	archKey := getArchKey(node)

	for _, entry := range entries {
		fullDir := filepath.Join(devicesPath, entry.Name())
		classFile := filepath.Join(fullDir, "class")
		content, err := os.ReadFile(classFile)
		if err != nil {
			continue
		}

		classID := strings.TrimSpace(string(content))
		isAccelerator := strings.HasPrefix(classID, classAccelerator)
		isBridge := strings.HasPrefix(classID, classBridge)
		if !isAccelerator && !isBridge {
			continue
		}

		item, err := parseItem(fullDir)
		if err != nil {
			continue
		}

		cardName := common.ResolveCardServerName(item, archKey)
		if cardName == "" {
			continue
		}

		if isBridge {
			tmpValue = cardName
			continue
		}

		return cardName, nil
	}
	return tmpValue, nil
}

func getArchKey(node *corev1.Node) string {
	arch, ok := node.Labels["kubernetes.io/arch"]
	if !ok {
		return unknown
	}

	switch arch {
	case "arm64":
		return "aarch64"
	default:
		return "x86_64"
	}
}

func parseItem(dirPath string) (common.PCIIdentity, error) {
	ids := [arrLength]string{}
	files := []string{"vendor", "device", "subsystem_vendor", "subsystem_device"}

	for i, name := range files {
		content, err := os.ReadFile(filepath.Join(dirPath, name))
		if err != nil {
			return common.PCIIdentity{}, err
		}
		ids[i] = strings.TrimSpace(string(content))
	}
	return common.PCIIdentity{
		Vendor:          ids[0],
		Device:          ids[1],
		SubsystemVendor: ids[2],
		SubsystemDevice: ids[3],
	}, nil
}

// ExecuteCommand executes a system command using `nsenter` in the specified namespaces.
// It returns the output of the command execution or an error if the command fails.
func ExecuteCommand(cmd string) (string, error) {
	cmdArgs := []string{
		"-i/proc/1/ns/ipc",
		"-m/proc/1/ns/mnt",
		"-n/proc/1/ns/net",
	}

	cmdParts := strings.Fields(cmd)

	cmdArgs = append(cmdArgs, cmdParts...)

	cmdOut := exec.Command("nsenter", cmdArgs...)
	var out bytes.Buffer
	cmdOut.Stdout = &out
	cmdOut.Stderr = &out

	err := cmdOut.Run()
	if err != nil {
		return "", fmt.Errorf("error executing command: %w\nOutput: %s", err, out.String())
	}

	return out.String(), nil
}
