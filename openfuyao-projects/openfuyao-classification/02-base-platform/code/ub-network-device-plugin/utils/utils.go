/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strconv"

	cri "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubelet/pkg/types"
)

// In Kubernetes versions 1.19 and earlier, the DevicesIDs field is of type []string.
type PodDeviceEntryV1 struct {
	PodUID        string   `json:"PodUID"`
	ContainerName string   `json:"ContainerName"`
	ResourceName  string   `json:"ResourceName"`
	DeviceIDs     []string `json:"DeviceIDs"`
	AllocResp     string   `json:"AllocResp"`
}

// In Kubernetes versions 1.20 and later, the DevicesIDs field is of type map[int64][]string.
type PodDeviceEntryV2 struct {
	PodUID        string             `json:"PodUID"`
	ContainerName string             `json:"ContainerName"`
	ResourceName  string             `json:"ResourceName"`
	DeviceIDs     map[int64][]string `json:"DeviceIDs"`
	AllocResp     string             `json:"AllocResp"`
}

type DataV1 struct {
	PodDeviceEntries []PodDeviceEntryV1 `json:"PodDeviceEntries"`
}

type DataV2 struct {
	PodDeviceEntries []PodDeviceEntryV2 `json:"PodDeviceEntries"`
}

type FileContentV1 struct {
	Data     DataV1 `json:"Data"`
	Checksum int    `json:"Checksum"`
}

type FileContentV2 struct {
	Data     DataV2 `json:"Data"`
	Checksum int    `json:"Checksum"`
}

func parseKubeletCheckpoint(filePath string) (interface{}, error) {
	file, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %v", err)
	}

	var contentV2 FileContentV2
	err = json.Unmarshal(file, &contentV2)
	if err == nil {
		klog.Infof("parseKubeletCheckpoint: parsed as V2 format")
		filteredEntries := filterUBEntriesV2(contentV2.Data.PodDeviceEntries)
		contentV2.Data.PodDeviceEntries = filteredEntries
		return &contentV2, nil
	}

	var contentV1 FileContentV1
	err = json.Unmarshal(file, &contentV1)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal file: %v", err)
	}

	klog.Infof("parseKubeletCheckpoint: parsed as V1 format")
	filteredEntries := filterUBEntriesV1(contentV1.Data.PodDeviceEntries)
	contentV1.Data.PodDeviceEntries = filteredEntries
	return &contentV1, nil
}

func filterUBEntriesV1(entries []PodDeviceEntryV1) []PodDeviceEntryV1 {
	result := make([]PodDeviceEntryV1, 0)
	for _, entry := range entries {
		if isUBDevice(entry.ResourceName) {
			result = append(result, entry)
		}
	}
	return result
}

func filterUBEntriesV2(entries []PodDeviceEntryV2) []PodDeviceEntryV2 {
	result := make([]PodDeviceEntryV2, 0)
	for _, entry := range entries {
		if isUBDevice(entry.ResourceName) {
			result = append(result, entry)
		}
	}
	return result
}

func GetAllDeviceIdsFromKubeletCheckpoint(filePath string) ([]string, error) {
	content, err := parseKubeletCheckpoint(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse kubelet checkpoint: %v", err)
	}

	deviceIDMap := make(map[string][]string)

	switch c := content.(type) {
	case *FileContentV1:
		for _, entry := range c.Data.PodDeviceEntries {
			for _, deviceID := range entry.DeviceIDs {
				deviceIDMap[deviceID] = append(deviceIDMap[deviceID], entry.PodUID)
			}
		}
	case *FileContentV2:
		for _, entry := range c.Data.PodDeviceEntries {
			for _, deviceIDs := range entry.DeviceIDs {
				for _, deviceID := range deviceIDs {
					deviceIDMap[deviceID] = append(deviceIDMap[deviceID], entry.PodUID)
				}
			}
		}
	default:
		return nil, fmt.Errorf("unknown checkpoint format")
	}

	res := make([]string, 0, len(deviceIDMap))
	for deviceID := range deviceIDMap {
		res = append(res, deviceID)
	}

	return res, nil
}

func GetPodUIDByDeviceID(deviceID string, filePath string) (string, error) {
	content, err := parseKubeletCheckpoint(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to parse kubelet checkpoint: %v", err)
	}

	switch c := content.(type) {
	case *FileContentV1:
		for _, entry := range c.Data.PodDeviceEntries {
			if slices.Contains(entry.DeviceIDs, deviceID) {
				klog.Infof("PreStartContainer get pod uid: %v", entry.PodUID)
				return entry.PodUID, nil
			}
		}
	case *FileContentV2:
		for _, entry := range c.Data.PodDeviceEntries {
			for _, deviceIDs := range entry.DeviceIDs {
				if slices.Contains(deviceIDs, deviceID) {
					klog.Infof("PreStartContainer get pod uid: %v", entry.PodUID)
					return entry.PodUID, nil
				}
			}
		}
	default:
		return "", fmt.Errorf("unknown checkpoint format")
	}

	return "", fmt.Errorf("PodUID not found")
}

func GetSandboxes(ctx context.Context, client cri.RuntimeService, podUid string) ([]*runtimeapi.PodSandbox, error) {
	filter := &runtimeapi.PodSandboxFilter{
		LabelSelector: map[string]string{types.KubernetesPodUIDLabel: podUid},
	}
	return client.ListPodSandbox(ctx, filter)
}

func GetSandboxPid(ctx context.Context, client cri.RuntimeService, sandboxId string) (string, error) {
	klog.Infof("GetSandboxPid sandboxId: %v", sandboxId)
	sandboxStatus, err := client.PodSandboxStatus(ctx, sandboxId, true)
	if err != nil {
		return "", err
	}
	data, err := parseJsonStr(sandboxStatus.Info["info"])
	if err != nil {
		return "", err
	}
	pid, ok := data["pid"].(float64)
	if !ok {
		return "", fmt.Errorf("failed to get pid")
	}
	klog.Infof("GetSandboxPid pid: %v", pid)

	return strconv.Itoa(int(pid)), nil
}

func parseJsonStr(jsonStr string) (map[string]interface{}, error) {
	var data map[string]interface{}
	err := json.Unmarshal([]byte(jsonStr), &data)
	if err != nil {
		return nil, err
	}

	return data, nil
}

func EnsureDevicePermission(path string, perm int) error {
	fileInfo, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("file %s does not exist", path)
		}
		return fmt.Errorf("failed to stat device file: %w", err)
	}

	currentPerm := fileInfo.Mode().Perm()
	if currentPerm == os.FileMode(perm) {
		klog.Infof("file %s already has permission %o, skipping", TidDevicePath, perm)
		return nil
	}

	if err = os.Chmod(TidDevicePath, os.FileMode(perm)); err != nil {
		return fmt.Errorf("failed to chmod %s to %o: %w", TidDevicePath, perm, err)
	}

	klog.Infof("Successfully set file %s permission to %o", TidDevicePath, perm)
	return nil
}

func isUBDevice(resName string) bool {
	return resName == ResourceName
}
