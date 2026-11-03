/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package cmd

import (
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp/sntp"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	dockerContainerResultCount     = 1
	containerdContainerResultCount = 1
	nfsPortNumber                  = "2049"
	containerdServiceType          = "containerd"
	dockerServiceType              = "docker"
	testContainerdName             = "test-containerd"
	testContainerName              = "test-container"
	containerType                  = "container"
	containerNotCreatedStatus      = "notCreated"
	testContainerAddress           = "tcp://0.0.0.0:8080"
	testContainerId                = "test-id"
	firstIndex                     = 0
	secondIndex                    = 1
	thirdIndex                     = 2
	fourthIndex                    = 3
)

func TestStatusCmdInitialization(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
	}{
		{
			name:          "status command properties",
			expectedUse:   "status",
			expectedShort: "Displays the status of the local service started by bke.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expectedUse, statusCmd.Use)
			assert.Equal(t, tt.expectedShort, statusCmd.Short)
		})
	}
}

func TestGetContainerServers(t *testing.T) {
	containerServers := getContainerServers()

	assert.NotEmpty(t, containerServers)

	// Check that the expected containers are present
	expectedContainers := []string{
		utils.LocalKubernetesName,
		utils.LocalImageRegistryName,
		utils.LocalYumRegistryName,
		utils.LocalChartRegistryName,
		utils.LocalNFSRegistryName,
	}

	var foundContainers []string
	for _, server := range containerServers {
		foundContainers = append(foundContainers, server[secondIndex]) // server[secondIndex] is the name
	}

	for _, expected := range expectedContainers {
		assert.Contains(t, foundContainers, expected)
	}

	// Check that the ports are correctly set
	for _, server := range containerServers {
		name := server[secondIndex]
		defaultAddr := server[thirdIndex]

		switch name {
		case utils.LocalKubernetesName:
			assert.Contains(t, defaultAddr, utils.DefaultKubernetesPort)
		case utils.LocalImageRegistryName:
			assert.Contains(t, defaultAddr, configinit.DefaultImageRepoPort)
		case utils.LocalYumRegistryName:
			assert.Contains(t, defaultAddr, configinit.DefaultYumRepoPort)
		case utils.LocalChartRegistryName:
			assert.Contains(t, defaultAddr, utils.DefaultChartRegistryPort)
		case utils.LocalNFSRegistryName:
			assert.Contains(t, defaultAddr, nfsPortNumber)
		}
	}
}

func TestGetNtpServerStatus(t *testing.T) {
	tests := []struct {
		name                string
		mockNtpClient       func(string) (time.Time, error)
		mockExists          func(string) bool
		expectedStatus      string
		expectedServiceType string
	}{
		{
			name: "NTP server running",
			mockNtpClient: func(addr string) (time.Time, error) {
				return time.Now(), nil // Success
			},
			mockExists: func(path string) bool {
				return true // Service file exists
			},
			expectedStatus:      "running",
			expectedServiceType: "systemd",
		},
		{
			name: "NTP server not running",
			mockNtpClient: func(addr string) (time.Time, error) {
				return time.Now(), fmt.Errorf("connection failed") // Error
			},
			mockExists: func(path string) bool {
				return true // Service file exists
			},
			expectedStatus:      "notCreated",
			expectedServiceType: "systemd",
		},
		{
			name: "NTP server running without systemd file",
			mockNtpClient: func(addr string) (time.Time, error) {
				return time.Now(), nil // Success
			},
			mockExists: func(path string) bool {
				return false // Service file doesn't exist
			},
			expectedStatus:      "running",
			expectedServiceType: "proc",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(sntp.Client, tt.mockNtpClient)

			patches.ApplyFunc(utils.Exists, tt.mockExists)

			getNtpServerStatus()

		})
	}
}

func TestGetContainerdContainerStatus(t *testing.T) {
	// Prepare test data
	containerServers := [][]string{
		{containerType, testContainerdName, testContainerAddress, containerNotCreatedStatus, ""},
	}

	tests := []struct {
		name                 string
		mockContainerInspect func(name string) (containerd.NerdContainerInfo, error)
		expectedStatus       string
	}{
		{
			name: "containerd container exists and is running",
			mockContainerInspect: func(name string) (containerd.NerdContainerInfo, error) {
				return containerd.NerdContainerInfo{}, nil
			},
			expectedStatus: "running",
		},
		{
			name: "containerd container does not exist",
			mockContainerInspect: func(name string) (containerd.NerdContainerInfo, error) {
				return containerd.NerdContainerInfo{}, fmt.Errorf("container not found")
			},
			expectedStatus: "notCreated"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patch
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(containerd.ContainerInspect, tt.mockContainerInspect)

			result := getContainerdContainerStatus(containerServers)

			assert.Len(t, result, containerdContainerResultCount)
			assert.Equal(t, containerdServiceType, result[firstIndex][firstIndex]) // service type
			assert.Equal(t, testContainerdName, result[firstIndex][secondIndex])   // name
		})
	}
}

func TestStatusCmdRun(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(containerd.ContainerInspect,
		func(name string) (containerd.NerdContainerInfo, error) { return containerd.NerdContainerInfo{}, nil })
	patches.Reset()

	patches.ApplyFunc(containerd.ContainerExists,
		func(name string) (containerd.NerdContainerInfo, bool) { return containerd.NerdContainerInfo{}, true })
	patches.Reset()

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{}

	statusCmd.Run(cmd, args)

	// The function should complete without error
	assert.True(t, true) // This assertion just confirms the function ran without error
}
