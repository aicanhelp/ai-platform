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
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	startImageRepoPort      = "5000"
	startYumRepoPort        = "8080"
	startChartRepoPort      = "8443"
	testImageContainerName  = "test-image"
	testImageAddress        = "test-image:latest"
	testYumContainerName    = "test-yum"
	testYumAddress          = "test-yum:latest"
	testNfsContainerName    = "test-nfs"
	testNfsAddress          = "test-nfs:latest"
	testChartContainerName  = "test-chart"
	testChartAddress        = "test-chart:latest"
	capturedParamNameIndex  = 0
	capturedParamImageIndex = 1
	capturedParamPortIndex  = 2
	capturedParamDataIndex  = 3
)

func TestStartCmdInitialization(t *testing.T) {
	tests := []struct {
		name           string
		cmd            *cobra.Command
		expectedUse    string
		expectedShort  string
		hasSubcommands bool
	}{
		{
			name:           "start command properties",
			cmd:            startCmd,
			expectedUse:    "start",
			expectedShort:  "Start some basic fixed services.",
			hasSubcommands: true,
		},
		{
			name:           "image server start command properties",
			cmd:            imageServerStartCmd,
			expectedUse:    "image",
			expectedShort:  "Starting the Mirror Repository.",
			hasSubcommands: false,
		},
		{
			name:           "yum server start command properties",
			cmd:            yumServerStartCmd,
			expectedUse:    "yum",
			expectedShort:  "Starting the yum Repository.",
			hasSubcommands: false,
		},
		{
			name:           "nfs server start command properties",
			cmd:            nfsServerStartCmd,
			expectedUse:    "nfs",
			expectedShort:  "Starting the nfs Repository.",
			hasSubcommands: false,
		},
		{
			name:           "chart server start command properties",
			cmd:            chartServerStartCmd,
			expectedUse:    "chart",
			expectedShort:  "Starting the chart Repository.",
			hasSubcommands: false,
		},
		{
			name:           "ntp server start command properties",
			cmd:            ntpServerStartCmd,
			expectedUse:    "ntpserver",
			expectedShort:  "Starting the NTP service.",
			hasSubcommands: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expectedUse, tt.cmd.Use)
			assert.Equal(t, tt.expectedShort, tt.cmd.Short)
		})
	}
}

func TestEnsureDataDir(t *testing.T) {
	tests := []struct {
		name         string
		path         string
		mockExists   func(string) bool
		mockMkdirAll func(string, os.FileMode) error
		expectError  bool
	}{
		{
			name: "directory exists",
			path: "/existing/path",
			mockExists: func(path string) bool {
				return true
			},
			mockMkdirAll: func(path string, perm os.FileMode) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "directory does not exist and creation succeeds",
			path: "/new/path",
			mockExists: func(path string) bool {
				return false
			},
			mockMkdirAll: func(path string, perm os.FileMode) error {
				return nil
			},
			expectError: false,
		},
		{
			name: "directory does not exist and creation fails",
			path: "/error/path",
			mockExists: func(path string) bool {
				return false
			},
			mockMkdirAll: func(path string, perm os.FileMode) error {
				return fmt.Errorf("mkdir error")
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, tt.mockExists)

			patches.ApplyFunc(os.MkdirAll, tt.mockMkdirAll)

			err := ensureDataDir(tt.path)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestImageServerStartCmdRun(t *testing.T) {
	tests := []struct {
		name                     string
		dataDir                  string
		mockExists               func(string) bool
		mockStartImageRegistry   func(string, string, string, string) error
		expectStartImageRegistry bool
	}{
		{
			name:    "successful image server start",
			dataDir: "/tmp/test-image-data",
			mockExists: func(path string) bool {
				return true // Data directory exists
			},
			mockStartImageRegistry: func(name, image, port, data string) error {
				return nil
			},
			expectStartImageRegistry: true,
		},
		{
			name:    "image server start with non-existent data dir",
			dataDir: "/tmp/test-new-image-data",
			mockExists: func(path string) bool {
				return false // Data directory doesn't exist initially
			},
			mockStartImageRegistry: func(name, image, port, data string) error {
				return nil
			},
			expectStartImageRegistry: true,
		},
		{
			name:    "image server start fails",
			dataDir: "/tmp/test-fail-data",
			mockExists: func(path string) bool {
				return true
			},
			mockStartImageRegistry: func(name, image, port, data string) error {
				return fmt.Errorf("start failed")
			},
			expectStartImageRegistry: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a temporary command with flags
			cmd := &cobra.Command{}
			cmd.Flags().String("data", tt.dataDir, "data directory")
			cmd.Flags().String("name", testImageContainerName, "container name")
			cmd.Flags().String("image", testImageAddress, "image address")
			cmd.Flags().String("port", startImageRepoPort, "service port")

			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.Exists, tt.mockExists)

			startCalled := false
			var capturedParams []string
			patches.ApplyFunc(server.StartImageRegistry, func(name, image, port, data string) error {
				startCalled = true
				capturedParams = []string{name, image, port, data}
				return tt.mockStartImageRegistry(name, image, port, data)
			})

			// Also mock os.MkdirAll to simulate directory creation when needed
			patches.ApplyFunc(os.MkdirAll, func(path string, perm os.FileMode) error {
				return nil
			})

			// Run the command
			imageServerStartCmd.Run(cmd, []string{})

			if tt.expectStartImageRegistry {
				assert.True(t, startCalled, "StartImageRegistry should have been called")
				if startCalled {
					assert.Equal(t, testImageContainerName, capturedParams[capturedParamNameIndex])
					assert.Equal(t, testImageAddress, capturedParams[capturedParamImageIndex])
					assert.Equal(t, startImageRepoPort, capturedParams[capturedParamPortIndex])
					assert.Equal(t, tt.dataDir, capturedParams[capturedParamDataIndex])
				}
			} else {
				assert.False(t, startCalled, "StartImageRegistry should not have been called")
			}
		})
	}
}

func TestYumServerStartCmdRun(t *testing.T) {
	// Create a temporary command with flags
	cmd := &cobra.Command{}
	cmd.Flags().String("data", "/tmp/test-yum-data", "data directory")
	cmd.Flags().String("name", testYumContainerName, "container name")
	cmd.Flags().String("image", testYumAddress, "image address")
	cmd.Flags().String("port", startYumRepoPort, "service port")

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true // Data directory exists
	})

	startCalled := false
	patches.ApplyFunc(server.StartYumRegistry, func(name, image, port, data string) error {
		startCalled = true
		return nil
	})

	// Run the command
	yumServerStartCmd.Run(cmd, []string{})

	assert.True(t, startCalled, "StartYumRegistry should have been called")
}

func TestNFSserverStartCmdRun(t *testing.T) {
	// Create a temporary command with flags
	cmd := &cobra.Command{}
	cmd.Flags().String("data", "/tmp/test-nfs-data", "data directory")
	cmd.Flags().String("name", testNfsContainerName, "container name")
	cmd.Flags().String("image", testNfsAddress, "image address")

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true // Data directory exists
	})

	startCalled := false
	patches.ApplyFunc(server.StartNFSServer, func(name, image, data string) error {
		startCalled = true
		return nil
	})

	// Run the command
	nfsServerStartCmd.Run(cmd, []string{})

	assert.True(t, startCalled, "StartNFSServer should have been called")
}

func TestChartServerStartCmdRun(t *testing.T) {
	// Create a temporary command with flags
	cmd := &cobra.Command{}
	cmd.Flags().String("data", "/tmp/test-chart-data", "data directory")
	cmd.Flags().String("name", testChartContainerName, "container name")
	cmd.Flags().String("image", testChartAddress, "image address")
	cmd.Flags().String("port", startChartRepoPort, "service port")

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(utils.Exists, func(path string) bool {
		return true // Data directory exists
	})

	startCalled := false
	patches.ApplyFunc(server.StartChartRegistry, func(name, image, port, data string) error {
		startCalled = true
		return nil
	})

	// Run the command
	chartServerStartCmd.Run(cmd, []string{})

	assert.True(t, startCalled, "StartChartRegistry should have been called")
}
