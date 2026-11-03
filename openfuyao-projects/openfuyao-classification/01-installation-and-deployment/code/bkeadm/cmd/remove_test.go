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

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestRemoveCmdInitialization(t *testing.T) {
	tests := []struct {
		name           string
		cmd            *cobra.Command
		expectedUse    string
		expectedShort  string
		hasSubcommands bool
	}{
		{
			name:           "remove command properties",
			cmd:            removeCmd,
			expectedUse:    "remove",
			expectedShort:  "Removing dependent Services.",
			hasSubcommands: true,
		},
		{
			name:           "image server remove command properties",
			cmd:            imageServerRemoveCmd,
			expectedUse:    "image",
			expectedShort:  "Removing the Mirror Repository.",
			hasSubcommands: false,
		},
		{
			name:           "yum server remove command properties",
			cmd:            yumServerRemoveCmd,
			expectedUse:    "yum",
			expectedShort:  "Removing the yum Repository.",
			hasSubcommands: false,
		},
		{
			name:           "nfs server remove command properties",
			cmd:            nfsServerRemoveCmd,
			expectedUse:    "nfs",
			expectedShort:  "Removing the nfs Repository.",
			hasSubcommands: false,
		},
		{
			name:           "chart server remove command properties",
			cmd:            chartServerRemoveCmd,
			expectedUse:    "chart",
			expectedShort:  "Removing the chart Repository.",
			hasSubcommands: false,
		},
		{
			name:           "ntp server remove command properties",
			cmd:            ntpServerRemoveCmd,
			expectedUse:    "ntpserver",
			expectedShort:  "Removing the ntp service.",
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

func TestImageServerRemoveCmdRun(t *testing.T) {
	tests := []struct {
		name                    string
		args                    []string
		mockRemoveImageRegistry func(string) error
		expectedName            string
	}{
		{
			name:         "remove with default name",
			args:         []string{},
			expectedName: utils.LocalImageRegistryName,
			mockRemoveImageRegistry: func(name string) error {
				return nil
			},
		},
		{
			name:         "remove with custom name",
			args:         []string{"custom-image-name"},
			expectedName: "custom-image-name",
			mockRemoveImageRegistry: func(name string) error {
				return nil
			},
		},
		{
			name:         "remove with error",
			args:         []string{},
			expectedName: utils.LocalImageRegistryName,
			mockRemoveImageRegistry: func(name string) error {
				return fmt.Errorf("removal failed")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			removeCalled := false
			var capturedName string
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(server.RemoveImageRegistry, func(name string) error {
				removeCalled = true
				capturedName = name
				return tt.mockRemoveImageRegistry(name)
			})

			// Create and run the command
			cmd := &cobra.Command{}
			imageServerRemoveCmd.Run(cmd, tt.args)

			assert.True(t, removeCalled, "RemoveImageRegistry should have been called")
			assert.Equal(t, tt.expectedName, capturedName)
		})
	}
}

func TestYumServerRemoveCmdRun(t *testing.T) {
	tests := []struct {
		name                  string
		args                  []string
		mockRemoveYumRegistry func(string) error
		expectedName          string
	}{
		{
			name:         "remove with default name",
			args:         []string{},
			expectedName: utils.LocalYumRegistryName,
			mockRemoveYumRegistry: func(name string) error {
				return nil
			},
		},
		{
			name:         "remove with custom name",
			args:         []string{"custom-yum-name"},
			expectedName: "custom-yum-name",
			mockRemoveYumRegistry: func(name string) error {
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			removeCalled := false
			var capturedName string
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(server.RemoveYumRegistry, func(name string) error {
				removeCalled = true
				capturedName = name
				return tt.mockRemoveYumRegistry(name)
			})

			// Create and run the command
			cmd := &cobra.Command{}
			yumServerRemoveCmd.Run(cmd, tt.args)

			assert.True(t, removeCalled, "RemoveYumRegistry should have been called")
			assert.Equal(t, tt.expectedName, capturedName)
		})
	}
}

func TestNFSServerRemoveCmdRun(t *testing.T) {
	tests := []struct {
		name                string
		args                []string
		mockRemoveNFSServer func(string) error
		expectedName        string
	}{
		{
			name:         "remove with default name",
			args:         []string{},
			expectedName: utils.LocalNFSRegistryName,
			mockRemoveNFSServer: func(name string) error {
				return nil
			},
		},
		{
			name:         "remove with custom name",
			args:         []string{"custom-nfs-name"},
			expectedName: "custom-nfs-name",
			mockRemoveNFSServer: func(name string) error {
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			removeCalled := false
			var capturedName string
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(server.RemoveNFSServer, func(name string) error {
				removeCalled = true
				capturedName = name
				return tt.mockRemoveNFSServer(name)
			})

			// Create and run the command
			cmd := &cobra.Command{}
			nfsServerRemoveCmd.Run(cmd, tt.args)

			assert.True(t, removeCalled, "RemoveNFSServer should have been called")
			assert.Equal(t, tt.expectedName, capturedName)
		})
	}
}

func TestChartServerRemoveCmdRun(t *testing.T) {
	tests := []struct {
		name                    string
		args                    []string
		mockRemoveChartRegistry func(string) error
		expectedName            string
	}{
		{
			name:         "remove with default name",
			args:         []string{},
			expectedName: utils.LocalChartRegistryName,
			mockRemoveChartRegistry: func(name string) error {
				return nil
			},
		},
		{
			name:         "remove with custom name",
			args:         []string{"custom-chart-name"},
			expectedName: "custom-chart-name",
			mockRemoveChartRegistry: func(name string) error {
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			removeCalled := false
			var capturedName string
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(server.RemoveChartRegistry, func(name string) error {
				removeCalled = true
				capturedName = name
				return tt.mockRemoveChartRegistry(name)
			})

			// Create and run the command
			cmd := &cobra.Command{}
			chartServerRemoveCmd.Run(cmd, tt.args)

			assert.True(t, removeCalled, "RemoveChartRegistry should have been called")
			assert.Equal(t, tt.expectedName, capturedName)
		})
	}
}

func TestNTPServerRemoveCmdRun(t *testing.T) {
	tests := []struct {
		name                string
		mockRemoveNTPServer func() error
	}{
		{
			name: "remove NTP server successfully",
			mockRemoveNTPServer: func() error {
				return nil
			},
		},
		{
			name: "remove NTP server with error",
			mockRemoveNTPServer: func() error {
				return fmt.Errorf("removal failed")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Apply patches
			removeCalled := false
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(server.RemoveNTPServer, func() error {
				removeCalled = true
				return tt.mockRemoveNTPServer()
			})

			// Create and run the command
			cmd := &cobra.Command{}
			ntpServerRemoveCmd.Run(cmd, []string{})

			assert.True(t, removeCalled, "RemoveNTPServer should have been called")
		})
	}
}
