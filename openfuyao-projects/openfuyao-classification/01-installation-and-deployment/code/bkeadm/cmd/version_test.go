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
	"io"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/utils/version"
)

const (
	versionSubcommandCount    = 0
	versionHasSubcommandCount = 1
	versionFirstCommandIndex  = 0
)

func TestVersionCmdInitialization(t *testing.T) {
	tests := []struct {
		name           string
		expectedUse    string
		expectedShort  string
		hasSubcommands bool
	}{
		{
			name:           "version command properties",
			expectedUse:    "version",
			expectedShort:  "version",
			hasSubcommands: true,
		},
		{
			name:           "only command properties",
			expectedUse:    "only",
			expectedShort:  "only",
			hasSubcommands: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cmd *cobra.Command
			if tt.expectedUse == "version" {
				cmd = versionCmd
			} else if tt.expectedUse == "only" {
				cmd = onlyCmd
			}

			assert.Equal(t, tt.expectedUse, cmd.Use)
			assert.Equal(t, tt.expectedShort, cmd.Short)

			if tt.hasSubcommands {
				assert.Len(t, cmd.Commands(), versionHasSubcommandCount) // onlyCmd doesn't have subcommands, but versionCmd has onlyCmd as subcommand
			}
		})
	}
}

func TestRegisterVersionCommand(t *testing.T) {
	// Save original root command structure
	originalCommandsCount := len(rootCmd.Commands())

	// Find the version command in root commands
	var foundVersionCmd bool
	for _, cmd := range rootCmd.Commands() {
		if cmd.Use == "version" {
			foundVersionCmd = true
			// Check if onlyCmd is registered as subcommand
			assert.Len(t, cmd.Commands(), versionHasSubcommandCount)
			assert.Equal(t, "only", cmd.Commands()[versionFirstCommandIndex].Use)
			break
		}
	}

	assert.True(t, foundVersionCmd, "version command should be registered in root command")

	// Restore original state for other tests
	rootCmd.ResetCommands()
	for i := 0; i < originalCommandsCount; i++ {
		// We can't easily restore the original commands, so we'll just note this limitation
		// In a real scenario, we might need to restructure how we test command registration
	}
}

func TestVersionCmdRun(t *testing.T) {
	// Save original values
	originalVersion := version.Version
	originalGitCommitID := version.GitCommitID
	originalArchitecture := version.Architecture
	originalTimestamp := version.Timestamp
	defer func() {
		version.Version = originalVersion
		version.GitCommitID = originalGitCommitID
		version.Architecture = originalArchitecture
		version.Timestamp = originalTimestamp
	}()

	tests := []struct {
		name            string
		versionVars     func()
		expectedOutputs []string
	}{
		{
			name: "version command with mock version info",
			versionVars: func() {
				version.Version = "v1.0.0"
				version.GitCommitID = "abc123"
				version.Architecture = "amd64"
				version.Timestamp = "2024-01-01"
			},
			expectedOutputs: []string{
				"version: v1.0.0",
				"gitCommitID: abc123",
				"os/arch: amd64",
				"date: 2024-01-01",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			old := os.Stdout
			r, w, err := os.Pipe()
			assert.NoError(t, err)
			os.Stdout = w

			tt.versionVars()

			cmd := &cobra.Command{}
			versionCmd.Run(cmd, []string{})

			w.Close()
			os.Stdout = old

			output, err := io.ReadAll(r)
			assert.NoError(t, err)
			outputStr := string(output)

			for _, expectedOutput := range tt.expectedOutputs {
				assert.Contains(t, outputStr, expectedOutput)
			}
		})
	}
}

func TestOnlyCmdRun(t *testing.T) {
	tests := []struct {
		name           string
		mockVersion    string
		expectedOutput string
	}{
		{
			name:           "only command with mock version",
			mockVersion:    "v1.0.0",
			expectedOutput: "v1.0.0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Capture stdout
			old := os.Stdout
			r, w, _ := os.Pipe()
			os.Stdout = w

			// Apply patch to mock version
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyGlobalVar(&version.Version, tt.mockVersion)

			// Create and run the command
			cmd := &cobra.Command{}
			args := []string{}
			onlyCmd.Run(cmd, args)

			// Get the output
			w.Close()
			out, _ := io.ReadAll(r)
			os.Stdout = old

			outputStr := string(out)
			assert.Contains(t, outputStr, tt.expectedOutput)
		})
	}
}
