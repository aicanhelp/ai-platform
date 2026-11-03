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
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/reset"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestResetCmdInitialization(t *testing.T) {
	tests := []struct {
		name          string
		expectedUse   string
		expectedShort string
		hasFlags      bool
	}{
		{
			name:          "reset command properties",
			expectedUse:   "reset",
			expectedShort: "Remove the local Kubernetes cluster",
			hasFlags:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expectedUse, resetCmd.Use)
			assert.Equal(t, tt.expectedShort, resetCmd.Short)

			if tt.hasFlags {
				// Check if required flags exist
				allFlag := resetCmd.Flags().Lookup("all")
				assert.NotNil(t, allFlag)

				mountFlag := resetCmd.Flags().Lookup("mount")
				assert.NotNil(t, mountFlag)

				confirmFlag := resetCmd.Flags().Lookup("confirm")
				assert.NotNil(t, confirmFlag)
			}
		})
	}
}

func TestRegisterResetCommand(t *testing.T) {
	// Find the reset command in root commands
	var foundResetCmd bool
	for _, cmd := range rootCmd.Commands() {
		if cmd.Use == "reset" {
			foundResetCmd = true
			break
		}
	}

	assert.True(t, foundResetCmd, "reset command should be registered in root command")
}

func TestResetCmdPreRunE(t *testing.T) {
	tests := []struct {
		name                      string
		allOption                 bool
		confirmOption             bool
		mockPromptForConfirmation func(bool) bool
		expectError               bool
		errorContains             string
	}{
		{
			name:                      "all option true with confirmation",
			allOption:                 true,
			confirmOption:             true,
			mockPromptForConfirmation: func(bool) bool { return true },
			expectError:               false,
		},
		{
			name:                      "all option true without confirmation should return error",
			allOption:                 true,
			confirmOption:             false,
			mockPromptForConfirmation: func(bool) bool { return false },
			expectError:               true,
			errorContains:             "operation cancelled by user",
		},
		{
			name:                      "all option false should not require confirmation",
			allOption:                 false,
			confirmOption:             false,
			mockPromptForConfirmation: func(bool) bool { return false }, // This shouldn't be called
			expectError:               false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalAll := resetOption.All
			originalConfirm := confirm
			defer func() {
				resetOption.All = originalAll
				confirm = originalConfirm
			}()

			// Set test values
			resetOption.All = tt.allOption
			confirm = tt.confirmOption

			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.PromptForConfirmation, tt.mockPromptForConfirmation)

			// Call PreRunE
			err := resetCmd.PreRunE(nil, nil)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestResetCmdRun(t *testing.T) {
	// Save original values
	originalOptions := resetOption.Options
	originalArgs := resetOption.Args
	defer func() {
		resetOption.Options = originalOptions
		resetOption.Args = originalArgs
	}()

	// Apply patch to mock the Reset method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*reset.Options).Reset, func(o *reset.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	resetCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, resetOption.Args)
	assert.Equal(t, options, resetOption.Options)
}

func TestResetCmdRunWithAllOption(t *testing.T) {
	// Save original values
	originalOptions := resetOption.Options
	originalArgs := resetOption.Args
	originalAll := resetOption.All
	defer func() {
		resetOption.Options = originalOptions
		resetOption.Args = originalArgs
		resetOption.All = originalAll
	}()

	// Set the All option to true
	resetOption.All = true

	// Apply patch to mock the Reset method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*reset.Options).Reset, func(o *reset.Options) {
		// Mock implementation - do nothing
	})

	// Apply patch to mock the PromptForConfirmation function
	patches.ApplyFunc(utils.PromptForConfirmation, func(confirmed bool) bool {
		return true // Simulate user confirmation
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	// First run PreRunE to handle the confirmation logic
	err := resetCmd.PreRunE(cmd, args)
	assert.NoError(t, err)

	// Then run the actual command
	resetCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, resetOption.Args)
	assert.Equal(t, options, resetOption.Options)
	assert.True(t, resetOption.All)
}
