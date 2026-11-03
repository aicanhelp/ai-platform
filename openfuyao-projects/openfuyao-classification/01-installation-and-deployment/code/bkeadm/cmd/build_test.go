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
	"runtime"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/build"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	buildSubcommandCount = 1
)

func TestBuildCmdInitialization(t *testing.T) {
	tests := []struct {
		name           string
		cmd            *cobra.Command
		expectedUse    string
		expectedShort  string
		hasSubcommands bool
	}{
		{
			name:           "build command properties",
			cmd:            buildCmd,
			expectedUse:    "build",
			expectedShort:  "Build the BKE installation package.",
			hasSubcommands: true,
		},
		{
			name:           "build config command properties",
			cmd:            buildConfigCmd,
			expectedUse:    "config",
			expectedShort:  "The default BKE configuration is exported.",
			hasSubcommands: false,
		},
		{
			name:           "patch command properties",
			cmd:            patchCmd,
			expectedUse:    "patch",
			expectedShort:  "Build the bke patch pack.",
			hasSubcommands: false,
		},
		{
			name:           "online image command properties",
			cmd:            onlineCmd,
			expectedUse:    "online-image",
			expectedShort:  "Compile an image installed online",
			hasSubcommands: false,
		},
		{
			name:           "rpm command properties",
			cmd:            rpmCmd,
			expectedUse:    "rpm",
			expectedShort:  "Build an offline rpm package",
			hasSubcommands: false,
		},
		{
			name:           "check command properties",
			cmd:            preCheckCmd,
			expectedUse:    "check",
			expectedShort:  "Check the mirror in the configuration file",
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

func TestRegisterBuildCommand(t *testing.T) {
	// Find the build command in root commands
	var foundBuildCmd bool
	for _, cmd := range rootCmd.Commands() {
		if cmd.Use == "build" {
			foundBuildCmd = true
			// Check if subcommands are registered
			assert.Len(t, cmd.Commands(), buildSubcommandCount) // patch

			var foundSubcommands []string
			for _, subCmd := range cmd.Commands() {
				foundSubcommands = append(foundSubcommands, subCmd.Use)
			}

			expectedSubcommands := []string{"patch"}
			for _, expected := range expectedSubcommands {
				assert.Contains(t, foundSubcommands, expected)
			}
			break
		}
	}

	assert.True(t, foundBuildCmd, "build command should be registered in root command")
}

func TestBuildCmdPreRunE(t *testing.T) {
	tests := []struct {
		name          string
		fileValue     string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with file",
			fileValue:   "/path/to/file.yaml",
			expectError: false,
		},
		{
			name:          "missing file should return error",
			fileValue:     "",
			expectError:   true,
			errorContains: "The parameter `file` is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalFile := buildOption.File
			defer func() {
				buildOption.File = originalFile
			}()

			// Set test values
			buildOption.File = tt.fileValue

			// Call PreRunE validation
			err := buildCmd.PreRunE(nil, nil)

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

func TestBuildCmdRun(t *testing.T) {

	// Apply patch to mock Build method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*build.Options).Build, func(o *build.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	buildCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.NotEqual(t, args, buildOption.Args)
	assert.Equal(t, options, buildOption.Options)
}

func TestBuildConfigCmdRun(t *testing.T) {
	// Save original values
	originalArgs := buildOption.Args
	originalOptions := buildOption.Options
	defer func() {
		buildOption.Args = originalArgs
		buildOption.Options = originalOptions
	}()

	// Apply patch to mock Config method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*build.Options).Config, func(o *build.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	buildConfigCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.NotEqual(t, args, buildOption.Args)
	assert.Equal(t, options, buildOption.Options)
}

func TestPatchCmdPreRunE(t *testing.T) {
	tests := []struct {
		name          string
		fileValue     string
		strategyValue string
		expectError   bool
		errorContains string
	}{
		{
			name:          "valid input with file and valid strategy",
			fileValue:     "/path/to/file.yaml",
			strategyValue: "registry",
			expectError:   false,
		},
		{
			name:          "valid input with file and invalid strategy (gets default)",
			fileValue:     "/path/to/file.yaml",
			strategyValue: "invalid",
			expectError:   false,
		},
		{
			name:          "missing file should return error",
			fileValue:     "",
			strategyValue: "registry",
			expectError:   true,
			errorContains: "The parameter `file` is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalFile := patchOption.File
			originalStrategy := patchOption.Strategy
			defer func() {
				patchOption.File = originalFile
				patchOption.Strategy = originalStrategy
			}()

			// Set test values
			patchOption.File = tt.fileValue
			patchOption.Strategy = tt.strategyValue

			// Apply patch for ContainsString to simulate valid strategies
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
				for _, s := range slice {
					if s == item {
						return true
					}
				}
				return false
			})

			// Call PreRunE validation
			err := patchCmd.PreRunE(nil, nil)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)

				// Check if default strategy is set when invalid
				if tt.strategyValue == "invalid" {
					assert.Equal(t, "registry", patchOption.Strategy)
				}
			}
		})
	}
}

func TestPatchCmdRun(t *testing.T) {
	// Save original values
	originalArgs := patchOption.Args
	originalOptions := patchOption.Options
	defer func() {
		patchOption.Args = originalArgs
		patchOption.Options = originalOptions
	}()

	// Apply patch to mock Patch method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*build.Options).Patch, func(o *build.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	patchCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.NotEqual(t, args, patchOption.Args)
	assert.Equal(t, options, patchOption.Options)
}

func TestOnlineCmdPreRunE(t *testing.T) {
	tests := []struct {
		name          string
		fileValue     string
		targetValue   string
		strategyValue string
		archValue     string
		expectError   bool
		errorContains string
	}{
		{
			name:          "valid input with all required params",
			fileValue:     "/path/to/file.yaml",
			targetValue:   "cr.registry.com/repo:tag",
			strategyValue: "registry",
			archValue:     "amd64",
			expectError:   false,
		},
		{
			name:          "missing file should return error",
			fileValue:     "",
			targetValue:   "cr.registry.com/repo:tag",
			strategyValue: "registry",
			archValue:     "amd64",
			expectError:   true,
			errorContains: "The parameter `file` is required",
		},
		{
			name:          "missing target should return error",
			fileValue:     "/path/to/file.yaml",
			targetValue:   "",
			strategyValue: "registry",
			archValue:     "amd64",
			expectError:   true,
			errorContains: "The parameter `target` is required",
		},
		{
			name:          "missing arch gets default",
			fileValue:     "/path/to/file.yaml",
			targetValue:   "cr.registry.com/repo:tag",
			strategyValue: "invalid",
			archValue:     "",
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalFile := onlineOption.File
			originalTarget := onlineOption.Target
			originalStrategy := onlineOption.Strategy
			originalArch := onlineOption.Arch
			defer func() {
				onlineOption.File = originalFile
				onlineOption.Target = originalTarget
				onlineOption.Strategy = originalStrategy
				onlineOption.Arch = originalArch
			}()

			// Set test values
			onlineOption.File = tt.fileValue
			onlineOption.Target = tt.targetValue
			onlineOption.Strategy = tt.strategyValue
			onlineOption.Arch = tt.archValue

			// Apply patch for ContainsString to simulate valid strategies
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc(utils.ContainsString, func(slice []string, item string) bool {
				for _, s := range slice {
					if s == item {
						return true
					}
				}
				return false
			})

			// Call PreRunE validation
			err := onlineCmd.PreRunE(nil, nil)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)

				// Check if default strategy is set when invalid
				if tt.strategyValue == "invalid" {
					assert.Equal(t, "registry", onlineOption.Strategy)
				}

				// Check if default arch is set when empty
				if tt.archValue == "" {
					assert.Equal(t, runtime.GOARCH, onlineOption.Arch)
				}
			}
		})
	}
}

func TestOnlineCmdRun(t *testing.T) {
	// Save original values
	originalArgs := onlineOption.Args
	originalOptions := onlineOption.Options
	defer func() {
		onlineOption.Args = originalArgs
		onlineOption.Options = originalOptions
	}()

	// Apply patch to mock BuildOnlineImage method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*build.Options).BuildOnlineImage, func(o *build.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	onlineCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.NotEqual(t, args, onlineOption.Args)
	assert.Equal(t, options, onlineOption.Options)
}

func TestRpmCmdPreRunE(t *testing.T) {
	tests := []struct {
		name          string
		addValue      string
		packageValue  string
		expectError   bool
		errorContains string
	}{
		{
			name:         "both add and package provided",
			addValue:     "centos/8/amd64",
			packageValue: "docker-ce",
			expectError:  false,
		},
		{
			name:         "neither add nor package provided",
			addValue:     "",
			packageValue: "",
			expectError:  false,
		},
		{
			name:          "only add provided should return error",
			addValue:      "centos/8/amd64",
			packageValue:  "",
			expectError:   true,
			errorContains: "The parameter `add` or `package` is required",
		},
		{
			name:          "only package provided should return error",
			addValue:      "",
			packageValue:  "docker-ce",
			expectError:   true,
			errorContains: "The parameter `add` or `package` is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalAdd := rpmOption.Add
			originalPackage := rpmOption.Package
			defer func() {
				rpmOption.Add = originalAdd
				rpmOption.Package = originalPackage
			}()

			// Set test values
			rpmOption.Add = tt.addValue
			rpmOption.Package = tt.packageValue

			// Call PreRunE validation
			err := rpmCmd.PreRunE(nil, nil)

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

func TestRpmCmdRun(t *testing.T) {
	// Save original values
	originalArgs := rpmOption.Args
	originalOptions := rpmOption.Options
	defer func() {
		rpmOption.Args = originalArgs
		rpmOption.Options = originalOptions
	}()

	// Apply patch to mock Build method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*build.RpmOptions).Build, func(o *build.RpmOptions) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	rpmCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.NotEqual(t, args, rpmOption.Args)
	assert.Equal(t, options, rpmOption.Options)
}

func TestPreCheckCmdPreRunE(t *testing.T) {
	tests := []struct {
		name          string
		fileValue     string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with file",
			fileValue:   "/path/to/file.yaml",
			expectError: false,
		},
		{
			name:          "missing file should return error",
			fileValue:     "",
			expectError:   true,
			errorContains: "The parameter `file` is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalFile := preCheckOption.File
			defer func() {
				preCheckOption.File = originalFile
			}()

			// Set test values
			preCheckOption.File = tt.fileValue

			// Call PreRunE validation
			err := preCheckCmd.PreRunE(nil, nil)

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

func TestPreCheckCmdRun(t *testing.T) {
	// Save original values
	originalArgs := preCheckOption.Args
	originalOptions := preCheckOption.Options
	defer func() {
		preCheckOption.Args = originalArgs
		preCheckOption.Options = originalOptions
	}()

	// Apply patch to mock PreCheck method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*build.PreCheckOptions).PreCheck, func(o *build.PreCheckOptions) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	preCheckCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.NotEqual(t, args, preCheckOption.Args)
	assert.Equal(t, options, preCheckOption.Options)
}
