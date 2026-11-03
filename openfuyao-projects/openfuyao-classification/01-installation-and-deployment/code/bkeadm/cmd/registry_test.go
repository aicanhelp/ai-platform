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
	"net"
	"os"
	"runtime"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/registry"
)

const (
	registrySubcommandCount = 3
	testRegistryPort        = "40443"
	firstArgIndex           = 0
)

var testRegistryIP = net.IPv4(
	testIPv4SegmentA,
	testIPv4SegmentB,
	testIPv4SegmentC,
	testIPv4SegmentD,
)

func TestRegistryCmdInitialization(t *testing.T) {
	tests := []struct {
		name           string
		cmd            *cobra.Command
		expectedUse    string
		expectedShort  string
		hasSubcommands bool
	}{
		{
			name:           "registry command properties",
			cmd:            registryCmd,
			expectedUse:    "registry",
			expectedShort:  "Synchronize images between two mirror repositories",
			hasSubcommands: true,
		},
		{
			name:           "sync command properties",
			cmd:            syncDep,
			expectedUse:    "sync",
			expectedShort:  "In the two mirror repositories, mirrors are synchronized by copying data blocks from one repository to another.",
			hasSubcommands: false,
		},
		{
			name:           "transfer command properties",
			cmd:            transferDep,
			expectedUse:    "transfer",
			expectedShort:  "Transfer images in docker pull / docker push mode",
			hasSubcommands: false,
		},
		{
			name:           "list-tags command properties",
			cmd:            listTagsDep,
			expectedUse:    "list-tags",
			expectedShort:  "Lists all tags the mirror repository",
			hasSubcommands: false,
		},
		{
			name:           "inspect command properties",
			cmd:            inspectDep,
			expectedUse:    "inspect",
			expectedShort:  "List information about images in the mirror repository",
			hasSubcommands: false,
		},
		{
			name:           "manifests command properties",
			cmd:            manifestsDep,
			expectedUse:    "manifests",
			expectedShort:  "Make a multi-architecture wake image",
			hasSubcommands: false,
		},
		{
			name:           "delete command properties",
			cmd:            deleteDep,
			expectedUse:    "delete",
			expectedShort:  "Delete a specified mirror",
			hasSubcommands: false,
		},
		{
			name:           "view command properties",
			cmd:            viewDep,
			expectedUse:    "view",
			expectedShort:  "View warehouse view",
			hasSubcommands: false,
		},
		{
			name:           "patch command properties",
			cmd:            patchDep,
			expectedUse:    "patch",
			expectedShort:  "Specially customized incremental packet mirror synchronization",
			hasSubcommands: false,
		},
		{
			name:           "download command properties",
			cmd:            downloadDep,
			expectedUse:    "download",
			expectedShort:  "Download the specified file in the image",
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

func TestRegisterRegistryCommand(t *testing.T) {
	// Find the registry command in root commands
	var foundRegistryCmd bool
	for _, cmd := range rootCmd.Commands() {
		if cmd.Use == "registry" {
			foundRegistryCmd = true
			// Check if subcommands are registered
			assert.Len(t, cmd.Commands(), registrySubcommandCount) // sync, view, patch

			var foundSubcommands []string
			for _, subCmd := range cmd.Commands() {
				foundSubcommands = append(foundSubcommands, subCmd.Use)
			}

			expectedSubcommands := []string{"sync", "view", "patch"}
			for _, expected := range expectedSubcommands {
				assert.Contains(t, foundSubcommands, expected)
			}
			break
		}
	}

	assert.True(t, foundRegistryCmd, "registry command should be registered in root command")
}

func TestSyncCmdArgs(t *testing.T) {
	tests := []struct {
		name           string
		sourceValue    string
		targetValue    string
		multiArchValue bool
		archValue      string
		expectError    bool
		errorContains  string
	}{
		{
			name:           "valid inputs",
			sourceValue:    "docker.io/library/busybox:1.35",
			targetValue:    "0.0.0.0:40443/library/busybox:1.35",
			multiArchValue: false,
			archValue:      "amd64",
			expectError:    false,
		},
		{
			name:           "missing source should return error",
			sourceValue:    "",
			targetValue:    "0.0.0.0:40443/library/busybox:1.35",
			multiArchValue: false,
			archValue:      "amd64",
			expectError:    true,
			errorContains:  "The `source` parameter is required",
		},
		{
			name:           "missing target should return error",
			sourceValue:    "docker.io/library/busybox:1.35",
			targetValue:    "",
			multiArchValue: false,
			archValue:      "amd64",
			expectError:    true,
			errorContains:  "The `target` parameter is required",
		},
		{
			name:           "multi-arch with arch should return error",
			sourceValue:    "docker.io/library/busybox:1.35",
			targetValue:    "0.0.0.0:40443/library/busybox:1.35",
			multiArchValue: true,
			archValue:      "amd64",
			expectError:    true,
			errorContains:  "The `arch` parameter is not allowed when `multi-arch` is true",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalSource := syncOption.Source
			originalTarget := syncOption.Target
			originalMultiArch := syncOption.MultiArch
			originalArch := syncOption.Arch
			defer func() {
				syncOption.Source = originalSource
				syncOption.Target = originalTarget
				syncOption.MultiArch = originalMultiArch
				syncOption.Arch = originalArch
			}()

			// Set test values
			syncOption.Source = tt.sourceValue
			syncOption.Target = tt.targetValue
			syncOption.MultiArch = tt.multiArchValue
			syncOption.Arch = tt.archValue

			// Call Args validation
			err := syncDep.Args(nil, nil)

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

func TestTransferCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		sourceValue   string
		targetValue   string
		imageValue    string
		fileValue     string
		archValue     string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid inputs with image",
			sourceValue: "docker.io/library/",
			targetValue: "registry.cloud.com/k8s",
			imageValue:  "busybox:1.28",
			fileValue:   "",
			archValue:   "amd64",
			expectError: false,
		},
		{
			name:        "valid inputs with file",
			sourceValue: "docker.io/library/",
			targetValue: "registry.cloud.com/k8s",
			imageValue:  "",
			fileValue:   "/path/to/file.txt",
			archValue:   "amd64",
			expectError: false,
		},
		{
			name:          "missing source should return error",
			sourceValue:   "",
			targetValue:   "registry.cloud.com/k8s",
			imageValue:    "busybox:1.28",
			fileValue:     "",
			archValue:     "amd64",
			expectError:   true,
			errorContains: "The `source` parameter is required",
		},
		{
			name:          "missing target should return error",
			sourceValue:   "docker.io/library/",
			targetValue:   "",
			imageValue:    "busybox:1.28",
			fileValue:     "",
			archValue:     "amd64",
			expectError:   true,
			errorContains: "The `target` parameter is required",
		},
		{
			name:          "missing both image and file should return error",
			sourceValue:   "docker.io/library/",
			targetValue:   "registry.cloud.com/k8s",
			imageValue:    "",
			fileValue:     "",
			archValue:     "amd64",
			expectError:   true,
			errorContains: "There must be one of the parameters `image` and `file`",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalSource := transferOption.Source
			originalTarget := transferOption.Target
			originalImage := transferOption.Image
			originalFile := transferOption.File
			originalArch := transferOption.Arch
			defer func() {
				transferOption.Source = originalSource
				transferOption.Target = originalTarget
				transferOption.Image = originalImage
				transferOption.File = originalFile
				transferOption.Arch = originalArch
			}()

			// Set test values
			transferOption.Source = tt.sourceValue
			transferOption.Target = tt.targetValue
			transferOption.Image = tt.imageValue
			transferOption.File = tt.fileValue
			transferOption.Arch = tt.archValue

			// Call Args validation
			err := transferDep.Args(nil, nil)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)

				// Check that arch is set to runtime.GOARCH if empty
				if tt.archValue == "" {
					assert.Equal(t, runtime.GOARCH, transferOption.Arch)
				}
			}
		})
	}
}

func TestListTagsCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with image arg",
			args:        []string{"registry.example.com/image:tag"},
			expectError: false,
		},
		{
			name:          "missing image arg should return error",
			args:          []string{},
			expectError:   true,
			errorContains: "The `image` is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalImage := listTagsOption.Image
			defer func() {
				listTagsOption.Image = originalImage
			}()

			// Call Args validation
			err := listTagsDep.Args(nil, tt.args)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				if len(tt.args) > 0 {
					assert.Equal(t, tt.args[firstArgIndex], listTagsOption.Image)
				}
			}
		})
	}
}

func TestInspectCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with image arg",
			args:        []string{"registry.example.com/image:tag"},
			expectError: false,
		},
		{
			name:          "missing image arg should return error",
			args:          []string{},
			expectError:   true,
			errorContains: "The `image` is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalImage := inspectOption.Image
			defer func() {
				inspectOption.Image = originalImage
			}()

			// Call Args validation
			err := inspectDep.Args(nil, tt.args)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				if len(tt.args) > 0 {
					assert.Equal(t, tt.args[firstArgIndex], inspectOption.Image)
				}
			}
		})
	}
}

func TestManifestsCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		imageValue    string
		args          []string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with image and sufficient args",
			imageValue:  "registry.example.com/image:tag",
			args:        []string{"arg1", "arg2"}, // At least 2 args as per MinManifestsImageArgs
			expectError: false,
		},
		{
			name:          "missing image should return error",
			imageValue:    "",
			args:          []string{"arg1", "arg2"},
			expectError:   true,
			errorContains: "The `image` is required",
		},
		{
			name:          "insufficient args should return error",
			imageValue:    "registry.example.com/image:tag",
			args:          []string{"arg1"}, // Less than MinManifestsImageArgs
			expectError:   true,
			errorContains: "There are at least two schema images",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalImage := manifestsOption.Image
			defer func() {
				manifestsOption.Image = originalImage
			}()

			// Set test values
			manifestsOption.Image = tt.imageValue

			// Call Args validation
			err := manifestsDep.Args(nil, tt.args)

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

func TestDeleteCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with sufficient args",
			args:        []string{"arg1", "arg2"}, // At least 2 args as per MinManifestsImageArgs
			expectError: false,
		},
		{
			name:          "insufficient args should return error",
			args:          []string{"arg1"}, // Less than MinManifestsImageArgs
			expectError:   true,
			errorContains: "There are at least two schema images",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Call Args validation
			err := deleteDep.Args(nil, tt.args)

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

func TestViewCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with registry address",
			args:        []string{testRegistryIP.String() + ":" + testRegistryPort},
			expectError: false,
		},
		{
			name:          "missing registry address should return error",
			args:          []string{},
			expectError:   true,
			errorContains: "The `registry address` is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Call Args validation
			err := viewDep.Args(nil, tt.args)

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

func TestDownloadCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		imageValue    string
		fileValue     string
		dirValue      string
		mockGetwd     func() (string, error)
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid inputs",
			imageValue:  "repository/kubectl:v1.23.17",
			fileValue:   "kubectl",
			dirValue:    "/opt",
			expectError: false,
		},
		{
			name:          "missing image should return error",
			imageValue:    "",
			fileValue:     "kubectl",
			dirValue:      "/opt",
			expectError:   true,
			errorContains: "The `image` parameter is required",
		},
		{
			name:          "missing file should return error",
			imageValue:    "repository/kubectl:v1.23.17",
			fileValue:     "",
			dirValue:      "/opt",
			expectError:   true,
			errorContains: "The `file` parameter is required",
		},
		{
			name:       "empty dir gets current working directory",
			imageValue: "repository/kubectl:v1.23.17",
			fileValue:  "kubectl",
			dirValue:   "",
			mockGetwd: func() (string, error) {
				return "/tmp/test", nil
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalImage := downloadOption.Image
			originalFile := downloadOption.DownloadInImageFile
			originalDir := downloadOption.DownloadToDir
			defer func() {
				downloadOption.Image = originalImage
				downloadOption.DownloadInImageFile = originalFile
				downloadOption.DownloadToDir = originalDir
			}()

			// Set test values
			downloadOption.Image = tt.imageValue
			downloadOption.DownloadInImageFile = tt.fileValue
			downloadOption.DownloadToDir = tt.dirValue

			// Apply patches if needed
			if tt.mockGetwd != nil {
				patches := gomonkey.NewPatches()
				defer patches.Reset()
				patches.ApplyFunc(os.Getwd, tt.mockGetwd)
			}

			// Call Args validation
			err := downloadDep.Args(nil, nil)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
				// If dir was empty and Getwd succeeded, it should be set
				if tt.dirValue == "" && tt.mockGetwd != nil {
					assert.Equal(t, "/tmp/test", downloadOption.DownloadToDir)
				}
			}
		})
	}
}

func TestSyncCmdRun(t *testing.T) {
	// Save original values
	originalArgs := syncOption.Args
	originalOptions := syncOption.Options
	defer func() {
		syncOption.Args = originalArgs
		syncOption.Options = originalOptions
	}()

	// Apply patch to mock Sync method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*registry.Options).Sync, func(o *registry.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	syncDep.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, syncOption.Args)
	assert.Equal(t, options, syncOption.Options)
}

func TestTransferCmdRun(t *testing.T) {
	// Save original values
	originalArgs := transferOption.Args
	originalOptions := transferOption.Options
	defer func() {
		transferOption.Args = originalArgs
		transferOption.Options = originalOptions
	}()

	// Apply patch to mock MigrateImage method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*registry.Options).MigrateImage, func(o *registry.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	transferDep.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, transferOption.Args)
	assert.Equal(t, options, transferOption.Options)
}

func TestInspectCmdRun(t *testing.T) {
	// Save original values
	originalArgs := inspectOption.Args
	originalOptions := inspectOption.Options
	defer func() {
		inspectOption.Args = originalArgs
		inspectOption.Options = originalOptions
	}()

	// Apply patch to mock Inspect method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*registry.Options).Inspect, func(o *registry.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	inspectDep.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, inspectOption.Args)
	assert.Equal(t, options, inspectOption.Options)
}

func TestManifestsCmdRun(t *testing.T) {
	// Save original values
	originalArgs := manifestsOption.Args
	originalOptions := manifestsOption.Options
	defer func() {
		manifestsOption.Args = originalArgs
		manifestsOption.Options = originalOptions
	}()

	// Apply patch to mock Manifests method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*registry.Options).Manifests, func(o *registry.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	manifestsDep.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, manifestsOption.Args)
	assert.Equal(t, options, manifestsOption.Options)
}

func TestDeleteCmdRun(t *testing.T) {
	// Save original values
	originalArgs := deleteOption.Args
	originalOptions := deleteOption.Options
	defer func() {
		deleteOption.Args = originalArgs
		deleteOption.Options = originalOptions
	}()

	// Apply patch to mock Delete method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*registry.Options).Delete, func(o *registry.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	deleteDep.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, deleteOption.Args)
	assert.Equal(t, options, deleteOption.Options)
}

func TestViewCmdRun(t *testing.T) {
	// Save original values
	originalArgs := viewOption.Args
	originalOptions := viewOption.Options
	defer func() {
		viewOption.Args = originalArgs
		viewOption.Options = originalOptions
	}()

	// Apply patch to mock View method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*registry.Options).View, func(o *registry.Options) {
		// Mock implementation - do nothing
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	viewDep.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, viewOption.Args)
	assert.Equal(t, options, viewOption.Options)
}

func TestDownloadCmdRun(t *testing.T) {
	// Save original values
	originalArgs := downloadOption.Args
	originalOptions := downloadOption.Options
	defer func() {
		downloadOption.Args = originalArgs
		downloadOption.Options = originalOptions
	}()

	// Apply patch to mock Download method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*registry.OptionsDownload).Download, func(o *registry.OptionsDownload) error {
		return nil
	})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	downloadDep.Run(cmd, args)

	// Verify that args and options were set
	assert.NotEqual(t, args, downloadOption.Args)
	assert.Equal(t, options, downloadOption.Options)
}
