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

	"gopkg.openfuyao.cn/bkeadm/pkg/cluster"
	"gopkg.openfuyao.cn/bkeadm/pkg/root"
)

const (
	clusterSubcommandCount = 2
)

func TestClusterCmdInitialization(t *testing.T) {
	tests := []struct {
		name           string
		cmd            *cobra.Command
		expectedUse    string
		expectedShort  string
		hasSubcommands bool
	}{
		{
			name:           "cluster command properties",
			cmd:            clusterCmd,
			expectedUse:    "cluster",
			expectedShort:  "Operating an existing cluster.",
			hasSubcommands: true,
		},
		{
			name:           "list command properties",
			cmd:            listDep,
			expectedUse:    "list",
			expectedShort:  "Cluster list",
			hasSubcommands: false,
		},
		{
			name:           "remove command properties",
			cmd:            removeDep,
			expectedUse:    "remove",
			expectedShort:  "Delete a Specified Cluster",
			hasSubcommands: false,
		},
		{
			name:           "create command properties",
			cmd:            createDep,
			expectedUse:    "create",
			expectedShort:  "Deploying a Cluster",
			hasSubcommands: false,
		},
		{
			name:           "scale command properties",
			cmd:            scaleDep,
			expectedUse:    "scale",
			expectedShort:  "shard cluster",
			hasSubcommands: false,
		},
		{
			name:           "logs command properties",
			cmd:            logsDep,
			expectedUse:    "logs",
			expectedShort:  "Obtain cluster deployment events",
			hasSubcommands: false,
		},
		{
			name:           "exist command properties",
			cmd:            existDep,
			expectedUse:    "exist",
			expectedShort:  "Manage existing clusters",
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

func TestRegisterClusterCommand(t *testing.T) {

	// Find the cluster command in root commands
	var foundClusterCmd bool
	for _, cmd := range rootCmd.Commands() {
		if cmd.Use == "cluster" {
			foundClusterCmd = true
			// Check if subcommands are registered
			assert.Len(t, cmd.Commands(), clusterSubcommandCount) // list, create

			var foundSubcommands []string
			for _, subCmd := range cmd.Commands() {
				foundSubcommands = append(foundSubcommands, subCmd.Use)
			}

			expectedSubcommands := []string{"list", "create"}
			for _, expected := range expectedSubcommands {
				assert.Contains(t, foundSubcommands, expected)
			}
			break
		}
	}

	assert.True(t, foundClusterCmd, "cluster command should be registered in root command")
}

func TestListCmdPreRunEAndRun(t *testing.T) {
	// Save original values
	originalArgs := listOption.Args
	originalOptions := listOption.Options
	defer func() {
		listOption.Args = originalArgs
		listOption.Options = originalOptions
	}()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*root.Options).ClusterPre, func(o *root.Options) error {
		return nil
	})

	patches.ApplyFunc((*cluster.Options).List, func(o *cluster.Options) {
		// Mock implementation - do nothing
	})

	// Test PreRunE
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	err := listDep.PreRunE(cmd, args)
	assert.NoError(t, err)
	assert.Equal(t, args, listOption.Args)
	assert.Equal(t, options, listOption.Options)

	// Test Run
	listDep.Run(cmd, args)
	assert.Equal(t, args, listOption.Args)
	assert.Equal(t, options, listOption.Options)
}

func TestRemoveCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with ns/name format",
			args:        []string{"namespace/name"},
			expectError: false,
		},
		{
			name:          "missing args should return error",
			args:          []string{},
			expectError:   true,
			errorContains: "Required parameters are missing",
		},
		{
			name:          "invalid format should return error",
			args:          []string{"invalid-format"},
			expectError:   true,
			errorContains: "The parameter format is invalid, The parameter format is ns/name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Call Args validation
			err := removeDep.Args(nil, tt.args)

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

func TestRemoveCmdPreRunEAndRun(t *testing.T) {
	// Save original values
	originalArgs := removeOption.Args
	originalOptions := removeOption.Options
	defer func() {
		removeOption.Args = originalArgs
		removeOption.Options = originalOptions
	}()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*root.Options).ClusterPre, func(o *root.Options) error {
		return nil
	})

	patches.ApplyFunc((*cluster.Options).Remove, func(o *cluster.Options) {
		// Mock implementation - do nothing
	})

	// Test PreRunE
	cmd := &cobra.Command{}
	args := []string{"namespace/name"}

	err := removeDep.PreRunE(cmd, args)
	assert.NoError(t, err)
	assert.Equal(t, args, removeOption.Args)
	assert.Equal(t, options, removeOption.Options)

	// Test Run
	removeDep.Run(cmd, args)
	assert.Equal(t, args, removeOption.Args)
	assert.Equal(t, options, removeOption.Options)
}

func TestCreateCmdArgs(t *testing.T) {
	tests := []struct {
		name           string
		fileValue      string
		nodesFileValue string
		expectError    bool
		errorContains  string
	}{
		{
			name:           "valid input with file",
			fileValue:      "/path/to/file.yaml",
			nodesFileValue: "/path/to/nodes.yaml",
			expectError:    false,
		},
		{
			name:          "missing file should return error",
			fileValue:     "",
			expectError:   true,
			errorContains: "The `file` parameter is required",
		},
		{
			name:           "missing nodes file should return error",
			fileValue:      "/path/to/file.yaml",
			nodesFileValue: "",
			expectError:    true,
			errorContains:  "The `nodes` parameter is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalFile := createOption.File
			originalNodesFile := createOption.NodesFile
			defer func() {
				createOption.File = originalFile
				createOption.NodesFile = originalNodesFile
			}()

			// Set test values
			createOption.File = tt.fileValue
			createOption.NodesFile = tt.nodesFileValue

			// Call Args validation
			err := createDep.Args(nil, nil)

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

func TestCreateCmdPreRunEAndRun(t *testing.T) {
	// Save original values
	originalArgs := createOption.Args
	originalOptions := createOption.Options
	originalFile := createOption.File
	defer func() {
		createOption.Args = originalArgs
		createOption.Options = originalOptions
		createOption.File = originalFile
	}()

	// Set required file value
	createOption.File = "/path/to/file.yaml"

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*root.Options).ClusterPre, func(o *root.Options) error {
		return nil
	})

	patches.ApplyFunc((*cluster.Options).Cluster, func(o *cluster.Options) {
		// Mock implementation - do nothing
	})

	// Test PreRunE
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	err := createDep.PreRunE(cmd, args)
	assert.NoError(t, err)
	assert.Equal(t, args, createOption.Args)
	assert.Equal(t, options, createOption.Options)

	// Test Run
	createDep.Run(cmd, args)
	assert.Equal(t, args, createOption.Args)
	assert.Equal(t, options, createOption.Options)
}

func TestScaleCmdArgs(t *testing.T) {
	tests := []struct {
		name           string
		fileValue      string
		nodesFileValue string
		expectError    bool
		errorContains  string
	}{
		{
			name:           "valid input with file",
			fileValue:      "/path/to/file.yaml",
			nodesFileValue: "/path/to/nodes.yaml",
			expectError:    false,
		},
		{
			name:          "missing file should return error",
			fileValue:     "",
			expectError:   true,
			errorContains: "The `file` parameter is required",
		},
		{
			name:           "missing nodes file should return error",
			fileValue:      "/path/to/file.yaml",
			nodesFileValue: "",
			expectError:    true,
			errorContains:  "The `nodes` parameter is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalFile := scaleOption.File
			originalNodesFile := scaleOption.NodesFile
			defer func() {
				scaleOption.File = originalFile
				scaleOption.NodesFile = originalNodesFile
			}()

			// Set test values
			scaleOption.File = tt.fileValue
			scaleOption.NodesFile = tt.nodesFileValue

			// Call Args validation
			err := scaleDep.Args(nil, nil)

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

func TestScaleCmdPreRunEAndRun(t *testing.T) {
	// Save original values
	originalArgs := scaleOption.Args
	originalOptions := scaleOption.Options
	originalFile := scaleOption.File
	defer func() {
		scaleOption.Args = originalArgs
		scaleOption.Options = originalOptions
		scaleOption.File = originalFile
	}()

	// Set required file value
	scaleOption.File = "/path/to/file.yaml"

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*root.Options).ClusterPre, func(o *root.Options) error {
		return nil
	})

	patches.ApplyFunc((*cluster.Options).Scale, func(o *cluster.Options) {
		// Mock implementation - do nothing
	})

	// Test PreRunE
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	err := scaleDep.PreRunE(cmd, args)
	assert.NoError(t, err)
	assert.Equal(t, args, scaleOption.Args)
	assert.Equal(t, options, scaleOption.Options)

	// Test Run
	scaleDep.Run(cmd, args)
	assert.Equal(t, args, scaleOption.Args)
	assert.Equal(t, options, scaleOption.Options)
}

func TestLogsCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		args          []string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with ns/name format",
			args:        []string{"namespace/name"},
			expectError: false,
		},
		{
			name:          "missing args should return error",
			args:          []string{},
			expectError:   true,
			errorContains: "Required parameters are missing",
		},
		{
			name:          "invalid format should return error",
			args:          []string{"invalid-format"},
			expectError:   true,
			errorContains: "The parameter format is invalid, The parameter format is ns/name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Call Args validation
			err := logsDep.Args(nil, tt.args)

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

func TestLogsCmdPreRunEAndRun(t *testing.T) {
	// Save original values
	originalArgs := logsOption.Args
	originalOptions := logsOption.Options
	defer func() {
		logsOption.Args = originalArgs
		logsOption.Options = originalOptions
	}()

	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*root.Options).ClusterPre, func(o *root.Options) error {
		return nil
	})

	patches.ApplyFunc((*cluster.Options).Log, func(o *cluster.Options) {
		// Mock implementation - do nothing
	})

	// Test PreRunE
	cmd := &cobra.Command{}
	args := []string{"namespace/name"}

	err := logsDep.PreRunE(cmd, args)
	assert.NoError(t, err)
	assert.Equal(t, args, logsOption.Args)
	assert.Equal(t, options, logsOption.Options)

	// Test Run
	logsDep.Run(cmd, args)
	assert.Equal(t, args, logsOption.Args)
	assert.Equal(t, options, logsOption.Options)
}

func TestExistCmdArgs(t *testing.T) {
	tests := []struct {
		name          string
		fileValue     string
		confValue     string
		expectError   bool
		errorContains string
	}{
		{
			name:        "valid input with both file and conf",
			fileValue:   "/path/to/file.yaml",
			confValue:   "/path/to/conf.yaml",
			expectError: false,
		},
		{
			name:          "missing file should return error",
			fileValue:     "",
			confValue:     "/path/to/conf.yaml",
			expectError:   true,
			errorContains: "The `file` parameter is required",
		},
		{
			name:          "missing conf should return error",
			fileValue:     "/path/to/file.yaml",
			confValue:     "",
			expectError:   true,
			errorContains: "The `conf` parameter is required",
		},
		{
			name:          "missing both should return error",
			fileValue:     "",
			confValue:     "",
			expectError:   true,
			errorContains: "The `file` parameter is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalFile := existOption.File
			originalConf := existOption.Conf
			defer func() {
				existOption.File = originalFile
				existOption.Conf = originalConf
			}()

			// Set test values
			existOption.File = tt.fileValue
			existOption.Conf = tt.confValue

			// Call Args validation
			err := existDep.Args(nil, nil)

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

func TestExistCmdPreRunEAndRun(t *testing.T) {
	// Apply patches
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*root.Options).ClusterPre, func(o *root.Options) error {
		return nil
	})

	patches.ApplyFunc((*cluster.Options).ExistsCluster, func(o *cluster.Options) {
		// Mock implementation - do nothing
	})

	// Test PreRunE
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	err := existDep.PreRunE(cmd, args)
	assert.NoError(t, err)
	assert.Equal(t, args, existOption.Args)
	assert.Equal(t, options, existOption.Options)

	// Test Run
	existDep.Run(cmd, args)
	assert.Equal(t, args, existOption.Args)
	assert.Equal(t, options, existOption.Options)
}
