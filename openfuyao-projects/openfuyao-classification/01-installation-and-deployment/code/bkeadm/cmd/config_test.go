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
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/spf13/cobra"
	"github.com/stretchr/testify/assert"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"

	"gopkg.openfuyao.cn/bkeadm/pkg/config"
)

func TestConfigCmdPreRunE(t *testing.T) {
	tests := []struct {
		name            string
		initialDir      string
		initialProduct  string
		mockGetwd       func() (dir string, err error)
		expectedDir     string
		expectedProduct string
		expectError     bool
	}{
		{
			name:           "empty directory gets current working directory",
			initialDir:     "",
			initialProduct: "",
			mockGetwd: func() (dir string, err error) {
				return "/tmp/test", nil
			},
			expectedDir:     "/tmp/test",
			expectedProduct: "fuyao-portal",
			expectError:     false,
		},
		{
			name:           "non-empty directory remains unchanged",
			initialDir:     "/custom/dir",
			initialProduct: "my-product",
			mockGetwd: func() (dir string, err error) {
				return "/tmp/test", nil // This shouldn't be used
			},
			expectedDir:     "/custom/dir",
			expectedProduct: "my-product",
			expectError:     false,
		},
		{
			name:           "empty product gets default value",
			initialDir:     "/some/dir",
			initialProduct: "",
			mockGetwd: func() (dir string, err error) {
				return "/tmp/test", nil
			},
			expectedDir:     "/some/dir",
			expectedProduct: "fuyao-portal",
			expectError:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalDir := configOption.Directory
			originalProduct := configOption.Product
			defer func() {
				configOption.Directory = originalDir
				configOption.Product = originalProduct
			}()

			// Set initial values
			configOption.Directory = tt.initialDir
			configOption.Product = tt.initialProduct

			// Apply patches
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			if tt.mockGetwd != nil {
				patches.ApplyFunc(os.Getwd, tt.mockGetwd)
			}

			// Call PreRunE
			err := configCmd.PreRunE(nil, nil)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedDir, configOption.Directory)
				assert.Equal(t, tt.expectedProduct, configOption.Product)
			}
		})
	}
}

func TestEncryptCmdArgs(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		fileValue   string
		expectError bool
	}{
		{
			name:        "no args and no file should return error",
			args:        []string{},
			fileValue:   "",
			expectError: true,
		},
		{
			name:        "with args but no file should not return error",
			args:        []string{"test"},
			fileValue:   "",
			expectError: false,
		},
		{
			name:        "no args but with file should not return error",
			args:        []string{},
			fileValue:   "test.yaml",
			expectError: false,
		},
		{
			name:        "both args and file should not return error",
			args:        []string{"test"},
			fileValue:   "test.yaml",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original value
			originalFile := encryptOption.File
			defer func() {
				encryptOption.File = originalFile
			}()

			// Set file value
			encryptOption.File = tt.fileValue

			// Call Args validation
			err := encryptCmd.Args(nil, tt.args)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestDecryptCmdArgs(t *testing.T) {
	tests := []struct {
		name        string
		args        []string
		fileValue   string
		expectError bool
	}{
		{
			name:        "no args and no file should return error",
			args:        []string{},
			fileValue:   "",
			expectError: true,
		},
		{
			name:        "with args but no file should not return error",
			args:        []string{"test"},
			fileValue:   "",
			expectError: false,
		},
		{
			name:        "no args but with file should not return error",
			args:        []string{},
			fileValue:   "test.yaml",
			expectError: false,
		},
		{
			name:        "both args and file should not return error",
			args:        []string{"test"},
			fileValue:   "test.yaml",
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original value
			originalFile := decryptOption.File
			defer func() {
				decryptOption.File = originalFile
			}()

			// Set file value
			decryptOption.File = tt.fileValue

			// Call Args validation
			err := decryptCmd.Args(nil, tt.args)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestConfigCmdRun(t *testing.T) {
	// Apply patches to mock the Config method
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc((*config.Options).Config,
		func(op *config.Options, customExtra map[string]string, imageRepo, yumRepo, chartRepo confv1beta1.Repo, ntpServer string) {
			// Mock implementation - do nothing
		})

	// Create and run the command
	cmd := &cobra.Command{}
	args := []string{"arg1", "arg2"}

	// Capture the state before running
	originalArgs := configOption.Args
	originalOptions := configOption.Options

	configCmd.Run(cmd, args)

	// Verify that args and options were set
	assert.Equal(t, args, configOption.Args)
	assert.Equal(t, options, configOption.Options)

	// Restore original values
	configOption.Args = originalArgs
	configOption.Options = originalOptions
}

func TestEncryptCmdRun(t *testing.T) {
	tests := []struct {
		name              string
		args              []string
		fileValue         string
		mockEncryptString func(*config.Options) error
		mockEncryptFile   func(*config.Options) error
	}{
		{
			name:      "run with args calls EncryptString",
			args:      []string{"test"},
			fileValue: "",
			mockEncryptString: func(o *config.Options) error {
				return nil
			},
			mockEncryptFile: func(o *config.Options) error {
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalArgs := encryptOption.Args
			originalOptions := encryptOption.Options
			originalFile := encryptOption.File
			defer func() {
				encryptOption.Args = originalArgs
				encryptOption.Options = originalOptions
				encryptOption.File = originalFile
			}()

			// Set values
			encryptOption.File = tt.fileValue

			// Apply patches to mock the encryption methods
			stringCalled := false
			fileCalled := false

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*config.Options).EncryptString, func(o *config.Options) error {
				stringCalled = true
				return tt.mockEncryptString(o)
			})

			patches.ApplyFunc((*config.Options).EncryptFile, func(o *config.Options) error {
				fileCalled = true
				return tt.mockEncryptFile(o)
			})

			// Create and run the command
			cmd := &cobra.Command{}
			encryptCmd.Run(cmd, tt.args)

			// Verify that the correct methods were called
			if len(tt.args) > 0 {
				assert.True(t, stringCalled, "EncryptString should be called when args are provided")
			} else {
				assert.False(t, stringCalled, "EncryptString should not be called when no args are provided")
			}

			if tt.fileValue != "" {
				assert.True(t, fileCalled, "EncryptFile should be called when file is provided")
			} else {
				assert.False(t, fileCalled, "EncryptFile should not be called when no file is provided")
			}

			// Verify that args and options were set
			assert.Equal(t, tt.args, encryptOption.Args)
			assert.Equal(t, options, encryptOption.Options)
		})
	}
}

func TestDecryptCmdRun(t *testing.T) {
	tests := []struct {
		name              string
		args              []string
		fileValue         string
		mockDecryptString func(*config.Options) error
		mockDecryptFile   func(*config.Options) error
	}{
		{
			name:      "run with args calls DecryptString",
			args:      []string{"test"},
			fileValue: "",
			mockDecryptString: func(o *config.Options) error {
				return nil
			},
			mockDecryptFile: func(o *config.Options) error {
				return nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Save original values
			originalArgs := decryptOption.Args
			originalOptions := decryptOption.Options
			originalFile := decryptOption.File
			defer func() {
				decryptOption.Args = originalArgs
				decryptOption.Options = originalOptions
				decryptOption.File = originalFile
			}()

			// Set values
			decryptOption.File = tt.fileValue

			// Apply patches to mock the decryption methods
			stringCalled := false
			fileCalled := false

			patches := gomonkey.NewPatches()

			defer patches.Reset()

			patches.ApplyFunc((*config.Options).DecryptString, func(o *config.Options) error {
				stringCalled = true
				return tt.mockDecryptString(o)
			})

			patches.ApplyFunc((*config.Options).DecryptFile, func(o *config.Options) error {
				fileCalled = true
				return tt.mockDecryptFile(o)
			})

			// Create and run the command
			cmd := &cobra.Command{}
			decryptCmd.Run(cmd, tt.args)

			// Verify that the correct methods were called
			if len(tt.args) > 0 {
				assert.True(t, stringCalled, "DecryptString should be called when args are provided")
			} else {
				assert.False(t, stringCalled, "DecryptString should not be called when no args are provided")
			}

			if tt.fileValue != "" {
				assert.True(t, fileCalled, "DecryptFile should be called when file is provided")
			} else {
				assert.False(t, fileCalled, "DecryptFile should not be called when no file is provided")
			}

			// Verify that args and options were set
			assert.Equal(t, tt.args, decryptOption.Args)
			assert.Equal(t, options, decryptOption.Options)
		})
	}
}
