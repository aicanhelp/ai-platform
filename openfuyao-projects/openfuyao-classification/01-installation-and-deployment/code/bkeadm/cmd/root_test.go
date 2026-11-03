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

	"gopkg.openfuyao.cn/bkeadm/pkg/root"
	blog "gopkg.openfuyao.cn/bkeadm/utils/log"
)

func TestRootCmdInitialization(t *testing.T) {
	tests := []struct {
		name               string
		expectedUse        string
		expectedShort      string
		hasPersistentFlags bool
	}{
		{
			name:               "root command properties",
			expectedUse:        "bke",
			expectedShort:      "Bocloud Enterprise Kubernetes deployment tool.",
			hasPersistentFlags: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expectedUse, rootCmd.Use)
			assert.Equal(t, tt.expectedShort, rootCmd.Short)

			// Check if persistent flags are registered
			if tt.hasPersistentFlags {
				kubeConfigFlag := rootCmd.PersistentFlags().Lookup("kubeconfig")
				assert.NotNil(t, kubeConfigFlag)

				docFlag := rootCmd.PersistentFlags().Lookup("doc")
				assert.NotNil(t, docFlag)

				logLevelFlag := rootCmd.PersistentFlags().Lookup("log-level")
				assert.NotNil(t, logLevelFlag)
			}
		})
	}
}

func TestRootCmdRun(t *testing.T) {
	tests := []struct {
		name         string
		args         []string
		docValue     bool
		expectedCall string // "Print" or "PrintDoc"
	}{
		{
			name:         "run with doc flag true",
			args:         []string{},
			docValue:     true,
			expectedCall: "PrintDoc",
		},
		{
			name:         "run with doc flag false",
			args:         []string{"arg1", "arg2"},
			docValue:     false,
			expectedCall: "Print",
		},
	}

	originalDoc := doc
	defer func() { doc = originalDoc }() // restore original value

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			doc = tt.docValue

			// Create a mock command for testing
			cmd := &cobra.Command{}

			// Capture the options.Args value after the run function executes
			var printCalled, printDocCalled bool

			// Apply patches to the Print and PrintDoc methods
			patches := gomonkey.NewPatches()
			defer patches.Reset()
			patches.ApplyFunc((*root.Options).Print, func(o *root.Options) {
				printCalled = true
			})

			patches.ApplyFunc((*root.Options).PrintDoc, func(o *root.Options) {
				printDocCalled = true
			})

			// Store original options to restore later
			originalOptions := options
			defer func() { options = originalOptions }()

			// Execute the run function
			rootCmd.Run(cmd, tt.args)

			// Check which method was called based on doc flag
			if tt.expectedCall == "PrintDoc" {
				assert.True(t, printDocCalled, "PrintDoc should have been called")
				assert.False(t, printCalled, "Print should not have been called")
			} else {
				assert.True(t, printCalled, "Print should have been called")
				assert.False(t, printDocCalled, "PrintDoc should not have been called")
			}

			// Check if args were set correctly
			if !tt.docValue {
				assert.Equal(t, tt.args, options.Args)
			}
		})
	}
}

func TestExecute(t *testing.T) {
	Execute()
}

func TestApplyGlobalLogLevel(t *testing.T) {
	originalLogLevel := logLevel
	defer func() { logLevel = originalLogLevel }()

	t.Run("empty log level", func(t *testing.T) {
		logLevel = ""
		assert.NoError(t, applyGlobalLogLevel())
	})

	t.Run("valid log level", func(t *testing.T) {
		logLevel = "debug"

		patches := gomonkey.NewPatches()
		defer patches.Reset()

		called := false
		patches.ApplyFunc(blog.SetLevel, func(level string) error {
			called = true
			assert.Equal(t, "debug", level)
			return nil
		})

		assert.NoError(t, applyGlobalLogLevel())
		assert.True(t, called)
	})

	t.Run("invalid log level", func(t *testing.T) {
		logLevel = "bad-level"

		patches := gomonkey.NewPatches()
		defer patches.Reset()

		patches.ApplyFunc(blog.SetLevel, func(level string) error {
			return assert.AnError
		})

		assert.ErrorIs(t, applyGlobalLogLevel(), assert.AnError)
	})
}
