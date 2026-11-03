/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package cmd

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"gopkg.openfuyao.cn/bkeadm/pkg/root"
	blog "gopkg.openfuyao.cn/bkeadm/utils/log"
	"gopkg.openfuyao.cn/preflight/api"
)

var (
	doc      bool
	logLevel string
	options  root.Options
)

// rootCmd is the base command executed when no subcommands are provided.
var rootCmd = &cobra.Command{
	Use:   "bke",
	Short: "Bocloud Enterprise Kubernetes deployment tool.",
	Long: `Bocloud Enterprise Kubernetes deployment tool.
It provides an integrated solution for Kubernetes cluster deployment,
operation, maintenance, and governance.`,
	PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
		return applyGlobalLogLevel()
	},
	Run: func(cmd *cobra.Command, args []string) {
		options.Args = args
		if doc {
			options.PrintDoc()
			return
		}
		options.Print()
	},
}

// Execute runs the root command.
// It should be called once in the main function.
func Execute() int {
	executedCommand, err := rootCmd.ExecuteC()
	if err == nil {
		return 0
	}
	var exitErr *commandExitError
	if errors.As(err, &exitErr) && exitErr != nil {
		if exitErr.err != nil {
			fmt.Fprintln(os.Stderr, exitErr.err)
		}
		return exitErr.code
	}
	if operation, ok := preflightOperation(executedCommand); ok {
		reportErr := writeAndRenderPreflightReport(singleInputIssueReport(
			operation,
			"PREFLIGHT-OPTIONS",
			preflightFlagField(err),
			err,
			"correct the command-line options, then rerun preflight",
		))
		if errors.As(reportErr, &exitErr) && exitErr != nil {
			if exitErr.err != nil {
				fmt.Fprintln(os.Stderr, exitErr.err)
			}
			return exitErr.code
		}
		fmt.Fprintln(os.Stderr, reportErr)
		return api.ExitInternalError
	}
	// Cobra has already rendered ordinary command errors and usage exactly as
	// before. Only preflight's typed errors are handled above.
	return 1
}

func preflightFlagField(err error) string {
	const fallback = "命令行参数"
	if err == nil {
		return fallback
	}
	message := err.Error()
	start := strings.Index(message, "--")
	if start < 0 {
		return fallback
	}
	field := message[start:]
	if end := strings.IndexAny(field, "\"' :=\t\r\n"); end >= 0 {
		field = field[:end]
	}
	if field == "--" {
		return fallback
	}
	return field
}

func preflightOperation(command *cobra.Command) (api.Operation, bool) {
	// Pointer comparison instead of name comparison: nested subcommands under
	// preflight keep resolving to their operation even if a deeper command
	// tree is added later.
	if command == nil || command.Parent() != preflightCmd {
		return "", false
	}
	switch command.Name() {
	case preflightInitCmd.Name():
		return api.OperationInit, true
	case preflightClusterCreateCmd.Name():
		return api.OperationClusterCreate, true
	default:
		return "", false
	}
}

func applyGlobalLogLevel() error {
	if logLevel == "" {
		return nil
	}
	return blog.SetLevel(logLevel)
}

func init() {
	cobra.OnInitialize()
	rootCmd.PersistentFlags().StringVar(
		&options.KubeConfig,
		"kubeconfig",
		"",
		"Path to the Kubernetes configuration file.",
	)
	rootCmd.PersistentFlags().BoolVar(
		&doc,
		"doc",
		false,
		"Display command documentation.",
	)
	rootCmd.PersistentFlags().StringVar(
		&logLevel,
		"log-level",
		"",
		"Override log level for this command. Valid values: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL.",
	)

	// Register all subcommands
	registerInitCommand()
	registerResetCommand()
	registerStartCommand()
	registerVersionCommand()
	registerRegistryCommand()
	registerBuildCommand()
	registerClusterCommand()
	registerPreflightCommands()
}
