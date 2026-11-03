/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Application entry point
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

const (
	defaultTimeout = 600
)

var (
	cfgFile string
)

// rootCmd root command
var rootCmd = &cobra.Command{
	Use:   "envCheck",
	Short: "Environment pre-check validation tool for installation",
	Long: `envCheck is a tool for checking deployment environments, supporting:
- Residual file query
- Conflicting application check
- Residual file cleanup
- Clock synchronization check`,
}

// queryCmd query subcommand
var queryCmd = &cobra.Command{
	Use:   "query",
	Short: "Query residual files",
	Long:  `Query residual file information under specified paths, including file existence, permissions, owner, etc.`,
	RunE:  runQuery,
}

// cleanCmd clean subcommand
var cleanCmd = &cobra.Command{
	Use:   "clean",
	Short: "Clean residual files",
	Long:  `Clean residual files under specified paths, supporting interactive confirmation and force clean`,
	RunE:  runClean,
}

// programCmd program subcommand
var programCmd = &cobra.Command{
	Use:   "program",
	Short: "Check conflicting applications",
	Long:  `Check if specified applications are installed, including version and path information`,
	RunE:  runProgram,
}

// clockCmd clock subcommand
var clockCmd = &cobra.Command{
	Use:   "clock",
	Short: "Check clock synchronization",
	Long:  `Check whether the clocks of others nodes are synchronized with the bootstrap node`,
	RunE:  runClock,
}

// routeCmd route subcommand
var routeCmd = &cobra.Command{
	Use:   "route",
	Short: "Check default route",
	Long:  `Check whether the default route exists and output detailed information`,
	RunE:  runRoute,
}

// iptablesCmd iptables subcommand
var iptablesCmd = &cobra.Command{
	Use:   "iptables",
	Short: "Check iptables rules",
	Long:  `Check whether the FORWARD chain default policy is ACCEPT`,
	RunE:  runIptables,
}

func init() {
	// Global flags
	rootCmd.PersistentFlags().StringVarP(&cfgFile, "config", "c", "./config.json", "Configuration file path")

	cleanCmd.Flags().Bool("force", false, "force clean (no confirmation prompt)")

	// Add subcommands
	rootCmd.AddCommand(queryCmd, cleanCmd, programCmd, clockCmd, routeCmd, iptablesCmd)
	rootCmd.AddCommand(diskCmd, kernelCmd, portCmd, runCmd, runLocalCmd)

	portCmd.Flags().StringVar(&roles, "roles", "", "Comma-separated list of roles to check (empty=all)")

	runCmd.Flags().StringVar(&checkList, "checks", "",
		"Comma-separated list of checks to run (kernel,port,disk,clock,fileQuery,programCheck,route,iptables)")
	runCmd.Flags().StringVar(&skipList, "skip", "", "Comma-separated list of checks to skip")
	runCmd.Flags().IntVar(&timeout, "timeout", defaultTimeout, "Timeout in seconds for remote execution")
	runCmd.Flags().StringVar(&roles, "roles", "", "Comma-separated list of roles to check (empty=all)")

	runLocalCmd.Flags().StringVar(&checkList, "checks", "", "Checks to run")
	runLocalCmd.Flags().BoolVar(&runLocal, "local", true, "Run in local mode")
	runLocalCmd.Flags().StringVar(&roles, "roles", "", "Comma-separated list of roles (auto-detected if empty)")
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
