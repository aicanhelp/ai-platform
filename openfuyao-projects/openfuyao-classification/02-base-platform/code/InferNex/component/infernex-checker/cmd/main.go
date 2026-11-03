/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package main is the entry point of the infernex-checker CLI tool, providing
// sub-commands to run Hardware, K8s, and Config-Env pre-flight checks before
// deploying InferNex via Helm.
package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/checker/all"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/checker/configenv"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/checker/hardware"
	k8schecker "gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/checker/k8s"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/log"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/output"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/parser"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

var (
	nodesFile               string
	valuesFile              string
	outputJSON              string
	logFile                 string
	enableConnectivityCheck bool
)

func main() {
	defer log.Close()

	root := &cobra.Command{
		Use:   "infernex-check",
		Short: "InferNex pre-flight check tool",
		Long: "Performs systematic checks on the InferNex deployment environment before helm install. " +
			"The tool only detects and reports; it does not trigger any deployment operations.",
	}

	root.PersistentFlags().StringVar(&nodesFile, "nodes", "", "path to node info file (required)")
	root.PersistentFlags().StringVar(&outputJSON, "output", "", "path to JSON result output file (optional)")
	root.PersistentFlags().StringVar(&logFile, "log", "infernex-checker.log", "path to log file")
	root.PersistentFlags().BoolVar(&enableConnectivityCheck, "enable-connectivity-check", false,
		"enable network connectivity checks (H-07 Step 2 and H-10 Step 3), disabled by default for faster checks")
	_ = root.MarkPersistentFlagRequired("nodes")

	root.AddCommand(
		newHardwareCmd(),
		newK8sCmd(),
		newConfigEnvCmd(),
		newAllCmd(),
	)

	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		log.Close()
		os.Exit(1)
	}
}

// newHardwareCmd infernex-check hardware
func newHardwareCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "hardware",
		Short: "Run Hardware Layer checks only (H-01~H-10)",
		RunE: func(cmd *cobra.Command, args []string) error {
			initLog()
			nodes, err := loadNodes()
			if err != nil {
				return err
			}
			result := hardware.CheckHardware(nodes, enableConnectivityCheck)
			report := &types.Report{Hardware: result}
			report.Summary = output.BuildSummary(report)
			output.PrintReport(report)
			return writeJSON(report)
		},
	}
}

// newK8sCmd infernex-check k8s
func newK8sCmd() *cobra.Command {
	var kubeconfigFile string
	var dnsCheckImage string
	cmd := &cobra.Command{
		Use:   "k8s",
		Short: "Run K8s Layer checks only (K-01~K-04)",
		RunE: func(cmd *cobra.Command, args []string) error {
			initLog()
			nodes, err := loadNodes()
			if err != nil {
				return err
			}
			k8sClient, err := loadK8sClient(kubeconfigFile)
			if err != nil {
				return err
			}
			result := k8schecker.CheckK8s(k8sClient, nodeNames(nodes), dnsCheckImage)
			report := &types.Report{K8s: result}
			report.Summary = output.BuildSummary(report)
			output.PrintReport(report)
			return writeJSON(report)
		},
	}
	cmd.Flags().StringVar(&kubeconfigFile, "kubeconfig", defaultKubeconfig(), "path to kubeconfig file")
	cmd.Flags().StringVar(&dnsCheckImage, "dns-check-image",
		"", "container image for DNS check pod (default: busybox:1.36)")
	return cmd
}

// newConfigEnvCmd infernex-check config-env
func newConfigEnvCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config-env",
		Short: "Run Config-Env Layer checks only (B-01~B-02)",
		RunE: func(cmd *cobra.Command, args []string) error {
			initLog()
			nodes, values, err := loadNodesAndValues()
			if err != nil {
				return err
			}
			result := configenv.CheckConfigEnv(nodes, values)
			report := &types.Report{ConfigEnv: result}
			report.Summary = output.BuildSummary(report)
			output.PrintReport(report)
			return writeJSON(report)
		},
	}
	cmd.Flags().StringVar(&valuesFile, "values", "", "path to Helm values.yaml file (required)")
	_ = cmd.MarkFlagRequired("values")
	return cmd
}

// newAllCmd infernex-check all
func newAllCmd() *cobra.Command {
	var kubeconfigFile string
	var dnsCheckImage string
	cmd := &cobra.Command{
		Use:   "all",
		Short: "Run all checks (Hardware Layer -> K8s Layer -> Config-Env Layer)",
		RunE: func(cmd *cobra.Command, args []string) error {
			initLog()
			nodes, values, err := loadNodesAndValues()
			if err != nil {
				return err
			}
			k8sClient, err := loadK8sClient(kubeconfigFile)
			if err != nil {
				return err
			}
			report := all.CheckAll(nodes, values, k8sClient, dnsCheckImage, enableConnectivityCheck)
			report.Summary = output.BuildSummary(report)
			output.PrintReport(report)
			return writeJSON(report)
		},
	}
	cmd.Flags().StringVar(&valuesFile, "values", "", "path to Helm values.yaml file (required)")
	_ = cmd.MarkFlagRequired("values")
	cmd.Flags().StringVar(&kubeconfigFile, "kubeconfig", defaultKubeconfig(), "path to kubeconfig file")
	cmd.Flags().StringVar(&dnsCheckImage, "dns-check-image",
		"", "container image for DNS check pod (default: busybox:1.36)")
	return cmd
}

func initLog() {
	if err := log.Init(logFile); err != nil {
		fmt.Fprintf(os.Stderr, "failed to initialize log: %v\n", err)
	}
}

func loadNodes() ([]types.NodeInfo, error) {
	nodes, err := parser.ParseNodes(nodesFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load node file: %w", err)
	}
	return nodes, nil
}

func loadNodesAndValues() ([]types.NodeInfo, map[string]interface{}, error) {
	nodes, err := loadNodes()
	if err != nil {
		return nil, nil, err
	}
	values, err := loadValues()
	if err != nil {
		return nil, nil, err
	}
	return nodes, values, nil
}

func loadValues() (map[string]interface{}, error) {
	if valuesFile == "" {
		return nil, fmt.Errorf("please specify a Helm values.yaml file via --values")
	}
	values, err := parser.ParseValues(valuesFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load values file: %w", err)
	}
	return values, nil
}

func loadK8sClient(kubeconfig string) (*executor.K8sClient, error) {
	client, err := executor.NewK8sClient(kubeconfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize K8s client: %w", err)
	}
	return client, nil
}

func nodeNames(nodes []types.NodeInfo) []string {
	names := make([]string, len(nodes))
	for i, n := range nodes {
		names[i] = n.Name
	}
	return names
}

func writeJSON(report *types.Report) error {
	if outputJSON == "" {
		return nil
	}
	if err := output.WriteJSON(report, outputJSON); err != nil {
		return fmt.Errorf("failed to write JSON result: %w", err)
	}
	fmt.Printf("\nJSON result written to: %s\n", outputJSON)
	return nil
}

func defaultKubeconfig() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".kube", "config")
}
