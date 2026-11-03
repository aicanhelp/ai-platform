/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package main

import (
	"github.com/spf13/cobra"

	"env-check/pkg/config"
	"env-check/pkg/kernel"
	"env-check/pkg/logger"
	"env-check/pkg/output"
)

// kernelCmd kernel subcommand
var kernelCmd = &cobra.Command{
	Use:   "kernel",
	Short: "Check kernel version",
	Long:  `Check if the kernel version meets the configured requirements.`,
	RunE:  runKernel,
}

// runKernel executes kernel check operation
func runKernel(cmd *cobra.Command, args []string) error {
	return runAction(cmd, args, config.OperationMode("kernelCheck"), actionHandler{
		execute: func(cmd *cobra.Command, cfg *config.AppConfig, log logger.Logger) (interface{}, error) {
			// Use default config if not specified
			if cfg.KernelCheck == nil {
				cfg.KernelCheck = &config.KernelCheckConfig{
					MinVersion: "4.18",
					Operator:   ">=",
				}
			}

			k := kernel.NewChecker(cfg.KernelCheck, log)
			result, err := k.Execute()
			if err != nil {
				log.Error("kernel check failed: " + err.Error())
				return nil, err
			}

			if !result.KernelInfo.IsValid {
				log.Warning("kernel version check failed: current=" + result.KernelInfo.Version +
					", expected=" + result.KernelInfo.Operator + " " + result.KernelInfo.ExpectedVer)
			} else {
				log.Info("kernel version check passed")
			}

			return result, nil
		},
		generateHTML: func(res interface{}) error {
			return output.GenerateKernelHTML(*(res.(*config.KernelCheckResult)), "kernel.html")
		},
		format: func(res interface{}, outputFormat string) (string, error) {
			formatter := output.NewFormatter()
			return formatter.FormatKernelCheck(res.(*config.KernelCheckResult), outputFormat)
		},
		baseName: "kernel-check-result",
	})
}
