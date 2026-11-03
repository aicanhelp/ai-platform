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
	"env-check/pkg/disk"
	"env-check/pkg/logger"
	"env-check/pkg/output"
)

// diskCmd disk subcommand
var diskCmd = &cobra.Command{
	Use:   "disk",
	Short: "Check disk space",
	Long:  `Check if the configured disk paths have sufficient free space.`,
	RunE:  runDisk,
}

// runDisk executes disk check operation
func runDisk(cmd *cobra.Command, args []string) error {
	return runAction(cmd, args, config.OperationMode("diskCheck"), actionHandler{
		execute: func(cmd *cobra.Command, cfg *config.AppConfig, log logger.Logger) (interface{}, error) {
			// Use default config if not specified
			if cfg.DiskCheck == nil {
				cfg.DiskCheck = &config.DiskCheckConfig{
					CheckItems: []config.DiskCheckItem{
						{Path: "/", MinFreeGB: 50},
						{Path: "/var/lib/docker", MinFreeGB: 80},
					},
				}
			}

			// Get current node roles from config
			var roles []string
			for _, host := range cfg.Hosts {
				roles = append(roles, host.Role...)
			}

			d := disk.NewChecker(cfg.DiskCheck, log, roles)
			result, err := d.Execute()
			if err != nil {
				log.Error("disk check failed: " + err.Error())
				return nil, err
			}

			if result.Summary.InsufficientPath > 0 {
				log.Warning("some disk paths have insufficient space")
			} else {
				log.Info("all disk paths have sufficient space")
			}

			return result, nil
		},
		generateHTML: func(res interface{}) error {
			return output.GenerateDiskHTML(*(res.(*config.DiskCheckResult)), "disk.html")
		},
		format: func(res interface{}, outputFormat string) (string, error) {
			formatter := output.NewFormatter()
			return formatter.FormatDiskCheck(res.(*config.DiskCheckResult), outputFormat)
		},
		baseName: "disk-check-result",
	})
}
