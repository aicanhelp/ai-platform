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
	"env-check/pkg/logger"
	"env-check/pkg/output"
	"env-check/pkg/port"
)

// portCmd port subcommand
var portCmd = &cobra.Command{
	Use:   "port",
	Short: "Check port occupancy",
	Long:  `Check if the configured ports are occupied on the local machine.`,
	RunE:  runPort,
}

// runPort executes port check operation
func runPort(cmd *cobra.Command, args []string) error {
	return runAction(cmd, args, config.OperationMode("portCheck"), actionHandler{
		execute: func(cmd *cobra.Command, cfg *config.AppConfig, log logger.Logger) (interface{}, error) {
			// Use default config if not specified
			if cfg.PortCheck == nil {
				cfg.PortCheck = &config.PortCheckConfig{
					Ports: map[string][]string{
						"bootstrap": {"36443", "40080", "40443", "38080"},
						"master":    {"6443", "30029", "30909", "30903", "3030", "30019", "9012"},
						"worker":    {"10250", "10256", "30000", "30001", "30022", "30033", "30024"},
					},
					Timeout: 3,
				}
			}

			p := port.NewChecker(cfg.PortCheck, log)
			nodeRoles := splitAndTrim(roles)
			result := p.Execute(nodeRoles)

			if result.Summary.Used > 0 {
				log.Warning("some ports are occupied, please check the report")
			} else {
				log.Info("all ports are free")
			}

			return result, nil
		},
		generateHTML: func(res interface{}) error {
			return output.GeneratePortHTML(*(res.(*config.PortCheckResult)), "port.html")
		},
		format: func(res interface{}, outputFormat string) (string, error) {
			formatter := output.NewFormatter()
			return formatter.FormatPortCheck(res.(*config.PortCheckResult), outputFormat)
		},
		baseName: "port-check-result",
	})
}
