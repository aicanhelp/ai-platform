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

package main

import (
	"github.com/spf13/cobra"

	"env-check/pkg/clean"
	"env-check/pkg/config"
	"env-check/pkg/logger"
	"env-check/pkg/output"
)

// runClean executes clean operation using the common runner
func runClean(cmd *cobra.Command, args []string) error {
	return runAction(cmd, args, config.ModeClean, actionHandler{
		execute: func(cmd *cobra.Command, cfg *config.AppConfig, log logger.Logger) (interface{}, error) {
			// Pass force flag into config
			cfg.CleanForce, _ = cmd.Flags().GetBool("force")

			c := clean.NewCleaner(cfg, log)
			result, err := c.Execute()
			if err != nil {
				log.Error("file clean failed: " + err.Error())
				return nil, err
			}
			return result, nil
		},
		generateHTML: func(res interface{}) error {
			return output.GenerateCleanHTML(*(res.(*config.CleanResult)), "clean.html")
		},
		format: func(res interface{}, outputFormat string) (string, error) {
			formatter := output.NewFormatter()
			return formatter.FormatClean(res.(*config.CleanResult), outputFormat)
		},
		baseName: "clean-result",
	})
}
