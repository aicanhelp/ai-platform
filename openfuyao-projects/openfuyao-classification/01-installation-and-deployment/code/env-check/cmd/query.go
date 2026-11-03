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

	"env-check/pkg/config"
	"env-check/pkg/logger"
	"env-check/pkg/output"
	"env-check/pkg/query"
)

// runQuery executes query operation using the common runner
func runQuery(cmd *cobra.Command, args []string) error {
	return runAction(cmd, args, config.ModeQuery, actionHandler{
		execute: func(cmd *cobra.Command, cfg *config.AppConfig, log logger.Logger) (interface{}, error) {
			q := query.NewQuery(cfg, log)
			result, err := q.Execute()
			if err != nil {
				log.Error("file query failed: " + err.Error())
				return nil, err
			}
			return result, nil
		},
		generateHTML: func(res interface{}) error {
			return output.GenerateQueryHTML(*(res.(*config.QueryResult)), "query.html")
		},
		format: func(res interface{}, outputFormat string) (string, error) {
			formatter := output.NewFormatter()
			return formatter.FormatQuery(res.(*config.QueryResult), outputFormat)
		},
		baseName: "query-result",
	})
}
