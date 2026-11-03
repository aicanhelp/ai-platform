/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package cmd

import (
	"github.com/spf13/cobra"

	"installer/cmd/add"
	"installer/cmd/create"
	"installer/cmd/destroy"
	"installer/cmd/export"
	"installer/cmd/version"
	"installer/pkg/common"
)

// rootCmd基本命令，不带任何子命令
var rootCmd = &cobra.Command{
	Use:   common.GetServiceName(),
	Short: "openFuyao root command",
	Long: `The fuyao root command is used to create and destroy Kubernetes cluster,
fuyao management surface, and external components. `,
}

// Execute is execute cmd
func Execute() error {
	return rootCmd.Execute()
}

func init() {
	rootCmd.AddCommand(export.NewExportCmd())
	rootCmd.AddCommand(create.NewCreateCmd())
	rootCmd.AddCommand(add.NewAddCmd())
	rootCmd.AddCommand(version.NewVersionCmd())
	rootCmd.AddCommand(destroy.NewDestroyCmd())
}
