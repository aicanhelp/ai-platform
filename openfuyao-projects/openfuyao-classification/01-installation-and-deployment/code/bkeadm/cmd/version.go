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
	"fmt"

	"github.com/spf13/cobra"

	"gopkg.openfuyao.cn/bkeadm/utils/version"
)

// versionCmd represents the version command
var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "version",
	Long:  `bke version.`,
	Example: `
# View the BKE version
bke version
`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println(fmt.Sprintf("version: %s", version.Version))
		fmt.Println(fmt.Sprintf("gitCommitID: %s", version.GitCommitID))
		fmt.Println(fmt.Sprintf("os/arch: %s", version.Architecture))
		fmt.Println(fmt.Sprintf("date: %s", version.Timestamp))
	},
}

// onlyCmd represents the version command
var onlyCmd = &cobra.Command{
	Use:   "only",
	Short: "only",
	Long:  `bke version only.`,
	Example: `
# View the BKE version
bke version only
`,
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Println(version.Version)
	},
}

func registerVersionCommand() {
	rootCmd.AddCommand(versionCmd)
	versionCmd.AddCommand(onlyCmd)
}
