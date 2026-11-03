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

	"gopkg.openfuyao.cn/bkeadm/pkg/reset"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

var resetOption reset.Options

// resetCmd represents the reset command
var resetCmd = &cobra.Command{
	Use:   "reset",
	Short: "Remove the local Kubernetes cluster",
	Long:  `Clean up the boot node service and restore the node to the bare state.`,
	Example: `
# Delete the local kubernetes service
bke reset
# Clear the boot node service and mount directory
bke reset --mount
# Empty the node container and runtime
bke reset --all
# Clear node services and remove decompressed data
bke reset --all --mount
`,
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if resetOption.All && !confirm {
			fmt.Println("This instruction deletes the contaienr service and the contaienr runtime, " +
				"returning it to an uninitialized state")
			if !utils.PromptForConfirmation(confirm) {
				return fmt.Errorf("operation cancelled by user")
			}
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		resetOption.Options = options
		resetOption.Args = args
		resetOption.Reset()
	},
}

func registerResetCommand() {
	rootCmd.AddCommand(resetCmd)

	// Here you will define your flags and configuration settings.
	resetCmd.Flags().BoolVar(&resetOption.All, "all", false, "Restore node")
	resetCmd.Flags().BoolVar(&resetOption.Mount, "mount", false, "remove decompress directories and services")
	resetCmd.Flags().BoolVar(&confirm, "confirm", false, "Skip deleting all confirmations")
}
