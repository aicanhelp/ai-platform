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

	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

// removeCmd Removing a specified Service
var removeCmd = &cobra.Command{
	Use:   "remove",
	Short: "Removing dependent Services.",
	Long:  `Removing dependent Services.`,
	Example: `
# Removing dependent Services
bke remove chart
`,
}

// imageServerRemoveCmd Remove the mirror repository service
var imageServerRemoveCmd = &cobra.Command{
	Use:   "image",
	Short: "Removing the Mirror Repository.",
	Long:  `Removing the Mirror Repository.`,
	Example: `
# Removing the Mirror Repository.
bke remove image
`,
	Run: func(cmd *cobra.Command, args []string) {
		name := utils.LocalImageRegistryName
		if len(args) > 0 {
			name = args[0]
		}
		err := server.RemoveImageRegistry(name)
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

// yumServerRemoveCmd Remove the yum repository service
var yumServerRemoveCmd = &cobra.Command{
	Use:   "yum",
	Short: "Removing the yum Repository.",
	Long:  `Removing the yum Repository.`,
	Example: `
# Removing the yum Repository.
bke remove yum
`,
	Run: func(cmd *cobra.Command, args []string) {
		name := utils.LocalYumRegistryName
		if len(args) > 0 {
			name = args[0]
		}
		err := server.RemoveYumRegistry(name)
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

// nfsServerRemoveCmd Remove the NFS warehouse service
var nfsServerRemoveCmd = &cobra.Command{
	Use:   "nfs",
	Short: "Removing the nfs Repository.",
	Long:  `Removing the nfs Repository.`,
	Example: `
# Removing the nfs Repository.
bke remove nfs
`,
	Run: func(cmd *cobra.Command, args []string) {
		name := utils.LocalNFSRegistryName
		if len(args) > 0 {
			name = args[0]
		}
		err := server.RemoveNFSServer(name)
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

// chartServerRemoveCmd Remove the Chart repository service
var chartServerRemoveCmd = &cobra.Command{
	Use:   "chart",
	Short: "Removing the chart Repository.",
	Long:  `Removing the chart Repository.`,
	Example: `
# Removing the chart Repository.
bke remove chart
`,
	Run: func(cmd *cobra.Command, args []string) {
		name := utils.LocalChartRegistryName
		if len(args) > 0 {
			name = args[0]
		}
		err := server.RemoveChartRegistry(name)
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

// ntpServerRemoveCmd Remove the NTP service
var ntpServerRemoveCmd = &cobra.Command{
	Use:   "ntpserver",
	Short: "Removing the ntp service.",
	Long:  `Removing the ntp service.`,
	Example: `
# Removing the ntp service.
bke remove ntpserver
`,
	Run: func(cmd *cobra.Command, args []string) {
		err := server.RemoveNTPServer()
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

func registerRemoveCommand() {
	rootCmd.AddCommand(removeCmd)
	removeCmd.AddCommand(imageServerRemoveCmd)
	removeCmd.AddCommand(yumServerRemoveCmd)
	removeCmd.AddCommand(nfsServerRemoveCmd)
	removeCmd.AddCommand(chartServerRemoveCmd)
	removeCmd.AddCommand(ntpServerRemoveCmd)
}
