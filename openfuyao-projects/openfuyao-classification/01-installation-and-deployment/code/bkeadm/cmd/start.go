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
	"os"

	"github.com/spf13/cobra"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp/sntp"

	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

// ensureDataDir ensures the data directory exists, creating it if necessary
func ensureDataDir(data string) error {
	if !utils.Exists(data) {
		if err := os.MkdirAll(data, utils.DefaultDirPermission); err != nil {
			return err
		}
	}
	return nil
}

// startCmd Start the base dependency service
var startCmd = &cobra.Command{
	Use:   "start",
	Short: "Start some basic fixed services.",
	Long:  `Start some basic fixed services`,
	Example: `
# Start some basic fixed services
bke start imageServer
`,
}

// imageServerStartCmd Start the mirror warehouse service
var imageServerStartCmd = &cobra.Command{
	Use:   "image",
	Short: "Starting the Mirror Repository.",
	Long:  `Starting the Mirror Repository.`,
	Example: `
# Starting the Mirror Repository.
bke start image
`,
	Run: func(cmd *cobra.Command, args []string) {
		data := cmd.Flag("data").Value.String()
		if err := ensureDataDir(data); err != nil {
			fmt.Println(err.Error())
			return
		}
		err := server.StartImageRegistry(cmd.Flag("name").Value.String(), cmd.Flag("image").Value.String(), cmd.Flag("port").Value.String(),
			data)
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

// yumServerStartCmd Start the yum repository service
var yumServerStartCmd = &cobra.Command{
	Use:   "yum",
	Short: "Starting the yum Repository.",
	Long:  `Starting the yum Repository.`,
	Example: `
# Starting the yum Repository.
bke start yum
`,
	Run: func(cmd *cobra.Command, args []string) {
		data := cmd.Flag("data").Value.String()
		if err := ensureDataDir(data); err != nil {
			fmt.Println(err.Error())
			return
		}
		err := server.StartYumRegistry(cmd.Flag("name").Value.String(), cmd.Flag("image").Value.String(), cmd.Flag("port").Value.String(), data)
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

// nfsServerStartCmd  Start the nfs warehouse service
var nfsServerStartCmd = &cobra.Command{
	Use:   "nfs",
	Short: "Starting the nfs Repository.",
	Long:  `Starting the nfs Repository.`,
	Example: `
# Starting the nfs Repository.
bke start nfs
`,
	Run: func(cmd *cobra.Command, args []string) {
		data := cmd.Flag("data").Value.String()
		if err := ensureDataDir(data); err != nil {
			fmt.Println(err.Error())
			return
		}
		err := server.StartNFSServer(cmd.Flag("name").Value.String(), cmd.Flag("image").Value.String(), data)
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

// chartServerStartCmd Start the Chart Repository service
var chartServerStartCmd = &cobra.Command{
	Use:   "chart",
	Short: "Starting the chart Repository.",
	Long:  `Starting the chart Repository.`,
	Example: `
# Starting the chart Repository.
bke start chart
`,
	Run: func(cmd *cobra.Command, args []string) {
		data := cmd.Flag("data").Value.String()
		if err := ensureDataDir(data); err != nil {
			fmt.Println(err.Error())
			return
		}
		err := server.StartChartRegistry(cmd.Flag("name").Value.String(), cmd.Flag("image").Value.String(), cmd.Flag("port").Value.String(),
			data)
		if err != nil {
			fmt.Println(err.Error())
		}
	},
}

// ntpServerStartCmd Start the NTP service
var ntpServerStartCmd = &cobra.Command{
	Use:   "ntpserver",
	Short: "Starting the NTP service.",
	Long:  `Starting the NTP service.`,
	Example: `
# Starting the NTP service.
bke start ntpserver
`,
	Run: func(cmd *cobra.Command, args []string) {
		_, err := sntp.Client(fmt.Sprintf("127.0.0.1:%d", utils.DefaultNTPServerPort))
		if err == nil {
			fmt.Println("The ntp server is running")
			return
		}
		if cmd.Flag("systemd").Value.String() == "true" {
			server.SystemdNTPServer()
			return
		}
		if cmd.Flag("foreground").Value.String() == "true" {
			server.SystemdDaemonNTPServer()
			return
		}
		server.DaemonNTPServer()
	},
}

func registerStartCommand() {
	rootCmd.AddCommand(startCmd)
	//mark startCmd.AddCommand imageServerStartCmd)
	//mark startCmd.AddCommand yumServerStartCmd)
	//mark startCmd.AddCommand nfsServerStartCmd)
	//mark startCmd.AddCommand chartServerStartCmd)
	startCmd.AddCommand(ntpServerStartCmd)

	imageServerStartCmd.Flags().String("name", utils.LocalImageRegistryName, "container name")
	imageServerStartCmd.Flags().String("image",
		utils.DefaultLocalRegistry+utils.DefaultLocalImageRegistry, "image address")
	imageServerStartCmd.Flags().String("port", configinit.DefaultImageRepoPort, "service Port")
	imageServerStartCmd.Flags().String("data", "/tmp/image", "data directory")

	yumServerStartCmd.Flags().String("name", utils.LocalYumRegistryName, "container name")
	yumServerStartCmd.Flags().String("image", utils.DefaultLocalRegistry+utils.DefaultLocalYumRegistry, "image address")
	yumServerStartCmd.Flags().String("port", configinit.DefaultYumRepoPort, "service Port")
	yumServerStartCmd.Flags().String("data", "/tmp/yum", "data directory")

	nfsServerStartCmd.Flags().String("name", utils.LocalNFSRegistryName, "container name")
	nfsServerStartCmd.Flags().String("image", utils.DefaultLocalRegistry+utils.DefaultLocalNFSRegistry, "image address")
	nfsServerStartCmd.Flags().String("data", "/tmp/nfs", "data directory")

	chartServerStartCmd.Flags().String("name", utils.LocalChartRegistryName, "container name")
	chartServerStartCmd.Flags().String("image",
		utils.DefaultLocalRegistry+utils.DefaultLocalChartRegistry, "image address")
	chartServerStartCmd.Flags().String("port", utils.DefaultChartRegistryPort, "service Port")
	chartServerStartCmd.Flags().String("data", "/tmp/chart", "data directory")

	ntpServerStartCmd.Flags().Bool("systemd", false, "systemd service")
	ntpServerStartCmd.Flags().Bool("foreground", false, "foreground service")
}
