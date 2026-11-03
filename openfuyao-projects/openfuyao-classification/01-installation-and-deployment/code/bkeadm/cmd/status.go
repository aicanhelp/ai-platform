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
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/ntp/sntp"

	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

// statusCmd Displays the status of the local service started by bke.
var statusCmd = &cobra.Command{
	Use:   "status",
	Short: "Displays the status of the local service started by bke.",
	Long:  `Displays the status of the local service started by bke.`,
	Example: `
# Displays the status of the local service started by bke.
bke status
`,
	Run: func(cmd *cobra.Command, args []string) {
		display()
	},
}

func registerStatusCommand() {
	rootCmd.AddCommand(statusCmd)
}

// getContainerServers 返回容器服务器的初始配置列表
func getContainerServers() [][]string {
	return [][]string{
		{"container", utils.LocalKubernetesName,
			fmt.Sprintf("tcp://0.0.0.0:%s", utils.DefaultKubernetesPort), "notCreated", ""},
		{"container", utils.LocalImageRegistryName,
			fmt.Sprintf("tcp://0.0.0.0:%s", configinit.DefaultImageRepoPort), "notCreated",
			fmt.Sprintf("%s/%s", global.Workspace, utils.ImageDataDirectory)},
		{"container", utils.LocalYumRegistryName,
			fmt.Sprintf("tcp://0.0.0.0:%s", configinit.DefaultYumRepoPort), "notCreated",
			fmt.Sprintf("%s/%s", global.Workspace, utils.SourceDataDirectory)},
		{"container", utils.LocalChartRegistryName,
			fmt.Sprintf("tcp://0.0.0.0:%s", utils.DefaultChartRegistryPort), "notCreated",
			fmt.Sprintf("%s/%s", global.Workspace, utils.ChartDataDirectory)},
		{"container", utils.LocalNFSRegistryName,
			fmt.Sprintf("tcp://0.0.0.0:2049"), "notCreated",
			fmt.Sprintf("%s/%s", global.Workspace, utils.NFSDataDirectory)},
	}
}

// getDockerContainerStatus 获取 Docker 容器状态
func getDockerContainerStatus(containerServers [][]string) [][]string {
	var rows [][]string
	for _, server := range containerServers {
		newServer := server
		if info, ok := global.Docker.ContainerExists(server[1]); ok {
			newServer[0] = "docker"
			newServer[3] = info.State.Status
		}
		rows = append(rows, newServer)
	}
	return rows
}

// getContainerdContainerStatus 获取 Containerd 容器状态
func getContainerdContainerStatus(containerServers [][]string) [][]string {
	var rows [][]string
	for _, server := range containerServers {
		newServer := server
		newServer[0] = "containerd"
		newServer[3] = "notCreated"
		if info, err := econd.ContainerInspect(server[1]); err == nil && len(info.Id) > 0 {
			newServer[3] = info.State.Status
		}
		rows = append(rows, newServer)
	}
	return rows
}

// getNtpServerStatus 获取 NTP 服务器状态
func getNtpServerStatus() []string {
	server := []string{"proc", "ntpserver", fmt.Sprintf("udp://0.0.0.0:%d", utils.DefaultNTPServerPort), "notCreated", ""}
	_, err := sntp.Client(fmt.Sprintf("127.0.0.1:%d", utils.DefaultNTPServerPort))
	if err == nil {
		server[3] = "running"
	}
	if utils.Exists("/etc/systemd/system/ntpserver.service") {
		server[0] = "systemd"
	}
	return server
}

// printStatusTable 打印状态表格
func printStatusTable(headers []string, rows [][]string) {
	const tabPadding = 2 // 列之间的最小空格数，用于tabwriter对齐
	w := tabwriter.NewWriter(os.Stdout, 0, 0, tabPadding, ' ', 0)
	fmt.Fprintln(w, strings.Join(headers, "\t"))
	for _, row := range rows {
		fmt.Fprintln(w, strings.Join(row, "\t"))
	}
	if err := w.Flush(); err != nil {
		fmt.Println("flush tablewriter failed:", err.Error())
	}
}

func display() {
	headers := []string{"server", "name", "default", "status", "mount"}
	var rows [][]string
	containerServers := getContainerServers()

	if infrastructure.IsDocker() {
		rows = append(rows, getDockerContainerStatus(containerServers)...)
	}

	if infrastructure.IsContainerd() {
		rows = append(rows, getContainerdContainerStatus(containerServers)...)
	}

	rows = append(rows, getNtpServerStatus())
	printStatusTable(headers, rows)
}
