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
	"errors"
	"time"

	"github.com/spf13/cobra"

	"gopkg.openfuyao.cn/bkeadm/pkg/agent"
)

// commandCmd represents the agent command
var commandCmd = &cobra.Command{
	Use:   "command",
	Short: "The machine executes remote instructions",
	Long:  `Manage the BKEAgent by submitting instructions to Kubernetes.`,
	Example: `
# Migrate the Agent listening cluster
bke command
`,
}

var execOption agent.Options

var execCmd = &cobra.Command{
	Use:   "exec",
	Short: "Execute specified command",
	Long:  `Executes the specified command on certain nodes.`,
	Example: `
# Migrate the Agent listening cluster
bke command exec --nodes ip1,ip2,node3 --command touch /tmp/m1
bke command exec --nodes ip1 -f shell.file
`,

	Args: func(cmd *cobra.Command, args []string) error {
		if execOption.Name == "" {
			execOption.Name = time.Now().Format("200601021504")
		}
		if execOption.Nodes == "" {
			return errors.New("The `nodes` parameter is required. ")
		}
		if execOption.File == "" && execOption.Command == "" {
			return errors.New("One of the parameters `file` and `command` must exist. ")
		}
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		execOption.Args = args
		execOption.Options = options
		return execOption.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		existOption.Args = args
		existOption.Options = options
		execOption.Exec()
	},
}

var liOption agent.Options

var liCmd = &cobra.Command{
	Use:   "list",
	Short: "List all the commands",
	Long:  `List all the commands. `,
	Example: `
# list all the commands
bke command list
`,

	Args: func(cmd *cobra.Command, args []string) error {
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		liOption.Args = args
		liOption.Options = options
		return liOption.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		liOption.Args = args
		liOption.Options = options
		liOption.List()
	},
}

var infoOptions agent.Options

var infoCmd = &cobra.Command{
	Use:   "info",
	Short: "Observe a command out",
	Long:  `Observe a command out `,
	Example: `
# Observe a command out
bke command info ns/name
`,

	Args: func(cmd *cobra.Command, args []string) error {
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		infoOptions.Args = args
		infoOptions.Options = options
		return infoOptions.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		infoOptions.Args = args
		infoOptions.Options = options
		infoOptions.Info()
	},
}

var rmOptions agent.Options

var rmCmd = &cobra.Command{
	Use:   "remove",
	Short: "Delete instruction",
	Long:  `Delete instruction `,
	Example: `
# Delete instruction
bke command remove ns/name
`,

	Args: func(cmd *cobra.Command, args []string) error {
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		rmOptions.Args = args
		rmOptions.Options = options
		return rmOptions.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		rmOptions.Args = args
		rmOptions.Options = options
		rmOptions.Remove()
	},
}

var syncTimeOptions agent.Options

var syncTimeCmd = &cobra.Command{
	Use:   "syncTime",
	Short: "Synchronization time",
	Long:  `Synchronization time`,
	Example: `
# Synchronization time
bke command syncTime 192.168.24.25:123
`,

	Args: func(cmd *cobra.Command, args []string) error {
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		syncTimeOptions.Args = args
		syncTimeOptions.Options = options
		syncTimeOptions.SyncTime()
	},
}

func registerCommandCommand() {
	rootCmd.AddCommand(commandCmd)
	commandCmd.AddCommand(execCmd)
	commandCmd.AddCommand(liCmd)
	commandCmd.AddCommand(infoCmd)
	commandCmd.AddCommand(rmCmd)
	commandCmd.AddCommand(syncTimeCmd)

	// Here you will define your flags and configuration settings.
	execCmd.Flags().StringVar(&execOption.Name, "name", "", "instruction name")
	execCmd.Flags().StringVarP(&execOption.File, "file", "f", "", "shell command file")
	execCmd.Flags().StringVar(&execOption.Command, "command", "", "shell command")
	execCmd.Flags().StringVarP(&execOption.Nodes, "nodes", "n", "", "node list, example 192.168.1.231,192.xxx.xxx.xxx")
}
