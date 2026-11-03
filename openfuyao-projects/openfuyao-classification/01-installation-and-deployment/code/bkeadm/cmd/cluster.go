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
	"strings"

	"github.com/spf13/cobra"

	"gopkg.openfuyao.cn/bkeadm/pkg/cluster"
)

// clusterCmd represents the cluster command
var clusterCmd = &cobra.Command{
	Use:   "cluster",
	Short: "Operating an existing cluster.",
	Long:  `Operating an existing cluster.`,
	Example: `
# Deploy the Kubernetes cluster
bke cluster -h
`,
}

var listOption = cluster.Options{}

// listDep represents the cluster command
var listDep = &cobra.Command{
	Use:   "list",
	Short: "Cluster list",
	Long:  `Cluster list`,
	Example: `
# Obtaining the cluster list
bke cluster list
`,
	Args: func(cmd *cobra.Command, args []string) error {
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		listOption.Args = args
		listOption.Options = options
		return listOption.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		listOption.Args = args
		listOption.Options = options
		listOption.List()
	},
}

var removeOption = cluster.Options{}

// removeDep represents the cluster command
var removeDep = &cobra.Command{
	Use:   "remove",
	Short: "Delete a Specified Cluster",
	Long:  `Delete a Specified Cluster`,
	Example: `
# Delete a Specified Cluster
bke cluster remove ns/name
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return errors.New("Required parameters are missing. ")
		}
		if len(strings.Split(args[0], "/")) != 2 {
			return errors.New("The parameter format is invalid, The parameter format is ns/name. ")
		}
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		removeOption.Args = args
		removeOption.Options = options
		return removeOption.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		removeOption.Args = args
		removeOption.Options = options
		removeOption.Remove()
	},
}

var createOption = cluster.Options{}

// createDep represents the cluster command
var createDep = &cobra.Command{
	Use:   "create",
	Short: "Deploying a Cluster",
	Long:  `Deploying a Cluster.`,
	Example: `
# Deploy the Kubernetes cluster
bke cluster create -f /bke/cluster/bkecluster.yaml -n /bke/cluster/bkenodes.yaml
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if createOption.File == "" {
			return errors.New("The `file` parameter is required. ")
		}
		if createOption.NodesFile == "" {
			return errors.New("The `nodes` parameter is required. ")
		}
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		createOption.Args = args
		createOption.Options = options
		return createOption.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		createOption.Args = args
		createOption.Options = options
		createOption.Cluster()
	},
}

var scaleOption = cluster.Options{}

// scaleDep represents the cluster command
var scaleDep = &cobra.Command{
	Use:   "scale",
	Short: "shard cluster",
	Long:  `shard cluster`,
	Example: `
# shard cluster
bke cluster scale -f /bke/cluster/bkecluster.yaml -n /bke/cluster/bkenodes.yaml
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if scaleOption.File == "" {
			return errors.New("The `file` parameter is required. ")
		}
		if scaleOption.NodesFile == "" {
			return errors.New("The `nodes` parameter is required. ")
		}
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		scaleOption.Args = args
		scaleOption.Options = options
		return scaleOption.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		scaleOption.Args = args
		scaleOption.Options = options
		scaleOption.Scale()
	},
}

var logsOption = cluster.Options{}

// logsDep represents the cluster command
var logsDep = &cobra.Command{
	Use:   "logs",
	Short: "Obtain cluster deployment events",
	Long:  `Obtain cluster deployment events`,
	Example: `
# Obtain cluster deployment events
bke cluster logs ns/name
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return errors.New("Required parameters are missing. ")
		}
		if len(strings.Split(args[0], "/")) != 2 {
			return errors.New("The parameter format is invalid, The parameter format is ns/name. ")
		}
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		logsOption.Args = args
		logsOption.Options = options
		return logsOption.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		logsOption.Args = args
		logsOption.Options = options
		logsOption.Log()
	},
}

var existOption = cluster.Options{}

// existDep represents the cluster command
var existDep = &cobra.Command{
	Use:   "exist",
	Short: "Manage existing clusters",
	Long:  `Manage existing clusters`,
	Example: `
# Upgrade the old k8s cluster and manage other types of clusters
bke cluster exist --conf kubeconf -f bkecluster.yaml 
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if existOption.File == "" {
			return errors.New("The `file` parameter is required. ")
		}
		if existOption.Conf == "" {
			return errors.New("The `conf` parameter is required. ")
		}
		return nil
	},
	PreRunE: func(cmd *cobra.Command, args []string) error {
		existOption.Args = args
		existOption.Options = options
		return existOption.ClusterPre()
	},
	Run: func(cmd *cobra.Command, args []string) {
		existOption.Args = args
		existOption.Options = options
		existOption.ExistsCluster()
	},
}

func registerClusterCommand() {
	rootCmd.AddCommand(clusterCmd)

	//mark clusterCmd AddCommand removeDep
	//mark clusterCmd AddCommand scaleDep
	//mark clusterCmd AddCommand logsDep
	//mark clusterCmd AddCommand existDep

	clusterCmd.AddCommand(listDep)
	clusterCmd.AddCommand(createDep)
	// Here you will define your flags and configuration settings.
	createDep.Flags().StringVarP(&createOption.File, "file", "f", "", "BKE Cluster Configuration File")
	createDep.Flags().StringVarP(&createOption.NodesFile, "nodes", "n", "", "BKE Nodes Configuration File")
	createDep.Flags().StringVar(&createOption.ImmutableOS, "immutable-os", "", "Set immutable OS type for worker nodes, e.g. 'kubeos'")
	scaleDep.Flags().StringVarP(&scaleOption.File, "file", "f", "", "BKE Cluster Configuration File")
	scaleDep.Flags().StringVarP(&scaleOption.NodesFile, "nodes", "n", "", "BKE Nodes Configuration File")

	existDep.Flags().StringVarP(&existOption.File, "file", "f", "", "BKE Cluster Configuration File")
	existDep.Flags().StringVarP(&existOption.Conf, "conf", "c", "", "Target Cluster kubeconfig File")
}
