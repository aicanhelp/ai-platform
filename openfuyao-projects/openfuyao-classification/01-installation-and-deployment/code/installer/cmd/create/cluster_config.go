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

package create

import (
	"github.com/spf13/cobra"

	"installer/pkg/common"
	"installer/pkg/pipelines"
)

// ClusterConfigFileOptions is cluster config file options
type ClusterConfigFileOptions struct {
	EnableAllParams      bool
	EnableAdvancedParams bool
	EnableHaParams       bool
	FileSaveFolder       string
}

// NewClusterConfigFileOptions is new ClusterConfigFileOptions
func NewClusterConfigFileOptions() *ClusterConfigFileOptions {
	return &ClusterConfigFileOptions{}
}

// NewClusterConfigFileCmd is new cmd
func NewClusterConfigFileCmd() *cobra.Command {
	o := NewClusterConfigFileOptions()
	cmd := &cobra.Command{
		Use:   "cluster-config",
		Short: "Create cluster configuration file",
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.Run()
		},
	}

	o.AddFlags(cmd)

	return cmd
}

// Run is run cmd
func (o *ClusterConfigFileOptions) Run() error {
	arg := common.Argument{
		EnableAllParams:      o.EnableAllParams,
		EnableAdvancedParams: o.EnableAdvancedParams,
		EnableHaParams:       o.EnableHaParams,
		FileSaveFolder:       o.FileSaveFolder,
	}

	return pipelines.CreateClusterConfigFile(arg)
}

// AddFlags is get flags
func (o *ClusterConfigFileOptions) AddFlags(cmd *cobra.Command) {
	cmd.Flags().BoolVarP(&o.EnableAllParams, "all-params", "", false, "The cluster configuration file contains all parameters")
	cmd.Flags().BoolVarP(&o.EnableAdvancedParams, "advanced-params", "", false, "The cluster configuration file contains advanced parameters")
	cmd.Flags().BoolVarP(&o.EnableHaParams, "ha", "", false, "The cluster configuration file contains highly available deployment parameters")
	cmd.Flags().StringVarP(&o.FileSaveFolder, "path", "p", "", "Cluster configuration file save path")
}
