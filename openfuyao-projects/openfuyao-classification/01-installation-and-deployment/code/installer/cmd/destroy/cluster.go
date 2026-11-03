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

package destroy

import (
	"github.com/spf13/cobra"

	cmdcommon "installer/cmd/common"
	"installer/cmd/options"
	"installer/pkg/common"
	"installer/pkg/pipelines"
)

// OptionsDestroyCluster 卸载集群标志
type OptionsDestroyCluster struct {
	CommonOptions *options.CommonOptions

	ClusterCfgFile     string
	RegistryNodeUser   string
	RegistryNodeHost   string
	RegistryNodePasswd string
	SkipDestroyHub     bool
	EnableOffline      bool
	AllComponent       bool
	// ArtifactPath 离线制品存放路径
	ArtifactPath string
}

// NewOptionsDestroyCluster is new OptionsDestroyCluster
func NewOptionsDestroyCluster() *OptionsDestroyCluster {
	return &OptionsDestroyCluster{
		CommonOptions: options.NewCommonOptions(),
	}
}

// NewDestroyClusterCmd 卸载集群命令
func NewDestroyClusterCmd() *cobra.Command {
	o := NewOptionsDestroyCluster()
	cmd := &cobra.Command{
		Use:   "cluster",
		Short: "Destroy a cluster",
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.run()
		},
	}

	o.CommonOptions.AddCommonFlags(cmd)
	o.addFlags(cmd)
	return cmd
}

func (o *OptionsDestroyCluster) run() error {
	args := common.Argument{
		FilePath:             o.ClusterCfgFile,
		RegistryNodeUsername: o.RegistryNodeUser,
		RegistryNodeIP:       o.RegistryNodeHost,
		RegistryNodePasswd:   []byte(o.RegistryNodePasswd),
		SkipDestroyHub:       o.SkipDestroyHub,
		Debug:                o.CommonOptions.Debug,
		DestroyCluster:       true,
		EnableOffline:        o.EnableOffline,
		AllComponent:         o.AllComponent,
		ArtifactPath:         o.ArtifactPath,
	}

	cmdcommon.InitLog(o.CommonOptions.Debug)
	return pipelines.DestroyCluster(args)
}

func (o *OptionsDestroyCluster) addFlags(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&o.ClusterCfgFile, "filename", "f", "", "Path to a configuration file")
	cmd.Flags().BoolVarP(&o.EnableOffline, "offline", "", false, "Offline uninstall")
	cmd.Flags().StringVarP(&o.RegistryNodeUser, "registry-node-user", "", "root", "Mirror warehouse node username")
	cmd.Flags().StringVarP(&o.RegistryNodeHost, "registry-node-host", "", "", "Mirror warehouse node ip")
	cmd.Flags().BoolVarP(&o.AllComponent, "all", "A", false, "Uninstall all components")
	cmd.Flags().StringVarP(&o.RegistryNodePasswd, "registry-node-passwd", "", "", "Mirror warehouse node password")
	cmd.Flags().BoolVarP(&o.SkipDestroyHub, "skip-rm-hub", "", false, "Skip destroy hub")
	cmd.Flags().StringVarP(&o.ArtifactPath, "artifact-path", "", "", "artifact save path")
}
