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

	cmdcommon "installer/cmd/common"
	"installer/cmd/options"
	"installer/pkg/common"
	"installer/pkg/pipelines"
)

// OptionsCreateCluster is create cluster options
type OptionsCreateCluster struct {
	// ClusterCfgFile 集群配置文件
	ClusterCfgFile string
	// EnableOffline 是否为离线安装
	EnableOffline bool
	// ArtifactPath 离线制品存放路径
	ArtifactPath string
	// InitCfg 集群初始化文件
	InitCfg string
	// AllComponent 安装所有组件
	AllComponent bool
	// FuyaoZone 时区
	FuyaoZone string
	// EnableAI 是否启用AI
	EnableAI bool
	// SkipRmTmpFile 跳过删除临时文件
	SkipRmTmpFile bool
	// CommonOptions 通用选项
	CommonOptions *options.CommonOptions
}

// NewOptionsCreateCluster is new OptionsCreateCluster
func NewOptionsCreateCluster() *OptionsCreateCluster {
	return &OptionsCreateCluster{
		CommonOptions: options.NewCommonOptions(),
	}
}

// NewCreateClusterCmd is new create cluster cmd
func NewCreateClusterCmd() *cobra.Command {
	o := NewOptionsCreateCluster()

	cmd := &cobra.Command{
		Use:   "cluster",
		Short: "Create a Kubernetes cluster or create a Kubernetes cluster with openfuyao-system",
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.run()
		},
	}

	o.CommonOptions.AddCommonFlags(cmd)
	o.addFlags(cmd)
	return cmd
}

func (o *OptionsCreateCluster) run() error {
	args := common.Argument{
		FilePath:      o.ClusterCfgFile,
		AllComponent:  o.AllComponent,
		Debug:         o.CommonOptions.Debug,
		FuyaoZone:     o.FuyaoZone,
		ArtifactPath:  o.ArtifactPath,
		EnableOffline: o.EnableOffline,
		InitCfg:       o.InitCfg,
		SkipRmTmpFile: o.SkipRmTmpFile,
		EnableAI:      o.EnableAI,
	}

	cmdcommon.InitLog(o.CommonOptions.Debug)
	return pipelines.CreateCluster(args)
}

func (o *OptionsCreateCluster) addFlags(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&o.FuyaoZone, "zone", "", "cn", "Time zone, used to identify the country")
	cmd.Flags().StringVarP(&o.ClusterCfgFile, "file", "f", "", "Path to a cluster config file")
	cmd.Flags().BoolVarP(&o.EnableOffline, "offline", "", false, "Offline install")
	cmd.Flags().BoolVarP(&o.AllComponent, "all", "A", false, "Install all components")
	cmd.Flags().StringVarP(&o.InitCfg, "init-config", "", "", "init cluster config file")
	cmd.Flags().StringVarP(&o.ArtifactPath, "artifact-path", "", "", "artifact save path")
	cmd.Flags().BoolVarP(&o.SkipRmTmpFile, "skip-rm-tmp-file", "", false, "Skip deleting temporary files")
	cmd.Flags().BoolVarP(&o.EnableAI, "ascend-ai", "", false, "Whether to install ascending plotter")
}
