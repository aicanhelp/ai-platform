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

package add

import (
	"github.com/spf13/cobra"

	cmdcommon "installer/cmd/common"
	"installer/cmd/options"
	"installer/pkg/common"
	"installer/pkg/pipelines"
)

// OptionsAddNodes 添加节点选项
type OptionsAddNodes struct {
	// 通用选项
	CommonOptions *options.CommonOptions
	// AddNodesCfgFile 添加节点配置文件
	AddNodesCfgFile string
	// EnableOffline 是否为离线安装
	EnableOffline bool
	// ArtifactPath 离线制品存放路径
	ArtifactPath string
	// FuyaoZone 时区
	FuyaoZone string
	// SkipRmTmpFile 跳过删除临时文件
	SkipRmTmpFile bool
}

// NewOptionsAddNodes is new OptionsAddNodes
func NewOptionsAddNodes() *OptionsAddNodes {
	return &OptionsAddNodes{
		CommonOptions: options.NewCommonOptions(),
	}
}

// NewAddNodesCmd is add node cmd
func NewAddNodesCmd() *cobra.Command {
	o := NewOptionsAddNodes()
	cmd := &cobra.Command{
		Use:   "nodes",
		Short: "Add nodes to the cluster based on the configuration file",
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.run()
		},
	}

	o.CommonOptions.AddCommonFlags(cmd)
	o.setFlags(cmd)
	return cmd
}

// run is run add nodes
func (o *OptionsAddNodes) run() error {
	args := common.Argument{
		FilePath:      o.AddNodesCfgFile,
		AddNodes:      true,
		Debug:         o.CommonOptions.Debug,
		FuyaoZone:     o.FuyaoZone,
		ArtifactPath:  o.ArtifactPath,
		EnableOffline: o.EnableOffline,
		SkipRmTmpFile: o.SkipRmTmpFile,
	}
	cmdcommon.InitLog(o.CommonOptions.Debug)
	return pipelines.AddNodes(args)
}

// setFlags is get add node flag
func (o *OptionsAddNodes) setFlags(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&o.AddNodesCfgFile, "file", "f", "", "Config file save path")
	cmd.Flags().StringVarP(&o.FuyaoZone, "zone", "", "cn", "Time zone, used to identify the country")
	cmd.Flags().BoolVarP(&o.EnableOffline, "offline", "", false, "Offline add nodes")
	cmd.Flags().StringVarP(&o.ArtifactPath, "artifact-path", "", "", "artifact save path")
	cmd.Flags().BoolVarP(&o.SkipRmTmpFile, "skip-rm-tmp-file", "", false, "Skip deleting temporary files")

	cmd.MarkFlagRequired("file")
}
