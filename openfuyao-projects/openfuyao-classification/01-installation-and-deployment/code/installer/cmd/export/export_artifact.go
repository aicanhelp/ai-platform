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

package export

import (
	"github.com/spf13/cobra"

	cmdcommon "installer/cmd/common"
	"installer/cmd/options"
	"installer/pkg/common"
	"installer/pkg/pipelines"
)

// OptionsExportArtifact is export artifact options
type OptionsExportArtifact struct {
	CommonOptions *options.CommonOptions

	FileSaveFolder string
	FuyaoZone      string
}

// NewOptionsExportArtifact is new OptionsExportArtifact
func NewOptionsExportArtifact() *OptionsExportArtifact {
	return &OptionsExportArtifact{
		CommonOptions: options.NewCommonOptions(),
	}
}

// NewExportArtifactCmd is export artifact cmd
func NewExportArtifactCmd() *cobra.Command {
	o := NewOptionsExportArtifact()
	cmd := &cobra.Command{
		Use:   "artifact",
		Short: "Download the required images and binary files for offline installation",
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.run()
		},
	}

	o.CommonOptions.AddCommonFlags(cmd)
	o.addFlags(cmd)
	return cmd
}

func (o *OptionsExportArtifact) run() error {
	arg := common.Argument{
		Debug:          o.CommonOptions.Debug,
		FuyaoZone:      o.FuyaoZone,
		FileSaveFolder: o.FileSaveFolder,
	}

	cmdcommon.InitLog(o.CommonOptions.Debug)
	return pipelines.ExportArtifact(arg)
}

func (o *OptionsExportArtifact) addFlags(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&o.FileSaveFolder, "path", "p", "", "Artifact file save path")
	cmd.Flags().StringVarP(&o.FuyaoZone, "zone", "", "cn", "Time zone, used to identify the country")
}
