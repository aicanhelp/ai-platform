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

	"installer/cmd/options"
)

// OptionsExport is export options
type OptionsExport struct {
	CommonOptions *options.CommonOptions
}

// NewOptionsExport is new OptionsExport
func NewOptionsExport() *OptionsExport {
	return &OptionsExport{
		CommonOptions: options.NewCommonOptions(),
	}
}

// NewExportCmd is new export cmd
func NewExportCmd() *cobra.Command {
	o := NewOptionsExport()
	cmd := &cobra.Command{
		Use:   "export",
		Short: "Export command",
	}

	o.CommonOptions.AddCommonFlags(cmd)
	cmd.AddCommand(NewExportArtifactCmd())
	return cmd
}
