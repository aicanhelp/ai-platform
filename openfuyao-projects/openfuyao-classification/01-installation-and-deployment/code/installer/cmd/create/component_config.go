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

// ComponentConfigFileOptions is component config options
type ComponentConfigFileOptions struct {
	componentManifestsSavePath string
}

// NewComponentConfigFileOptions is new ComponentConfigFileOptions
func NewComponentConfigFileOptions() *ComponentConfigFileOptions {
	return &ComponentConfigFileOptions{}
}

// NewComponentConfigFileCmd is new cmd
func NewComponentConfigFileCmd() *cobra.Command {
	o := NewComponentConfigFileOptions()
	cmd := &cobra.Command{
		Use:   "component-config",
		Short: "Create component configuration file",
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.Run()
		},
	}

	o.AddFlags(cmd)

	return cmd
}

// Run is run cmd
func (o *ComponentConfigFileOptions) Run() error {
	arg := common.Argument{
		FileSaveFolder: o.componentManifestsSavePath,
	}

	return pipelines.CreateComponentConfigFile(arg)
}

// AddFlags is get flags
func (o *ComponentConfigFileOptions) AddFlags(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&o.componentManifestsSavePath, "path", "p", "", "Component configuration file save path")
}
