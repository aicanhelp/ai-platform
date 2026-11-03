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

// ComponentPackageOptions is component package options
type ComponentPackageOptions struct {
	componentManifestsFile string
	enableSubpackageTar    bool
}

// NewComponentPackageOptions is new ComponentPackageOptions
func NewComponentPackageOptions() *ComponentPackageOptions {
	return &ComponentPackageOptions{}
}

// NewComponentPackageCmd is new cmd
func NewComponentPackageCmd() *cobra.Command {
	o := NewComponentPackageOptions()
	cmd := &cobra.Command{
		Use:   "component-package",
		Short: "Create component installation package",
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.Run()
		},
	}

	o.AddFlags(cmd)

	return cmd
}

// Run is run cmd
func (o *ComponentPackageOptions) Run() error {
	arg := common.Argument{
		FilePath:            o.componentManifestsFile,
		EnableSubpackageTar: o.enableSubpackageTar,
	}

	return pipelines.CreateComponentPackage(arg)
}

// AddFlags is get flag
func (o *ComponentPackageOptions) AddFlags(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&o.componentManifestsFile, "file", "f", "", "Component installation package save path")
	cmd.Flags().BoolVarP(&o.enableSubpackageTar, "subpackage-tar", "", false,
		"Package and compress according to the operating system and CPU architecture")
}
