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

	"installer/cmd/options"
)

// OptionsCreate is create options
type OptionsCreate struct {
	CommonOptions *options.CommonOptions
}

// NewOptionsCreate is new OptionsCreate
func NewOptionsCreate() *OptionsCreate {
	return &OptionsCreate{
		CommonOptions: options.NewCommonOptions(),
	}
}

// NewCreateCmd is create cmd
func NewCreateCmd() *cobra.Command {
	o := NewOptionsCreate()
	cmd := &cobra.Command{
		Use:   "create",
		Short: "create a cluster or a cluster configuration file",
	}

	o.CommonOptions.AddCommonFlags(cmd)

	cmd.AddCommand(NewCreateClusterCmd())
	cmd.AddCommand(NewClusterConfigFileCmd())
	cmd.AddCommand(NewComponentConfigFileCmd())
	cmd.AddCommand(NewComponentPackageCmd())
	cmd.AddCommand(NewCreateRegistryCmd())
	return cmd
}
