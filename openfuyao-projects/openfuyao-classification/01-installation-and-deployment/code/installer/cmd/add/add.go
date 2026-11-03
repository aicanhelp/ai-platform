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

	"installer/cmd/options"
)

type AddOptions struct {
	CommonOptions *options.CommonOptions
}

func NewAddOptions() *AddOptions {
	return &AddOptions{
		CommonOptions: options.NewCommonOptions(),
	}
}

func NewAddCmd() *cobra.Command {
	o := NewAddOptions()
	cmd := &cobra.Command{
		Use:   "add",
		Short: "Add command",
	}

	o.CommonOptions.AddCommonFlags(cmd)
	cmd.AddCommand(NewAddNodesCmd())
	return cmd
}
