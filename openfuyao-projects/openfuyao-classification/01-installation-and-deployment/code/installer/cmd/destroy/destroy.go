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

	"installer/cmd/options"
)

// OptionsDestroy 销毁命令的标志
type OptionsDestroy struct {
	CommonOptions *options.CommonOptions
}

// NewOptionsDestroy is new OptionsDestroy
func NewOptionsDestroy() *OptionsDestroy {
	return &OptionsDestroy{
		CommonOptions: options.NewCommonOptions(),
	}
}

// NewDestroyCmd 销毁命令
func NewDestroyCmd() *cobra.Command {
	o := NewOptionsDestroy()
	cmd := &cobra.Command{
		Use:   "destroy",
		Short: "Destroy command",
	}

	o.CommonOptions.AddCommonFlags(cmd)

	cmd.AddCommand(NewDestroyClusterCmd())
	return cmd
}
