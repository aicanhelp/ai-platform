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

// OptionsCreateRegistry is registry options
type OptionsCreateRegistry struct {
	CommonOptions *options.CommonOptions

	ArtifactPath       string
	RegistryNodeUser   string
	RegistryNodeHost   string
	RegistryNodePasswd string
	RegistryDomain     string
	RegistryPasswd     string
	RegistryDBPasswd   string
	CertExpiryDays     string
	SkipRmTmpFile      bool
}

// NewOptionsCreateRegistry is new OptionsCreateRegistry
func NewOptionsCreateRegistry() *OptionsCreateRegistry {
	return &OptionsCreateRegistry{
		CommonOptions: options.NewCommonOptions(),
	}
}

// NewCreateRegistryCmd is new cmd
func NewCreateRegistryCmd() *cobra.Command {
	o := NewOptionsCreateRegistry()
	cmd := &cobra.Command{
		Use:   "registry",
		Short: "Create an image warehouse and upload the image",
		RunE: func(cmd *cobra.Command, args []string) error {
			return o.run()
		},
	}

	o.CommonOptions.AddCommonFlags(cmd)
	o.addFlags(cmd)

	return cmd
}

// Run is run cmd
func (o *OptionsCreateRegistry) run() error {
	arg := common.Argument{
		FilePath:               o.ArtifactPath,
		RegistryNodeUsername:   o.RegistryNodeUser,
		RegistryNodeIP:         o.RegistryNodeHost,
		RegistryNodePasswd:     []byte(o.RegistryNodePasswd),
		RegistryDomain:         o.RegistryDomain,
		RegistryPasswd:         []byte(o.RegistryPasswd),
		RegistryDBPasswd:       []byte(o.RegistryDBPasswd),
		RegistryCertExpiryDays: o.CertExpiryDays,
		Debug:                  o.CommonOptions.Debug,
		SkipRmTmpFile:          o.SkipRmTmpFile,
	}

	defer o.clearSensitiveData()

	cmdcommon.InitLog(o.CommonOptions.Debug)
	return pipelines.CreateRegistry(arg)
}

// AddFlags is get flags
func (o *OptionsCreateRegistry) addFlags(cmd *cobra.Command) {
	cmd.Flags().StringVarP(&o.ArtifactPath, "artifact-path", "", "", "Artifact save address")
	cmd.Flags().StringVarP(&o.RegistryNodeUser, "registry-node-user", "", "root", "Mirror warehouse node username")
	cmd.Flags().StringVarP(&o.RegistryNodeHost, "registry-node-host", "", "", "Mirror warehouse node host")
	cmd.Flags().StringVarP(&o.RegistryNodePasswd, "registry-node-passwd", "", "", "Mirror warehouse node password")
	cmd.Flags().StringVarP(&o.RegistryDomain, "registry-domain", "", "", "Mirror warehouse domain name")
	cmd.Flags().StringVarP(&o.RegistryPasswd, "registry-passwd", "", "", "Mirror warehouse password")
	cmd.Flags().StringVarP(&o.RegistryDBPasswd, "registry-db-passwd", "", "", "Mirror warehouse database password")
	cmd.Flags().StringVarP(&o.CertExpiryDays, "cert-expiry-days", "", "3650",
		"The expiration days of the mirror warehouse certificate")
	cmd.Flags().BoolVarP(&o.SkipRmTmpFile, "skip-rm-tmp-file", "", false, "Skip deleting temporary files")

	cmd.MarkFlagRequired("registry-passwd")
	cmd.MarkFlagRequired("registry-db-passwd")
	cmd.MarkFlagRequired("artifact-path")
}

func (o *OptionsCreateRegistry) clearSensitiveData() {
	o.RegistryNodePasswd = ""
	o.RegistryPasswd = ""
	o.RegistryDBPasswd = ""
}
