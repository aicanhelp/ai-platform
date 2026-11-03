/******************************************************************
 * Copyright (c) 2024 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package cmd

import (
	"errors"
	"runtime"

	"github.com/spf13/cobra"

	"gopkg.openfuyao.cn/bkeadm/pkg/build"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

var buildOption build.Options

// buildCmd represents the build command
var buildCmd = &cobra.Command{
	Use:   "build",
	Short: "Build the BKE installation package.",
	Long:  `Build the BKE installation package.`,
	Example: `
# Build the BKE installation package.
bke build -f bke.yaml -t bke.tar.gz
`,
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if buildOption.File == "" {
			return errors.New("The parameter `file` is required. ")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		buildOption.Args = args
		buildOption.Options = options
		buildOption.Build()
	},
}

// buildCmd represents the build command
var buildConfigCmd = &cobra.Command{
	Use:   "config",
	Short: "The default BKE configuration is exported.",
	Long:  `The default BKE configuration is exported.`,
	Example: `
# Build the BKE installation package.
bke build config
`,
	Run: func(cmd *cobra.Command, args []string) {
		buildOption.Args = args
		buildOption.Options = options
		buildOption.Config()
	},
}

var patchOption build.Options

// patchCmd represents the build command
var patchCmd = &cobra.Command{
	Use:   "patch",
	Short: "Build the bke patch pack.",
	Long:  `Build the bke patch pack.`,
	Example: `
# Build the bke patch pack.
bke build patch -f bke.yaml -t bke.tar.gz
`,
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if patchOption.File == "" {
			return errors.New("The parameter `file` is required. ")
		}
		if !utils.ContainsString([]string{"registry", "oci"}, patchOption.Strategy) {
			patchOption.Strategy = "registry"
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		patchOption.Args = args
		patchOption.Options = options
		patchOption.Patch()
	},
}

var onlineOption build.Options

// onlineCmd represents the build command
var onlineCmd = &cobra.Command{
	Use:   "online-image",
	Short: "Compile an image installed online",
	Long:  `Compile an image installed online`,
	Example: `
# Compile an image installed online
bke build online-image -f bke.yaml -t cr.openfuyao.cn/openfuyao/bke-online-installed:latest

# Compile an multi arch image, the default arch is amd64
# The host should have docker buildx installed an working properly
bke build online-image -f bke.yaml --arch amd64,arm64 -t cr.openfuyao.cn/openfuyao/bke-online-installed:latest
`,
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if onlineOption.File == "" {
			return errors.New("The parameter `file` is required. ")
		}
		if onlineOption.Target == "" {
			return errors.New("The parameter `target` is required. ")
		}
		if !utils.ContainsString([]string{"registry", "docker"}, onlineOption.Strategy) {
			onlineOption.Strategy = "registry"
		}
		if onlineOption.Arch == "" {
			onlineOption.Arch = runtime.GOARCH
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		onlineOption.Args = args
		onlineOption.Options = options
		onlineOption.BuildOnlineImage()
	},
}

var rpmOption build.RpmOptions

// rpmCmd represents the build command
var rpmCmd = &cobra.Command{
	Use:   "rpm",
	Short: "Build an offline rpm package",
	Long:  `Build an offline rpm package`,
	Example: `
# init rpm package
# in this case, rpm is a directory that is empty or has many packages to source
bke build rpm --source rpm

# Add a new rpm file for the already make rpm.tar.gz
bke build rpm --source rpm.tar.gz --add centos/8/amd64 --package docker-ce

# custom image ware house
bke build rpm --source rpm.tar.gz --add centos/8/amd64 --package docker-ce --registry cr.openfuyao.cn/openfuyao

# only build the rpm
bke build rpm --add centos/8/amd64 --package docker-ce
`,
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if (rpmOption.Add == "") != (rpmOption.Package == "") {
			return errors.New("The parameter `add` or `package` is required. ")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		rpmOption.Args = args
		rpmOption.Options = options
		rpmOption.Build()
	},
}

var preCheckOption build.PreCheckOptions

// rpmCmd represents the build command
var preCheckCmd = &cobra.Command{
	Use:   "check",
	Short: "Check the mirror in the configuration file",
	Long:  `Check the mirror in the configuration file`,
	Example: `
# example
bke build check -f bke.yaml
# only check image repo
bke build check -f bke.yaml --only-image
`,
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if preCheckOption.File == "" {
			return errors.New("The parameter `file` is required. ")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		preCheckOption.Args = args
		preCheckOption.Options = options
		preCheckOption.PreCheck()
	},
}

func registerBuildCommand() {
	rootCmd.AddCommand(buildCmd)
	buildCmd.AddCommand(patchCmd)

	//mark buildCmd.AddCommand buildConfigCmd
	//mark buildCmd.AddCommand onlineCmd
	//mark buildCmd.AddCommand rpmCmd
	//mark buildCmd.AddCommand preCheckCmd
	// Here you will define your flags and configuration settings.
	buildCmd.Flags().StringVarP(&buildOption.File, "file", "f", "", "Configuration file path")
	buildCmd.Flags().StringVarP(&buildOption.Target, "target", "t", "", "Packaged BKE files")
	buildCmd.Flags().BoolVar(&buildOption.SkipTLSVerify, "skip-tls-verify", false,
		"Skip TLS verification when syncing registry images if not configured in YAML")

	// build patch
	patchCmd.Flags().StringVarP(&patchOption.File, "file", "f", "", "Configuration file path")
	patchCmd.Flags().StringVarP(&patchOption.Target, "target", "t", "", "Packaged BKE files")
	patchCmd.Flags().StringVarP(&patchOption.Strategy, "strategy", "s", "registry",
		"Mirror sync policy: oci (no Docker) or registry (default)")
	patchCmd.Flags().BoolVar(&patchOption.SkipTLSVerify, "skip-tls-verify", false,
		"Skip TLS verification when syncing registry images if not configured in YAML")

	// build online image
	onlineCmd.Flags().StringVarP(&onlineOption.File, "file", "f", "", "Configuration file path")
	onlineCmd.Flags().StringVarP(&onlineOption.Target, "target", "t", "", "Destination image name")
	onlineCmd.Flags().StringVar(&onlineOption.Arch, "arch", "", "Destination image architecture, such as amd64,arm64")

	// build rpm package
	rpmCmd.Flags().StringVar(&rpmOption.Source, "source", "", "Source rpm file path, example rpm.tar.gz")
	rpmCmd.Flags().StringVar(&rpmOption.Add, "add", "", "Add rpm file path, example centos/8/amd64")
	rpmCmd.Flags().StringVar(&rpmOption.Registry,
		"registry", "registry.cn-hangzhou.aliyuncs.com/bocloud", "Registry address")
	rpmCmd.Flags().StringVar(&rpmOption.Package, "package", "", "Package name")

	// preCheck config file
	preCheckCmd.Flags().StringVarP(&preCheckOption.File, "file", "f", "", "Configuration file path")
	preCheckCmd.Flags().BoolVar(&preCheckOption.OnlyImage,
		"only-image", false, "Only check the image mirror in the configuration file")
}
