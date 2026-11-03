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
	"fmt"
	"os"

	"github.com/spf13/cobra"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	configinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"

	"gopkg.openfuyao.cn/bkeadm/pkg/config"
	"gopkg.openfuyao.cn/bkeadm/pkg/securityinit"
)

var configOption config.Options

// configCmd represents the config command
var configCmd = &cobra.Command{
	Use:   "config",
	Short: "Generate bke configuration.",
	Long:  `Generate bke configuration.`,
	Example: `
# Generate a cluster configuration file
bke config
`,
	PreRunE: func(cmd *cobra.Command, args []string) error {
		if configOption.Directory == "" {
			var err error
			configOption.Directory, err = os.Getwd()
			if err != nil {
				fmt.Printf("Warning: failed to get current working directory: %v\n", err)
			}
		}
		if configOption.Product == "" {
			configOption.Product = "fuyao-portal"
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		configOption.Args = args
		configOption.Options = options
		configOption.Config(map[string]string{}, v1beta1.Repo{
			Domain: configinit.DefaultImageRepo,
			Ip:     "0.0.0.0",
			Port:   configinit.DefaultImageRepoPort,
			Prefix: configinit.ImageRegistryKubernetes,
		}, v1beta1.Repo{
			Domain: configinit.DefaultYumRepo,
			Ip:     "0.0.0.0",
			Port:   configinit.DefaultYumRepoPort,
		}, v1beta1.Repo{
			Domain: configinit.DefaultChartRepo,
			Ip:     "0.0.0.0",
			Port:   configinit.DefaultOnlineChartRepoPort,
			Prefix: configinit.DefaultChartRepoPrefix,
		}, "")
	},
}

var encryptOption config.Options

// encryptCmd represents the encryptCmd command
var encryptCmd = &cobra.Command{
	Use:   "encrypt",
	Short: "Encryption configuration file.",
	Long: `Encryption configuration file. The management cluster key is loaded
from the cluster-system/bke-password-encryption-key Secret; a kubeconfig
(--kubeconfig or ~/.kube/config) with access to the management cluster
is required.`,
	Example: `
# Encryption configuration file.
bke config encrypt -f bkecluster.yaml
# Encryption a single string 
bke config encrypt test
`,
	Args:    requireFileOrArg(&encryptOption),
	PreRunE: loadManagementClusterKey(),
	Run: func(cmd *cobra.Command, args []string) {
		encryptOption.Args = args
		encryptOption.Options = options
		if len(args) > 0 {
			err := encryptOption.EncryptString()
			if err != nil {
				fmt.Println(err.Error())
			}
		}
		if len(encryptOption.File) > 0 {
			err := encryptOption.EncryptFile()
			if err != nil {
				fmt.Println(err.Error())
			}
		}
	},
}

var decryptOption config.Options

// decryptCmd represents the decryptCmd command
var decryptCmd = &cobra.Command{
	Use:   "decrypt",
	Short: "Decrypting configuration files.",
	Long: `Decrypting configuration files. The management cluster key is loaded
from the cluster-system/bke-password-encryption-key Secret; a kubeconfig
(--kubeconfig or ~/.kube/config) with access to the management cluster
is required.`,
	Example: `
# Decrypting configuration files
bke config decrypt -f bkecluster.yaml
# Decrypting a single string
bke config decrypt xxxxx
`,
	Args:    requireFileOrArg(&decryptOption),
	PreRunE: loadManagementClusterKey(),
	Run: func(cmd *cobra.Command, args []string) {
		decryptOption.Args = args
		decryptOption.Options = options
		if len(args) > 0 {
			err := decryptOption.DecryptString()
			if err != nil {
				fmt.Println(err.Error())
			}
		}
		if len(decryptOption.File) > 0 {
			err := decryptOption.DecryptFile()
			if err != nil {
				fmt.Println(err.Error())
			}
		}
	},
}

// requireFileOrArg returns an Args validator for the encrypt/decrypt commands:
// either a positional argument or the --file flag must be provided.
func requireFileOrArg(opt *config.Options) func(cmd *cobra.Command, args []string) error {
	return func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 && len(opt.File) == 0 {
			return errors.New("The `file` parameter is required. ")
		}
		return nil
	}
}

// loadManagementClusterKey returns a PreRunE for the encrypt/decrypt commands:
// it initializes the K8s client (--kubeconfig, default ~/.kube/config) and
// loads the password encryption key from the management cluster Secret, so
// local encrypt/decrypt uses the same key as the cluster.
func loadManagementClusterKey() func(cmd *cobra.Command, args []string) error {
	return func(cmd *cobra.Command, args []string) error {
		if err := options.ClusterPre(); err != nil {
			return err
		}
		return securityinit.Init()
	}
}

func registerConfigCommand() {
	rootCmd.AddCommand(configCmd)
	configCmd.AddCommand(encryptCmd)
	configCmd.AddCommand(decryptCmd)

	// Here you will define your flags and configuration settings.
	configCmd.Flags().StringVarP(&configOption.Directory, "directory", "d", "", "Configuration file Directory")
	configCmd.Flags().StringVarP(&configOption.Product, "product", "p", "fuyao-portal",
		"enum: fuyao-portal/fuyao-business/fuyao-allinone")

	encryptCmd.Flags().StringVarP(&encryptOption.File, "file", "f", "", "bkecluster configuration file")
	decryptCmd.Flags().StringVarP(&decryptOption.File, "file", "f", "", "bkecluster configuration file")
}
