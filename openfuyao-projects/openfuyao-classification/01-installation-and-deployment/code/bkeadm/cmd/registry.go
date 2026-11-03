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
	"runtime"

	"github.com/spf13/cobra"

	"gopkg.openfuyao.cn/bkeadm/pkg/build"
	"gopkg.openfuyao.cn/bkeadm/pkg/registry"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

// migrateCmd represents the sync command
var registryCmd = &cobra.Command{
	Use:   "registry",
	Short: "Synchronize images between two mirror repositories",
	Long:  "Synchronize images between two mirror repositories，by way of block transfer",
	Example: `
# Get help
bke registry -h
`,
}

var syncOption = registry.Options{}

// syncDep synchronous mirror
var syncDep = &cobra.Command{
	Use:   "sync",
	Short: "In the two mirror repositories, mirrors are synchronized by copying data blocks from one repository to another.",
	Long:  "In the two mirror repositories, mirrors are synchronized by copying data blocks from one repository to another.",
	Example: `
# Migration multi-architecture images.
bke registry sync --source docker.io/library/busybox:1.35 --target 127.0.0.1:40443/library/busybox:1.35 --multi-arch
bke registry sync --source docker.io/library/busybox:1.35 --target 127.0.0.1:40443/library/busybox:1.35 --arch arm64
# Migration multi-architecture images in batches.
bke registry sync --source docker.io/library -f image-list.txt --target registry.cloud.com/k8s
$ cat image-list.txt
busybox:1.28
alpine:3.14
. . .
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(syncOption.Source) == 0 {
			return errors.New("The `source` parameter is required. ")
		}
		if len(syncOption.Target) == 0 {
			return errors.New("The `target` parameter is required. ")
		}
		if syncOption.MultiArch && len(syncOption.Arch) > 0 {
			return errors.New("The `arch` parameter is not allowed when `multi-arch` is true. ")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		syncOption.Args = args
		syncOption.Options = options
		syncOption.Sync()
	},
}

var transferOption = registry.Options{}

// migrateDep represents the image command
var transferDep = &cobra.Command{
	Use:   "transfer",
	Short: "Transfer images in docker pull / docker push mode",
	Long:  "Transfer images in docker pull / docker push mode",
	Example: `
# Transfer multi-architecture images.
bke registry transfer --source docker.io/library/ --image busybox:1.28 --target registry.cloud.com/k8s --arch amd64,arm64
# Transfer multi-architecture images in batches.
bke registry transfer --source docker.io/library -f image-list.txt --target registry.cloud.com/k8s
$ cat image-list.txt
busybox:1.28
alpine:3.14
. . .
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(transferOption.Source) == 0 {
			return errors.New("The `source` parameter is required. ")
		}
		if len(transferOption.Target) == 0 {
			return errors.New("The `target` parameter is required. ")
		}
		if len(transferOption.Image) == 0 && len(transferOption.File) == 0 {
			return errors.New("There must be one of the parameters `image` and `file`. ")
		}
		if len(transferOption.Arch) == 0 {
			transferOption.Arch = runtime.GOARCH
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		transferOption.Args = args
		transferOption.Options = options
		transferOption.MigrateImage()
	},
}

var listTagsOption = registry.Options{}

// migrateDep represents the image command
var listTagsDep = &cobra.Command{
	Use:   "list-tags",
	Short: "Lists all tags the mirror repository",
	Long:  "Lists all tags the mirror repository",
	Example: `
# Lists all tags the mirror repository
bke registry list-tags registry.cn-hangzhou.aliyuncs.com/bocloud/pause
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return errors.New("The `image` is required. ")
		}
		listTagsOption.Image = args[0]
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		listTagsOption.Args = args
		listTagsOption.Options = options
	},
}

var inspectOption = registry.Options{}

// inspectDep represents the image command
var inspectDep = &cobra.Command{
	Use:   "inspect",
	Short: "List information about images in the mirror repository",
	Long:  "List information about images in the mirror repository",
	Example: `
# inspect the image
bke registry inspect registry.bocloud.com/kubernetes/pause:3.8
bke registry inspect --dest-tls-verify registry.bocloud.com/kubernetes/pause:3.8 
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return errors.New("The `image` is required. ")
		}
		inspectOption.Image = args[0]
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		inspectOption.Args = args
		inspectOption.Options = options
		inspectOption.Inspect()
	},
}

var manifestsOption = registry.Options{}

// manifestsDep represents the image command
var manifestsDep = &cobra.Command{
	Use:   "manifests",
	Short: "Make a multi-architecture wake image",
	Long:  "Make a multi-architecture wake image",
	Example: `
# manifests the image
bke registry manifests --image=127.0.0.1:40443/library/busybox:1.35 127.0.0.1:40443/library/busybox:1.35-amd64 127.0.0.1:40443/library/busybox:1.35-arm64
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if manifestsOption.Image == "" {
			return errors.New("The `image` is required. ")
		}
		if len(args) < utils.MinManifestsImageArgs {
			return errors.New("There are at least two schema images. ")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		manifestsOption.Args = args
		manifestsOption.Options = options
		manifestsOption.Manifests()
	},
}

var deleteOption = registry.Options{}

// deleteDep represents the image command
var deleteDep = &cobra.Command{
	Use:   "delete",
	Short: "Delete a specified mirror",
	Long:  "Delete a specified mirror",
	Example: `
# delete the image
bke registry delete 192.168.2.111:40443/library/busybox:1.35
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) < utils.MinManifestsImageArgs {
			return errors.New("There are at least two schema images. ")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		deleteOption.Args = args
		deleteOption.Options = options
		deleteOption.Delete()
	},
}

var viewOption = registry.Options{}

// viewDep represents the image command
var viewDep = &cobra.Command{
	Use:   "view",
	Short: "View warehouse view",
	Long:  "View information such as the image tag of the warehouse",
	Example: `
# view warehouse
bke registry view 192.168.2.111:40443
# default https
bke registry view https://192.168.2.111:40443
# prefix 
bke registry view http://192.168.2.120:40443 --prefix kubernetes/kube
bke registry view http://192.168.2.120:40443 --prefix kubernetes --tags 3
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 {
			return errors.New("The `registry address` is required. ")
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		viewOption.Args = args
		viewOption.Options = options
		viewOption.View()
	},
}

// patchDep represents the image command
var patchDep = &cobra.Command{
	Use:   "patch",
	Short: "Specially customized incremental packet mirror synchronization",
	Long:  "Specially customized incremental packet mirror synchronization",
	Example: `
# synchronous incremental image
bke registry patch --source /bke-patch --target 127.0.0.1:40443
`,
	Run: func(cmd *cobra.Command, args []string) {
		source := cmd.Flag("source").Value.String()
		target := cmd.Flag("target").Value.String()
		if len(source) == 0 || len(target) == 0 {
			fmt.Println("The `source` and `target` parameters are required. ")
			return
		}
		build.SpecificSync(source, target)
	},
}

var downloadOption = registry.OptionsDownload{}

// downloadDep represents the image command
var downloadDep = &cobra.Command{
	Use:   "download",
	Short: "Download the specified file in the image",
	Long:  "Complete the file download in memory without pulling the image",
	Example: `
# Download the file through the absolute path
bke registry download --image repository/kubectl:v1.23.17 -f /opt/bocloud/kubectl
# Fuzzy download file
bke registry download --image repository/kubectl:v1.23.17 -f kubectl -d /opt
# Download multiple files
bke registry download --image repository/kubectl:v1.23.17 -f kubectl,kubectl-ns -d /opt
`,
	Args: func(cmd *cobra.Command, args []string) error {
		if len(downloadOption.Image) == 0 {
			return errors.New("The `image` parameter is required. ")
		}
		if len(downloadOption.DownloadInImageFile) == 0 {
			return errors.New("The `file` parameter is required. ")
		}
		if len(downloadOption.DownloadToDir) == 0 {
			var err error
			downloadOption.DownloadToDir, err = os.Getwd()
			if err != nil {
				fmt.Printf("Warning: failed to get current working directory: %v\n", err)
			}
		}
		return nil
	},
	Run: func(cmd *cobra.Command, args []string) {
		downloadOption.Args = args
		downloadOption.Options = options
		err := downloadOption.Download()
		if err != nil {
			fmt.Println("download error: " + err.Error())
		}
	},
}

func registerRegistryCommand() {
	rootCmd.AddCommand(registryCmd)
	registryCmd.AddCommand(syncDep)
	registryCmd.AddCommand(viewDep)
	registryCmd.AddCommand(patchDep)

	//mark registryCmd.AddCommand transferDep)
	//mark registryCmd.AddCommand listTagsDep)
	//mark registryCmd.AddCommand inspectDep)
	//mark registryCmd.AddCommand manifestsDep)
	//mark registryCmd.AddCommand deleteDep)
	//mark registryCmd.AddCommand downloadDep)

	// Here you will define your flags and configuration settings.
	// sync
	syncDep.Flags().StringVarP(&syncOption.File, "file", "f", "", "Image names are arranged in rows, example name:tag")
	syncDep.Flags().BoolVar(&syncOption.MultiArch, "multi-arch", false, "Synchronize multi-schema images")
	syncDep.Flags().StringVar(&syncOption.Source, "source", "", "Mirroring source address, example docker.io/library/")
	syncDep.Flags().StringVar(&syncOption.Target, "target", "", "Target warehouse address, example registry.cloud.com/k8s")
	syncDep.Flags().BoolVar(&syncOption.SrcTLSVerify, "src-tls-verify", false, "Verify the source TLS certificate")
	syncDep.Flags().BoolVar(&syncOption.DestTLSVerify, "dest-tls-verify", false, "Verify the destination TLS certificate")
	syncDep.Flags().StringVar(&syncOption.Arch, "arch", "", "Specifies the synchronous mirror schema amd64/arm64")
	syncDep.Flags().BoolVar(&syncOption.SyncRepo, "sync-repo", false, "Synchronous repository")

	// transferDep
	transferDep.Flags().StringVarP(&transferOption.File,
		"file", "f", "", "Image names are arranged in rows, example name:tag")
	transferDep.Flags().StringVar(&transferOption.Arch,
		"arch", "", "Example amd64,arm64 , when not specified as the current system architecture")
	transferDep.Flags().StringVar(&transferOption.Source,
		"source", "", "Mirroring source address, example docker.io/library/")
	transferDep.Flags().StringVar(&transferOption.Image,
		"image", "", "Name of the mirror, example busybox:1.28")
	transferDep.Flags().StringVar(&transferOption.Target,
		"target", "", "Target warehouse address, example registry.cloud.com/k8s")

	// listTag
	listTagsDep.Flags().BoolVar(&listTagsOption.DestTLSVerify,
		"dest-tls-verify", false, "Verify the destination TLS certificate")
	// inspect
	inspectDep.Flags().BoolVar(&inspectOption.DestTLSVerify,
		"dest-tls-verify", false, "Verify the destination TLS certificate")
	// manifest
	manifestsDep.Flags().StringVar(&manifestsOption.Image, "image", "", "multi-architecture image")
	// delete
	deleteDep.Flags().BoolVar(&deleteOption.DestTLSVerify,
		"dest-tls-verify", false, "Verify the destination TLS certificate")
	// view
	viewDep.Flags().StringVar(&viewOption.Prefix, "prefix", "", "Prefix of the image path")
	viewDep.Flags().IntVar(&viewOption.Tags, "tags", utils.DefaultImageTags, "Tags of the image")
	viewDep.Flags().BoolVar(&viewOption.Export, "export", false, "Export the image list to the file")

	// patchDep
	patchDep.Flags().String("source", "", "bke incremental package directory")
	patchDep.Flags().String("target", "127.0.0.1:40443", "Address of the target mirror warehouse")

	// downloadDep
	downloadDep.Flags().StringVar(&downloadOption.Image, "image", "", "Download files from the image")
	downloadDep.Flags().StringVarP(&downloadOption.Username, "username", "u", "", "User name of the mirror warehouse")
	downloadDep.Flags().StringVarP(&downloadOption.Password, "password", "p", "", "Mirror warehouse password")
	downloadDep.Flags().StringVar(&downloadOption.CertDir, "certDir", "", "Mirror warehouse certificate")
	downloadDep.Flags().BoolVar(&downloadOption.SrcTLSVerify, "src-tls-verify", false, "Verify the source TLS certificate")
	downloadDep.Flags().StringVarP(&downloadOption.DownloadInImageFile,
		"downloadInImageFile", "f", "", "Download the image file")
	downloadDep.Flags().StringVarP(&downloadOption.DownloadToDir,
		"downloadToDir", "d", "", "Download the file to the specified directory")
}
