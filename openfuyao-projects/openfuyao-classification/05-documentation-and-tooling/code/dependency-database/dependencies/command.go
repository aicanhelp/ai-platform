// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

// Package dependencies 是依赖数据库所有工具的实现
package dependencies

import (
	"fmt"

	"github.com/spf13/cobra"
)

const initialHelmURL = "https://helm.openfuyao.cn/_dev/index.yaml" // 拉取chart包的默认url
const initialTargetStoragePath = "../json"                         // 最终识别结果存储目录
const initialLoadFilePath = "../json/componentDependencyFile.json" // 读取json文件的默认路径（相对main.go）

// 定义命令行需要的参数
var (
	helmURL                string
	componentName          string
	componentVersion       string
	needDevelopmentVersion bool
	targetStoragePath      string
	loadFilePath           string
)

// defineRootCommand 定义根命令
func defineRootCommand() *cobra.Command {
	var rootCmd = &cobra.Command{
		Use:   "dependency", // 命令名
		Short: "依赖管理工具",
		Long:  `一个依赖管理工具，用于查询组件被依赖情况`,
		// Run 是命令执行的核心逻辑
		Run: func(cmd *cobra.Command, args []string) {
			// 显示帮助信息（当没有子命令时）
			cmd.Help()
		},
	}
	return rootCmd
}

// defineBuildCommand 定义构建命令
func defineBuildCommand() *cobra.Command {
	// buildCmd 定义 build 命令
	var buildCmd = &cobra.Command{
		Use:   "build",
		Short: "构建组件被依赖情况",
		Long:  `构建所有组件的被依赖情况，并将结果输出至指定路径下的json文件中`,
		Run: func(cmd *cobra.Command, args []string) {
			if helmURL == "" {
				helmURL = initialHelmURL
				fmt.Printf("默认index.yaml地址：%s\n", helmURL)
			}

			if targetStoragePath == "" {
				targetStoragePath = initialTargetStoragePath
				fmt.Printf("默认最终识别结果存储的目录：%s\n", targetStoragePath)
			}

			// 调用依赖分析函数
			dependencyComponentFilePath, componentDependencyFilePath, err :=
				IdentifyDependencies(helmURL, targetStoragePath)
			if err != nil {
				fmt.Printf("错误: %v\n", err)
				return
			}

			fmt.Printf("依赖JSON文件存储在：%s\n", dependencyComponentFilePath)
			fmt.Printf("被依赖JSON文件存储在：%s\n", componentDependencyFilePath)
		},
	}

	// 添加标志
	buildCmd.Flags().StringVarP(&helmURL, "url", "u", "",
		"index.yaml所在的网址（不填写默认为 https://helm.openfuyao.cn/_dev/index.yaml ）")
	buildCmd.Flags().StringVarP(&targetStoragePath, "path", "p", "",
		"最终识别结果存储的目录（不填写默认为 ../json ）")

	return buildCmd
}

// defineCheckCommand 定义查询命令
func defineCheckCommand() *cobra.Command {
	var checkCmd = &cobra.Command{
		Use:   "check",
		Short: "检查组件被依赖情况",
		Long:  `查询指定组件的被依赖情况`,
		Run: func(cmd *cobra.Command, args []string) {
			if componentName == "" {
				fmt.Println("错误: 组件名称不能为空")
				cmd.Help()
				return
			}
			if loadFilePath == "" {
				loadFilePath = initialLoadFilePath
				fmt.Printf("默认读取json文件地址：%s\n", loadFilePath)
			}

			results, err :=
				FindComponentDependency(componentName, componentVersion, needDevelopmentVersion, loadFilePath)
			if err != nil {
				fmt.Printf("错误: %v\n", err)
				return
			}
			if componentVersion == "" {
				componentVersion = "all-versions"
			}
			fmt.Printf("组件 %s (版本: %s，是否需要开发者版本：%t) 的被依赖情况:\n",
				componentName, componentVersion, needDevelopmentVersion)
			if len(results) > 0 {
				for _, r := range results {
					fmt.Println(r)
				}
			} else {
				fmt.Println("无任何组件依赖该组件")
			}
		},
	}

	// 添加标志
	checkCmd.Flags().StringVarP(&componentName, "name", "n", "",
		"组件名称（必填）")
	checkCmd.Flags().StringVarP(&componentVersion, "version", "v", "",
		"组件版本（填写为严格查询，不填写为宽松查询）")
	checkCmd.Flags().BoolVarP(&needDevelopmentVersion, "dev", "d", false,
		"是否包含开发版本（使用 -d=false 或 -d=true，默认 false）")
	checkCmd.Flags().StringVarP(&loadFilePath, "path", "p", "",
		"读取json文件的路径（不填写默认为 ../json/componentDependencyFile.json ）")

	return checkCmd
}

// SetUpCommand 完成命令构建
func SetUpCommand() *cobra.Command {
	var rootCmd = defineRootCommand()

	var buildCmd = defineBuildCommand()
	rootCmd.AddCommand(buildCmd)

	var checkCmd = defineCheckCommand()
	rootCmd.AddCommand(checkCmd)

	return rootCmd
}
