// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package dependencies

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestDefineRootCommand 测试根命令
func TestDefineRootCommand(t *testing.T) {
	// 测试根命令的基本属性
	rootCmd := defineRootCommand()

	assert.Equal(t, "dependency", rootCmd.Use)
	assert.Equal(t, "依赖管理工具", rootCmd.Short)
	assert.Contains(t, rootCmd.Long, "依赖管理工具")

	// 测试没有子命令时显示帮助信息
	buf := new(bytes.Buffer)
	rootCmd.SetOut(buf)
	rootCmd.SetErr(buf)
	rootCmd.Run(rootCmd, []string{})

	assert.Contains(t, buf.String(), "Usage:")
}

// TestSetUpCommand 测试命令构建
func TestSetUpCommand(t *testing.T) {
	// 创建命令
	rootCmd := SetUpCommand()

	// 测试根命令属性
	assert.Equal(t, "dependency", rootCmd.Use)
	assert.Equal(t, "依赖管理工具", rootCmd.Short)
	assert.Contains(t, rootCmd.Long, "依赖管理工具")

	// 测试子命令数量
	var subcommandNum = 2
	assert.Len(t, rootCmd.Commands(), subcommandNum, "根命令应包含2个子命令")

	// 测试子命令存在性
	buildCmd := rootCmd.Commands()[0]
	checkCmd := rootCmd.Commands()[1]

	assert.Equal(t, "build", buildCmd.Use)
	assert.Equal(t, "构建组件被依赖情况", buildCmd.Short)
	assert.Equal(t, "check", checkCmd.Use)
	assert.Equal(t, "检查组件被依赖情况", checkCmd.Short)

	// 测试子命令的标志
	buildURLFlag := buildCmd.Flag("url")
	assert.NotNil(t, buildURLFlag, "build 命令应有 --url 标志")
	assert.Equal(t, "u", buildURLFlag.Shorthand)

	buildPathFlag := buildCmd.Flag("path")
	assert.NotNil(t, buildPathFlag, "build 命令应有 --path 标志")
	assert.Equal(t, "p", buildPathFlag.Shorthand)

	checkNameFlag := checkCmd.Flag("name")
	assert.NotNil(t, checkNameFlag, "check 命令应有 --name 标志")
	assert.Equal(t, "n", checkNameFlag.Shorthand)

	checkVersionFlag := checkCmd.Flag("version")
	assert.NotNil(t, checkVersionFlag, "check 命令应有 --version 标志")
	assert.Equal(t, "v", checkVersionFlag.Shorthand)

	checkDevFlag := checkCmd.Flag("dev")
	assert.NotNil(t, checkDevFlag, "check 命令应有 --dev 标志")
	assert.Equal(t, "d", checkDevFlag.Shorthand)

	checkPathFlag := checkCmd.Flag("path")
	assert.NotNil(t, checkPathFlag, "check 命令应有 --path 标志")
	assert.Equal(t, "p", checkPathFlag.Shorthand)
}
