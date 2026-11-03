/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2024-11-25
 */

// Package version
package version

import (
	"fmt"
	"os"
	"runtime"
)

var (
	gitVersion string // taged version $(git describe --tags --dirty), use latest by default
	gitCommit  string // sha1 from git, output of $(git rev-parse HEAD)
	buildDate  string // build date in ISO8601 format, output of $(date -u +'%Y-%m-%dT%H:%M:%SZ')
	version    string // version 版本号
	release    string // Release number
)

func init() {
	const maxArgLen = 2
	if len(os.Args) == maxArgLen && os.Args[1] == "-v" {
		fmt.Println("Version:      ", version)
		fmt.Println("Release:      ", release)
		fmt.Println("Go Version:   ", runtime.Version())
		fmt.Println("Git Commit:   ", gitCommit)
		fmt.Println("Built:        ", buildDate)
		fmt.Println("OS/Arch:      ", fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH))
		fmt.Println("Compiler:     ", runtime.Compiler)
	}
}
