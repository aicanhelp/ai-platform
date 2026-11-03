/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package preflightlogenv configures process-wide logging before command and
// Provider packages initialize their legacy ologger dependencies.
package preflightlogenv

import (
	"os"
	"strings"
)

func init() {
	if !isPreflightCommand(os.Args) {
		return
	}
	if err := os.Setenv("OLOGGER_CONSOLE", "0"); err != nil {
		return
	}
	if err := os.Setenv("OLOGGER_ENABLE_FILE", "0"); err != nil {
		return
	}
}

func isPreflightCommand(args []string) bool {
	// The root command accepts persistent flags before the subcommand. Skip the
	// known value-taking root flags without treating arbitrary flag values or
	// filenames containing "preflight" as the command name.
	skipValue := false
	for _, arg := range args[1:] {
		if skipValue {
			skipValue = false
			continue
		}
		if strings.HasPrefix(arg, "--kubeconfig=") || strings.HasPrefix(arg, "--log-level=") {
			continue
		}
		switch arg {
		case "--kubeconfig", "--log-level":
			skipValue = true
		case "--doc", "--help", "-h":
			continue
		default:
			return arg == "preflight"
		}
	}
	return false
}
