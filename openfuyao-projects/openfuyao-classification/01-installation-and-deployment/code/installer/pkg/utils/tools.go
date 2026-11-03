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

package utils

import (
	"fmt"
	"os/exec"

	"installer/pkg/constants"
	"installer/pkg/zlog"
)

// ExecLocalSudoCmd is exec local sudo command
func ExecLocalSudoCmd(command string) (string, error) {
	cmd := fmt.Sprintf("sudo -E /bin/bash <<EOF\n%s\nEOF", command)
	return ExecLocalCmd(cmd)
}

// ExecLocalCmd is exec local command
func ExecLocalCmd(command string) (string, error) {
	zlog.Infof("exec command: %s", command)
	output, err := ExecLocalCmdWithoutLog(command)
	if err != nil {
		zlog.Errorf("error is %s", err)
	}
	zlog.Infof("output is %s", output)
	return output, err
}

// ExecLocalCmdWithoutLog exec local cmd without log
func ExecLocalCmdWithoutLog(command string) (string, error) {
	cmd := exec.Command("bash", "-c", command)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(output), err
}

// LocalInstallTool is local install tool
func LocalInstallTool(name string) error {
	cmd := exec.Command("which", name)
	err := cmd.Run()
	if err != nil {
		baseCmd := constants.AptGet
		_, err := ExecLocalCmd(fmt.Sprintf("command -v %s", baseCmd))
		if err != nil {
			baseCmd = constants.Yum
			zlog.Infof("exec command 'command -v %s', error: %v", baseCmd, err)
		}

		_, err = ExecLocalCmd(fmt.Sprintf("%s install %s -y", baseCmd, name))
		if err != nil {
			return fmt.Errorf("failed to install %s: %v", name, err)
		}
	}
	return nil
}

// DestroyPassword is destroy password
func DestroyPassword(bt []byte) {
	for i := range bt {
		bt[i] = 0
	}
}
