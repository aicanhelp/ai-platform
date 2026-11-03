/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package utils

import (
	"fmt"
	"os/exec"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

// TestExecLocalCmd test ExecLocalCmd
func TestExecLocalCmd(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethod(
		reflect.TypeOf(&exec.Cmd{}), "CombinedOutput", func(_ *exec.Cmd) ([]byte, error) {
			return []byte{1, 2, 3, 4, 5}, nil
		},
	)
	exe, err := ExecLocalCmd("12345")
	assert.Nil(t, err)
	assert.NotNil(t, exe)

	patches.Reset()
	patches.ApplyMethod(
		reflect.TypeOf(&exec.Cmd{}), "CombinedOutput", func(_ *exec.Cmd) ([]byte, error) {
			return nil, fmt.Errorf("break")
		},
	)
	exe, err = ExecLocalCmd("12345")
	assert.Equal(t, "", exe)
	assert.NotNil(t, err)
}

// TestLocalInstallTool test LocalInstallTool
func TestLocalInstallTool(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethod(
		reflect.TypeOf(&exec.Cmd{}), "Run", func(_ *exec.Cmd) error {
			return fmt.Errorf("break")
		},
	)
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		return nil
	})
	patches.ApplyFunc(ExecLocalCmd, func(command string) (string, error) {
		return "", nil
	})

	err := LocalInstallTool("test")
	assert.Nil(t, err)

	patches.Reset()
	patches.ApplyMethod(
		reflect.TypeOf(&exec.Cmd{}), "Run", func(_ *exec.Cmd) error {
			return fmt.Errorf("break")
		},
	)
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		return nil
	})
	patches.ApplyFunc(ExecLocalCmd, func(command string) (string, error) {
		return "", fmt.Errorf("break")
	})
	err = LocalInstallTool("test")
	assert.NotNil(t, err)
}
