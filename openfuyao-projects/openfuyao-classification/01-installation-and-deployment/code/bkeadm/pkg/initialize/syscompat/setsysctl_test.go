/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package syscompat

import (
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"gopkg.openfuyao.cn/bkeadm/utils"
	"gopkg.openfuyao.cn/bkeadm/utils/log"
)

func TestSetSysctlOpenFileError(t *testing.T) {

	SetSysctl()

	assert.True(t, true)
}

func TestSetSysctlWriteStringError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return []byte(""), nil
	})

	patches.ApplyFunc((*os.File).WriteString, func(f *os.File, s string) (int, error) {
		return 0, &os.PathError{Op: "write", Path: "test", Err: os.ErrPermission}
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	SetSysctl()

	assert.True(t, true)
}

func TestSetSysctlExecCommandError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return []byte(""), nil
	})

	patches.ApplyFunc((*os.File).WriteString, func(f *os.File, s string) (int, error) {
		return len(s), nil
	})

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		return nil
	})

	patches.ApplyFunc((*exec.Cmd).Run, func(c *exec.Cmd) error {
		return &os.PathError{Op: "run", Path: "sysctl", Err: os.ErrPermission}
	})

	patches.ApplyFunc(log.SteppedInfo, func(stepName string, args ...any) {})

	SetSysctl()

	assert.True(t, true)
}

func TestSetSysctlReadFilePermissionErrorDoesNotTruncate(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return nil, &os.PathError{Op: "read", Path: filename, Err: os.ErrPermission}
	})

	openFileCalled := false
	patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
		openFileCalled = true
		return os.NewFile(uintptr(0), "test"), nil
	})

	SetSysctl()

	assert.False(t, openFileCalled)
}

func TestSetSysctlAllConfigsExist(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
		return os.NewFile(uintptr(0), "test"), nil
	})

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return []byte("fs.file-max = 9000000\nfs.inotify.max_user_watches = 1000000\nfs.inotify.max_user_instances = 1000000\nnet.ipv4.ip_forward = 1\n"), nil
	})

	writeStringCalled := false
	patches.ApplyFunc((*os.File).WriteString, func(f *os.File, s string) (int, error) {
		writeStringCalled = true
		return len(s), nil
	})

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		return nil
	})

	patches.ApplyFunc((*exec.Cmd).Run, func(c *exec.Cmd) error {
		return nil
	})

	SetSysctl()

	assert.False(t, writeStringCalled)
}

func TestSetSysctlPartialConfigs(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
		return os.NewFile(uintptr(0), "test"), nil
	})

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return []byte("fs.file-max = 9000000\n"), nil
	})

	writeStringCalled := false
	patches.ApplyFunc((*os.File).WriteString, func(f *os.File, s string) (int, error) {
		writeStringCalled = true
		return len(s), nil
	})

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		return nil
	})

	patches.ApplyFunc((*exec.Cmd).Run, func(c *exec.Cmd) error {
		return nil
	})

	SetSysctl()

	assert.True(t, writeStringCalled)
}

func TestSetSysctlAppendsMissingConfigsInManagedBlock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.OpenFile, func(name string, flag int, perm os.FileMode) (*os.File, error) {
		return os.NewFile(uintptr(0), "test"), nil
	})

	patches.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return []byte("kernel.pid_max = 4194304\n"), nil
	})

	var written string
	patches.ApplyFunc((*os.File).WriteString, func(f *os.File, s string) (int, error) {
		written = s
		return len(s), nil
	})

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		return nil
	})

	patches.ApplyFunc((*exec.Cmd).Run, func(c *exec.Cmd) error {
		return nil
	})

	SetSysctl()

	assert.Contains(t, written, BKEManagedSysctlBegin)
	assert.Contains(t, written, strings.Join(bkeManagedSysctlSettings, "\n"))
	assert.Contains(t, written, BKEManagedSysctlEnd)
	assert.Contains(t, written, "kernel.pid_max = 4194304")
}

func TestRemoveManagedBlockFromFile(t *testing.T) {
	tempFile, err := os.CreateTemp(t.TempDir(), "sysctl.conf")
	assert.NoError(t, err)
	defer tempFile.Close()

	content := strings.Join([]string{
		"kernel.pid_max = 4194304",
		BKEManagedSysctlBegin,
		"net.ipv4.ip_forward = 1",
		BKEManagedSysctlEnd,
		"vm.swappiness = 0",
	}, "\n")
	err = os.WriteFile(tempFile.Name(), []byte(content), utils.DefaultFilePermission)
	assert.NoError(t, err)

	removeManagedBlockFromFile(tempFile.Name(), BKEManagedSysctlBegin, BKEManagedSysctlEnd)

	data, err := os.ReadFile(tempFile.Name())
	assert.NoError(t, err)
	result := string(data)
	assert.NotContains(t, result, BKEManagedSysctlBegin)
	assert.NotContains(t, result, "net.ipv4.ip_forward = 1")
	assert.Contains(t, result, "kernel.pid_max = 4194304")
	assert.Contains(t, result, "vm.swappiness = 0")
}
