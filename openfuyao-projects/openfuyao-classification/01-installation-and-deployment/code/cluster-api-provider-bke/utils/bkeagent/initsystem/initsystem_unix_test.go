//go:build !windows

/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package initsystem

import (
	"os"
	"os/exec"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
)

const (
	testService      = "test-service"
	kubeletService   = "kubelet"
	activeStatus     = "active"
	activatingStatus = "activating"
	inactiveStatus   = "inactive"
	stoppedStatus    = "stopped"
)

func TestOpenRCServiceStartSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "rc-service", name)
		assert.Equal(t, 2, len(arg))
		assert.Equal(t, testService, arg[0])
		assert.Equal(t, "start", arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceStart(testService)

	assert.NoError(t, err)
}

func TestOpenRCServiceStartError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceStart(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestOpenRCServiceStopSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "rc-service", name)
		assert.Equal(t, testService, arg[0])
		assert.Equal(t, "stop", arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceStop(testService)

	assert.NoError(t, err)
}

func TestOpenRCServiceStopError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceStop(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestOpenRCServiceRestartSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "rc-service", name)
		assert.Equal(t, testService, arg[0])
		assert.Equal(t, "restart", arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceRestart(testService)

	assert.NoError(t, err)
}

func TestOpenRCServiceRestartError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceRestart(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestOpenRCServiceExistsTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "rc-service", name)
		assert.Equal(t, testService, arg[0])
		assert.Equal(t, "status", arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte("service is running"), nil
	})

	openrc := &OpenRCInitSystem{}
	exists := openrc.ServiceExists(testService)

	assert.True(t, exists)
}

func TestOpenRCServiceExistsFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte("service does not exist"), nil
	})

	openrc := &OpenRCInitSystem{}
	exists := openrc.ServiceExists(testService)

	assert.False(t, exists)
}

func TestOpenRCServiceIsEnabledTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "rc-update", name)
		assert.Equal(t, "show", arg[0])
		assert.Equal(t, "default", arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte("service1 | default\n" + testService + " | default\nservice2 | default"), nil
	})

	openrc := &OpenRCInitSystem{}
	enabled := openrc.ServiceIsEnabled(testService)

	assert.True(t, enabled)
}

func TestOpenRCServiceIsEnabledFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte("service1 | default\nservice2 | default"), nil
	})

	openrc := &OpenRCInitSystem{}
	enabled := openrc.ServiceIsEnabled(testService)

	assert.False(t, enabled)
}

func TestOpenRCServiceIsActiveTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "rc-service", name)
		assert.Equal(t, testService, arg[0])
		assert.Equal(t, "status", arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte("service is running"), nil
	})

	openrc := &OpenRCInitSystem{}
	active := openrc.ServiceIsActive(testService)

	assert.True(t, active)
}

func TestOpenRCServiceIsActiveStopped(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte("service is " + stoppedStatus), nil
	})

	openrc := &OpenRCInitSystem{}
	active := openrc.ServiceIsActive(testService)

	assert.False(t, active)
}

func TestOpenRCServiceIsActiveDoesNotExist(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "CombinedOutput", func() ([]byte, error) {
		return []byte("service does not exist"), nil
	})

	openrc := &OpenRCInitSystem{}
	active := openrc.ServiceIsActive(testService)

	assert.False(t, active)
}

func TestOpenRCEnableCommand(t *testing.T) {
	openrc := &OpenRCInitSystem{}
	cmd := openrc.EnableCommand(testService)

	assert.Equal(t, "rc-update add "+testService+" default", cmd)
}

func TestOpenRCServiceDisableSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "rc-update", name)
		assert.Equal(t, testService, arg[0])
		assert.Equal(t, "stop", arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceDisable(testService)

	assert.NoError(t, err)
}

func TestOpenRCServiceDisableError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceDisable(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestOpenRCServiceEnableSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "rc-update", name)
		assert.Equal(t, testService, arg[0])
		assert.Equal(t, "start", arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceEnable(testService)

	assert.NoError(t, err)
}

func TestOpenRCServiceEnableError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	openrc := &OpenRCInitSystem{}
	err := openrc.ServiceEnable(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestSystemdEnableCommand(t *testing.T) {
	sysd := &SystemdInitSystem{}
	cmd := sysd.EnableCommand(testService)

	assert.Equal(t, "systemctl enable "+testService+".service", cmd)
}

func TestSystemdReloadSystemdSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "systemctl", name)
		assert.Equal(t, "daemon-reload", arg[0])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	sysd := &SystemdInitSystem{}
	err := sysd.reloadSystemd()

	assert.NoError(t, err)
}

func TestSystemdReloadSystemdError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	sysd := &SystemdInitSystem{}
	err := sysd.reloadSystemd()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to reload systemd")
}

func TestSystemdServiceStartSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	runCount := 0
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		if name == "systemctl" && arg[0] == "daemon-reload" {
			return cmd
		}
		assert.Equal(t, "systemctl", name)
		assert.Equal(t, "start", arg[0])
		assert.Equal(t, testService, arg[1])
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		runCount++
		return nil
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceStart(testService)

	assert.NoError(t, err)
	assert.Equal(t, 2, runCount)
}

func TestSystemdServiceStartReloadError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceStart(testService)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to reload systemd")
}

func TestSystemdServiceStartRunError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	callCount := 0
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		callCount++
		if callCount == 1 {
			return nil
		}
		return execError
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceStart(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestSystemdServiceRestartSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	runCount := 0
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		runCount++
		return nil
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceRestart(testService)

	assert.NoError(t, err)
	assert.Equal(t, 2, runCount)
}

func TestSystemdServiceRestartReloadError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceRestart(testService)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to reload systemd")
}

func TestSystemdServiceRestartRunError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	callCount := 0
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		callCount++
		if callCount == 1 {
			return nil
		}
		return execError
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceRestart(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestSystemdServiceStopSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "systemctl", name)
		assert.Equal(t, "stop", arg[0])
		assert.Equal(t, testService, arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceStop(testService)

	assert.NoError(t, err)
}

func TestSystemdServiceStopError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceStop(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestSystemdServiceExistsTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "systemctl", name)
		assert.Equal(t, "status", arg[0])
		assert.Equal(t, testService, arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte("● " + testService + ".service - loaded"), nil
	})

	sysd := &SystemdInitSystem{}
	exists := sysd.ServiceExists(testService)

	assert.True(t, exists)
}

func TestSystemdServiceExistsNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte("could not be found"), nil
	})

	sysd := &SystemdInitSystem{}
	exists := sysd.ServiceExists(testService)

	assert.False(t, exists)
}

func TestSystemdServiceExistsEmpty(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte(""), nil
	})

	sysd := &SystemdInitSystem{}
	exists := sysd.ServiceExists(testService)

	assert.False(t, exists)
}

func TestSystemdServiceIsEnabledTrue(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "systemctl", name)
		assert.Equal(t, "is-enabled", arg[0])
		assert.Equal(t, testService, arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	sysd := &SystemdInitSystem{}
	enabled := sysd.ServiceIsEnabled(testService)

	assert.True(t, enabled)
}

func TestSystemdServiceIsEnabledFalse(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	sysd := &SystemdInitSystem{}
	enabled := sysd.ServiceIsEnabled(testService)

	assert.False(t, enabled)
}

func TestSystemdServiceIsActiveActive(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "systemctl", name)
		assert.Equal(t, "is-active", arg[0])
		assert.Equal(t, testService, arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte(activeStatus), nil
	})

	sysd := &SystemdInitSystem{}
	active := sysd.ServiceIsActive(testService)

	assert.True(t, active)
}

func TestSystemdServiceIsActiveActivating(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte(activatingStatus), nil
	})

	sysd := &SystemdInitSystem{}
	active := sysd.ServiceIsActive(testService)

	assert.True(t, active)
}

func TestSystemdServiceIsActiveInactive(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte(inactiveStatus), nil
	})

	sysd := &SystemdInitSystem{}
	active := sysd.ServiceIsActive(testService)

	assert.False(t, active)
}

func TestSystemdServiceIsActiveWithWhitespace(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Output", func() ([]byte, error) {
		return []byte("  " + activeStatus + "  \n"), nil
	})

	sysd := &SystemdInitSystem{}
	active := sysd.ServiceIsActive(testService)

	assert.True(t, active)
}

func TestSystemdServiceDisableSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "systemctl", name)
		assert.Equal(t, "disable", arg[0])
		assert.Equal(t, testService, arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceDisable(testService)

	assert.NoError(t, err)
}

func TestSystemdServiceDisableError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceDisable(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestSystemdServiceEnableSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		assert.Equal(t, "systemctl", name)
		assert.Equal(t, "enable", arg[0])
		assert.Equal(t, testService, arg[1])
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return nil
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceEnable(testService)

	assert.NoError(t, err)
}

func TestSystemdServiceEnableError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyFunc(exec.Command, func(name string, arg ...string) *exec.Cmd {
		cmd := &exec.Cmd{}
		return cmd
	})
	patches.ApplyMethodFunc(&exec.Cmd{}, "Run", func() error {
		return execError
	})

	sysd := &SystemdInitSystem{}
	err := sysd.ServiceEnable(testService)

	assert.Error(t, err)
	assert.Equal(t, execError, err)
}

func TestGetInitSystemSystemd(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.LookPath, func(name string) (string, error) {
		if name == "systemctl" {
			return "/usr/bin/systemctl", nil
		}
		return "", os.ErrNotExist
	})

	initSystem, err := GetInitSystem()

	assert.NoError(t, err)
	assert.NotNil(t, initSystem)
	assert.IsType(t, &SystemdInitSystem{}, initSystem)
}

func TestGetInitSystemOpenRC(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.LookPath, func(name string) (string, error) {
		if name == "systemctl" {
			return "", os.ErrNotExist
		}
		if name == "openrc" {
			return "/usr/bin/openrc", nil
		}
		return "", os.ErrNotExist
	})

	initSystem, err := GetInitSystem()

	assert.NoError(t, err)
	assert.NotNil(t, initSystem)
	assert.IsType(t, &OpenRCInitSystem{}, initSystem)
}

func TestGetInitSystemNone(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.LookPath, func(name string) (string, error) {
		return "", os.ErrNotExist
	})

	initSystem, err := GetInitSystem()

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no supported init system detected")
	assert.Nil(t, initSystem)
}

func TestGetInitSystemSystemctlError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(exec.LookPath, func(name string) (string, error) {
		if name == "systemctl" {
			return "", os.ErrPermission
		}
		if name == "openrc" {
			return "/usr/bin/openrc", nil
		}
		return "", os.ErrNotExist
	})

	initSystem, err := GetInitSystem()

	assert.NoError(t, err)
	assert.NotNil(t, initSystem)
	assert.IsType(t, &OpenRCInitSystem{}, initSystem)
}
