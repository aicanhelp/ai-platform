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

package httprepo

import (
	"os"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/exec"
)

const (
	testPackage     = "test-package"
	testOutput      = "test output"
	testErrorOutput = "test error output"
	ubuntuOS        = "ubuntu"
	debianOS        = "debian"
	centosOS        = "centos"
	kylinOS         = "kylin"
	redhatOS        = "redhat"
	fedoraOS        = "fedora"
	openeulerOS     = "openeuler"
	hopeosOS        = "hopeos"
	unknownOS       = "unknownos"
	aptManager      = "apt"
	yumManager      = "yum"
	unknownManager  = "unknown"
)

func TestRepoUpdateAptSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = aptManager
	defer func() {
		packageManager = originalPackageManager
	}()

	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithTimeout",
		func(timeout time.Duration, command string, arg ...string) (string, error) {
			assert.Equal(t, repoUpdateTimeout, timeout)
			return testOutput, nil
		})

	err := RepoUpdate()

	assert.NoError(t, err)
}

func TestRepoUpdateAptErrorBestEffort(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = aptManager
	defer func() {
		packageManager = originalPackageManager
	}()

	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithTimeout",
		func(timeout time.Duration, command string, arg ...string) (string, error) {
			return testErrorOutput, os.ErrPermission
		})

	err := RepoUpdate()

	assert.NoError(t, err)
}

func TestRepoUpdateYumSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = yumManager
	defer func() {
		packageManager = originalPackageManager
	}()

	var capturedCmd string
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithTimeout",
		func(timeout time.Duration, command string, arg ...string) (string, error) {
			assert.Equal(t, repoUpdateTimeout, timeout)
			if len(arg) >= 2 {
				capturedCmd = arg[1]
			}
			return testOutput, nil
		})

	err := RepoUpdate()

	assert.NoError(t, err)
	assert.Contains(t, capturedCmd, "makecache --timer")
	assert.Contains(t, capturedCmd, "skip_if_unavailable=true")
	assert.NotContains(t, capturedCmd, "clean all")
}

func TestRepoUpdateYumErrorBestEffort(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = yumManager
	defer func() {
		packageManager = originalPackageManager
	}()

	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithTimeout",
		func(timeout time.Duration, command string, arg ...string) (string, error) {
			return testErrorOutput, os.ErrPermission
		})

	err := RepoUpdate()

	assert.NoError(t, err)
}

func TestRepoUpdateUnknownPackageManager(t *testing.T) {
	originalPackageManager := packageManager
	packageManager = unknownManager
	defer func() {
		packageManager = originalPackageManager
	}()

	err := RepoUpdate()

	assert.NoError(t, err)
}

func TestRepoSearchSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithOutput", func(command string, arg ...string) (string, error) {
		return testPackage, nil
	})

	err := RepoSearch(testPackage)

	assert.NoError(t, err)
}

func TestRepoSearchExecuteError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithOutput", func(command string, arg ...string) (string, error) {
		return testErrorOutput, execError
	})

	err := RepoSearch(testPackage)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "search package")
}

func TestRepoSearchNotFound(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithOutput", func(command string, arg ...string) (string, error) {
		return "other-package", nil
	})

	err := RepoSearch(testPackage)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestRepoSearchEmptyOutput(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithOutput", func(command string, arg ...string) (string, error) {
		return "", nil
	})

	err := RepoSearch(testPackage)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestRepoInstallSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	packages := []string{testPackage, "package2", "package3"}
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		return testOutput, nil
	})

	err := RepoInstall(packages...)

	assert.NoError(t, err)
}

func TestRepoInstallSinglePackage(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		return testOutput, nil
	})

	err := RepoInstall(testPackage)

	assert.NoError(t, err)
}

func TestRepoInstallError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	execError := os.ErrPermission
	packages := []string{testPackage, "package2"}
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		return testErrorOutput, execError
	})

	err := RepoInstall(packages...)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "install packages")
	assert.Contains(t, err.Error(), testPackage)
}

func TestRepoRemoveAptSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = aptManager
	defer func() {
		packageManager = originalPackageManager
	}()

	packages := []string{testPackage, "package2"}
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		return testOutput, nil
	})

	err := RepoRemove(packages...)

	assert.NoError(t, err)
}

func TestRepoRemoveYumSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = yumManager
	defer func() {
		packageManager = originalPackageManager
	}()

	packages := []string{testPackage, "package2"}
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		return testOutput, nil
	})

	err := RepoRemove(packages...)

	assert.NoError(t, err)
}

func TestRepoRemoveAptRemoveError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = aptManager
	defer func() {
		packageManager = originalPackageManager
	}()

	execError := os.ErrPermission
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		return testErrorOutput, execError
	})

	err := RepoRemove(testPackage)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "remove packages")
}

func TestRepoRemoveAptPurgeError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = aptManager
	defer func() {
		packageManager = originalPackageManager
	}()

	callCount := 0
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		callCount++
		if callCount == 1 {
			return testOutput, nil
		}
		return testErrorOutput, os.ErrPermission
	})

	err := RepoRemove(testPackage)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "purge packages")
}

func TestRepoRemoveYumError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = yumManager
	defer func() {
		packageManager = originalPackageManager
	}()

	execError := os.ErrPermission
	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		return testErrorOutput, execError
	})

	err := RepoRemove(testPackage)

	assert.Error(t, err)
	assert.Contains(t, err.Error(), "remove packages")
}

func TestRepoRemoveSinglePackage(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	originalPackageManager := packageManager
	packageManager = yumManager
	defer func() {
		packageManager = originalPackageManager
	}()

	patches.ApplyMethodFunc(&exec.CommandExecutor{}, "ExecuteCommandWithCombinedOutput", func(command string, arg ...string) (string, error) {
		return testOutput, nil
	})

	err := RepoRemove(testPackage)

	assert.NoError(t, err)
}
