/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package runtime

import (
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	econd "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/containerd"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/executor/docker"
)

func TestDetectRuntimeFromEnvironmentDocker(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		if key == "RUNTIME" {
			return ContainerRuntimeDocker
		}
		return ""
	})

	result := DetectRuntime()
	assert.Equal(t, ContainerRuntimeDocker, result)
}

func TestDetectRuntimeFromEnvironmentContainerd(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		if key == "RUNTIME" {
			return ContainerRuntimeContainerd
		}
		return ""
	})

	result := DetectRuntime()
	assert.Equal(t, ContainerRuntimeContainerd, result)
}

func TestDetectRuntimeFromEnvironmentInvalid(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		if key == "RUNTIME" {
			return "invalid-runtime"
		}
		return ""
	})

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})

	result := DetectRuntime()
	assert.Equal(t, "", result)
}

func TestDetectRuntimeDockerSockExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return ""
	})

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		if name == dockerSockLinux {
			return nil, nil
		}
		return nil, os.ErrNotExist
	})

	patches.ApplyFunc(docker.NewDockerClient, func() (interface{}, error) {
		return nil, nil
	})

	result := DetectRuntime()
	assert.Equal(t, ContainerRuntimeDocker, result)
}

func TestDetectRuntimeDockerSockNotExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return ""
	})

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})

	result := DetectRuntime()
	assert.Equal(t, "", result)
}

func TestDetectRuntimeContainerdSockExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return ""
	})

	callCount := 0
	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		callCount++
		if name == containerdSockLinux {
			return nil, nil
		}
		if name == dockerSockLinux {
			return nil, os.ErrNotExist
		}
		return nil, os.ErrNotExist
	})

	patches.ApplyFunc(docker.NewDockerClient, func() (interface{}, error) {
		return nil, os.ErrNotExist
	})

	patches.ApplyFunc(econd.NewContainedClient, func() (interface{}, error) {
		return nil, nil
	})

	result := DetectRuntime()
	assert.Equal(t, ContainerRuntimeContainerd, result)
}

func TestDetectRuntimeBothSockNotExist(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return ""
	})

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})

	result := DetectRuntime()
	assert.Equal(t, "", result)
}

func TestDetectRuntimeDockerClientError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return ""
	})

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		if name == dockerSockLinux {
			return nil, nil
		}
		return nil, os.ErrNotExist
	})

	patches.ApplyFunc(docker.NewDockerClient, func() (interface{}, error) {
		return nil, os.ErrPermission
	})

	patches.ApplyFunc(econd.NewContainedClient, func() (interface{}, error) {
		return nil, os.ErrPermission
	})

	result := DetectRuntime()
	assert.Equal(t, "", result)
}

func TestDetectRuntimeContainerdClientError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(os.Getenv, func(key string) string {
		return ""
	})

	patches.ApplyFunc(os.Stat, func(name string) (os.FileInfo, error) {
		if name == dockerSockLinux {
			return nil, os.ErrNotExist
		}
		if name == containerdSockLinux {
			return nil, nil
		}
		return nil, os.ErrNotExist
	})

	patches.ApplyFunc(docker.NewDockerClient, func() (interface{}, error) {
		return nil, os.ErrNotExist
	})

	patches.ApplyFunc(econd.NewContainedClient, func() (interface{}, error) {
		return nil, os.ErrPermission
	})

	result := DetectRuntime()
	assert.Equal(t, "", result)
}

func TestConstants(t *testing.T) {
	assert.Equal(t, "docker", ContainerRuntimeDocker)
	assert.Equal(t, "containerd", ContainerRuntimeContainerd)
	assert.Equal(t, "/var/run/docker.sock", dockerSockLinux)
	assert.Equal(t, "/var/run/containerd/containerd.sock", containerdSockLinux)
}
