/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package infrastructure

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/dockerd"
	cond "gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure/k3s"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

func TestStartLocalKubernetesDockerFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return true })
	patches.ApplyFunc(IsContainerd, func() bool { return false })
	patches.ApplyFunc(k3s.StartK3sWithDocker, func(k3s.Config, string) error {
		return assert.AnError
	})

	err := StartLocalKubernetes(k3s.Config{}, "")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "start k3s with docker")
}

func TestStartLocalKubernetesContainerdFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })
	patches.ApplyFunc(IsContainerd, func() bool { return true })
	patches.ApplyFunc(k3s.EnsureDirExists, func(string) error { return nil })
	patches.ApplyFunc(k3s.StartK3sWithContainerd, func(k3s.Config, string) error {
		return assert.AnError
	})

	err := StartLocalKubernetes(k3s.Config{}, "")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "start k3s with containerd")
}

func TestStartLocalKubernetesContainerdManifestDirFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })
	patches.ApplyFunc(IsContainerd, func() bool { return true })
	patches.ApplyFunc(k3s.EnsureDirExists, func(dir string) error {
		if dir == utils.DefaultExtendManifestsDir {
			return assert.AnError
		}
		return nil
	})

	err := StartLocalKubernetes(k3s.Config{}, "")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "prepare k3s manifests directory")
}

func TestRuntimeInstallDockerInstallFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })
	patches.ApplyFunc(IsContainerd, func() bool { return false })
	patches.ApplyFunc(dockerd.EnsureDockerServer, func(string, string, string, string) error {
		return assert.AnError
	})

	err := RuntimeInstall(RuntimeConfig{Runtime: "docker"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "install docker")
}

func TestRuntimeInstallContainerdFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })
	patches.ApplyFunc(IsContainerd, func() bool { return false })
	patches.ApplyFunc(cond.Install, func(string, string, string, string, string) error {
		return assert.AnError
	})

	err := RuntimeInstall(RuntimeConfig{Runtime: "containerd"})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "install containerd")
}

func TestRuntimeInstallCniPluginFailure(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(IsDocker, func() bool { return false })
	patches.ApplyFunc(IsContainerd, func() bool { return false })
	patches.ApplyFunc(cond.Install, func(string, string, string, string, string) error {
		return nil
	})
	patches.ApplyFunc(cond.CniPluginInstall, func(string) error {
		return assert.AnError
	})

	err := RuntimeInstall(RuntimeConfig{
		Runtime:       "containerd",
		CniPluginFile: "/tmp/cni.tgz",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "install CNI plugin")
}
