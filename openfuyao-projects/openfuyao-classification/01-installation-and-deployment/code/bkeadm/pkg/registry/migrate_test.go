/*
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          <http://license.coscl.org.cn/MulanPSL2>
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package registry

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/executor/exec"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

const (
	testOneValue = 1
)

func setupMigrateDockerMocks(t *testing.T, patches *gomonkey.Patches) {
	t.Helper()
	mockDocker := &docker.Client{}
	global.Docker = mockDocker
	global.Command = &exec.CommandExecutor{}

	patches.ApplyFunc((*docker.Client).Pull, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
		return nil
	})
	patches.ApplyFunc((*docker.Client).Tag, func(_ *docker.Client, _, _ string) error {
		return nil
	})
	patches.ApplyFunc((*docker.Client).Remove, func(_ *docker.Client, _ docker.ImageRef) error {
		return nil
	})
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
		func(_ *exec.CommandExecutor, _ string, _ ...string) (string, error) {
			return "ok", nil
		})
}

func TestSyncImageSingleArch(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	setupMigrateDockerMocks(t, patches)
	needRemoveImage = []string{}

	err := syncImage("source.example.com/image:v1.0", "target.example.com/image:v1.0", []string{"amd64"})
	assert.NoError(t, err)
	assert.Len(t, needRemoveImage, 2)
}

func TestSyncImageMultiArch(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	setupMigrateDockerMocks(t, patches)
	needRemoveImage = []string{}

	patches.ApplyFunc(processArchImage, func(source, target, arch string) (string, error) {
		return target + "-" + arch, nil
	})
	patches.ApplyFunc(global.ExecuteManifestCommands, func(string, string, []string) error {
		return nil
	})

	err := syncImage("source.example.com/image:v1.0", "target.example.com/image:v1.0", []string{"amd64", "arm64"})
	assert.NoError(t, err)
}

func TestSyncSingleArchImagePullFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Docker = &docker.Client{}
	patches.ApplyFunc((*docker.Client).Pull, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
		return errors.New("pull failed")
	})

	err := syncSingleArchImage("src/img", "tgt/img", "amd64")
	assert.Error(t, err)
}

func TestSyncSingleArchImageTagFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Docker = &docker.Client{}
	patches.ApplyFunc((*docker.Client).Pull, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
		return nil
	})
	patches.ApplyFunc((*docker.Client).Tag, func(_ *docker.Client, _, _ string) error {
		return errors.New("tag failed")
	})

	err := syncSingleArchImage("src/img", "tgt/img", "amd64")
	assert.Error(t, err)
}

func TestPullImageWithRetryFallback(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Docker = &docker.Client{}

	call := 0
	patches.ApplyFunc((*docker.Client).Pull, func(_ *docker.Client, ref docker.ImageRef, _ utils.RetryOptions) error {
		call++
		if call == 1 {
			return errors.New("not found")
		}
		assert.Contains(t, ref.Image, "-amd64")
		return nil
	})

	addr, err := pullImageWithRetry("src/img", "amd64")
	assert.NoError(t, err)
	assert.Equal(t, "src/img-amd64", addr)
}

func TestPullImageWithRetryBothFail(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Docker = &docker.Client{}
	patches.ApplyFunc((*docker.Client).Pull, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
		return errors.New("pull failed")
	})

	_, err := pullImageWithRetry("src/img", "amd64")
	assert.Error(t, err)
}

func TestPushImageSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Command = &exec.CommandExecutor{}
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
		func(_ *exec.CommandExecutor, command string, _ ...string) (string, error) {
			assert.Contains(t, command, "/bin/sh")
			return "pushed", nil
		})

	assert.NoError(t, pushImage("tgt/img:tag"))
}

func TestPushImageFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Command = &exec.CommandExecutor{}
	patches.ApplyFunc((*exec.CommandExecutor).ExecuteCommandWithOutput,
		func(_ *exec.CommandExecutor, _ string, _ ...string) (string, error) {
			return "", errors.New("push failed")
		})

	assert.Error(t, pushImage("tgt/img:tag"))
}

func TestProcessArchImageSuccess(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	setupMigrateDockerMocks(t, patches)
	needRemoveImage = []string{}

	targetArch, err := processArchImage("src/img", "tgt/img", "amd64")
	assert.NoError(t, err)
	assert.Equal(t, "tgt/img-amd64", targetArch)
	assert.Len(t, needRemoveImage, 1)
}

func TestProcessArchImageRemoveFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Docker = &docker.Client{}
	patches.ApplyFunc((*docker.Client).Pull, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
		return nil
	})
	patches.ApplyFunc((*docker.Client).Tag, func(_ *docker.Client, _, _ string) error {
		return nil
	})
	patches.ApplyFunc((*docker.Client).Remove, func(_ *docker.Client, _ docker.ImageRef) error {
		return errors.New("remove failed")
	})

	_, err := processArchImage("src/img", "tgt/img", "amd64")
	assert.Error(t, err)
}

func TestSyncMultiArchImageProcessArchFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(processArchImage, func(string, string, string) (string, error) {
		return "", errors.New("arch failed")
	})

	err := syncMultiArchImage("src/img", "tgt/img", []string{"amd64"})
	assert.Error(t, err)
}

func TestMigrateImageFromFile(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	patches.ApplyFunc(syncImage, func(string, string, []string) error { return nil })
	patches.ApplyFunc(cleanBuildImage, func() {})

	tmpDir := t.TempDir()
	listFile := filepath.Join(tmpDir, "images.txt")
	require.NoError(t, os.WriteFile(listFile, []byte("app:v1\n\napp:v2\n"), 0o644))

	op := &Options{
		Source: "src.registry.com",
		Target: "tgt.registry.com",
		File:   listFile,
	}
	op.MigrateImage()
}

func TestMigrateImageSingleImage(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })

	var synced int
	patches.ApplyFunc(syncImage, func(source, target string, _ []string) error {
		synced++
		assert.Contains(t, source, "src.registry.com/app:v1")
		assert.Contains(t, target, "tgt.registry.com/app:v1")
		return nil
	})
	patches.ApplyFunc(cleanBuildImage, func() {})

	op := &Options{
		Source: "src.registry.com",
		Target: "tgt.registry.com",
		Image:  "app:v1",
		Arch:   "amd64",
	}
	op.MigrateImage()
	assert.Equal(t, 1, synced)
}

func TestMigrateImageSyncFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })
	patches.ApplyFunc(syncImage, func(string, string, []string) error {
		return errors.New("sync failed")
	})

	op := &Options{
		Source: "src.registry.com",
		Target: "tgt.registry.com",
		Image:  "app:v1",
	}
	op.MigrateImage()
}

func TestMigrateImageOpenFileFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return true })

	op := &Options{
		Source: "src.registry.com",
		Target: "tgt.registry.com",
		File:   "/nonexistent/images.txt",
	}
	op.MigrateImage()
}

func TestSyncSingleArchImagePushFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Docker = &docker.Client{}
	patches.ApplyFunc((*docker.Client).Pull, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
		return nil
	})
	patches.ApplyFunc((*docker.Client).Tag, func(_ *docker.Client, _, _ string) error {
		return nil
	})
	patches.ApplyFunc(pushImage, func(string) error {
		return errors.New("push failed")
	})

	err := syncSingleArchImage("src/img", "tgt/img", "amd64")
	assert.Error(t, err)
}

func TestProcessArchImagePushFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	global.Docker = &docker.Client{}
	patches.ApplyFunc((*docker.Client).Pull, func(_ *docker.Client, _ docker.ImageRef, _ utils.RetryOptions) error {
		return nil
	})
	patches.ApplyFunc((*docker.Client).Tag, func(_ *docker.Client, _, _ string) error {
		return nil
	})
	patches.ApplyFunc((*docker.Client).Remove, func(_ *docker.Client, _ docker.ImageRef) error {
		return nil
	})
	patches.ApplyFunc(pushImage, func(string) error {
		return errors.New("push failed")
	})

	_, err := processArchImage("src/img", "tgt/img", "amd64")
	assert.Error(t, err)
}

func TestSyncMultiArchImageExecuteManifestFails(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	setupMigrateDockerMocks(t, patches)
	patches.ApplyFunc(processArchImage, func(source, target, arch string) (string, error) {
		return target + "-" + arch, nil
	})
	patches.ApplyFunc(global.ExecuteManifestCommands, func(string, string, []string) error {
		return errors.New("manifest cmd failed")
	})

	err := syncMultiArchImage("src/img", "tgt/img", []string{"amd64"})
	assert.Error(t, err)
}

func TestNeedRemoveImageVariable(t *testing.T) {
	needRemoveImage = []string{}
	assert.NotPanics(t, func() {
		needRemoveImage = append(needRemoveImage, "test-image:v1.0")
	})
	assert.Len(t, needRemoveImage, testOneValue)
}
