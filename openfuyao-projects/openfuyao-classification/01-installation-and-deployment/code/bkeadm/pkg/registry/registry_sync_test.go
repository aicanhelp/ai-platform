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

package registry

import (
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/bkeadm/pkg/executor/docker"
	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/infrastructure"
)

func TestParseImageReferences(t *testing.T) {
	t.Run("valid docker references", func(t *testing.T) {
		srcRef, destRef, err := parseImageReferences(
			"docker://src.example.com/ns/img:tag",
			"docker://dest.example.com/ns/img:tag",
			"src.example.com/ns/img:tag",
		)
		require.NoError(t, err)
		assert.NotNil(t, srcRef)
		assert.NotNil(t, destRef)
	})

	t.Run("invalid source", func(t *testing.T) {
		_, _, err := parseImageReferences("://bad", "docker://dest.example.com/img:tag", "://bad")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid source name")
	})

	t.Run("invalid destination", func(t *testing.T) {
		_, _, err := parseImageReferences("docker://src.example.com/img:tag", "://bad", "src.example.com/img:tag")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid destination name")
	})
}

func TestTrySyncWithArchSuffix(t *testing.T) {
	t.Run("success with arch suffix", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(CopyRegistry, func(op Options) error {
			assert.Contains(t, op.Source, "-amd64")
			return nil
		})

		op := Options{Source: "repo.example.com/img", Target: "dest.example.com/img"}
		err := trySyncWithArchSuffix(op, op.Source, op.Target, "amd64")
		require.NoError(t, err)
	})

	t.Run("failure propagates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(CopyRegistry, func(Options) error {
			return errors.New("copy failed")
		})

		op := Options{Source: "repo.example.com/img", Target: "dest.example.com/img"}
		err := trySyncWithArchSuffix(op, op.Source, op.Target, "amd64")
		assert.Error(t, err)
	})
}

func TestSyncRepoImageRouting(t *testing.T) {
	t.Run("single arch delegates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		called := false
		patches.ApplyFunc(syncSingleArchRepoImage, func(Options, string) error {
			called = true
			return nil
		})

		err := syncRepoImage(Options{}, []string{"amd64"})
		require.NoError(t, err)
		assert.True(t, called)
	})

	t.Run("multi arch delegates", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		called := false
		patches.ApplyFunc(syncMultiArchRepoImage, func(Options, []string) error {
			called = true
			return nil
		})

		err := syncRepoImage(Options{}, []string{"amd64", "arm64"})
		require.NoError(t, err)
		assert.True(t, called)
	})
}

func TestMigrateImageNonDockerEnvironment(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(infrastructure.IsDocker, func() bool { return false })

	op := &Options{Image: "test/image:tag"}
	op.MigrateImage()
}

func TestCleanBuildImage(t *testing.T) {
	needRemoveImage = []string{"image-a", "image-b"}
	originalDocker := global.Docker
	global.Docker = &docker.Client{}
	defer func() {
		global.Docker = originalDocker
		needRemoveImage = nil
	}()

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	removed := 0
	patches.ApplyFunc((*docker.Client).Remove, func(_ *docker.Client, ref docker.ImageRef) error {
		removed++
		assert.NotEmpty(t, ref.Image)
		return nil
	})

	cleanBuildImage()
	assert.Equal(t, 2, removed)
}
