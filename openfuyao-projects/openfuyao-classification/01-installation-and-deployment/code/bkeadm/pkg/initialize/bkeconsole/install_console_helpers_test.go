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

package bkeconsole

import (
	"embed"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	econd "gopkg.openfuyao.cn/bkeadm/pkg/executor/containerd"
	"gopkg.openfuyao.cn/bkeadm/pkg/common/types"
	"gopkg.openfuyao.cn/bkeadm/utils"
)

//go:embed testdata/*
var testEmbedFS embed.FS

func TestCopyEmbeddedFS(t *testing.T) {
	dst := t.TempDir()
	err := copyEmbeddedFS(testEmbedFS, "testdata", dst)
	require.NoError(t, err)

	content, err := os.ReadFile(filepath.Join(dst, "sample.txt"))
	require.NoError(t, err)
	assert.Equal(t, "console-test\n", string(content))
}

func TestResolveImageRepoIP(t *testing.T) {
	t.Run("other repo matches image repo", func(t *testing.T) {
		cfg := types.K3sRestartConfig{
			OtherRepo:   "registry.example.com:443/openfuyao/",
			ImageRepo:   "registry.example.com",
			OtherRepoIp: "203.0.113.5",
			HostIP:      "192.168.1.1",
		}
		assert.Equal(t, "203.0.113.5", resolveImageRepoIP(cfg))
	})

	t.Run("container inspect returns IP", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(econd.ContainerInspect, func(string) (econd.NerdContainerInfo, error) {
			info := econd.NerdContainerInfo{Id: "abc"}
			info.NetworkSettings.IPAddress = "172.17.0.2"
			return info, nil
		})

		cfg := types.K3sRestartConfig{HostIP: "192.168.1.1", ImageRepo: "local.domain"}
		assert.Equal(t, "172.17.0.2", resolveImageRepoIP(cfg))
	})

	t.Run("fallback host IP", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(econd.ContainerInspect, func(string) (econd.NerdContainerInfo, error) {
			return econd.NerdContainerInfo{}, assert.AnError
		})

		cfg := types.K3sRestartConfig{HostIP: "192.168.1.99", ImageRepo: "local.domain"}
		assert.Equal(t, "192.168.1.99", resolveImageRepoIP(cfg))
	})
}

func TestConfigureK3sImages(t *testing.T) {
	origImage, origPause := k3sImage, k3sPause
	defer func() {
		k3sImage = origImage
		k3sPause = origPause
	}()

	t.Run("other repo override", func(t *testing.T) {
		configureK3sImages(types.K3sRestartConfig{
			OtherRepo: "cr.example.com:443/openfuyao/",
		})
		assert.Contains(t, k3sImage, "cr.example.com:443/openfuyao/")
		assert.Contains(t, k3sPause, "cr.example.com:443/openfuyao/")
	})

	t.Run("online image uses third-party mirror defaults", func(t *testing.T) {
		configureK3sImages(types.K3sRestartConfig{OnlineImage: "online.tar"})
		assert.Contains(t, k3sImage, utils.DefaultThirdMirror)
		assert.Contains(t, k3sPause, utils.DefaultThirdMirror)
	})

	t.Run("offline local registry", func(t *testing.T) {
		configureK3sImages(types.K3sRestartConfig{
			ImageRepoPort: "5000",
			ImageRepo:     "registry.local",
		})
		assert.Contains(t, k3sImage, "127.0.0.1:5000")
		assert.Contains(t, k3sPause, "registry.local:443")
	})
}

func TestEnsureK3sContainerReleased(t *testing.T) {
	t.Run("container already gone", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(econd.ContainerExists, func(string) (econd.NerdContainerInfo, bool) {
			return econd.NerdContainerInfo{}, false
		})
		require.NoError(t, ensureK3sContainerReleased())
	})

	t.Run("remove succeeds after retry", func(t *testing.T) {
		attempt := 0
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		patches.ApplyFunc(econd.ContainerExists, func(string) (econd.NerdContainerInfo, bool) {
			attempt++
			if attempt >= 2 {
				return econd.NerdContainerInfo{}, false
			}
			return econd.NerdContainerInfo{Id: "cid-1"}, true
		})
		patches.ApplyFunc(econd.Run, func([]string) error { return nil })
		patches.ApplyFunc(time.Sleep, func(time.Duration) {})

		require.NoError(t, ensureK3sContainerReleased())
	})
}
