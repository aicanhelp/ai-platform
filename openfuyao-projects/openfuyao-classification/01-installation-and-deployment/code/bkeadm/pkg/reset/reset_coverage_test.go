/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package reset

import (
	"errors"
	"os"
	"path"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/bkeadm/pkg/global"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/syscompat"
	"gopkg.openfuyao.cn/bkeadm/pkg/server"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/source"
)

func TestRemoveContainerService(t *testing.T) {
	called := 0
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(server.RemoveImageRegistry, func(string) error { called++; return nil })
	patches.ApplyFunc(server.RemoveChartRegistry, func(string) error { called++; return nil })
	patches.ApplyFunc(server.RemoveYumRegistry, func(string) error { called++; return nil })
	patches.ApplyFunc(server.RemoveNFSServer, func(string) error { called++; return nil })

	RemoveContainerService()
	assert.Equal(t, 4, called)
}

func TestRemoveNtpService(t *testing.T) {
	called := false
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(server.RemoveNTPServer, func() error {
		called = true
		return nil
	})

	RemoveNtpService()
	assert.True(t, called)
}

func TestRemoveDataDir(t *testing.T) {
	origWorkspace := global.Workspace
	global.Workspace = t.TempDir()
	t.Cleanup(func() { global.Workspace = origWorkspace })

	mountDir := path.Join(global.Workspace, "mount")
	require.NoError(t, os.MkdirAll(mountDir, 0o755))

	RemoveDataDir()
	_, err := os.Stat(mountDir)
	assert.True(t, os.IsNotExist(err))
}

func TestResetAllWithSourceError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(RemoveLocalKubernetes, func() {})
	patches.ApplyFunc(RemoveContainerService, func() {})
	patches.ApplyFunc(RemoveNtpService, func() {})
	patches.ApplyFunc(removeAllInOne, func() {})
	patches.ApplyFunc(source.ResetSource, func() error {
		return errors.New("reset source failed")
	})
	patches.ApplyFunc(syscompat.RepoUpdate, func() error { return nil })

	op := Options{All: true}
	op.Reset()
}
