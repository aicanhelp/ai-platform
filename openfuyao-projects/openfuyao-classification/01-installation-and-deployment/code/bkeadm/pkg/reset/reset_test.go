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

package reset

import (
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/syscompat"
	"gopkg.openfuyao.cn/bkeadm/pkg/initialize/timezone"
	"gopkg.openfuyao.cn/bkeadm/pkg/root"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common/source"
)

func TestReset(t *testing.T) {

	op := Options{
		Options: root.Options{
			KubeConfig: "/root/.kube/config",
		},
	}
	op.Reset()
}

func TestResetAllMountRemovesBKEManagedConfigAndRestoresTimezone(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(RemoveLocalKubernetes, func() {})
	patches.ApplyFunc(RemoveContainerService, func() {})
	patches.ApplyFunc(RemoveNtpService, func() {})
	patches.ApplyFunc(removeAllInOne, func() {})
	patches.ApplyFunc(RemoveDataDir, func() {})
	patches.ApplyFunc(source.ResetSource, func() error {
		return nil
	})
	patches.ApplyFunc(syscompat.RepoUpdate, func() error {
		return nil
	})

	removeManagedConfigCalled := false
	restoreTimezoneCalled := false
	patches.ApplyFunc(syscompat.RemoveBKEManagedConfig, func() {
		removeManagedConfigCalled = true
	})
	patches.ApplyFunc(timezone.RestoreTimeZone, func() {
		restoreTimezoneCalled = true
	})

	op := Options{All: true, Mount: true}
	op.Reset()

	assert.True(t, removeManagedConfigCalled)
	assert.True(t, restoreTimezoneCalled)
}

func TestResetMountOnlyDoesNotRestoreInitGeneratedConfig(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(RemoveLocalKubernetes, func() {})
	patches.ApplyFunc(RemoveDataDir, func() {})
	patches.ApplyFunc(RemoveContainerService, func() {})
	patches.ApplyFunc(RemoveNtpService, func() {})
	patches.ApplyFunc(source.ResetSource, func() error {
		return nil
	})

	removeManagedConfigCalled := false
	restoreTimezoneCalled := false
	patches.ApplyFunc(syscompat.RemoveBKEManagedConfig, func() {
		removeManagedConfigCalled = true
	})
	patches.ApplyFunc(timezone.RestoreTimeZone, func() {
		restoreTimezoneCalled = true
	})

	op := Options{Mount: true}
	op.Reset()

	assert.False(t, removeManagedConfigCalled)
	assert.False(t, restoreTimezoneCalled)
}
