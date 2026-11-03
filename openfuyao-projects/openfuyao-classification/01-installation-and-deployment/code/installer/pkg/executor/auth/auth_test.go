/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package auth

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"

	"installer/pkg/constants"
)

// 轻量级Mock客户端
type LightMockClient struct{}

func (m *LightMockClient) SudoExecCmd(cmd string) (string, error) {
	return "success", nil
}

func (m *LightMockClient) DownloadFile(local, remote string) error {
	return nil
}

func (m *LightMockClient) CopyFilesToRemote(localDir, remoteDir string) error {
	return nil
}

func (m *LightMockClient) ExecCmd(cmd string) (string, error) {
	return "success", nil
}

func TestNewOauthWebHook_Success(t *testing.T) {
	oauth := NewOauthWebHook()

	assert.NotNil(t, oauth)
	expectedRemoteDir := filepath.Join(constants.EtcPath, constants.Kubernetes, constants.WebHookPath)
	expectedLocalDir := filepath.Join(constants.OptPath, constants.OpenFuyao, constants.Tmp, constants.WebHookPath)

	assert.Equal(t, expectedRemoteDir, oauth.webhookRemoteDir)
	assert.Equal(t, expectedLocalDir, oauth.webhookLocalDir)
}
