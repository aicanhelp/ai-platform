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

package upgradepath

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRepoFromBKEConfigData_OfflineDefault(t *testing.T) {
	repo, ok := RepoFromBKEConfigData(map[string]string{
		"domain":        "deploy.bocloud.k8s",
		"host":          "192.168.1.10",
		"imageRepoPort": "40443",
	})
	require.True(t, ok)
	assert.Equal(t, "deploy.bocloud.k8s", repo.Domain)
	assert.Equal(t, "192.168.1.10", repo.Ip)
	assert.Equal(t, "40443", repo.Port)
	assert.Equal(t, "kubernetes", repo.Prefix)
}

func TestRepoFromBKEConfigData_OtherRepo(t *testing.T) {
	repo, ok := RepoFromBKEConfigData(map[string]string{
		"otherRepo":   "cr.openfuyao.cn/openfuyao/",
		"otherRepoIp": "119.3.216.97",
	})
	require.True(t, ok)
	assert.Equal(t, "cr.openfuyao.cn", repo.Domain)
	assert.Equal(t, "119.3.216.97", repo.Ip)
	assert.Equal(t, "443", repo.Port)
	assert.Equal(t, "openfuyao", repo.Prefix)
}

func TestRepoFromBKEConfigData_OnlineImageOnly(t *testing.T) {
	repo, ok := RepoFromBKEConfigData(map[string]string{
		"domain":        "deploy.bocloud.k8s",
		"host":          "10.0.0.1",
		"imageRepoPort": "40443",
		"onlineImage":   "cr.openfuyao.cn/openfuyao/bke-online-installed:latest",
	})
	require.True(t, ok)
	assert.Equal(t, "default", repo.Domain)
	assert.Equal(t, "10.0.0.1", repo.Ip)
	assert.Equal(t, "40443", repo.Port)
	assert.Empty(t, repo.Prefix)
}

func TestRepoFromBKEConfigData_Empty(t *testing.T) {
	_, ok := RepoFromBKEConfigData(map[string]string{})
	assert.False(t, ok)
}
