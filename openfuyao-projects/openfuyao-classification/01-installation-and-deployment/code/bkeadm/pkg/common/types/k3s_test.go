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

package types

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestK3sRestartConfigZeroValue(t *testing.T) {
	var cfg K3sRestartConfig
	assert.Empty(t, cfg.OnlineImage)
	assert.Empty(t, cfg.OtherRepo)
	assert.Empty(t, cfg.OtherRepoIp)
	assert.Empty(t, cfg.OtherSource)
	assert.Empty(t, cfg.HostIP)
	assert.Empty(t, cfg.ImageRepo)
	assert.Empty(t, cfg.ImageRepoPort)
	assert.Empty(t, cfg.KubernetesPort)
	assert.Empty(t, cfg.ConsolePort)
}

func TestK3sRestartConfigFields(t *testing.T) {
	cfg := K3sRestartConfig{
		OnlineImage:    "online.tar",
		OtherRepo:      "repo.example.com",
		OtherRepoIp:    "10.0.0.1",
		OtherSource:    "http://source",
		HostIP:         "192.168.1.10",
		ImageRepo:      "registry.local",
		ImageRepoPort:  "5000",
		KubernetesPort: "6443",
		ConsolePort:    "8443",
	}

	assert.Equal(t, "online.tar", cfg.OnlineImage)
	assert.Equal(t, "repo.example.com", cfg.OtherRepo)
	assert.Equal(t, "10.0.0.1", cfg.OtherRepoIp)
	assert.Equal(t, "http://source", cfg.OtherSource)
	assert.Equal(t, "192.168.1.10", cfg.HostIP)
	assert.Equal(t, "registry.local", cfg.ImageRepo)
	assert.Equal(t, "5000", cfg.ImageRepoPort)
	assert.Equal(t, "6443", cfg.KubernetesPort)
	assert.Equal(t, "8443", cfg.ConsolePort)
}

func TestK3sRestartConfigJSONRoundTrip(t *testing.T) {
	original := K3sRestartConfig{
		OnlineImage:    "image.tar",
		OtherRepo:      "other.repo",
		OtherRepoIp:    "172.16.0.1",
		OtherSource:    "offline",
		HostIP:         "192.168.0.2",
		ImageRepo:      "image.repo",
		ImageRepoPort:  "5001",
		KubernetesPort: "6443",
		ConsolePort:    "443",
	}

	data, err := json.Marshal(original)
	require.NoError(t, err)

	var decoded K3sRestartConfig
	require.NoError(t, json.Unmarshal(data, &decoded))
	assert.Equal(t, original, decoded)
}
