/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package sshutil

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
)

func TestDefaultPort(t *testing.T) {
	assert.Equal(t, "22", DefaultPort(config.Host{}))
	assert.Equal(t, "2222", DefaultPort(config.Host{Port: "2222"}))
}

func TestAddress(t *testing.T) {
	assert.Equal(t, "10.0.0.1:22", Address(config.Host{IP: "10.0.0.1"}))
	assert.Equal(t, "10.0.0.1:2222", Address(config.Host{IP: "10.0.0.1", Port: "2222"}))
}

func TestBuildClientConfig(t *testing.T) {
	host := config.Host{UserName: "root", Password: "secret"}
	cfg := BuildClientConfig(host, 5*time.Second)
	assert.Equal(t, "root", cfg.User)
	assert.Len(t, cfg.Auth, 1)
	assert.Equal(t, 5*time.Second, cfg.Timeout)
	assert.NotNil(t, cfg.HostKeyCallback)
}
