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

package clock

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func TestAbsPositive(t *testing.T) {
	assert.Equal(t, int64(5), abs(5))
}

func TestAbsNegative(t *testing.T) {
	assert.Equal(t, int64(5), abs(-5))
}

func TestAbsZero(t *testing.T) {
	assert.Equal(t, int64(0), abs(0))
}

func TestIsLocalHost127(t *testing.T) {
	assert.True(t, isLocalHost("127.0.0.1"))
}

func TestIsLocalHostLocalhost(t *testing.T) {
	assert.True(t, isLocalHost("localhost"))
}

func TestIsLocalHostRemote(t *testing.T) {
	assert.NotPanics(t, func() {
		_ = isLocalHost("192.168.1.100")
	})
}

func TestGetRemoteTimeInvalidHost(t *testing.T) {
	origTimeout := timeoutSec
	timeoutSec = 1
	defer func() { timeoutSec = origTimeout }()

	cfg := &config.AppConfig{}
	c := NewClockChecker(cfg, logger.NewLogger(""))
	host := config.Host{IP: "999.999.999.999", Port: "22", UserName: "test", Password: "test"}

	_, err := c.getRemoteTime(host)
	assert.Error(t, err)
}

func TestExecuteOnlyBootstrap(t *testing.T) {
	cfg := &config.AppConfig{
		Hosts: []config.Host{
			{IP: "127.0.0.1", Role: []string{"bootstrap"}},
			{IP: "127.0.0.1", Role: []string{"worker"}},
		},
		ClockThreshold: 10,
	}
	log := logger.NewLogger("")
	checker := NewClockChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)
}
