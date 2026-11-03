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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

const hostNum = 2

func TestNewClockChecker(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	c := NewClockChecker(cfg, log)
	assert.NotNil(t, c)
	assert.Equal(t, cfg, c.config)
	assert.Equal(t, log, c.logger)
}

func TestIsBootstrap(t *testing.T) {
	c := NewClockChecker(&config.AppConfig{}, logger.NewLogger(""))
	assert.True(t, c.isBootstrap(config.Host{Role: []string{"bootstrap"}}))
	assert.False(t, c.isBootstrap(config.Host{Role: []string{"worker"}}))
}

func TestGetHostSets(t *testing.T) {
	cfg := &config.AppConfig{Hosts: []config.Host{
		{IP: "b", Role: []string{"bootstrap"}}, {IP: "n1", Role: []string{"node"}},
	}}
	c := NewClockChecker(cfg, logger.NewLogger(""))
	b, o, err := c.getHostSets()
	assert.NoError(t, err)
	assert.Len(t, b, 1)
	assert.Len(t, o, 1)

	// no bootstrap
	c2 := NewClockChecker(&config.AppConfig{Hosts: []config.Host{{IP: "x", Role: []string{"node"}}}},
		logger.NewLogger(""))
	if _, _, err2 := c2.getHostSets(); err2 == nil {
		assert.Error(t, err2)
	}

	// no non-bootstrap
	c3 := NewClockChecker(&config.AppConfig{Hosts: []config.Host{{IP: "b", Role: []string{"bootstrap"}}}},
		logger.NewLogger(""))
	if _, _, err3 := c3.getHostSets(); err3 == nil {
		assert.Error(t, err3)
	}
}

func TestGetLocalAndHostTime(t *testing.T) {
	c := NewClockChecker(&config.AppConfig{}, logger.NewLogger(""))
	lt, err := c.getLocalTime()
	assert.NoError(t, err)
	assert.Greater(t, lt, int64(0))
	// local host path via getHostTime
	h := config.Host{IP: "127.0.0.1"}
	ht, err := c.getHostTime(h)
	assert.NoError(t, err)
	assert.Greater(t, ht, int64(0))
}

func TestUpdateCheckResult(t *testing.T) {
	cfg := &config.AppConfig{ClockThreshold: 5}
	c := NewClockChecker(cfg, logger.NewLogger(""))
	ref := time.Now().Unix()
	all := []HostResult{{host: "a", role: []string{"node"}, time: ref, err: nil},
		{host: "b", role: []string{"node"}, time: ref + 100, err: fmt.Errorf("fail")}}
	res := &config.ClockCheckResult{Timestamp: time.Now(), Clocks: []config.ClockCheckInfo{}, Result: "pass"}
	c.updateCheckResult(all, ref, res)
	assert.Len(t, res.Clocks, hostNum)
	assert.True(t, res.Clocks[0].IsSynced)
	assert.Equal(t, "fail", res.Result)
}

func TestExecuteErrorPaths(t *testing.T) {
	// no bootstrap -> Execute should return error
	c1 := NewClockChecker(&config.AppConfig{Hosts: []config.Host{{IP: "n1", Role: []string{"node"}}}},
		logger.NewLogger(""))
	if _, err := c1.Execute(); err == nil {
		assert.Error(t, err)
	}

	// only bootstrap -> Execute should return error (no other hosts)
	c2 := NewClockChecker(&config.AppConfig{Hosts: []config.Host{{IP: "b", Role: []string{"bootstrap"}}}},
		logger.NewLogger(""))
	if _, err := c2.Execute(); err == nil {
		assert.Error(t, err)
	}
}
