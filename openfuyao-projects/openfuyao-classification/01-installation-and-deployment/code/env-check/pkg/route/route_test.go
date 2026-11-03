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

package route

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func createTestRouteConfig() *config.AppConfig {
	return &config.AppConfig{}
}

func TestNewRouteChecker(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")

	checker := NewRouteChecker(cfg, log)

	assert.NotNil(t, checker)
	assert.Equal(t, cfg, checker.config)
}

func TestRouteCheckerExecute(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotNil(t, result.DefaultRoutes)
}

func TestParseIPRouteLine(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	line := "default via 192.168.1.1 dev eth0 metric 100"
	route := checker.parseIPRouteLine(line)

	assert.True(t, route.Exists)
	assert.Equal(t, "default", route.Destination)
	assert.Equal(t, "192.168.1.1", route.Gateway)
	assert.Equal(t, "eth0", route.Interface)
	assert.Equal(t, 100, route.Metric)
}

func TestParseIPRouteLineMultiple(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	line := "default via 192.168.2.1 dev eth1 metric 200"
	route := checker.parseIPRouteLine(line)

	assert.True(t, route.Exists)
	assert.Equal(t, "192.168.2.1", route.Gateway)
	assert.Equal(t, "eth1", route.Interface)
	assert.Equal(t, 200, route.Metric)
}

func TestParseRouteNLine(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	line := "0.0.0.0         192.168.1.1     0.0.0.0         UG    100    0        0 eth0"
	route := checker.parseRouteNLine(line)

	assert.True(t, route.Exists)
	assert.Equal(t, "default", route.Destination)
	assert.Equal(t, "192.168.1.1", route.Gateway)
	assert.Equal(t, "eth0", route.Interface)
	assert.Equal(t, 100, route.Metric)
}

func TestParseRouteOutput(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	output := `default via 192.168.1.1 dev eth0 metric 100
default via 192.168.2.1 dev eth1 metric 200`

	routes := checker.parseRouteOutput(output)

	assert.Equal(t, 2, len(routes))
	assert.Equal(t, "192.168.1.1", routes[0].Gateway)
	assert.Equal(t, "192.168.2.1", routes[1].Gateway)
}

func TestParseRouteOutputEmpty(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	output := `192.168.1.0/24 dev eth0 proto kernel scope link src 192.168.1.100
192.168.2.0/24 dev eth1 proto kernel scope link src 192.168.2.100`

	routes := checker.parseRouteOutput(output)

	assert.Equal(t, 0, len(routes))
}

func TestParseRouteOutputRouteN(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	output := `Kernel IP routing table
Destination     Gateway         Genmask         Flags Metric Ref    Use Iface
0.0.0.0         192.168.1.1     0.0.0.0         UG    100    0        0 eth0
0.0.0.0         192.168.2.1     0.0.0.0         UG    200    0        0 eth1
192.168.1.0     0.0.0.0         255.255.255.0   U     0      0        0 eth0`

	routes := checker.parseRouteOutput(output)

	assert.GreaterOrEqual(t, len(routes), 1)
}

func TestRouteCheckResultSummary(t *testing.T) {
	cfg := createTestRouteConfig()
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)

	if result.Summary.TotalRoutes > 0 {
		assert.True(t, result.Summary.CheckPassed)
		assert.Equal(t, "default route exists", result.Summary.Message)
	} else {
		assert.False(t, result.Summary.CheckPassed)
		// 命令可用但无默认路由 → "no default route found"
		// 命令不可用 → 错误原因（如 "no route command available..."）
		assert.NotEmpty(t, result.Summary.Message)
	}
}