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

func TestParseMetricEmpty(t *testing.T) {
	result := parseMetric("")
	assert.Equal(t, 0, result)
}

func TestParseMetricNoDigits(t *testing.T) {
	result := parseMetric("abc")
	assert.Equal(t, 0, result)
}

func TestParseMetricLargeNumber(t *testing.T) {
	result := parseMetric("99999")
	assert.Equal(t, 99999, result)
}

func TestParseMetricWithSuffix(t *testing.T) {
	result := parseMetric("100 metric")
	assert.Equal(t, 100, result)
}

func TestParseRouteNLineShortLine(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	route := checker.parseRouteNLine("0.0.0.0 0.0.0.0")

	assert.True(t, route.Exists)
	assert.Equal(t, "default", route.Destination)
	assert.Equal(t, "", route.Gateway)
}

func TestParseRouteNLineNonDefault(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	route := checker.parseRouteNLine("192.168.1.0 0.0.0.0 255.255.255.0 UG 0 0 0 eth0")

	assert.True(t, route.Exists)
	assert.Equal(t, "default", route.Destination)
	assert.Equal(t, "", route.Gateway)
}

func TestParseRouteNLineWithAllFields(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	route := checker.parseRouteNLine("0.0.0.0 192.168.1.1 0.0.0.0 UG 100 0 0 eth0")

	assert.True(t, route.Exists)
	assert.Equal(t, "default", route.Destination)
	assert.Equal(t, "192.168.1.1", route.Gateway)
	assert.Equal(t, "eth0", route.Interface)
	assert.Equal(t, 100, route.Metric)
}

func TestParseRouteNLineGatewayStar(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	route := checker.parseRouteNLine("0.0.0.0 * 0.0.0.0 U 0 0 0 eth0")

	assert.True(t, route.Exists)
	assert.Equal(t, "default", route.Destination)
	assert.Equal(t, "", route.Gateway)
}

func TestExecuteRouteCommandNoCommand(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	_, err := checker.executeRouteCommand()
	assert.Error(t, err)
}

func TestExecuteNoCommand(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewRouteChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.NotEmpty(t, result.Summary.Message)
}
