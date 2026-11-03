/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package port

import (
	"net"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func TestNewChecker(t *testing.T) {
	cfg := &config.PortCheckConfig{
		Ports: map[string][]string{
			"master": {"6443"},
		},
		Timeout: 3,
	}
	log := logger.NewLogger("")
	checker := NewChecker(cfg, log)
	assert.NotNil(t, checker)
	assert.Len(t, checker.items, 1)
	assert.Equal(t, cfg, checker.cfg)
}

func TestBuildCheckItems(t *testing.T) {
	tests := []struct {
		name      string
		ports     map[string][]string
		wantItems int
	}{
		{"allRoles", map[string][]string{"bootstrap": {"80"}, "master": {"6443"}, "worker": {"10250"}}, 3},
		{"bootstrapOnly", map[string][]string{"bootstrap": {"80"}}, 1},
		{"masterOnly", map[string][]string{"master": {"6443"}}, 1},
		{"workerOnly", map[string][]string{"worker": {"10250"}}, 1},
		{"emptyPorts", map[string][]string{}, 0},
		{"nilPorts", nil, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.PortCheckConfig{Ports: tt.ports}
			items := buildCheckItems(cfg)
			assert.Len(t, items, tt.wantItems)
		})
	}
}

func TestCheckerAssign(t *testing.T) {
	cfg := &config.PortCheckConfig{
		Ports: map[string][]string{
			"bootstrap": {"80"},
			"master":    {"6443"},
			"worker":    {"10250"},
		},
	}
	checker := NewChecker(cfg, logger.NewLogger(""))
	assert.Len(t, checker.items, 3)

	tests := []struct {
		name  string
		roles []string
		item  *CheckItem
		want  bool
	}{
		{"emptyRoles", []string{}, checker.items[0], true},
		{"bootstrapMatch", []string{"bootstrap"}, checker.items[0], true},
		{"masterMatch", []string{"master"}, checker.items[1], true},
		{"workerMatch", []string{"worker"}, checker.items[2], true},
		{"noMatchBootstrapVsMaster", []string{"master"}, checker.items[0], false},
		{"multipleRolesMatch", []string{"master", "worker"}, checker.items[1], true},
		{"unknownRole", []string{"unknown"}, checker.items[0], false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := checker.assign(tt.item, tt.roles)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestCheckPortFree(t *testing.T) {
	cfg := &config.PortCheckConfig{Timeout: 1}
	checker := NewChecker(cfg, logger.NewLogger(""))

	info := checker.checkPort("65432")
	assert.Equal(t, "65432", info.Port)
	assert.Equal(t, "tcp", info.Protocol)
	assert.False(t, info.IsUsed)
}

func TestCheckPortUsed(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	assert.NoError(t, err)
	defer listener.Close()

	port := strconv.Itoa(listener.Addr().(*net.TCPAddr).Port)
	cfg := &config.PortCheckConfig{Timeout: 1}
	checker := NewChecker(cfg, logger.NewLogger(""))

	info := checker.checkPort(port)
	assert.Equal(t, port, info.Port)
	assert.True(t, info.IsUsed)
}

func TestCheckPortDefaultTimeout(t *testing.T) {
	cfg := &config.PortCheckConfig{Timeout: 0}
	checker := NewChecker(cfg, logger.NewLogger(""))

	info := checker.checkPort("65433")
	assert.Equal(t, "65433", info.Port)
	assert.False(t, info.IsUsed)
}

func TestRunCheckItem(t *testing.T) {
	cfg := &config.PortCheckConfig{Timeout: 1}
	checker := NewChecker(cfg, logger.NewLogger(""))
	item := NewCheckItem("test", []string{"all"}, 1, []string{"65431", "65430"})
	result := &config.PortCheckResult{
		Ports:   make([]config.PortCheckInfo, 0),
		Summary: config.PortSummary{},
	}

	checker.runCheckItem(item, result)
	assert.Len(t, result.Ports, 2)
	assert.Equal(t, 2, result.Summary.TotalChecked)
	assert.Equal(t, 2, result.Summary.Free)
	assert.Equal(t, 0, result.Summary.Used)
}

func TestCheckerExecute(t *testing.T) {
	cfg := &config.PortCheckConfig{
		Ports: map[string][]string{
			"master": {"65429"},
		},
		Timeout: 1,
	}
	checker := NewChecker(cfg, logger.NewLogger(""))
	result := checker.Execute([]string{"master"})
	assert.NotNil(t, result)
	assert.Equal(t, 1, result.Summary.TotalChecked)
}

func TestCheckerExecuteWithRoles(t *testing.T) {
	cfg := &config.PortCheckConfig{
		Ports: map[string][]string{
			"bootstrap": {"65428"},
			"master":    {"65427"},
		},
		Timeout: 1,
	}
	checker := NewChecker(cfg, logger.NewLogger(""))
	result := checker.Execute([]string{"worker"})
	assert.NotNil(t, result)
	assert.Equal(t, 0, result.Summary.TotalChecked)
}
