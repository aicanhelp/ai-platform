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

package main

import (
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func TestSplitAndTrim(t *testing.T) {
	t.Run("empty string returns nil", func(t *testing.T) {
		assert.Nil(t, splitAndTrim(""))
	})

	t.Run("normal comma-separated", func(t *testing.T) {
		result := splitAndTrim("a,b,c")
		assert.Equal(t, []string{"a", "b", "c"}, result)
	})

	t.Run("spaces around items", func(t *testing.T) {
		result := splitAndTrim(" a , b , c ")
		assert.Equal(t, []string{"a", "b", "c"}, result)
	})

	t.Run("empty items between commas", func(t *testing.T) {
		result := splitAndTrim("a,,b,")
		assert.Equal(t, []string{"a", "b"}, result)
	})

	t.Run("single item", func(t *testing.T) {
		result := splitAndTrim("single")
		assert.Equal(t, []string{"single"}, result)
	})
}

func TestFilterHostsByRole(t *testing.T) {
	hosts := []config.Host{
		{IP: "192.168.1.1", Role: []string{"master"}},
		{IP: "192.168.1.2", Role: []string{"worker"}},
		{IP: "192.168.1.3", Role: []string{"master", "worker"}},
	}

	t.Run("empty hosts returns nil", func(t *testing.T) {
		assert.Nil(t, filterHostsByRole(nil, []string{"master"}))
	})

	t.Run("empty roles returns nil", func(t *testing.T) {
		assert.Nil(t, filterHostsByRole(hosts, nil))
	})

	t.Run("single match", func(t *testing.T) {
		result := filterHostsByRole(hosts, []string{"worker"})
		assert.Len(t, result, 2)
		assert.Equal(t, "192.168.1.2", result[0].IP)
		assert.Equal(t, "192.168.1.3", result[1].IP)
	})

	t.Run("multiple roles match", func(t *testing.T) {
		result := filterHostsByRole(hosts, []string{"master", "worker"})
		assert.Len(t, result, 3)
	})

	t.Run("no match", func(t *testing.T) {
		result := filterHostsByRole(hosts, []string{"unknown"})
		assert.Nil(t, result)
	})

	t.Run("host with multiple roles matches one", func(t *testing.T) {
		result := filterHostsByRole(hosts, []string{"master"})
		assert.Len(t, result, 2)
		assert.Equal(t, "192.168.1.1", result[0].IP)
		assert.Equal(t, "192.168.1.3", result[1].IP)
	})
}

func TestBuildDispatchConfig(t *testing.T) {
	origTimeout := timeout
	origCheckList := checkList
	origSkipList := skipList
	t.Cleanup(func() {
		timeout = origTimeout
		checkList = origCheckList
		skipList = origSkipList
	})

	t.Run("nil dispatch creates default", func(t *testing.T) {
		timeout = 0
		checkList = ""
		skipList = ""
		cfg := &config.AppConfig{}
		dc := buildDispatchConfig(cfg)
		assert.Equal(t, defaultPollInterval, dc.PollInterval)
		assert.Equal(t, defaultConcurrentLimit, dc.ConcurrentLimit)
		assert.Equal(t, "/tmp/envcheck", dc.WorkDir)
	})

	t.Run("existing dispatch is preserved", func(t *testing.T) {
		timeout = 0
		checkList = ""
		skipList = ""
		existing := &config.DispatchConfig{
			PollInterval:    30,
			ConcurrentLimit: 5,
			WorkDir:         "/custom/work",
		}
		cfg := &config.AppConfig{Dispatch: existing}
		dc := buildDispatchConfig(cfg)
		assert.Equal(t, 30, dc.PollInterval)
		assert.Equal(t, 5, dc.ConcurrentLimit)
		assert.Equal(t, "/custom/work", dc.WorkDir)
		assert.Same(t, existing, dc)
	})

	t.Run("timeout overrides", func(t *testing.T) {
		timeout = 120
		checkList = ""
		skipList = ""
		cfg := &config.AppConfig{}
		dc := buildDispatchConfig(cfg)
		assert.Equal(t, 120, dc.Timeout)
	})

	t.Run("checkList sets checks", func(t *testing.T) {
		timeout = 0
		checkList = "kernel,port"
		skipList = ""
		cfg := &config.AppConfig{}
		dc := buildDispatchConfig(cfg)
		assert.Equal(t, []string{"kernel", "port"}, dc.Checks)
	})

	t.Run("skipList sets skipChecks", func(t *testing.T) {
		timeout = 0
		checkList = ""
		skipList = "disk,clock"
		cfg := &config.AppConfig{}
		dc := buildDispatchConfig(cfg)
		assert.Equal(t, []string{"disk", "clock"}, dc.SkipChecks)
	})
}

func TestGetTargetHosts(t *testing.T) {
	origRoles := roles
	t.Cleanup(func() {
		roles = origRoles
	})

	hosts := []config.Host{
		{IP: "192.168.1.1", Role: []string{"master"}},
		{IP: "192.168.1.2", Role: []string{"worker"}},
		{IP: "192.168.1.3", Role: []string{"master", "worker"}},
	}
	cfg := &config.AppConfig{Hosts: hosts}

	t.Run("empty roles returns all hosts", func(t *testing.T) {
		roles = ""
		result := getTargetHosts(cfg)
		assert.Len(t, result, 3)
	})

	t.Run("master role returns only master hosts", func(t *testing.T) {
		roles = "master"
		result := getTargetHosts(cfg)
		assert.Len(t, result, 2)
		assert.Equal(t, "192.168.1.1", result[0].IP)
		assert.Equal(t, "192.168.1.3", result[1].IP)
	})

	t.Run("master,worker returns master and worker hosts", func(t *testing.T) {
		roles = "master,worker"
		result := getTargetHosts(cfg)
		assert.Len(t, result, 3)
	})

	t.Run("unknown role returns empty", func(t *testing.T) {
		roles = "unknown"
		result := getTargetHosts(cfg)
		assert.Nil(t, result)
	})
}

func TestRunKernelCheck(t *testing.T) {
	log := logger.NewLogger("")

	t.Run("nil kernel check returns skip", func(t *testing.T) {
		cfg := &config.AppConfig{}
		result := runKernelCheck(cfg, log)
		assert.Equal(t, "kernel", result.CheckType)
		assert.Equal(t, "skip", result.Status)
	})

	t.Run("valid kernel version passes", func(t *testing.T) {
		cfg := &config.AppConfig{
			KernelCheck: &config.KernelCheckConfig{
				MinVersion: "0.0.1",
				Operator:   ">=",
			},
		}
		result := runKernelCheck(cfg, log)
		assert.Equal(t, "kernel", result.CheckType)
		assert.Equal(t, "pass", result.Status)
	})

	t.Run("impossible kernel version fails", func(t *testing.T) {
		cfg := &config.AppConfig{
			KernelCheck: &config.KernelCheckConfig{
				MinVersion: "999.999.999",
				Operator:   ">=",
			},
		}
		result := runKernelCheck(cfg, log)
		assert.Equal(t, "kernel", result.CheckType)
		assert.Equal(t, "fail", result.Status)
	})
}

func TestRunPortCheck(t *testing.T) {
	log := logger.NewLogger("")

	t.Run("nil port check returns skip", func(t *testing.T) {
		cfg := &config.AppConfig{}
		result := runPortCheck(cfg, log, nil)
		assert.Equal(t, "port", result.CheckType)
		assert.Equal(t, "skip", result.Status)
	})

	t.Run("free port returns pass", func(t *testing.T) {
		cfg := &config.AppConfig{
			PortCheck: &config.PortCheckConfig{
				Ports: map[string][]string{
					"bootstrap": {"65530"},
				},
				Timeout: 1,
			},
		}
		result := runPortCheck(cfg, log, []string{"bootstrap"})
		assert.Equal(t, "port", result.CheckType)
		assert.Equal(t, "pass", result.Status)
	})
}

func TestRunDiskCheck(t *testing.T) {
	log := logger.NewLogger("")

	t.Run("nil disk check returns skip", func(t *testing.T) {
		cfg := &config.AppConfig{}
		result, _ := runDiskCheck(cfg, log, nil)
		assert.Equal(t, "disk", result.CheckType)
		assert.Equal(t, "skip", result.Status)
	})

	t.Run("sufficient space returns pass", func(t *testing.T) {
		dir := t.TempDir()
		cfg := &config.AppConfig{
			DiskCheck: &config.DiskCheckConfig{
				CheckItems: []config.DiskCheckItem{
					{Path: dir, MinFreeGB: 0},
				},
			},
		}
		result, _ := runDiskCheck(cfg, log, []string{"master"})
		assert.Equal(t, "disk", result.CheckType)
		assert.Equal(t, "pass", result.Status)
	})

	t.Run("insufficient space returns fail", func(t *testing.T) {
		dir := t.TempDir()
		cfg := &config.AppConfig{
			DiskCheck: &config.DiskCheckConfig{
				CheckItems: []config.DiskCheckItem{
					{Path: dir, MinFreeGB: 999999999},
				},
			},
		}
		result, _ := runDiskCheck(cfg, log, []string{"master"})
		assert.Equal(t, "disk", result.CheckType)
		assert.Equal(t, "fail", result.Status)
	})
}

func TestRunClockCheck(t *testing.T) {
	result := runClockCheck()
	assert.Equal(t, "clock", result.CheckType)
	assert.Equal(t, "pass", result.Status)
	assert.Contains(t, result.Detail, "local time:")
}

func TestRunRouteCheck(t *testing.T) {
	log := logger.NewLogger("")
	cfg := &config.AppConfig{}
	result := runRouteCheck(cfg, log)
	assert.Equal(t, "route", result.CheckType)
	assert.NotEmpty(t, result.Status)
}

func TestRunIptablesCheck(t *testing.T) {
	log := logger.NewLogger("")
	cfg := &config.AppConfig{}
	result := runIptablesCheck(cfg, log)
	assert.Equal(t, "iptables", result.CheckType)
	assert.NotEmpty(t, result.Status)
}

func TestRunChecks(t *testing.T) {
	log := logger.NewLogger("")

	t.Run("empty checks returns nil", func(t *testing.T) {
		cfg := &config.AppConfig{}
		results := runChecks(cfg, log, nil, []string{})
		assert.Nil(t, results)
	})

	t.Run("kernel check returns one result", func(t *testing.T) {
		cfg := &config.AppConfig{}
		results := runChecks(cfg, log, nil, []string{"kernel"})
		assert.Len(t, results, 1)
		assert.Equal(t, "kernel", results[0].CheckType)
	})

	t.Run("clock check returns pass", func(t *testing.T) {
		cfg := &config.AppConfig{}
		results := runChecks(cfg, log, nil, []string{"clock"})
		assert.Len(t, results, 1)
		assert.Equal(t, "clock", results[0].CheckType)
		assert.Equal(t, "pass", results[0].Status)
	})

	t.Run("unknown check returns skip", func(t *testing.T) {
		cfg := &config.AppConfig{}
		results := runChecks(cfg, log, nil, []string{"unknown"})
		assert.Len(t, results, 1)
		assert.Equal(t, "skip", results[0].Status)
	})

	t.Run("multiple checks returns multiple results", func(t *testing.T) {
		cfg := &config.AppConfig{}
		results := runChecks(cfg, log, nil, []string{"kernel", "clock"})
		assert.Len(t, results, 2)
	})
}

func TestLoadLocalConfig(t *testing.T) {
	dir := t.TempDir()
	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)
	os.Chdir(dir)

	cfg, err := loadLocalConfig()
	assert.NoError(t, err)
	assert.NotNil(t, cfg)
	assert.True(t, cfg.LogFile != "" || cfg.OutputFormat != "")
}

func TestFormatFailedPrograms(t *testing.T) {
	t.Run("empty list returns empty string", func(t *testing.T) {
		result := formatFailedPrograms(nil)
		assert.Equal(t, "", result)
	})

	t.Run("one failed program", func(t *testing.T) {
		programs := []config.ProgramInfo{
			{Name: "test-prog", Passed: false, Installed: false},
		}
		result := formatFailedPrograms(programs)
		assert.Contains(t, result, "test-prog")
	})

	t.Run("multiple failed programs", func(t *testing.T) {
		programs := []config.ProgramInfo{
			{Name: "prog1", Passed: false, Installed: false},
			{Name: "prog2", Passed: false, Installed: true},
		}
		result := formatFailedPrograms(programs)
		assert.Contains(t, result, "prog1")
		assert.Contains(t, result, "prog2")
	})
}

func TestGenerateDispatchHTML(t *testing.T) {
	dir := t.TempDir()
	origDir, _ := os.Getwd()
	defer os.Chdir(origDir)
	os.Chdir(dir)

	res := &config.DispatchResult{
		Timestamp: time.Now(),
		Duration:  "1s",
		Nodes: []config.NodeResult{
			{IP: "192.168.1.1", Role: []string{"master"}, Status: "success"},
		},
		Summary: config.DispatchSummary{
			TotalNodes:   1,
			SuccessNodes: 1,
			Result:       "success",
		},
	}

	err := generateDispatchHTML(res)
	assert.NoError(t, err)

	_, err = os.Stat("dispatch.html")
	assert.NoError(t, err)
}

func TestFormatDispatchResult(t *testing.T) {
	res := &config.DispatchResult{
		Timestamp: time.Now(),
		Duration:  "1s",
		Nodes: []config.NodeResult{
			{IP: "192.168.1.1", Role: []string{"master"}, Status: "success"},
		},
		Summary: config.DispatchSummary{
			TotalNodes:   1,
			SuccessNodes: 1,
			Result:       "success",
		},
	}

	t.Run("text format", func(t *testing.T) {
		output, err := formatDispatchResult(res, "text")
		assert.NoError(t, err)
		assert.NotEmpty(t, output)
	})

	t.Run("json format", func(t *testing.T) {
		output, err := formatDispatchResult(res, "json")
		assert.NoError(t, err)
		assert.NotEmpty(t, output)
		assert.True(t, json.Valid([]byte(output)))
	})
}

func TestDoDispatch_EmptyHosts(t *testing.T) {
	log := logger.NewLogger("")
	cfg := &config.AppConfig{}
	result, err := doDispatch(nil, cfg, log)
	assert.Nil(t, result)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no hosts configured")
}

func TestRunDiskCommand(t *testing.T) {
	origCfgFile := cfgFile
	origDir, _ := os.Getwd()
	dir, _ := os.MkdirTemp("", "envcheck-disk-*")
	t.Cleanup(func() {
		cfgFile = origCfgFile
		os.Chdir(origDir)
		os.RemoveAll(dir)
	})
	cfgFile = "/nonexistent/config.json"
	os.Chdir(dir)

	err := runDisk(nil, nil)
	assert.NoError(t, err)
	_, err = os.Stat("disk.html")
	assert.NoError(t, err)
}

func TestRunKernelCommand(t *testing.T) {
	origCfgFile := cfgFile
	origDir, _ := os.Getwd()
	dir, _ := os.MkdirTemp("", "envcheck-kernel-*")
	t.Cleanup(func() {
		cfgFile = origCfgFile
		os.Chdir(origDir)
		os.RemoveAll(dir)
	})
	cfgFile = "/nonexistent/config.json"
	os.Chdir(dir)

	err := runKernel(nil, nil)
	assert.NoError(t, err)
	_, err = os.Stat("kernel.html")
	assert.NoError(t, err)
}

func TestRunPortCommand(t *testing.T) {
	origCfgFile := cfgFile
	origRoles := roles
	origDir, _ := os.Getwd()
	dir, _ := os.MkdirTemp("", "envcheck-port-*")
	t.Cleanup(func() {
		cfgFile = origCfgFile
		roles = origRoles
		os.Chdir(origDir)
		os.RemoveAll(dir)
	})
	cfgFile = "/nonexistent/config.json"
	roles = ""
	os.Chdir(dir)

	err := runPort(nil, nil)
	assert.NoError(t, err)
	_, err = os.Stat("port.html")
	assert.NoError(t, err)
}

func TestRunDispatchCommand(t *testing.T) {
	origCfgFile := cfgFile
	origDir, _ := os.Getwd()
	dir, _ := os.MkdirTemp("", "envcheck-dispatch-*")
	t.Cleanup(func() {
		cfgFile = origCfgFile
		os.Chdir(origDir)
		os.RemoveAll(dir)
	})
	cfgFile = "/nonexistent/config.json"
	os.Chdir(dir)

	err := runDispatch(nil, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no hosts configured")
}

func TestRunChecksAllTypes(t *testing.T) {
	log := logger.NewLogger("")
	cfg := &config.AppConfig{
		Paths: []string{"/nonexistent/path"},
		ProgramList: []config.ProgramConfig{
			{Name: "nonexistent-prog-test", ShouldExist: false},
		},
	}

	checks := []string{"kernel", "port", "disk", "clock", "fileQuery", "programCheck", "route", "iptables"}
	results := runChecks(cfg, log, []string{"master"}, checks)
	assert.Len(t, results, len(checks))
	for _, r := range results {
		assert.NotEmpty(t, r.CheckType)
		assert.NotEmpty(t, r.Status)
	}
}
