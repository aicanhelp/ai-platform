/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package remote

import (
	"context"
	"testing"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

func TestNewMultiCli(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	if mc == nil {
		t.Fatal("NewMultiCli returned nil")
	}
	if mc.remotes == nil {
		t.Error("remotes should not be nil")
	}
	if mc.ctx == nil {
		t.Error("ctx should not be nil")
	}
}

func TestMultiCli_SetLogger(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	var logger *log.Logger = nil
	mc.SetLogger(logger)
	if mc.log != logger {
		t.Error("logger not set correctly")
	}
}

func TestMultiCli_RegisterHosts_Empty(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	errs := mc.RegisterHosts([]Host{})
	if len(errs) != 0 {
		t.Errorf("expected 0 errors, got %d", len(errs))
	}
}

func TestMultiCli_AvailableHosts(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	hosts := mc.AvailableHosts()
	if len(hosts) != 0 {
		t.Errorf("expected 0 hosts, got %d", len(hosts))
	}
}

func TestMultiCli_RemoveHost(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	mc.RemoveHost("192.168.1.1")
}

func TestMultiCli_Run_EmptyRemotes(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	cmd := Command{Cmds: Commands{"ls"}}
	stdErrs, stdOuts := mc.Run(cmd)
	if stdErrs.Len() != 0 {
		t.Errorf("expected 0 errors, got %d", stdErrs.Len())
	}
	if stdOuts.Len() != 0 {
		t.Errorf("expected 0 outputs, got %d", stdOuts.Len())
	}
}

func TestMultiCli_Close(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	mc.Close()
}

func TestMultiCli_RegisterCustomCmdFunc(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	mc.remotes["192.168.1.1"] = &HostRemoteClient{
		host: &Host{Address: "192.168.1.1"},
	}
	mc.RegisterCustomCmdFunc("192.168.1.1", func(host *Host) Command {
		return Command{Cmds: Commands{"echo test"}}
	})
	if mc.remotes["192.168.1.1"].host.ExtraCustomCmdFunc == nil {
		t.Error("custom cmd func not set")
	}
}

func TestMultiCli_RegisterHostsCustomCmdFunc(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	mc.remotes["192.168.1.1"] = &HostRemoteClient{
		host: &Host{Address: "192.168.1.1"},
	}
	mc.RegisterHostsCustomCmdFunc(func(host *Host) Command {
		return Command{Cmds: Commands{"echo test"}}
	})
	if mc.remotes["192.168.1.1"].host.ExtraCustomCmdFunc == nil {
		t.Error("custom cmd func not set")
	}
}

func TestMultiCli_RemoveCustomCmdFunc(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	mc.remotes["192.168.1.1"] = &HostRemoteClient{
		host: &Host{
			Address: "192.168.1.1",
			ExtraCustomCmdFunc: func(host *Host) Command {
				return Command{}
			},
		},
	}
	mc.RemoveCustomCmdFunc("192.168.1.1")
	if mc.remotes["192.168.1.1"].host.ExtraCustomCmdFunc != nil {
		t.Error("custom cmd func not removed")
	}
}

func TestMultiCli_RemoveHostsCustomCmdFunc(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	mc.remotes["192.168.1.1"] = &HostRemoteClient{
		host: &Host{
			Address: "192.168.1.1",
			ExtraCustomCmdFunc: func(host *Host) Command {
				return Command{}
			},
		},
	}
	mc.RemoveHostsCustomCmdFunc()
	if mc.remotes["192.168.1.1"].host.ExtraCustomCmdFunc != nil {
		t.Error("custom cmd func not removed")
	}
}

func TestMultiCli_RegisterHosts_InvalidHost(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	hosts := []Host{
		{User: "", Password: "pass", Address: "192.168.1.1", Port: "22"},
	}
	errs := mc.RegisterHosts(hosts)
	if len(errs) == 0 {
		t.Error("expected errors for invalid host")
	}
}

func TestMultiCli_AvailableHosts_WithHosts(t *testing.T) {
	ctx := context.Background()
	mc := NewMultiCli(ctx)
	mc.remotes["192.168.1.1"] = &HostRemoteClient{
		host: &Host{Address: "192.168.1.1"},
	}
	mc.remotes["192.168.1.2"] = &HostRemoteClient{
		host: &Host{Address: "192.168.1.2"},
	}
	hosts := mc.AvailableHosts()
	if len(hosts) != 2 {
		t.Errorf("expected 2 hosts, got %d", len(hosts))
	}
}
