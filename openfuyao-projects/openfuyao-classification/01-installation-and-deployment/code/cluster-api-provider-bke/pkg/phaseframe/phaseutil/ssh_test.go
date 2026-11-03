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

package phaseutil

import (
	"testing"

	"github.com/stretchr/testify/assert"

	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkessh "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/remote"
)

func TestNodeToRemoteHost(t *testing.T) {
	nodes := bkenode.Nodes{
		{IP: "192.168.1.1", Username: "root", Port: "22", Password: "pass", Hostname: "node1"},
	}
	hosts := NodeToRemoteHost(nodes)
	assert.Len(t, hosts, 1)
	assert.Equal(t, "192.168.1.1", hosts[0].Address)
	assert.Equal(t, "root", hosts[0].User)
}

func TestSetRemoteHostArch(t *testing.T) {
	hosts := []bkessh.Host{
		{Address: "192.168.1.1", Extra: map[string]string{"arch": "unknown"}},
	}
	stdout := map[string]bkessh.CombineOuts{
		"192.168.1.1": {{Command: CheckArchCommand, Out: "amd64"}},
	}
	result := SetRemoteHostArch(hosts, stdout)
	if len(result) == 0 {
		t.Error("expected result")
	}
}

func TestHostArchExit(t *testing.T) {
	hosts := []bkessh.Host{
		{Address: "192.168.1.1", Extra: map[string]string{"arch": "unknown"}},
		{Address: "192.168.1.2", Extra: map[string]string{"arch": "amd64"}},
	}
	result := HostArchExit(hosts)
	if len(result) != 1 {
		t.Errorf("expected 1, got %d", len(result))
	}
}

func TestHostCustomCmdFunc(t *testing.T) {
	host := &bkessh.Host{
		Address: "192.168.1.1",
		Extra:   map[string]string{"hostname": "node1", "arch": "amd64"},
	}
	cmd := HostCustomCmdFunc(host)
	if len(cmd.Cmds) == 0 {
		t.Error("expected commands")
	}
}

func TestImmutableHostNodeFileCmdFunc(t *testing.T) {
	host := &bkessh.Host{
		Address: "192.168.1.1",
		Extra:   map[string]string{"hostname": "worker-1", "arch": "amd64"},
	}
	cmd := ImmutableHostNodeFileCmdFunc(host)

	// 应该有 2 条命令：写 node 文件 + restart bkeagent
	assert.Len(t, cmd.Cmds, 2)
	// 第一条命令写 node 文件，包含正确的 hostname
	assert.Contains(t, cmd.Cmds[0], "worker-1")
	assert.Contains(t, cmd.Cmds[0], "/etc/openFuyao/bkeagent/node")
	// 第二条命令重启 bkeagent
	assert.Contains(t, cmd.Cmds[1], "systemctl restart bkeagent")
	// 不应上传二进制文件
	assert.Empty(t, cmd.FileUp)
}

func TestImmutableHostNodeFileCmdFunc_EmptyHostname(t *testing.T) {
	host := &bkessh.Host{
		Address: "192.168.1.1",
		Extra:   map[string]string{"hostname": "", "arch": "amd64"},
	}
	cmd := ImmutableHostNodeFileCmdFunc(host)
	// 即使 hostname 为空，命令结构也应正确
	assert.Len(t, cmd.Cmds, 2)
	assert.Empty(t, cmd.FileUp)
}
