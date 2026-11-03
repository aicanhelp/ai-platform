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
	"fmt"

	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	bkessh "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/remote"
)

const (
	CheckArchCommand = "echo $(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/;s/^unknown$/amd64/')"
)

// NodeToRemoteHost converts node struct to remote host struct
func NodeToRemoteHost(nodes bkenode.Nodes) []bkessh.Host {
	var hosts []bkessh.Host
	for _, node := range nodes.Decrypt() {
		host := bkessh.Host{
			User:     node.Username,
			Address:  node.IP,
			Port:     node.Port,
			Password: node.Password,
			SSHKey:   nil,
			Extra: map[string]string{
				"hostname": node.Hostname,
				"arch":     "unknown",
			},
		}
		hosts = append(hosts, host)
	}
	return hosts
}

// SetRemoteHostArch sets remote host arch
func SetRemoteHostArch(srcHosts []bkessh.Host, stdout map[string]bkessh.CombineOuts) []bkessh.Host {
	for _, host := range srcHosts {
		if v, ok := stdout[host.Address]; ok {
			if v[0].Command == CheckArchCommand && len(v) > 0 {
				host.Extra["arch"] = v[0].Out
			}
		}
	}
	return srcHosts
}

func HostArchExit(hosts []bkessh.Host) []string {
	var hostsExit []string
	for _, host := range hosts {
		if host.Extra["arch"] == "unknown" {
			hostsExit = append(hostsExit, host.Address)
		}
	}
	return hostsExit
}

func HostCustomCmdFunc(host *bkessh.Host) bkessh.Command {
	c := []string{
		fmt.Sprintf("echo %s > /etc/openFuyao/bkeagent/node", host.Extra["hostname"]),
	}
	expectAgent := fmt.Sprintf("/bkeagent_linux_%s", host.Extra["arch"])
	f := []bkessh.File{
		{Src: expectAgent, Dst: "/usr/local/bin/"},
	}
	return bkessh.Command{
		Cmds:   c,
		FileUp: f,
	}
}

// ImmutableHostNodeFileCmdFunc 不可变 OS 模式下专用的 custom cmd func。
// KubeOS 镜像内置 bkeagent，开机即启动，此时 hostname 仍为模板默认值（如 localhost.localdomain），
// 会被写入 /etc/openFuyao/bkeagent/node 文件。此处用 BKENode 配置的正确 hostname 覆写 node 文件，
// 避免 checkHost 阶段 expectedBKENodeName 对比失败。不上传二进制（镜像已内置）。
// 同时负责重启 bkeagent，确保重启时读取到正确的 node 文件。
func ImmutableHostNodeFileCmdFunc(host *bkessh.Host) bkessh.Command {
	c := []string{
		fmt.Sprintf("echo %s > /etc/openFuyao/bkeagent/node", host.Extra["hostname"]),
		"systemctl restart bkeagent 2>&1 >/dev/null",
	}
	return bkessh.Command{
		Cmds: c,
	}
}
