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

package agentssh

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"

	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	bkessh "gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/remote"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

// DiscoverArchs connects via SSH and returns unique architectures for reachable nodes.
func DiscoverArchs(ctx context.Context, nodes bkenode.Nodes, logger *log.Logger) (map[string]struct{}, map[string]error, error) {
	multiCli, pushErrs, err := newSSHMultiCli(ctx, nodes, logger, "no available hosts for arch discovery")
	if err != nil {
		return nil, pushErrs, err
	}
	defer multiCli.Close()

	for nodeIP, err := range multiCli.RegisterHostsInfo() {
		pushErrs[nodeIP] = err
	}

	archs := make(map[string]struct{})
	for hostIP, arch := range multiCli.NodeArchByAddress() {
		arch = strings.TrimSpace(arch)
		if arch == "" || arch == "unknown" {
			pushErrs[hostIP] = errors.Errorf("unknown architecture for node %s", hostIP)
			continue
		}
		archs[arch] = struct{}{}
	}
	return archs, pushErrs, nil
}

// SSHUpgrade pushes staged binaries and service files to nodes and restarts bkeagent.
func SSHUpgrade(ctx context.Context, nodes bkenode.Nodes, staging *Staging, logger *log.Logger) (map[string]error, error) {
	if staging == nil {
		return nil, errors.New("staging is required")
	}

	multiCli, pushErrs, err := newSSHMultiCli(ctx, nodes, logger, "no available hosts to upgrade")
	if err != nil {
		return pushErrs, err
	}
	defer multiCli.Close()

	for nodeIP, err := range multiCli.RegisterHostsInfo() {
		pushErrs[nodeIP] = err
	}
	if len(multiCli.AvailableHosts()) == 0 {
		return pushErrs, errors.New("no available hosts after arch detection")
	}

	if err := executeUpgradePreCommand(multiCli, pushErrs, logger); err != nil {
		return pushErrs, err
	}

	multiCli.RegisterHostsCustomCmdFunc(upgradeHostFileFunc(staging.Dir))
	defer multiCli.RemoveHostsCustomCmdFunc()

	if err := executeUpgradeStartCommand(multiCli, staging.ServicePath, pushErrs, logger); err != nil {
		return pushErrs, err
	}

	postCommand := bkessh.Command{
		Cmds: bkessh.Commands{
			"chmod 755 /usr/local/bin/",
			"chmod 755 /etc/systemd/system/",
		},
	}
	stdErrs, _ := multiCli.Run(postCommand)
	for nodeIP, serrs := range stdErrs.Out() {
		if logger != nil {
			logger.Warn("BKEAgentSSHPush", "node %s post command err: %v", nodeIP, serrs.String())
		}
		pushErrs[nodeIP] = errors.Errorf("post command failed on %s: %s", nodeIP, serrs.String())
	}

	return pushErrs, nil
}

func newSSHMultiCli(
	ctx context.Context,
	nodes bkenode.Nodes,
	logger *log.Logger,
	noHostsMsg string,
) (*bkessh.MultiCli, map[string]error, error) {
	hosts := phaseutil.NodeToRemoteHost(nodes)
	multiCli := bkessh.NewMultiCli(ctx)
	if logger != nil {
		multiCli.SetLogger(logger)
	}

	pushErrs := make(map[string]error)
	for hostIP, err := range multiCli.RegisterHosts(hosts) {
		pushErrs[hostIP] = err
	}
	if len(multiCli.AvailableHosts()) == 0 {
		multiCli.Close()
		return nil, pushErrs, errors.New(noHostsMsg)
	}
	return multiCli, pushErrs, nil
}

func upgradeHostFileFunc(stagingDir string) func(host *bkessh.Host) bkessh.Command {
	return func(host *bkessh.Host) bkessh.Command {
		arch := host.Extra["arch"]
		binaryName := fmt.Sprintf("bkeagent_linux_%s", arch)
		src := filepath.Join(stagingDir, arch, binaryName)
		return bkessh.Command{
			FileUp: []bkessh.File{{Src: src, Dst: "/usr/local/bin/"}},
			Cmds: bkessh.Commands{
				fmt.Sprintf("echo %s > /etc/openFuyao/bkeagent/node", host.Extra["hostname"]),
			},
		}
	}
}

func executeUpgradePreCommand(multiCli *bkessh.MultiCli, pushErrs map[string]error, logger *log.Logger) error {
	if pushErrs == nil {
		return errors.New("pushErrs map is nil")
	}

	preCommand := bkessh.Command{
		Cmds: bkessh.Commands{
			"systemctl stop bkeagent 2>&1 >/dev/null || true",
			"cp -f /usr/local/bin/bkeagent /usr/local/bin/bkeagent.bak.$(date +%s) 2>/dev/null || true",
		},
	}

	stdErrs, _ := multiCli.Run(preCommand)
	for nodeIP, serrs := range stdErrs.Out() {
		if logger != nil {
			logger.Warn("BKEAgentSSHPush", "node %s pre command err: %v", nodeIP, serrs.String())
		}
		pushErrs[nodeIP] = errors.Errorf("pre command failed on %s: %s", nodeIP, serrs.String())
		multiCli.RemoveHost(nodeIP)
	}

	if len(multiCli.AvailableHosts()) == 0 {
		return errors.New("no available hosts after pre command")
	}
	return nil
}

func executeUpgradeStartCommand(
	multiCli *bkessh.MultiCli,
	servicePath string,
	pushErrs map[string]error,
	logger *log.Logger,
) error {
	if pushErrs == nil {
		return errors.New("pushErrs map is nil")
	}

	startCommand := bkessh.Command{
		FileUp: []bkessh.File{
			{Src: servicePath, Dst: "/etc/systemd/system"},
		},
		Cmds: bkessh.Commands{
			"mv -f /usr/local/bin/bkeagent_* /usr/local/bin/bkeagent",
			"chmod +x /usr/local/bin/bkeagent",
			"systemctl daemon-reload 2>&1 >/dev/null",
			"systemctl enable bkeagent 2>&1 >/dev/null || true",
			"systemctl restart bkeagent 2>&1 >/dev/null",
		},
	}

	stdErrs, _ := multiCli.Run(startCommand)
	for nodeIP, serrs := range stdErrs.Out() {
		if logger != nil {
			logger.Warn("BKEAgentSSHPush", "node %s start err: %v", nodeIP, serrs.String())
		}

		var tmpErrs []string
		for _, err := range serrs {
			if strings.Contains(err.Out, "Created symlink") || strings.Contains(err.Out, "File exists") {
				continue
			}
			tmpErrs = append(tmpErrs, err.Out)
		}
		if len(tmpErrs) == 0 {
			delete(pushErrs, nodeIP)
			continue
		}
		pushErrs[nodeIP] = errors.New(strings.Join(tmpErrs, ";"))
		multiCli.RemoveHost(nodeIP)
	}

	if len(multiCli.AvailableHosts()) == 0 {
		return errors.New("no available hosts after upgrade start")
	}
	return nil
}

// ArchsFromMap converts arch set to slice.
func ArchsFromMap(archs map[string]struct{}) []string {
	out := make([]string, 0, len(archs))
	for arch := range archs {
		out = append(out, arch)
	}
	return out
}
