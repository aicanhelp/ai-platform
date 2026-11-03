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
	"fmt"
	"sync"

	"github.com/pkg/errors"
	"golang.org/x/sync/semaphore"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/config"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	CheckArchCommand     = "echo $(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/;s/^unknown$/amd64/')"
	SudoCheckArchCommand = "sudo echo $(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/;s/^unknown$/amd64/')"
)

type MultiCli struct {
	remotes     map[string]*HostRemoteClient
	ctx         context.Context
	cancleFunc  context.CancelFunc
	concurrency int
	log         *log.Logger
}

// NewMultiCli creates a new multi-client.
func NewMultiCli(ctx context.Context) *MultiCli {
	cancelCtx, cancelFunc := context.WithCancel(ctx)
	return &MultiCli{
		remotes:     make(map[string]*HostRemoteClient),
		ctx:         cancelCtx,
		cancleFunc:  cancelFunc,
		concurrency: config.BkeClusterConcurrency,
		log:         log.With("module", "MultiCli"),
	}
}

// SetLogger sets the logger for the multi-client.
func (c *MultiCli) SetLogger(l *log.Logger) {
	c.log = l
}

// RegisterHosts registers hosts for the multi-client.
func (c *MultiCli) RegisterHosts(hosts []Host) map[string]error {
	if len(hosts) == 0 {
		return map[string]error{}
	}

	errs := make(map[string]error)

	for i := range hosts {
		// 直接使用hosts切片中的元素地址，避免range循环变量重用导致的指针问题
		host := &hosts[i]

		client, err := NewRemoteClient(host)
		if err != nil {
			errs[host.Address] = err
			continue
		}
		client.SetLogger(c.log)
		c.remotes[host.Address] = client
	}

	log.Infof("registered %d hosts, %d failed", len(c.remotes), len(errs))
	return errs
}

// AvailableHosts returns the available hosts for the multi-client.
//
//go:noinline
func (c *MultiCli) AvailableHosts() []string {
	var hosts []string
	for k := range c.remotes {
		hosts = append(hosts, k)
	}
	return hosts
}

// RemoveHost removes a host from the multi-client.
func (c *MultiCli) RemoveHost(hostIP string) {
	delete(c.remotes, hostIP)
}

// RegisterCustomCmdFunc registers a custom command function for a host.
func (c *MultiCli) RegisterCustomCmdFunc(hostIP string, f func(host *Host) Command) {
	c.remotes[hostIP].host.ExtraCustomCmdFunc = f
}

// RegisterHostsCustomCmdFunc registers a custom command function for all hosts.
func (c *MultiCli) RegisterHostsCustomCmdFunc(f func(host *Host) Command) {
	for _, remote := range c.remotes {
		remote.host.ExtraCustomCmdFunc = f
	}
}

// RemoveCustomCmdFunc removes a custom command function for a host.
func (c *MultiCli) RemoveCustomCmdFunc(hostIP string) {
	c.remotes[hostIP].host.ExtraCustomCmdFunc = nil
}

// RemoveHostsCustomCmdFunc removes a custom command function for all hosts.
func (c *MultiCli) RemoveHostsCustomCmdFunc() {
	for _, remote := range c.remotes {
		remote.host.ExtraCustomCmdFunc = nil
	}
}

// Run runs a command on all hosts.
func (c *MultiCli) Run(cmd Command) (stdErrs StdCombine, stdOuts StdCombine) {
	stdErrs = NewStdCombine()
	stdOuts = NewStdCombine()

	if len(c.remotes) == 0 {
		return
	}

	stdErrChan := make(chan CombineOut)
	stdOutChan := make(chan CombineOut)
	stopChan := make(chan struct{})

	go func() {
		for {
			select {
			case stdErr := <-stdErrChan:
				stdErrs.Add(stdErr)
			case stdOut := <-stdOutChan:
				stdOuts.Add(stdOut)
			case <-stopChan:
				return
			case <-c.ctx.Done():
				return
			default:
				// do nothing
			}
		}
	}()

	var wg sync.WaitGroup
	sem := semaphore.NewWeighted(int64(c.concurrency))

	for _, remoteCli := range c.remotes {
		// 深拷贝cmd，避免多个goroutine共享同一个FileUp和Cmds切片
		cmdBak := Command{
			Cmds:   make(Commands, len(cmd.Cmds)),
			FileUp: make([]File, len(cmd.FileUp)),
		}
		copy(cmdBak.Cmds, cmd.Cmds)
		copy(cmdBak.FileUp, cmd.FileUp)
		// 添加额外命令,针对单个节点自己的
		if remoteCli.host.ExtraCustomCmdFunc != nil {
			extraCmd := remoteCli.host.ExtraCustomCmdFunc(remoteCli.host)
			cmdBak.Cmds = append(cmdBak.Cmds, extraCmd.Cmds...)
			cmdBak.FileUp = append(cmdBak.FileUp, extraCmd.FileUp...)
		}
		cmdBak.Sudo(remoteCli.host.User)

		wg.Add(1)
		go func(remoteCli *HostRemoteClient) {
			defer wg.Done()
			if err := sem.Acquire(c.ctx, 1); err != nil {
				log.Errorf("Failed to acquire semaphore: %v", err)
				stdErrs.Add(NewCombineOut(remoteCli.host.Address, "", "failed to acquire semaphore"))
			}
			defer sem.Release(1)
			remoteCli.Exec(c.ctx, cmdBak, stdErrChan, stdOutChan)

		}(remoteCli)
	}

	wg.Wait()

	stopChan <- struct{}{}
	return stdErrs, stdOuts
}

// RegisterHostsInfo registers hosts info for the multi-client.
func (c *MultiCli) RegisterHostsInfo() map[string]error {
	checkCommand := Command{
		Cmds: Commands{
			CheckArchCommand,
		},
	}

	stdErrs, stdOuts := c.Run(checkCommand)

	errs := make(map[string]error)

	for nodeIP, err := range stdErrs.Out() {
		errs[nodeIP] = fmt.Errorf("Register host %s info failed, err: %v", nodeIP, err)
	}

	stdout := stdOuts.Out()
	var hostsExit []string

	for _, remotecli := range c.remotes {
		host := remotecli.host
		if v, ok := stdout[host.Address]; ok {
			if (v[0].Command == CheckArchCommand || v[0].Command == SudoCheckArchCommand) && len(v) > 0 {
				host.Extra["arch"] = v[0].Out
			}
			if host.Extra["arch"] == "unknown" {
				hostsExit = append(hostsExit, host.Address)
			}
		}
	}

	if len(hostsExit) > 0 {
		for _, nodeIP := range hostsExit {
			errs[nodeIP] = errors.Errorf("Register host %s info failed, unknown arch", nodeIP)
		}
	}

	for errHost, _ := range errs {
		delete(c.remotes, errHost)
	}

	return errs
}

// NodeArchByAddress returns architecture per connected host after RegisterHostsInfo.
func (c *MultiCli) NodeArchByAddress() map[string]string {
	archs := make(map[string]string, len(c.remotes))
	for addr, remote := range c.remotes {
		if remote.host == nil {
			continue
		}
		archs[addr] = remote.host.Extra["arch"]
	}
	return archs
}

// Close closes the multi-client.
func (c *MultiCli) Close() {
	log.Infof("closing multi-client with %d remotes", len(c.remotes))
	c.cancleFunc()
	for _, remote := range c.remotes {
		_ = remote.CloseRemoteCli()
	}
}
