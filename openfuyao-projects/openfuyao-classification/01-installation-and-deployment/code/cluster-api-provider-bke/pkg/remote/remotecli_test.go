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
	"time"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const defaultTimeout = 10 * time.Second

func TestRemoteSSHExec(t *testing.T) {
	// Skip this test as it requires network access to a real SSH server
	t.Skip("Skipping test that requires network access to real SSH server")

	host := &Host{
		User:     "root",
		Port:     "22",
		Password: "Ccc51521!",
		Address:  "172.28.100.206",
	}

	cmd := Command{
		Cmds: []string{
			"echo $(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/;s/^unknown$/amd64/')",
		},
	}

	cli, err := NewRemoteClient(host)
	if err != nil {
		t.Fatal(err)
	}

	stdErrChan := make(chan CombineOut)
	stdOutChan := make(chan CombineOut)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel() // 确保函数结束时取消 context

	go func() {
		// 创建一个可复用的定时器
		timer := time.NewTimer(defaultTimeout)
		defer timer.Stop() // 确保定时器被正确停止

		for {
			// 重置定时器
			if !timer.Stop() {
				select {
				case <-timer.C:
				default:
				}
			}
			timer.Reset(defaultTimeout)

			select {
			case <-ctx.Done():
				t.Log("goroutine exiting due to context cancellation")
				return
			case stdErr := <-stdErrChan:
				t.Logf("stdErr: %s", stdErr)
			case stdOut := <-stdOutChan:
				t.Logf("stdOut: %s", stdOut)
			case <-timer.C:
				return
			}
		}
	}()

	cli.Exec(context.Background(), cmd, stdErrChan, stdOutChan)
}

func TestMultiCli(t *testing.T) {
	commands1 := Command{
		Cmds: []string{
			"sleep 2",
			"echo $(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/;s/^unknown$/amd64/')",
			"echo abc",
		},
	}

	commands2 := Command{
		Cmds: []string{
			"sleep 5",
			"sleep 10",
		},
	}

	hosts := []Host{
		{
			User:     "root",
			Address:  "172.28.100.206",
			Port:     "22",
			Password: "Ccc51521!",
		},
		{
			User:     "root",
			Address:  "172.28.100.220",
			Port:     "22",
			Password: "Ccc51521!",
		},
	}

	ctx, cancel := context.WithCancel(context.Background())

	multiCli := NewMultiCli(ctx)
	errs := multiCli.RegisterHosts(hosts)
	for _, err := range errs {
		log.Errorf("failed to register hosts: %v", err)
	}

	go func() {
		log.Infof("start")
		<-time.After(8 * time.Second)
		log.Infof("cancel")
		cancel()
	}()

	stdErrs, stdOuts := multiCli.Run(commands1)
	for _, err := range stdErrs.Out() {
		for _, line := range err {
			log.Error(line)
		}
	}
	for _, outs := range stdOuts.Out() {
		for _, line := range outs {
			log.Info(line.String())
		}
	}

	stdErrs, stdOuts = multiCli.Run(commands2)
	for _, err := range stdErrs.Out() {
		for _, line := range err {
			log.Error(line)
		}
	}
	for _, outs := range stdOuts.Out() {
		for _, line := range outs {
			log.Info(line.String())
		}
	}
}
