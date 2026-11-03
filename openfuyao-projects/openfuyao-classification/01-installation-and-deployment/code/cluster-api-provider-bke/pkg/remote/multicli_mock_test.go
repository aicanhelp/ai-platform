/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package remote

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	gosftp "github.com/pkg/sftp"
	gossh "golang.org/x/crypto/ssh"
)

func TestMultiCli_RegisterHosts_Mock(t *testing.T) {
	patches := gomonkey.ApplyFunc(NewRemoteClient, func(h *Host) (*HostRemoteClient, error) {
		return &HostRemoteClient{
			host:       h,
			SshClient:  &Ssh{sshClient: &gossh.Client{}, alive: true},
			SftpClient: &Sftp{sftpClient: &gosftp.Client{}, alive: true},
			log:        nil,
		}, nil
	})
	defer patches.Reset()

	ctx := context.Background()
	mc := NewMultiCli(ctx)

	hosts := []Host{
		{User: "root", Password: "pass", Address: "192.168.1.1", Port: "22"},
	}

	errs := mc.RegisterHosts(hosts)
	if len(errs) != 0 {
		t.Errorf("expected no errors, got %d", len(errs))
	}
	if len(mc.remotes) != 1 {
		t.Errorf("expected 1 remote, got %d", len(mc.remotes))
	}
}
