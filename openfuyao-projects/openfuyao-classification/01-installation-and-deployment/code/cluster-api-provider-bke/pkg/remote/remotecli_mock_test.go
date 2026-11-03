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
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	gosftp "github.com/pkg/sftp"
	gossh "golang.org/x/crypto/ssh"
)

func TestNewRemoteClient_Mock(t *testing.T) {
	patches := gomonkey.ApplyFunc(NewSSHClient, func(host *Host) (*Ssh, error) {
		return &Ssh{sshClient: &gossh.Client{}, alive: true}, nil
	})
	patches.ApplyFunc(NewSFTPClient, func(sshClient *gossh.Client) (*Sftp, error) {
		return &Sftp{sftpClient: &gosftp.Client{}, alive: true}, nil
	})
	defer patches.Reset()

	host := &Host{
		User:     "root",
		Password: "pass",
		Address:  "192.168.1.1",
		Port:     "22",
	}

	client, err := NewRemoteClient(host)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if client == nil {
		t.Error("expected non-nil client")
	}
}
