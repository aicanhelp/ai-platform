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
	"testing"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

func TestNewRemoteClient_InvalidHost(t *testing.T) {
	host := &Host{
		User:     "",
		Password: "pass",
		Address:  "192.168.1.1",
		Port:     "22",
	}
	_, err := NewRemoteClient(host)
	if err == nil {
		t.Error("expected error for invalid host")
	}
}

func TestHostRemoteClient_SetLogger(t *testing.T) {
	client := &HostRemoteClient{
		host: &Host{Address: "192.168.1.1"},
	}
	var logger *log.Logger = log.With("test", "remotecli")
	client.SetLogger(logger)
	if client.log == nil {
		t.Error("logger not set")
	}
}

func TestHostRemoteClient_CloseRemoteCli(t *testing.T) {
	client := &HostRemoteClient{
		SshClient:  &Ssh{sshClient: nil, alive: false},
		SftpClient: &Sftp{sftpClient: nil, alive: false},
	}
	err := client.CloseRemoteCli()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
