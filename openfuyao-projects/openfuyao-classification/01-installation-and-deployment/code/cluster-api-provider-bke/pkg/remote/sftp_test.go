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

	"github.com/agiledragon/gomonkey/v2"
	gosftp "github.com/pkg/sftp"
	gossh "golang.org/x/crypto/ssh"
)

func TestSftp_Close(t *testing.T) {
	sftp := &Sftp{
		sftpClient: nil,
		alive:      false,
	}
	err := sftp.Close()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestSftp_UploadFile_NotAlive(t *testing.T) {
	sftp := &Sftp{
		sftpClient: nil,
		alive:      false,
	}
	err := sftp.UploadFile("/tmp/test", "/remote/path")
	if err == nil {
		t.Error("expected error when sftp not alive")
	}
}

func TestSftp_UploadFile_NilClient(t *testing.T) {
	sftp := &Sftp{
		sftpClient: nil,
		alive:      true,
	}
	err := sftp.UploadFile("/tmp/test", "/remote/path")
	if err == nil {
		t.Error("expected error when sftp client is nil")
	}
}

func TestNewSFTPClient_Mock(t *testing.T) {
	patches := gomonkey.ApplyFunc(gosftp.NewClient, func(conn *gossh.Client, opts ...gosftp.ClientOption) (*gosftp.Client, error) {
		return &gosftp.Client{}, nil
	})
	defer patches.Reset()

	client, err := NewSFTPClient(&gossh.Client{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if client == nil {
		t.Error("expected non-nil client")
	}
	if !client.alive {
		t.Error("expected client to be alive")
	}
}
